#!/usr/bin/env python3
"""Field-testing plots for the hook estimator, with NO ground truth available.

Everything here is derived from signals the drone produces on its own, so it
works on the real rig where hook_ground_truth_base_flat does not exist. The
questions it is built to answer, in order of how often they bite:

  1. Is YOLO actually seeing the hook?      -> detection health
  2. Does the filter follow the detections? -> detection vs estimate, innovation
  3. Is the filter confident or diverging?  -> sigma over time
  4. Is the payload swinging as modelled?   -> swing angles vs pendulum period
  5. Does the drone follow the commands?    -> command tracking

Nothing is required. Run the filter alone and you get plots 1-4; add the
mission and plot 5 appears too. Any stream that produced no samples is skipped
with a log line instead of raising, so a partial test still writes everything
it can.

Writes on Ctrl+C, and whenever the save_live_testing_plots service is called.
"""

import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')            # headless: field laptop, no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

import rclpy                                                        # noqa: E402
from rclpy.node import Node                                         # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy  # noqa: E402
from rclpy.signals import SignalHandlerOptions                      # noqa: E402

from geometry_msgs.msg import Vector3Stamped                        # noqa: E402
from nav_msgs.msg import Odometry                                   # noqa: E402
from sensor_msgs.msg import CameraInfo, JointState                  # noqa: E402
from std_msgs.msg import Float64MultiArray                          # noqa: E402
from std_srvs.srv import Trigger                                    # noqa: E402

from dji_msgs.msg import Topics                                     # noqa: E402
from smarc_msgs.msg import Topics as SmarcTopics                    # noqa: E402

# yolo_msgs lives in a submodule that is not always present (CI does not check
# it out). The detection-health plots are the only thing that needs it, so a
# missing package costs those plots and nothing else.
try:
    from yolo_msgs.msg import DetectionArray
    _HAVE_YOLO_MSGS = True
except ImportError:                     # pragma: no cover
    DetectionArray = None
    _HAVE_YOLO_MSGS = False

NAN = float('nan')


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _col(samples: list, key: str) -> np.ndarray:
    return np.array([s.get(key, NAN) for s in samples], dtype=float)


def _rel_time(samples: list, t0: float) -> np.ndarray:
    return _col(samples, 't') - t0


def _rate_hz(t: np.ndarray, window: float = 2.0) -> "tuple[np.ndarray, np.ndarray]":
    """Sliding-window message rate. Returns (t, hz), empty if too few samples."""
    if t.size < 2:
        return np.array([]), np.array([])
    out_t, out_hz = [], []
    for i, ti in enumerate(t):
        lo = ti - window
        n = int(np.count_nonzero((t >= lo) & (t <= ti)))
        span = min(window, ti - t[0])
        if span > 0.2:
            out_t.append(ti)
            out_hz.append(n / span)
    return np.array(out_t), np.array(out_hz)


class LiveTestingDebugPlots:

    def __init__(self, node: Node, robot_name: str, output_dir: str):
        self._node = node
        self._robot_name = robot_name
        self._output_dir = output_dir

        self._est: list = []        # hook_state          - filtered estimate
        self._raw: list = []        # hook_raw_measurement- per-detection, pre-filter
        self._swing: list = []      # hook_swing_state    - angles/rates + variances
        self._det: list = []        # yolo/detections     - pixels + score
        self._cmd: list = []        # cmd_vel_drone_frame - what the mission asked for
        self._odom: list = []       # smarc/odom          - what the drone did
        self._L: "float|None" = None
        self._xi: "float|None" = None
        self._image_wh: "tuple[int, int]|None" = None

        self._create_subscriptions()

        self._save_service = node.create_service(
            Trigger, 'save_live_testing_plots', self._on_save_requested)

        node.get_logger().info(
            f'Recording field-test plots for {robot_name} -> {output_dir}. '
            f'Call {node.get_namespace().rstrip("/")}/save_live_testing_plots, '
            f'or press Ctrl+C, to write them.')
        if not _HAVE_YOLO_MSGS:
            node.get_logger().warning(
                'yolo_msgs not importable - detection-health plots will be skipped.')

    # ---------------------------------------------------------------- inputs

    def _create_subscriptions(self):
        node = self._node

        # The hook topics are published BEST_EFFORT. A RELIABLE subscriber gets
        # NOTHING from them, silently - this has already cost a full test run.
        best_effort = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                                 durability=QoSDurabilityPolicy.VOLATILE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        node.create_subscription(Odometry, Topics.HOOK_STATE_CARTESIAN,
                                 self._hook_state_cb, best_effort)
        node.create_subscription(Odometry, Topics.HOOK_RAW_MEASUREMENT,
                                 self._hook_raw_cb, best_effort)
        node.create_subscription(JointState, Topics.HOOK_STATE_ANGULAR,
                                 self._swing_cb, best_effort)
        node.create_subscription(Float64MultiArray, Topics.HOOK_PENDULUM_PARAMETERS,
                                 self._params_cb, latched)
        node.create_subscription(Vector3Stamped, Topics.CMD_VELOCITY_DRONE_FRAME,
                                 self._cmd_cb, best_effort)
        node.create_subscription(Odometry, SmarcTopics.ODOM_TOPIC,
                                 self._odom_cb, 10)
        node.create_subscription(CameraInfo, Topics.GIMBAL_CAMERA_INFO_TOPIC,
                                 self._camera_info_cb, 10)

        if _HAVE_YOLO_MSGS:
            node.create_subscription(DetectionArray, Topics.YOLO_DETECTIONS,
                                     self._detections_cb, 10)

    def _hook_state_cb(self, msg: Odometry):
        self._est.append({
            't': _stamp_to_sec(msg.header.stamp),
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'vx': msg.twist.twist.linear.x,
            'vy': msg.twist.twist.linear.y,
            # HookKalmanFilter writes the linearised cartesian variances here.
            'var_x': msg.pose.covariance[0],
            'var_y': msg.pose.covariance[7],
        })

    def _hook_raw_cb(self, msg: Odometry):
        self._raw.append({
            't': _stamp_to_sec(msg.header.stamp),
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
        })

    def _swing_cb(self, msg: JointState):
        def at(seq, i):
            return float(seq[i]) if len(seq) > i else NAN
        self._swing.append({
            't': _stamp_to_sec(msg.header.stamp),
            'theta_x': at(msg.position, 0), 'theta_y': at(msg.position, 1),
            'omega_x': at(msg.velocity, 0), 'omega_y': at(msg.velocity, 1),
            # effort carries the UNlinearised angle variances.
            'var_theta_x': at(msg.effort, 0), 'var_theta_y': at(msg.effort, 1),
        })

    def _params_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 2:
            self._L, self._xi = float(msg.data[0]), float(msg.data[1])
            self._node.get_logger().info(
                f'Pendulum params for the plots: L={self._L:.3f} m, xi={self._xi:.4f}')

    def _cmd_cb(self, msg: Vector3Stamped):
        self._cmd.append({'t': _stamp_to_sec(msg.header.stamp),
                          'vx': msg.vector.x, 'vy': msg.vector.y})

    def _odom_cb(self, msg: Odometry):
        self._odom.append({'t': _stamp_to_sec(msg.header.stamp),
                           'vx': msg.twist.twist.linear.x,
                           'vy': msg.twist.twist.linear.y})

    def _camera_info_cb(self, msg: CameraInfo):
        if msg.width > 0 and msg.height > 0:
            self._image_wh = (int(msg.width), int(msg.height))

    def _detections_cb(self, msg):
        hooks = [d for d in msg.detections if d.class_name == 'hook']
        t = _stamp_to_sec(msg.header.stamp)
        if not hooks:
            # Recorded as a frame with zero hooks: dropouts are the single most
            # useful thing to see in the field, so they must not be invisible.
            self._det.append({'t': t, 'u': NAN, 'v': NAN, 'score': NAN, 'n': 0})
            return
        self._det.append({
            't': t,
            'u': sum(float(d.bbox.center.position.x) for d in hooks) / len(hooks),
            'v': sum(float(d.bbox.center.position.y) for d in hooks) / len(hooks),
            'score': max(float(d.score) for d in hooks),
            'n': len(hooks),
        })

    # ----------------------------------------------------------------- plots

    def _t0(self) -> "float|None":
        starts = [s[0]['t'] for s in (self._est, self._raw, self._swing,
                                      self._det, self._cmd, self._odom) if s]
        return min(starts) if starts else None

    def plot_detection_vs_estimate(self, out: str, t0: float):
        """The one that matters most: does the filter follow the detections?"""
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        t_est, x_est, y_est = _rel_time(self._est, t0), _col(self._est, 'x'), _col(self._est, 'y')
        sx = np.sqrt(np.clip(_col(self._est, 'var_x'), 0, None))
        sy = np.sqrt(np.clip(_col(self._est, 'var_y'), 0, None))
        t_raw, x_raw, y_raw = _rel_time(self._raw, t0), _col(self._raw, 'x'), _col(self._raw, 'y')

        for ax, est, raw, sig, name in ((axes[0], x_est, x_raw, sx, 'x'),
                                        (axes[1], y_est, y_raw, sy, 'y')):
            if self._raw:
                ax.plot(t_raw, raw, color='tab:blue', marker='.', linestyle='none',
                        markersize=4, alpha=0.6, label='hook_raw_measurement (per detection)')
            if self._est:
                ax.plot(t_est, est, color='crimson', linewidth=1.4, label='hook_state (estimate)')
                if np.any(np.isfinite(sig)):
                    ax.fill_between(t_est, est - sig, est + sig, color='crimson',
                                    alpha=0.18, linewidth=0, label='estimate ±1σ')
            ax.set_ylabel(f'{name} [m]')
            ax.grid(alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        axes[1].set_xlabel('time [s]')
        fig.suptitle(f'Hook position in {self._robot_name}/base_flat_link — '
                     f'detection vs filtered estimate (no ground truth)')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'detection_vs_estimate.png'), dpi=150)
        plt.close(fig)

    def plot_filter_health(self, out: str, t0: float):
        """Without ground truth, the filter's own consistency is the evidence:
        innovation should look like zero-mean noise, sigma should settle."""
        fig, axs = plt.subplots(2, 2, figsize=(13, 8))

        t_est = _rel_time(self._est, t0)
        t_raw = _rel_time(self._raw, t0)

        # -- innovation: measurement minus estimate, interpolated onto the raw stamps
        ax = axs[0][0]
        if self._est and self._raw and len(self._est) > 1:
            ex = np.interp(t_raw, t_est, _col(self._est, 'x'))
            ey = np.interp(t_raw, t_est, _col(self._est, 'y'))
            ix, iy = _col(self._raw, 'x') - ex, _col(self._raw, 'y') - ey
            ax.plot(t_raw, ix, color='tab:blue', marker='.', linestyle='none',
                    markersize=3, label=f'x  (mean {np.nanmean(ix):+.3f}, std {np.nanstd(ix):.3f})')
            ax.plot(t_raw, iy, color='tab:orange', marker='.', linestyle='none',
                    markersize=3, label=f'y  (mean {np.nanmean(iy):+.3f}, std {np.nanstd(iy):.3f})')
            ax.axhline(0.0, color='gray', linewidth=0.8)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'needs hook_state + hook_raw_measurement',
                    ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_title('Innovation (measurement − estimate)\nshould be zero-mean noise')
        ax.set_xlabel('time [s]'); ax.set_ylabel('[m]'); ax.grid(alpha=0.3)

        # -- estimate uncertainty over time
        ax = axs[0][1]
        if self._est:
            ax.plot(t_est, np.sqrt(np.clip(_col(self._est, 'var_x'), 0, None)),
                    color='crimson', label='σx')
            ax.plot(t_est, np.sqrt(np.clip(_col(self._est, 'var_y'), 0, None)),
                    color='darkorange', label='σy')
            ax.legend(fontsize=8)
        ax.set_title('Estimate uncertainty\nrising = filter losing the hook')
        ax.set_xlabel('time [s]'); ax.set_ylabel('σ [m]'); ax.grid(alpha=0.3)

        # -- measurement rate: gaps are the field failure mode
        ax = axs[1][0]
        if self._raw:
            tr, hz = _rate_hz(t_raw)
            if tr.size:
                ax.plot(tr, hz, color='tab:blue', label='accepted measurements')
        if self._det:
            td, hzd = _rate_hz(_rel_time(self._det, t0))
            if td.size:
                ax.plot(td, hzd, color='tab:green', linestyle='--', label='yolo frames')
        ax.set_title('Update rate\ngap between the two = detections rejected')
        ax.set_xlabel('time [s]'); ax.set_ylabel('Hz'); ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

        # -- estimate trajectory, top down
        ax = axs[1][1]
        if self._raw:
            ax.plot(_col(self._raw, 'x'), _col(self._raw, 'y'), color='tab:blue',
                    marker='.', linestyle='none', markersize=3, alpha=0.5,
                    label='raw measurement')
        if self._est:
            ax.plot(_col(self._est, 'x'), _col(self._est, 'y'), color='crimson',
                    linewidth=1.2, label='estimate')
        if self._L:
            circle = plt.Circle((0, 0), self._L, fill=False, color='gray',
                                linestyle=':', label=f'rope L={self._L:.2f} m')
            ax.add_patch(circle)
        ax.set_title('Hook path in base_flat_link (top down)')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        fig.suptitle(f'Filter health — {self._robot_name}')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'filter_health.png'), dpi=150)
        plt.close(fig)

    def plot_detection_health(self, out: str, t0: float):
        """Is YOLO seeing the hook, how confidently, and is it near the edge of
        frame (about to be lost)?"""
        fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        t_det = _rel_time(self._det, t0)
        u, v = _col(self._det, 'u'), _col(self._det, 'v')
        n, score = _col(self._det, 'n'), _col(self._det, 'score')

        ax = axs[0]
        ax.plot(t_det, u, color='tab:purple', marker='.', linestyle='none',
                markersize=3, label='u (px)')
        ax.plot(t_det, v, color='tab:brown', marker='.', linestyle='none',
                markersize=3, label='v (px)')
        if self._image_wh:
            w, h = self._image_wh
            ax.axhline(w, color='tab:purple', linestyle=':', linewidth=0.8)
            ax.axhline(h, color='tab:brown', linestyle=':', linewidth=0.8)
            ax.axhline(0, color='gray', linewidth=0.8)
            ax.set_title(f'Hook centre in image ({w}×{h}) — near a dotted line = about to leave frame')
        else:
            ax.set_title('Hook centre in image [px]')
        ax.set_ylabel('pixel'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

        ax = axs[1]
        ax.plot(t_det, score, color='tab:green', marker='.', linestyle='none', markersize=3)
        ax.set_ylim(0, 1.02)
        ax.set_title('Detection confidence — a downward drift means the model is losing the hook')
        ax.set_ylabel('score'); ax.grid(alpha=0.3)

        ax = axs[2]
        ax.step(t_det, n, where='post', color='tab:red')
        missed = int(np.count_nonzero(n == 0))
        ax.set_title(f'Hooks detected per frame — {missed} of {n.size} frames had none')
        ax.set_ylabel('count'); ax.set_xlabel('time [s]'); ax.grid(alpha=0.3)

        fig.suptitle(f'YOLO detection health — {self._robot_name}')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'detection_health.png'), dpi=150)
        plt.close(fig)

    def plot_swing(self, out: str, t0: float):
        fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        t = _rel_time(self._swing, t0)

        ax = axs[0]
        tx, ty = _col(self._swing, 'theta_x'), _col(self._swing, 'theta_y')
        stx = np.sqrt(np.clip(_col(self._swing, 'var_theta_x'), 0, None))
        sty = np.sqrt(np.clip(_col(self._swing, 'var_theta_y'), 0, None))
        ax.plot(t, tx, color='tab:blue', label='theta_x')
        ax.plot(t, ty, color='tab:orange', label='theta_y')
        if np.any(np.isfinite(stx)):
            ax.fill_between(t, tx - stx, tx + stx, color='tab:blue', alpha=0.15, linewidth=0)
            ax.fill_between(t, ty - sty, ty + sty, color='tab:orange', alpha=0.15, linewidth=0)
        ax.axhline(0.0, color='gray', linewidth=0.8)
        ax.set_ylabel('angle [rad]'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        title = 'Swing angles ±1σ'
        if self._L:
            period = 2 * np.pi * np.sqrt(self._L / 9.81)
            title += f' — pendulum period for L={self._L:.2f} m is {period:.2f} s'
        ax.set_title(title)

        ax = axs[1]
        ax.plot(t, _col(self._swing, 'omega_x'), color='tab:blue', label='omega_x')
        ax.plot(t, _col(self._swing, 'omega_y'), color='tab:orange', label='omega_y')
        ax.axhline(0.0, color='gray', linewidth=0.8)
        ax.set_ylabel('rate [rad/s]'); ax.set_xlabel('time [s]')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        fig.suptitle(f'Payload swing — {self._robot_name}')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'swing_state.png'), dpi=150)
        plt.close(fig)

    def plot_command_tracking(self, out: str, t0: float):
        """Only meaningful once a mission is running - skipped otherwise."""
        fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for ax, key, name in ((axs[0], 'vx', 'x'), (axs[1], 'vy', 'y')):
            if self._cmd:
                ax.plot(_rel_time(self._cmd, t0), _col(self._cmd, key),
                        color='tab:red', label='cmd_vel_drone_frame (commanded)')
            if self._odom:
                ax.plot(_rel_time(self._odom, t0), _col(self._odom, key),
                        color='tab:blue', alpha=0.8, label='smarc/odom (actual)')
            ax.set_ylabel(f'v{name} [m/s]'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axs[1].set_xlabel('time [s]')
        fig.suptitle(f'Command tracking — {self._robot_name}')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'command_tracking.png'), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ save

    def _run(self, label: str, fn) -> str:
        try:
            fn()
            return f'{label}: ok'
        except Exception as exc:                        # never lose the other plots
            self._node.get_logger().error(f'{label} plot failed: {exc}')
            return f'{label}: FAILED ({exc})'

    def save_everything(self) -> "tuple[bool, str]":
        t0 = self._t0()
        if t0 is None:
            msg = 'No samples on any topic - nothing to plot.'
            self._node.get_logger().warning(msg)
            return False, msg

        out = os.path.join(self._output_dir,
                           datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(out, exist_ok=True)

        results = []
        # Each block is gated on the data it needs, so running the filter alone
        # still produces everything except command tracking.
        if self._est or self._raw:
            results.append(self._run('detection_vs_estimate',
                                     lambda: self.plot_detection_vs_estimate(out, t0)))
            results.append(self._run('filter_health',
                                     lambda: self.plot_filter_health(out, t0)))
        else:
            results.append('detection_vs_estimate/filter_health: skipped (no hook_state)')

        if self._det:
            results.append(self._run('detection_health',
                                     lambda: self.plot_detection_health(out, t0)))
        else:
            results.append('detection_health: skipped (no yolo detections)')

        if self._swing:
            results.append(self._run('swing_state',
                                     lambda: self.plot_swing(out, t0)))
        else:
            results.append('swing_state: skipped (no hook_swing_state)')

        if self._cmd or self._odom:
            results.append(self._run('command_tracking',
                                     lambda: self.plot_command_tracking(out, t0)))
        else:
            results.append('command_tracking: skipped (no mission running)')

        counts = (f'samples: est={len(self._est)} raw={len(self._raw)} '
                  f'swing={len(self._swing)} det={len(self._det)} '
                  f'cmd={len(self._cmd)} odom={len(self._odom)}')
        summary = f'Wrote plots to {out}. {counts}. ' + '; '.join(results)
        self._node.get_logger().info(summary)
        return True, summary

    def _on_save_requested(self, request, response):
        ok, message = self.save_everything()
        response.success = ok
        response.message = message
        return response


def main():
    # rclpy's own SIGINT handler races with the `except KeyboardInterrupt`
    # below, and losing that race means losing the plots for the whole run.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    node = Node('live_testing_debug_plots')
    node.declare_parameter('robot_name', 'M350')
    node.declare_parameter('plot_output_dir', '/home/aleba/field_test_plots')

    plotter = LiveTestingDebugPlots(
        node,
        robot_name=node.get_parameter('robot_name').value,
        output_dir=node.get_parameter('plot_output_dir').value,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C received - writing plots before shutdown...')
        plotter.save_everything()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
