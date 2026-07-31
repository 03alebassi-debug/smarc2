from enum import Enum, auto

import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import TwistStamped
from dji_msgs.msg import Topics as DJITopics
from dji_msgs.msg import Links as DJILinks
from dji_msgs.msg import LabeledOBBs

from smarc_action_base.gentler_action_server import GentlerActionServer


class _Phase(Enum):
    CAPTURE_EQUILIBRIUM = auto()
    EXCITING = auto()
    COLLECTING = auto()


class EstimateLengthAndDamping:
    """Action server that identifies the hook's effective pendulum length (L)
    and damping ratio (xi): commands a short velocity step to excite a swing,
    then fits the free-decay oscillation observed via YOLO hook detections.
    Meant to be run once, before HookKalmanFilter, to obtain real L/xi values
    instead of hand-picked guesses.

    Action: "estimate_length_and_damping" (smarc_msgs/action/BaseAction, JSON goal/result)

    Goal fields (all optional):
        excitation_speed    (float, m/s, default 1.0)
        excitation_duration (float, s,   default 2.0)
        collection_duration (float, s,   default 20.0)  - max time to observe decay
        min_periods         (int,        default 4)      - stop early once this many periods are seen
        refractory_window   (float, s,   default 1.5)    - minimum spacing between accepted extrema
        smoothing_window    (int,        default 5)      - moving-average window (in samples) applied
                                                             to the raw measurement before extremum detection,
                                                             to reject detection jitter mistaken for real swings

    Result fields:
        success (bool), length (float, m), damping (float), message (str)
    """

    G = 9.81

    def __init__(self, node: Node, robot_name: str, output_path: "str|None" = None):
        self._node: Node = node
        self._robot_name: str = robot_name
        self._output_path: "str|None" = output_path

        self.BASE_FLAT_FRAME: str = self._robot_name + '/' + DJILinks.BASE_FLAT

        # Both image axes are tracked, and which one carries the swing is decided
        # from the data - see _select_axis. Fitting the period on a hardcoded
        # image axis is fragile: with the gimbal pointed down, image-horizontal
        # maps to base_flat Y while the excitation below drives body X (which
        # appears on image-vertical), so the original code measured the period on
        # the axis it does NOT excite - the one dominated by detection noise.
        self._last_x: "float|None" = None   # image horizontal
        self._last_y: "float|None" = None   # image vertical
        self._new_detection: bool = False   # set by _detection_callback, consumed once
        self._axis_index: "int|None" = None  # 0 = horizontal, 1 = vertical
        self._axis_buffer: "list[tuple[float, float, float]]" = []

        self._create_subscriptions()
        self._create_publishers()

        self._as = GentlerActionServer(
            self._node,
            self._robot_name + '/estimate_length_and_damping',
            self._on_goal_received,
            self._on_cancel_received,
            self._prepare_loop,
            self._loop_inner,
            self._give_feedback,
            loop_frequency=50
        )

    @property
    def now_time(self) -> float:
        t = self._node.get_clock().now().to_msg()
        return t.sec + t.nanosec * 1e-9

    def log(self, msg: str):
        self._node.get_logger().info(msg)

    def _create_subscriptions(self):
        _detection_topic_name = self._robot_name + '/' + DJITopics.LABELED_OBBS_TOPIC
        self._detection_subscription = self._node.create_subscription(
            LabeledOBBs, _detection_topic_name, self._detection_callback, 10
        )

    def _create_publishers(self):
        qos_best_effort10 = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                                        durability=QoSDurabilityPolicy.VOLATILE)
        self._ref_publisher = self._node.create_publisher(
            TwistStamped, self._robot_name + '/' + DJITopics.VELOCITY_SETPOINT_TOPIC, qos_profile=qos_best_effort10
        )

    def _detection_callback(self, msg):
        hook_indices = [i for i, cls_id in enumerate(msg.ids) if cls_id == "hook"]
        if not hook_indices:
            return

        norm_x = 0.0
        norm_y = 0.0
        for idx in hook_indices:
            pts = msg.obbs[idx].points
            norm_x += sum(p.x for p in pts) / len(pts)
            norm_y += sum(p.y for p in pts) / len(pts)
        norm_x /= len(hook_indices)
        norm_y /= len(hook_indices)

        self._last_x = norm_x
        self._last_y = norm_y
        self._new_detection = True

    def _publish_velocity_setpoint(self, vx: float, vy: float):
        setpoint = TwistStamped()
        setpoint.header.stamp = self._node.get_clock().now().to_msg()
        setpoint.header.frame_id = self.BASE_FLAT_FRAME
        setpoint.twist.linear.x = vx
        setpoint.twist.linear.y = vy
        self._ref_publisher.publish(setpoint)

    # --- GentlerActionServer callbacks ---

    def _on_goal_received(self, goal_request: dict) -> bool:
        try:
            self._excitation_speed    = float(goal_request.get('excitation_speed', 1.0))
            self._excitation_duration = float(goal_request.get('excitation_duration', 3.0))
            self._collection_duration = float(goal_request.get('collection_duration', 20.0))
            self._min_periods         = int(goal_request.get('min_periods', 4))
            self._refractory_window   = float(goal_request.get('refractory_window', 1.5))
            self._smoothing_window    = int(goal_request.get('smoothing_window', 5))
            # How long to watch both image axes before deciding which one the
            # swing is on (see _select_axis). Needs to cover a decent fraction of
            # a period; the pendulum here is ~4-5s.
            self._axis_selection_duration = float(goal_request.get('axis_selection_duration', 2.0))
            return True
        except Exception:
            self._node.get_logger().error("Failed to parse goal request")
            return False

    def _on_cancel_received(self) -> bool:
        self._publish_velocity_setpoint(0.0, 0.0)
        return True

    def _prepare_loop(self):
        self._phase = _Phase.CAPTURE_EQUILIBRIUM
        self._phase_start_time = self.now_time
        self._equilibrium: "float|None" = None
        self._result = {"success": False, "length": 0.0, "damping": 0.0, "message": ""}
        self._reset_period_estimation()

    def _give_feedback(self) -> str:
        if self._phase == _Phase.CAPTURE_EQUILIBRIUM:
            return "Waiting for a hook detection to establish equilibrium..."
        if self._phase == _Phase.EXCITING:
            return f"Exciting: {self.now_time - self._phase_start_time:.1f}/{self._excitation_duration:.1f}s"
        if self._phase == _Phase.COLLECTING:
            return (f"Collecting: {self.now_time - self._phase_start_time:.1f}/{self._collection_duration:.1f}s, "
                    f"periods seen: {self._n_periods}, current period estimate: {self._period_estimate}")
        return "Done"

    def _loop_inner(self) -> "bool|None":
        now = self.now_time
        elapsed = now - self._phase_start_time

        if self._phase == _Phase.CAPTURE_EQUILIBRIUM:
            if self._last_x is None or self._last_y is None:
                if elapsed > 5.0:
                    self._result["message"] = "No hook detection available to establish equilibrium"
                    return False
                return None
            # Set once the swing axis is known, in _select_axis.
            self._equilibrium = None
            self._phase = _Phase.EXCITING
            self._phase_start_time = now
            return None

        if self._phase == _Phase.EXCITING:
            if elapsed < self._excitation_duration:
                self._publish_velocity_setpoint(self._excitation_speed, 0.0)
                return None
            self._publish_velocity_setpoint(0.0, 0.0)
            self._phase = _Phase.COLLECTING
            self._phase_start_time = now
            return None

        if self._phase == _Phase.COLLECTING:
            # Only consume a sample when the detector has actually produced a NEW
            # one. This loop runs far faster than detections arrive, and feeding
            # the same value repeatedly builds flat plateaus into the smoothed
            # signal, which defeats the strict slope-sign extremum test below
            # (it needs slope_before > 0 AND slope_after < 0; a plateau gives 0)
            # - so no extrema are found, no periods are measured, and L comes out
            # as "no full period observed". This was masked while the node was
            # clock-starved to ~3Hz, i.e. roughly the detection rate itself.
            if self._last_x is not None and self._last_y is not None and self._new_detection:
                self._new_detection = False
                if self._axis_index is None:
                    # Still deciding which image axis actually carries the swing.
                    self._axis_buffer.append((now, self._last_x, self._last_y))
                    if now - self._phase_start_time >= self._axis_selection_duration:
                        self._select_axis()
                else:
                    self._process_measurement(now, (self._last_x, self._last_y)[self._axis_index])

            enough_periods = self._n_periods >= self._min_periods
            if elapsed >= self._collection_duration or enough_periods:
                return self._finalize()
            return None

        return None

    def _finalize(self) -> bool:
        if self._period_estimate is None:
            self._result["message"] = "No full period observed during collection window"
            return False

        wn = 2 * np.pi / self._period_estimate
        length = self.G / wn**2
        xi = self._estimate_damping(wn)

        self._result = {
            "success": True,
            "length": float(length),
            "damping": float(xi),
            "message": f"L={length:.3f}m, xi={xi:.4f}, from {self._n_periods} periods"
        }
        self.log(self._result["message"])

        if self._output_path is not None:
            self._save_result(length, xi)

        return True

    def _save_result(self, length: float, xi: float):
        import yaml
        data = {"robot_name": self._robot_name, "length": float(length), "damping": float(xi)}
        with open(self._output_path, 'w') as f:
            yaml.safe_dump(data, f)
        self.log(f"Saved identified L/xi to {self._output_path}")

    # --- Period estimation: moving-average smoothing + slope-sign extremum
    #     detection + refractory window, first period taken as prior,
    #     subsequent periods incrementally averaged ---

    def _select_axis(self):
        """Decide which image axis the swing is actually on, from the first
        `axis_selection_duration` of free decay, then replay that buffer through
        the extremum detector so nothing is lost.

        Measured from the data rather than hardcoded: the image->body mapping
        depends on the gimbal's current orientation (with it pointed down,
        image-horizontal is body Y and image-vertical is body X), so the axis the
        excitation lands on is not fixed. Picking the quieter axis is not a small
        error - in the test run its swing was ~60x smaller than the excited one
        and sat right at the detection noise floor (SNR ~1), so the period fit
        would be measuring noise."""
        if not self._axis_buffer:
            self._axis_index = 0
            self._equilibrium = self._last_x
            return

        xs = [b[1] for b in self._axis_buffer]
        ys = [b[2] for b in self._axis_buffer]
        ptp_x = max(xs) - min(xs)
        ptp_y = max(ys) - min(ys)
        self._axis_index = 0 if ptp_x >= ptp_y else 1
        name = "image-horizontal" if self._axis_index == 0 else "image-vertical"
        self.log(f'Fitting the period on the {name} axis '
                 f'(peak-to-peak: horizontal {ptp_x:.4f}, vertical {ptp_y:.4f})')
        if min(ptp_x, ptp_y) > 0.0 and max(ptp_x, ptp_y) / max(min(ptp_x, ptp_y), 1e-9) < 2.0:
            self.log('WARNING: the two axes swing by similar amounts - the motion may '
                     'not be planar, so the period fit could be unreliable')

        series = xs if self._axis_index == 0 else ys
        self._equilibrium = series[0]
        for t, bx, by in self._axis_buffer:
            self._process_measurement(t, bx if self._axis_index == 0 else by)
        self._axis_buffer = []

    def _reset_period_estimation(self):
        self._raw_buffer: "list[float]" = []
        self._window: "list[tuple[float,float]]" = []
        self._extrema: "list[tuple[float,float,bool]]" = []
        self._last_extremum_time: "float|None" = None
        self._period_estimate: "float|None" = None
        self._n_periods: int = 0
        self._axis_index = None
        self._axis_buffer = []
        self._new_detection = False

    def _process_measurement(self, t: float, x: float):
        # Smooth first: raw single-frame detections are noisy enough that,
        # applied directly, the slope-sign check below fires on jitter many
        # times per real swing instead of once - a moving average over the
        # last few samples suppresses that without needing a huge refractory
        # window to compensate.
        self._raw_buffer.append(x)
        if len(self._raw_buffer) > self._smoothing_window:
            self._raw_buffer.pop(0)
        smoothed_x = float(np.mean(self._raw_buffer))

        self._window.append((t, smoothed_x))
        if len(self._window) > 3:
            self._window.pop(0)
        if len(self._window) < 3:
            return

        (_, x0), (t1, x1), (_, x2) = self._window
        slope_before = x1 - x0
        slope_after  = x2 - x1
        is_max = slope_before > 0 and slope_after < 0
        is_min = slope_before < 0 and slope_after > 0
        if not (is_max or is_min):
            return

        if self._last_extremum_time is not None and (t1 - self._last_extremum_time) < self._refractory_window:
            return  # too soon since the last accepted extremum - likely noise, reject

        self._accept_extremum(t1, x1, is_max)

    def _accept_extremum(self, t: float, x: float, is_max: bool):
        self._extrema.append((t, x, is_max))
        self._last_extremum_time = t

        # Pair maxima with maxima (and minima with minima) using the slope-sign
        # classification computed above, NOT sign(x - equilibrium). The
        # equilibrium is a single instantaneous pre-excitation sample, and the
        # hook's rest position in base_flat_link shifts once the drone moves to
        # excite it - so that comparison could put peaks *and* troughs on the
        # same side, making this a peak-to-trough (i.e. HALF) period and
        # underestimating L by ~4x (L scales with T^2). The slope-sign
        # classification is immune to any such offset or drift.
        same_side_prev = next(
            ((pt, px) for pt, px, pmax in reversed(self._extrema[:-1]) if pmax == is_max),
            None
        )
        if same_side_prev is None:
            return  # first extremum of this type - nothing to compare against yet

        period = t - same_side_prev[0]

        if self._period_estimate is None:
            self._period_estimate = period
            self._n_periods = 1
            self.log(f'First period (prior): {period:.3f}s')
        else:
            self._n_periods += 1
            # incremental mean: avg += (new - avg) / n
            self._period_estimate += (period - self._period_estimate) / self._n_periods
            self.log(f'Period #{self._n_periods}: {period:.3f}s, running average: {self._period_estimate:.3f}s')

    def _estimate_damping(self, wn: float) -> float:
        """The decaying-amplitude envelope |x(t) - centre| = A0*exp(-xi*wn*t)
        passes through *every* extremum (peaks and troughs alike), so fit it via
        linear regression of ln(amplitude) vs time across all collected extrema
        rather than a single peak-to-peak ratio. Slope of that fit = -xi*wn."""
        if len(self._extrema) < 3:
            return 0.0

        # Measure amplitudes about the midpoint of the observed swing, not about
        # self._equilibrium: that is a single instantaneous pre-excitation sample
        # and the rest position shifts once the drone moves, so an offset there
        # biases every amplitude and can flatten the fit (a plausible cause of
        # the "xi always 0" results seen before).
        maxima = [x for _, x, is_max in self._extrema if is_max]
        minima = [x for _, x, is_max in self._extrema if not is_max]
        if maxima and minima:
            centre = 0.5 * (float(np.mean(maxima)) + float(np.mean(minima)))
        else:
            centre = float(np.mean([x for _, x, _ in self._extrema]))

        times = np.array([t for t, _, _ in self._extrema])
        amps  = np.array([abs(x - centre) for _, x, _ in self._extrema])
        amps  = np.clip(amps, 1e-9, None)  # guard against log(0)

        slope, _ = np.polyfit(times, np.log(amps), 1)
        return max(0.0, float(-slope / wn))
