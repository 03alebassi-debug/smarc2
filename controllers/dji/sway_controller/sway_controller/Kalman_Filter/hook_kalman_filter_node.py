#!/usr/bin/env python3

import os

import control as ct
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from smarc_msgs.action import BaseAction
from std_msgs.msg import String

from sway_controller.Kalman_Filter.ekf_ground_truth_plotter import save_all_plots
from sway_controller.HookKalmanFilter import HookKalmanFilter


def _load_identified_gains(model_path: str):
    """Extract (k_x, tau_x, k_y, tau_y) from the same identified continuous
    model used by alars_move_to_dumped_action_server._build_transfer_function /
    _get_node_parameters, so both controllers always agree on the drone's
    velocity-response dynamics."""
    d = np.load(model_path)

    tf_x = ct.tf(
        d['b__FLU_axes_0__to__FLUvelocity_ground_fused_x'],
        d['a__FLU_axes_0__to__FLUvelocity_ground_fused_x'],
    )
    tf_y = ct.tf(
        d['b__FLU_axes_1__to__FLUvelocity_ground_fused_y'],
        d['a__FLU_axes_1__to__FLUvelocity_ground_fused_y'],
    )

    k_x = float(tf_x.num[0][0][0])
    tau_x = float(tf_x.den[0][0][1])
    k_y = float(tf_y.num[0][0][0])
    tau_y = float(tf_y.den[0][0][1])

    return k_x, tau_x, k_y, tau_y


def _run_identification_action(node: Node, robot_name: str, timeout_sec: float = 60.0) -> bool:
    """Calls estimate_length_and_damping and blocks (via spin_until_future_complete,
    since nothing is spinning the node yet at this point in startup) until it
    finishes. BaseAction's result only carries a plain bool - the actual L/xi
    values are read from the file estimate_length_and_damping_node saves on
    success, not from this action's result itself. Returns True on success."""
    action_name = f'{robot_name}/estimate_length_and_damping'
    client = ActionClient(node, BaseAction, action_name)

    node.get_logger().info(f'Waiting for action server {action_name}...')
    if not client.wait_for_server(timeout_sec=timeout_sec):
        node.get_logger().warning(f'Action server {action_name} not available after {timeout_sec}s')
        return False

    goal = BaseAction.Goal()
    goal.goal = String(data='{}')

    node.get_logger().info('Requesting hook pendulum identification (this will excite a swing)...')
    send_goal_future = client.send_goal_async(
        goal,
        feedback_callback=lambda fb: node.get_logger().info(
            f'[identification] {fb.feedback.feedback.data}', throttle_duration_sec=1.0
        )
    )
    rclpy.spin_until_future_complete(node, send_goal_future, timeout_sec=timeout_sec)
    goal_handle = send_goal_future.result()
    if goal_handle is None or not goal_handle.accepted:
        node.get_logger().warning('estimate_length_and_damping goal was rejected')
        return False

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_sec)
    response = result_future.result()
    if response is None:
        node.get_logger().warning('estimate_length_and_damping did not complete in time')
        return False

    return bool(response.result.success)


def _odom_to_sample(msg: Odometry) -> dict:
    return {
        't': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        'x': msg.pose.pose.position.x,
        'y': msg.pose.pose.position.y,
        'vx': msg.twist.twist.linear.x,
        'vy': msg.twist.twist.linear.y,
        'vz': msg.twist.twist.linear.z,
    }


def main():
    # Disable rclpy's own automatic SIGINT handling: by default it can call
    # rclpy.shutdown() internally on Ctrl+C, racing with our own except
    # KeyboardInterrupt below and causing "rcl_shutdown already called" plus
    # "publisher's context is invalid" once it wins that race. With this,
    # our own try/except/finally is the only thing that shuts things down.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    node = Node("hook_kalman_filter_node")

    node.declare_parameter("robot_name", "M350")
    # use_sim_time is auto-declared by rclpy's Node base class already.
    node.declare_parameter("loop_freq", 50)
    # Negative = "not explicitly overridden" - falls back to the identified
    # values from estimate_length_and_damping_node if available, see below.
    node.declare_parameter("L", -1.0)
    node.declare_parameter("xi", -1.0)
    node.declare_parameter("qc", 0.01)
    node.declare_parameter("sigma_initial", 1.0)
    node.declare_parameter("mahalanobis_thr", 16.0)
    # Detections are dropped when the camera boresight is further than this from
    # straight-down: the pendulum-angle measurement silently degenerates there.
    node.declare_parameter("max_boresight_tilt_deg", 45.0)
    # Empty by default: resolved below against dji_captain's installed share
    # directory (same model alars_move_to_dumped_action_server uses), unless
    # explicitly overridden.
    node.declare_parameter("continuous_model_path", "")
    node.declare_parameter("ground_truth_topic", "hook_ground_truth_base_flat")
    node.declare_parameter("plot_output_dir", "/home/aleba/ekf_plots")

    robot_name = node.get_parameter("robot_name").value

    # Recording is set up before identification even starts (not just before the
    # final spin), and L is bound before entering the try block - so a Ctrl+C at
    # *any* point below (identification included) hits the except clause cleanly
    # with well-defined variables, instead of crashing with a raw traceback and
    # nothing saved.
    ground_truth_topic = node.get_parameter("ground_truth_topic").value
    plot_output_dir = node.get_parameter("plot_output_dir").value

    gt_samples: list = []
    est_samples: list = []
    raw_samples: list = []
    L = None

    # hook_state is published BEST_EFFORT (HookKalmanFilter._create_publishers) -
    # a RELIABLE subscriber (the default for a plain integer QoS depth) is
    # incompatible with a BEST_EFFORT publisher in ROS2/DDS and silently
    # receives nothing, so this must match.
    qos_best_effort10 = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                                    durability=QoSDurabilityPolicy.VOLATILE)

    node.create_subscription(
        Odometry, f'{robot_name}/{ground_truth_topic}',
        lambda msg: gt_samples.append(_odom_to_sample(msg)), 10
    )
    node.create_subscription(
        Odometry, f'{robot_name}/hook_state',
        lambda msg: est_samples.append(_odom_to_sample(msg)), qos_best_effort10
    )
    # Raw per-detection measurement (pre-fusion, pre-gating) - the clean signal
    # for checking the camera->base_flat_link axis mapping against ground truth.
    node.create_subscription(
        Odometry, f'{robot_name}/hook_raw_measurement',
        lambda msg: raw_samples.append(_odom_to_sample(msg)), qos_best_effort10
    )

    try:
        continuous_model_path = node.get_parameter("continuous_model_path").value
        if not continuous_model_path:
            continuous_model_path = os.path.join(
                get_package_share_directory("dji_captain"),
                "models", robot_name,
                "continuous_model", "model_bla_diag_cmdvle.npz",
            )

        k_x, tau_x, k_y, tau_y = _load_identified_gains(continuous_model_path)
        node.get_logger().info(
            f"Loaded identified gains from {continuous_model_path}: "
            f"k_x={k_x}, tau_x={tau_x}, k_y={k_y}, tau_y={tau_y}"
        )

        L = node.get_parameter("L").value
        xi = node.get_parameter("xi").value
        if L < 0 or xi < 0:
            if not _run_identification_action(node, robot_name):
                node.get_logger().warning(
                    'estimate_length_and_damping did not succeed - falling back to '
                    'a previously saved fit (if any) or placeholder defaults'
                )

            fitted_path = os.path.expanduser(f"~/.ros/hook_pendulum_params_{robot_name}.yaml")
            if os.path.exists(fitted_path):
                with open(fitted_path) as f:
                    fitted = yaml.safe_load(f)
                if L < 0:
                    L = fitted["length"]
                if xi < 0:
                    xi = fitted["damping"]
                node.get_logger().info(f"Loaded L={L}, xi={xi} from {fitted_path}")
            else:
                if L < 0:
                    L = 10.0
                if xi < 0:
                    xi = 0.1
                node.get_logger().warning(
                    f"No fitted pendulum params found at {fitted_path} and L/xi not "
                    f"overridden - using placeholder defaults L={L}, xi={xi}. Run "
                    f"estimate_length_and_damping_node first for real values."
                )

        HookKalmanFilter(
            node,
            robot_name=robot_name,
            use_simtime=node.get_parameter("use_sim_time").value,
            loop_freq=node.get_parameter("loop_freq").value,
            L=L,
            xi=xi,
            taux=tau_x,
            tauy=tau_y,
            kx=k_x,
            ky=k_y,
            qc=node.get_parameter("qc").value,
            sigma_initial=node.get_parameter("sigma_initial").value,
            mahalanobis_thr=node.get_parameter("mahalanobis_thr").value,
            max_boresight_tilt_deg=node.get_parameter("max_boresight_tilt_deg").value,
        )

        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C received - generating comparison plots before shutdown...')
        ok, message = save_all_plots(gt_samples, est_samples, plot_output_dir, robot_name,
                                     L=L, raw_samples=raw_samples)
        node.get_logger().info(message) if ok else node.get_logger().warning(message)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
