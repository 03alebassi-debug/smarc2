from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Vector3

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped

from dji_msgs.msg import Links


class HookGroundTruthComparator:
    """Subscribes to the hook's ground-truth Odometry (published by Unity in the
    unity_origin frame) and re-publishes it transformed into <robot_name>/base_flat_link -
    the same frame HookKalmanFilter reports its hook_state estimate in - so the two
    topics can be compared directly (same frame, same message type)."""

    def __init__(self, node: Node, robot_name: str,
                 ground_truth_topic: str = "hook_ground_truth",
                 output_topic: str = "hook_ground_truth_base_flat",
                 velocity_smoothing_window: int = 3):
        self._node: Node = node
        self._robot_name: str = robot_name
        self._target_frame: str = self._robot_name + '/' + Links.BASE_FLAT

        # The incoming twist is DELIBERATELY IGNORED - see _ground_truth_callback.
        self._prev_pos: "tuple[float, float, float]|None" = None
        self._prev_t: "float|None" = None
        self._vel_window: "list[tuple[float, float, float]]" = []
        self._velocity_smoothing_window: int = max(1, int(velocity_smoothing_window))

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

        self._gt_sub = self._node.create_subscription(
            Odometry,
            ground_truth_topic,
            self._ground_truth_callback,
            10
        )
        self._node.get_logger().info(
            f'Subscribed to:{ground_truth_topic}'
        )

        self._out_pub = self._node.create_publisher(
            Odometry,
            output_topic,
            10
        )
        self._node.get_logger().info(
            f'Publishing transformed ground truth on:{output_topic}'
        )

    def _ground_truth_callback(self, msg: Odometry):
        
        transform = None
        try:
            transform = self._tf_buffer.lookup_transform(
                self._target_frame,
                msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.05)
            )
        except Exception:
            
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._target_frame,
                    msg.header.frame_id,
                    Time(),
                    timeout=Duration(seconds=0.2)
                )
                self._node.get_logger().warning(
                    'TF not available at the ground truth stamp, using the latest '
                    'instead - differentiated velocity may show spikes while the '
                    'drone is moving',
                    throttle_duration_sec=5.0
                )
            except Exception as e:
                self._node.get_logger().warning(
                    f'Could not transform {msg.header.frame_id} -> {self._target_frame}: {e}',
                    throttle_duration_sec=1.0
                )
                return

        pose_in = PoseStamped()
        pose_in.header = msg.header
        pose_in.pose = msg.pose.pose
        pose_out = do_transform_pose_stamped(pose_in, transform)

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._target_frame
        out.child_frame_id = msg.child_frame_id

        out.pose.pose = pose_out.pose
        out.twist.twist.linear = self._velocity_from_position(pose_out, msg.header.stamp)

        self._out_pub.publish(out)

    def _velocity_from_position(self, pose_out: PoseStamped, stamp) -> Vector3:
        
        t = stamp.sec + stamp.nanosec * 1e-9
        p = (pose_out.pose.position.x, pose_out.pose.position.y, pose_out.pose.position.z)

        vel = Vector3()
        if self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 0.0:
                self._vel_window.append(tuple((c - pc) / dt for c, pc in zip(p, self._prev_pos)))
                if len(self._vel_window) > self._velocity_smoothing_window:
                    self._vel_window.pop(0)
                n = len(self._vel_window)
                vel.x = sum(v[0] for v in self._vel_window) / n
                vel.y = sum(v[1] for v in self._vel_window) / n
                vel.z = sum(v[2] for v in self._vel_window) / n

        self._prev_pos = p
        self._prev_t = t
        return vel
