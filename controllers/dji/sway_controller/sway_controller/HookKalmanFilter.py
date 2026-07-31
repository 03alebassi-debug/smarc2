from rclpy.node     import Node
from rclpy.qos      import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy
from rclpy.time     import Time
from rclpy.duration import Duration

from dji_msgs.msg       import Topics, Links, LabeledOBBs
from geometry_msgs.msg  import Vector3Stamped
from nav_msgs.msg       import Odometry
from sensor_msgs.msg    import CameraInfo, JointState
from tf2_ros            import Buffer, TransformListener
from tf2_geometry_msgs  import do_transform_vector3

import os
import yaml
import numpy   as np
import control as ct

from ament_index_python import get_package_share_directory

class HookKalmanFilter:
    def __init__(self, node:Node ,robot_name:str, 
                 use_simtime:bool,loop_freq:int, 
                 L:float, xi:float,
                 taux:float, tauy:float,
                 kx:float, ky:float, 
                 qc:float, sigma_initial:float,
                 mahalanobis_thr:float,
                 max_boresight_tilt_deg:float = 45.0):

        self._node:Node = node
        self._robot_name:str = robot_name
        self._T:float = 1/loop_freq
        self._mahalanobis_thr = mahalanobis_thr
        self._max_boresight_tilt_deg:float = max_boresight_tilt_deg

        self._camera_config_path:None|str = None 
        self._image_width:float  
        self._image_height:float 
        self._fx:float
        self._fy:float
        self._cx:float 
        self._cy:float 
        self._loaded_camera_parameters: bool = False

        pkg_share = get_package_share_directory('auv_state_estimation')
        filename = 'sim_1080p_cam_params.yaml' if use_simtime else 'z1_720p_cam_params.yaml'
        self._camera_config_path = os.path.join(pkg_share, 'config', filename) 

        self._read_camera_params()

        # theta_x/theta_y are swing angles from vertical *in base_flat_link*, but
        # detections arrive as pixel offsets in the gimbal's optical frame - the
        # gimbal is free to pitch/yaw, so that frame's axes don't stay aligned
        # with base_flat_link's. Rotate the pixel ray through TF (same pattern as
        # HookGroundTruthComparator/ProjectionNode) instead of assuming identity,
        # which was silently swapping/mixing x and y whenever the gimbal wasn't at
        # its assumed neutral orientation.
        # Links.GIMBAL_OPTICAL_FRAME ('z1_global_optical_frame') is only ever
        # published by z1_pro_driver's global_gimbal_pose_pub.py, which
        # dji_bringup.sh only launches in real-hardware mode. In sim, Unity's
        # own gimbal prefab live-publishes the same rotation via its own
        # ROSTransformTreePublisher, but under the plain (non-"global") name
        # 'z1_optical_frame' - so the frame to look up depends on use_simtime.
        camera_optical_frame = 'z1_optical_frame' if use_simtime else Links.GIMBAL_OPTICAL_FRAME
        self._camera_frame:str    = self._robot_name + '/' + camera_optical_frame
        self._base_flat_frame:str = self._robot_name + '/' + Links.BASE_FLAT
        # The rope attachment point - the actual pendulum pivot, which is offset
        # from both the camera and the base_flat origin (see _detection_callback).
        self._pivot_frame:str     = self._robot_name + '/' + Links.ROPE_BASE_LINK
        # Filled in per detection; the published hook position is expressed
        # relative to base_flat_link's origin, so the pivot offset has to be added
        # back to the pendulum displacement.
        self._pivot_in_base_flat:np.ndarray = np.zeros(3)
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

        self._last_meas:np.ndarray[None|float] = np.full(2, None) #[theta_x, theta_y]
        self._vmeas:np.ndarray[None|float]     = np.full(2, None)

        wn:float         = np.sqrt(9.81/L)
        self._L:float    = L
        self._taux:float = taux
        self._tauy:float = tauy
        self._kx:float   = kx
        self._ky:float   = ky
        self._qc:float   = qc

        # Continuous-time model, kept around (not just its discretization) so
        # _prediction() can re-discretize every tick using the *actual* elapsed
        # time instead of assuming loop_freq is really being achieved. A busy
        # executor (many nodes competing for CPU) can make the timer fire far
        # slower than requested, and silently reusing a fixed Ad/Bd computed for
        # 1/loop_freq while real steps take much longer corrupts the whole model.
        self._Ac:np.ndarray = np.array([
                                    [0,       1,       0,      0      ],   #theta_x
                                    [-wn**2, -2*xi*wn, 0,      0      ],   #omega_x
                                    [0,       0,       0,      1      ],   #theta_y
                                    [0,       0,      -wn**2, -2*xi*wn]    #omega_x
                                 ])

        self._Bc:np.ndarray = np.array([
                                    [0,    0  ],
                                    [-1/L, 0  ],
                                    [0,    0  ],
                                    [0,   -1/L]
                                 ])

        self._Cd:np.ndarray = np.array([
                                            [1, 0, 0,   0],
                                            [0, 0, 1.0, 0]
                                       ])
        self._Dc:np.ndarray = np.zeros((2, 2))

        self._last_tick_time = None
        # When a detection was last ACCEPTED (not merely received - rejected
        # ones leave the filter running open-loop on its model). See
        # _publish_hook_state for why this is the number that matters.
        self._last_accepted_meas_time = None
        # Rolling estimate of the achieved tick interval; see _prediction.
        self._typical_dt: "float|None" = None

        # [cmd_x, cmd_y], cmd_z dropped since it is assume the drone to move on the X-Y plane
        self._last_input:np.ndarray[float] = np.zeros(2)  
        self._mu_bar:np.ndarray[float]     = np.zeros(4) # estimated state after the prediction step
        self._mu:np.ndarray[float]         = np.zeros(4) # estimated state after the update step 

        self._sigma_px:float = 10.0 #pixels
        self._sigma_py:float = 10.0 #pixels

        self._R:np.ndarray[float]
        self._update_measurement_noise()

        self._Sigma_bar:np.ndarray[float] = sigma_initial * np.eye(4)
        self._Sigma:np.ndarray[float] = sigma_initial * np.eye(4)

        self._create_node_subscriptions()
        self._create_publishers()

        self._timer = self._node.create_timer(timer_period_sec=self._T, callback=self._prediction)

    def _read_camera_params(self):
        
        with open(self._camera_config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                self._node.get_logger().info(f"Error reading YAML file: {exc}")

        self._image_width  = config['image_width']
        self._image_height = config['image_height']
        self._fx           = config['camera_matrix']['data'][0]
        self._fy           = config['camera_matrix']['data'][4]
        self._cx           = config['camera_matrix']['data'][2]
        self._cy           = config['camera_matrix']['data'][5]

        self._loaded_camera_parameters = True 
        self._node.get_logger().info(f'Camera parameter loaded from:{self._camera_config_path}')
        

    def _create_publishers(self):
        qos_best_effort10 = QoSProfile(depth=10, 
                                               reliability=ReliabilityPolicy.BEST_EFFORT, 
                                               durability=QoSDurabilityPolicy.VOLATILE)
        _hook_state_topic:str = self._robot_name + '/hook_state'
        self._hook_state_pub = self._node.create_publisher(Odometry, _hook_state_topic, qos_best_effort10)
        self._node.get_logger().info(f'Publishing hook state on:{_hook_state_topic}')

        _hook_raw_meas_topic:str = self._robot_name + '/hook_raw_measurement'
        self._hook_raw_meas_pub = self._node.create_publisher(Odometry, _hook_raw_meas_topic, qos_best_effort10)
        self._node.get_logger().info(f'Publishing raw hook measurement on:{_hook_raw_meas_topic}')

        # The filter state itself, for controllers (LQG) rather than for humans.
        # hook_state carries the CARTESIAN hook pose, which a controller would
        # have to invert back through x = pivot_x + L*sin(theta) - needing L and
        # the live pivot offset, i.e. duplicating this node's geometry. Publishing
        # [theta, omega] directly avoids that coupling entirely.
        # JointState is used rather than a bare array because it is stamped (a
        # controller must be able to reject a stale estimate) and self-describing,
        # and a 2-axis pendulum genuinely is two revolute joints.
        _hook_swing_topic:str = self._robot_name + '/hook_swing_state'
        self._hook_swing_pub = self._node.create_publisher(JointState, _hook_swing_topic, qos_best_effort10)
        self._node.get_logger().info(f'Publishing hook swing state on:{_hook_swing_topic}')

    def _create_node_subscriptions(self):
        _detection_topic_name:str = self._robot_name + '/' + Topics.LABELED_OBBS_TOPIC
        self._detection_subscription = self._node.create_subscription(LabeledOBBs, 
                                                                      _detection_topic_name, 
                                                                      self._detection_callback, 
                                                                      10)
        self._node.get_logger().info(f'Succesfully subscribed to:{_detection_topic_name}')

        qos_best_effort10 = QoSProfile(depth=10, 
                                       reliability=ReliabilityPolicy.BEST_EFFORT, 
                                       durability=QoSDurabilityPolicy.VOLATILE)

        _cmd_vel_topic:str = self._robot_name + '/' + 'cmd_vel_drone_frame'
        self._cmd_vel_subscriber = self._node.create_subscription(Vector3Stamped, 
                                                                  _cmd_vel_topic, 
                                                                  self._cmd_vel_callback, 
                                                                  qos_profile=qos_best_effort10)
        self._node.get_logger().info(f'Succesfully subscribed to:{_cmd_vel_topic}')

        # The yaml loaded in _read_camera_params is only a starting point: on the
        # sim it is a REAL camera's calibration (distortion coefficients and all)
        # while Unity renders an ideal pinhole, so its principal point and focal
        # lengths are simply wrong there. Prefer whatever the camera itself
        # reports. See _camera_info_callback for the measured discrepancy.
        _camera_info_topic:str = self._robot_name + '/' + Topics.GIMBAL_CAMERA_INFO_TOPIC
        self._camera_info_subscription = self._node.create_subscription(CameraInfo,
                                                                        _camera_info_topic,
                                                                        self._camera_info_callback,
                                                                        10)
        self._node.get_logger().info(f'Succesfully subscribed to:{_camera_info_topic}')

        _odom_topic:str = self._robot_name + '/smarc/odom'
        self._odom_subscription = self._node.create_subscription(Odometry,
                                                                 _odom_topic,
                                                                 self._odom_callback,
                                                                 10)
        self._node.get_logger().info(f'Succesfully subscribed to:{_odom_topic}')

    def _lookup_pivot(self) -> "np.ndarray|None":
        """Position of the rope attachment point (the pendulum pivot) in
        base_flat_link. Not cached: rope_base_link is rigid w.r.t. base_link, but
        base_flat_link differs from base_link by the drone's roll/pitch, so the
        pivot does move slightly in this frame as the drone tilts."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_flat_frame, self._pivot_frame, Time(), timeout=Duration(seconds=0.05)
            )
        except Exception as e:
            self._node.get_logger().warning(
                f'Could not transform {self._pivot_frame} -> {self._base_flat_frame}, '
                f'treating the camera as the pivot (adds a systematic offset): {e}',
                throttle_duration_sec=5.0
            )
            return None
        return np.array([tf.transform.translation.x,
                         tf.transform.translation.y,
                         tf.transform.translation.z])

    def _camera_info_callback(self, msg:CameraInfo):
        """Adopt the intrinsics the camera itself publishes, overriding the yaml.

        This matters a lot on the sim: `sim_1080p_cam_params.yaml` is a real
        camera calibration, but Unity's virtual camera is an ideal pinhole, and
        measured against its own CameraInfo the yaml is off by
        fx 1402.8 vs 1768.1 (-20.7%), fy 1398.0 vs 1491.8 (-6.3%),
        cx 940.2 vs 960.0 (exact image centre), cy 539.3 vs 540.0.
        With the gimbal pointed down, image-horizontal maps to base_flat Y and
        image-vertical to base_flat X, so that cx error alone put a constant
        +0.058m bias on the estimated Y (while X, via the near-perfect cy, stayed
        unbiased) - which is most of the offset seen between the raw measurement
        and ground truth in y. The focal-length errors additionally scaled the
        angles by +26% on Y and +6.7% on X."""
        fx, fy = float(msg.k[0]), float(msg.k[4])
        cx, cy = float(msg.k[2]), float(msg.k[5])
        if fx <= 0.0 or fy <= 0.0:
            self._node.get_logger().warning(
                f'Ignoring CameraInfo with non-positive focal lengths (fx={fx}, fy={fy})',
                throttle_duration_sec=10.0
            )
            return

        unchanged = (self._loaded_camera_parameters
                     and abs(fx - self._fx) < 1e-6 and abs(fy - self._fy) < 1e-6
                     and abs(cx - self._cx) < 1e-6 and abs(cy - self._cy) < 1e-6)
        if unchanged:
            return

        self._node.get_logger().info(
            f'Using intrinsics from CameraInfo: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} '
            f'(yaml had fx={self._fx:.2f} fy={self._fy:.2f} cx={self._cx:.2f} cy={self._cy:.2f})'
        )
        self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy
        if msg.width > 0 and msg.height > 0:
            self._image_width, self._image_height = float(msg.width), float(msg.height)
        self._loaded_camera_parameters = True
        # R is expressed in normalized-camera units (pixels / focal length), so it
        # has to follow the focal lengths.
        self._update_measurement_noise()

    def _update_measurement_noise(self):
        self._R = np.array([
            [(self._sigma_px / self._fx)**2, 0                             ],
            [0,                              (self._sigma_py / self._fy)**2]
        ])

    def _cmd_vel_callback(self, msg:Vector3Stamped):
        self._last_input[0] = msg.vector.x
        self._last_input[1] = msg.vector.y

    def _detection_callback(self, msg):
        if not self._loaded_camera_parameters:
            self._node.get_logger().warning('Camera parameters not loaded yet, skipping measurement')
            return 

        hook_indices = [i for i, cls_id in enumerate(msg.ids) if cls_id == "hook"]
        if not hook_indices:
            self._node.get_logger().info(f'No hook detection in this frame', throttle_duration_sec=1.0)
            return 

        norm_x:float = 0.0
        norm_y:float = 0.0
        for idx in hook_indices:
            pts = msg.obbs[idx].points
            pts_norm_x:float = sum(p.x for p in pts) / len(pts)
            pts_norm_y:float = sum(p.y for p in pts) / len(pts)
            norm_x += pts_norm_x
            norm_y += pts_norm_y

        norm_x /= len(hook_indices)
        norm_y /= len(hook_indices)
    
        u:float = norm_x * (self._image_width / 2) + (self._image_width / 2)
        v:float = norm_y * (self._image_height / 2) + (self._image_height / 2)

        # Ray towards the hook in the camera's optical frame (REP103 convention:
        # x right, y down, z forward - see z1_pro_driver/global_gimbal_pose_pub.py
        # which publishes this frame's TF).
        ray_in = Vector3Stamped()
        # Set fields individually rather than `ray_in.header = msg.header`, which
        # would alias (and then mutate) the incoming message's own header.
        ray_in.header.stamp = msg.header.stamp
        ray_in.header.frame_id = self._camera_frame
        ray_in.vector.x = (u - self._cx) / self._fx
        ray_in.vector.y = (v - self._cy) / self._fy
        ray_in.vector.z = 1.0

        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_flat_frame, self._camera_frame, Time(), timeout=Duration(seconds=0.2)
            )
        except Exception as e:
            self._node.get_logger().warning(
                f'Could not transform {self._camera_frame} -> {self._base_flat_frame}, '
                f'skipping this detection: {e}',
                throttle_duration_sec=1.0
            )
            return

        ray_out = do_transform_vector3(ray_in, transform)

        # base_flat_link is gravity-aligned with z up, and the gimbal is expected
        # to be pointed roughly straight down at the hook, so -ray_out.vector.z is
        # the "downward" (boresight) component the swing angles are measured from.
        # That assumption is load-bearing: with a horizontal camera the ray's z
        # component goes to zero and atan2(x, -z) degenerates towards +-90deg, i.e.
        # silent garbage rather than an obvious failure. The sim in particular
        # starts with the gimbal horizontal until something commands it down, so
        # check it explicitly instead of trusting it.
        boresight = Vector3Stamped()
        boresight.header.frame_id = self._camera_frame
        boresight.vector.z = 1.0
        bs = do_transform_vector3(boresight, transform).vector
        # bs is a unit vector in base_flat_link; bs.z = -1 means straight down.
        tilt_from_down_deg = float(np.degrees(np.arccos(np.clip(-bs.z, -1.0, 1.0))))
        if tilt_from_down_deg > self._max_boresight_tilt_deg:
            self._node.get_logger().warning(
                f'Camera boresight is {tilt_from_down_deg:.0f}deg off straight-down '
                f'(limit {self._max_boresight_tilt_deg:.0f}deg) - the pendulum-angle '
                f'measurement is not valid in this pose, skipping this detection. '
                f'Is the gimbal pointed down? '
                f'(sim: ros2 topic pub -r 2 -t 5 {self._robot_name}/gimbal_camera/gimbal_cmd '
                f'geometry_msgs/msg/Vector3 "{{x: 0.0, y: 90.0, z: 0.0}}")',
                throttle_duration_sec=5.0
            )
            return

        # The camera is NOT at the pendulum pivot: measured in base_flat_link the
        # optical centre sits at ~(-0.034, -0.007, -0.259) while rope_base_link is
        # at ~(+0.161, 0.000, -0.254), i.e. ~0.195m apart in x. Taking the ray
        # angle as the swing angle (and L*sin of it as the position) therefore
        # both measured the angle about the wrong point and reported the hook's
        # offset from below the CAMERA rather than its actual position - a
        # systematic error, plus parallax that varies as the hook swings.
        # Instead intersect the ray with the sphere of radius L about the pivot,
        # which is where the hook must physically be.
        pivot = self._lookup_pivot()
        if pivot is None:
            # Fall back to the old camera-as-pivot behaviour rather than dropping
            # the detection outright.
            self._last_meas[0] = np.arctan2(ray_out.vector.x, -ray_out.vector.z)
            self._last_meas[1] = np.arctan2(ray_out.vector.y, -ray_out.vector.z)
            self._pivot_in_base_flat = np.zeros(3)
        else:
            cam = np.array([transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z])
            d = np.array([ray_out.vector.x, ray_out.vector.y, ray_out.vector.z])
            norm = np.linalg.norm(d)
            if norm <= 0.0:
                return
            d = d / norm

            # |cam + s*d - pivot| = L  ->  s^2 + 2(w.d)s + |w|^2 - L^2 = 0, w = cam - pivot.
            w = cam - pivot
            wd = float(w @ d)
            disc = wd * wd - float(w @ w) + self._L * self._L
            if disc < 0.0:
                # Ray misses the sphere entirely: the detection is inconsistent
                # with L (bad L, bad detection, or the hook is not on this rope).
                self._node.get_logger().warning(
                    'Hook detection ray does not intersect the pendulum sphere '
                    f'(L={self._L:.2f}m) - skipping it',
                    throttle_duration_sec=5.0
                )
                return
            s = -wd + np.sqrt(disc)   # far intersection = the hook below the drone
            if s <= 0.0:
                self._node.get_logger().warning(
                    'Hook intersection resolved behind the camera - skipping it',
                    throttle_duration_sec=5.0
                )
                return

            r = cam + s * d - pivot   # hook position relative to the pivot
            self._last_meas[0] = np.arctan2(r[0], -r[2])
            self._last_meas[1] = np.arctan2(r[1], -r[2])
            self._pivot_in_base_flat = pivot

        # Publish the raw single-detection measurement (pre-fusion, pre-gating)
        # for a clean axis comparison against hook_ground_truth_base_flat - see
        # _publish_raw_measurement for why hook_state itself can't be trusted here.
        self._publish_raw_measurement(msg.header.stamp)

        self._update()

    def _publish_raw_measurement(self, stamp):
        """Publishes the hook position implied by the latest single detection
        (theta_x/theta_y already rotated into base_flat_link), BEFORE any Kalman
        prediction/update or Mahalanobis gating. This is the clean signal to
        compare against hook_ground_truth_base_flat to check the
        camera->base_flat_link axis mapping in isolation: hook_state is
        confounded both by rejected updates (which fall back to pure prediction)
        and by the prediction step being forced with real cmd_vel/odom, so if the
        frames are consistent it's *this* topic - not hook_state - that should
        land on the same axis as the ground truth."""
        theta_x, theta_y = self._last_meas
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self._base_flat_frame
        msg.child_frame_id = self._robot_name + '/hook'
        # Pendulum displacement is about the PIVOT, so add the pivot's own offset
        # to report a position in base_flat_link - what hook_ground_truth_base_flat
        # carries, and what makes the two directly comparable.
        msg.pose.pose.position.x = float(self._pivot_in_base_flat[0] + self._L * np.sin(theta_x))
        msg.pose.pose.position.y = float(self._pivot_in_base_flat[1] + self._L * np.sin(theta_y))
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        self._hook_raw_meas_pub.publish(msg)

    def _odom_callback(self, msg:Odometry):
        # This twist is expressed in the ODOM frame, not in the child frame that
        # nav_msgs/Odometry conventionally implies: dji_captain fills it straight
        # from _velocity_ground, which it stamps with ODOM_FRAME
        # (_velocity_ground_callback), even though it sets child_frame_id to
        # base_link. Meanwhile the pendulum states - and the cmd_vel_drone_frame
        # input this gets combined with in _prediction's `a` - live in
        # base_flat_link, which carries the drone's yaw. Adding an odom-frame
        # velocity to a body-frame command therefore misassigns the forcing
        # between the two axes, completely so at a 90deg heading (where body x
        # is odom y) - which showed up as a large spurious spike on the estimated
        # y while the true swing was all in x. So rotate it into base_flat_link.
        vel_in = Vector3Stamped()
        vel_in.header.stamp = msg.header.stamp
        vel_in.header.frame_id = msg.header.frame_id
        vel_in.vector = msg.twist.twist.linear

        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_flat_frame, msg.header.frame_id, Time(), timeout=Duration(seconds=0.1)
            )
        except Exception as e:
            self._node.get_logger().warning(
                f'Could not transform odom velocity {msg.header.frame_id} -> '
                f'{self._base_flat_frame}, keeping previous value: {e}',
                throttle_duration_sec=5.0
            )
            return

        vel_body = do_transform_vector3(vel_in, transform).vector
        self._vmeas[0] = vel_body.x
        self._vmeas[1] = vel_body.y

    def _prediction(self):
        now = self._node.get_clock().now()

        if self._last_tick_time is None:
            # First tick: no meaningful elapsed interval to discretize over yet.
            self._last_tick_time = now
            return

        dt:float = (now - self._last_tick_time).nanoseconds * 1e-9
        self._last_tick_time = now

        if dt == 0.0:
            # Expected under use_sim_time, not an error: /clock advances in
            # discrete steps (currently 50ms, set by Unity's maximumDeltaTime)
            # while this timer asks for 20ms, so rclpy fires it several times per
            # clock update to catch up and the extra firings read an identical
            # stamp. No time has passed, so there is simply nothing to propagate.
            return

        if dt < 0.0:
            # Genuinely wrong - the clock went backwards (sim reset / bag loop).
            # Re-sync rather than integrating a negative interval.
            self._node.get_logger().warning(
                f'Clock went backwards ({dt:.3f}s between predictions), resynchronising',
                throttle_duration_sec=5.0
            )
            return

        # Compare against the rate actually being ACHIEVED, not against loop_freq.
        # Under use_sim_time the achievable tick rate is set by how finely /clock
        # advances (Unity's maximumDeltaTime), which is normally coarser than
        # loop_freq - so testing against self._T alone flags every single tick as
        # a fault and buries any real problem. Tracking a typical dt means this
        # only fires on a genuine stall relative to the current steady rate.
        if self._typical_dt is None:
            self._typical_dt = dt
        else:
            self._typical_dt += 0.05 * (dt - self._typical_dt)   # slow EWMA

        if dt > 3 * self._typical_dt and dt > 3 * self._T:
            self._node.get_logger().warning(
                f'_prediction() stalled: this tick took {dt:.3f}s vs a typical '
                f'{self._typical_dt:.3f}s ({1/self._typical_dt:.1f} Hz)',
                throttle_duration_sec=10.0
            )

        if self._vmeas[0] is None or self._vmeas[1] is None:
            self._node.get_logger().warning('No Odometry msg recieved yet. Skipping the update')
            return

        # Re-discretize every tick using the *actual* elapsed dt, rather than a
        # fixed Ad/Bd computed once for the nominal loop period.
        discrete_ss = ct.c2d(ct.ss(self._Ac, self._Bc, self._Cd, self._Dc), dt, 'zoh')
        Ad:np.ndarray = np.asarray(discrete_ss.A)
        Bd:np.ndarray = np.asarray(discrete_ss.B)

        Q_axis:np.ndarray = self._qc * np.array([[dt**3 / 3, dt**2 / 2],
                                                   [dt**2 / 2, dt       ]])
        Q:np.ndarray = np.block([
            [Q_axis,               np.zeros_like(Q_axis)],
            [np.zeros_like(Q_axis), Q_axis]
        ])

        a:np.ndarray = np.array([
                                   -self._taux * self._vmeas[0] + self._kx * self._last_input[0],
                                   -self._tauy * self._vmeas[1] + self._ky * self._last_input[1]
                                ])

        self._mu_bar = Ad @ self._mu + Bd @ a
        sigma_bar_update:np.ndarray[float] = Ad @ self._Sigma @ Ad.T + Q
        self._Sigma_bar = (sigma_bar_update + sigma_bar_update.T) / 2

        self._mu = self._mu_bar
        self._Sigma = self._Sigma_bar

        self._publish_hook_state()

    def _publish_hook_state(self):
        theta_x, omega_x, theta_y, omega_y = self._mu

        sin_tx, cos_tx = np.sin(theta_x), np.cos(theta_x)
        sin_ty, cos_ty = np.sin(theta_y), np.cos(theta_y)

        # theta is the swing angle about the PIVOT, so L*sin(theta) is a
        # displacement from the pivot - offset it to report a base_flat_link
        # position, directly comparable to hook_ground_truth_base_flat. The
        # velocities need no such term: the pivot is fixed in this frame.
        x  = self._pivot_in_base_flat[0] + self._L * sin_tx
        y  = self._pivot_in_base_flat[1] + self._L * sin_ty
        vx = self._L * cos_tx * omega_x
        vy = self._L * cos_ty * omega_y

        # Linearized (Jacobian) covariance propagation through
        # x  = L*sin(theta), vx = L*cos(theta)*omega  (and same for y)
        Sigma_x = self._Sigma[0:2, 0:2]  # cov of [theta_x, omega_x]
        Jx  = np.array([self._L * cos_tx, 0.0])
        Jvx = np.array([-self._L * sin_tx * omega_x, self._L * cos_tx])
        var_x  = float(Jx  @ Sigma_x @ Jx.T)
        var_vx = float(Jvx @ Sigma_x @ Jvx.T)

        Sigma_y = self._Sigma[2:4, 2:4]  # cov of [theta_y, omega_y]
        Jy  = np.array([self._L * cos_ty, 0.0])
        Jvy = np.array([-self._L * sin_ty * omega_y, self._L * cos_ty])
        var_y  = float(Jy  @ Sigma_y @ Jy.T)
        var_vy = float(Jvy @ Sigma_y @ Jvy.T)

        msg = Odometry()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._robot_name + '/' + Links.BASE_FLAT
        msg.child_frame_id = self._robot_name + '/hook'

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.w = 1.0  

        msg.twist.twist.linear.x = float(vx)
        msg.twist.twist.linear.y = float(vy)
        msg.twist.twist.linear.z = 0.0

        msg.pose.covariance[0]  = var_x   # (x, x)
        msg.pose.covariance[7]  = var_y   # (y, y)
        msg.twist.covariance[0] = var_vx  # (vx, vx)
        msg.twist.covariance[7] = var_vy  # (vy, vy)

        self._hook_state_pub.publish(msg)
        self._publish_swing_state(msg.header.stamp)

        # The DOMINANT term in the feedback delay. hook_swing_state is published
        # at the filter rate, but between accepted detections the filter is
        # predicting, so the newest real information can be far older than the
        # message stamp suggests - and it is the information age, not the
        # message age, that rotates the phase of a damping loop.
        if self._last_accepted_meas_time is not None:
            meas_age = (self._node.get_clock().now() - self._last_accepted_meas_time).nanoseconds * 1e-9
            self._node.get_logger().info(
                f'[meas] newest accepted detection is {meas_age*1000:.0f}ms old',
                throttle_duration_sec=2.0
            )

    def _publish_swing_state(self, stamp):
        """The raw filter state [theta_x, omega_x, theta_y, omega_y] - what a
        controller consumes. Ordering matches LQG.LQR.IDX's theta_x/omega_x/
        theta_y/omega_y entries, so it slices straight into the LQR state
        vector with no conversion.

        Variances come from the same Sigma the cartesian covariances are
        linearised from, but UNlinearised - these are the angle/rate variances
        themselves, so a controller can gate on estimate quality directly."""
        theta_x, omega_x, theta_y, omega_y = self._mu

        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = self._base_flat_frame
        msg.name = ['hook_theta_x', 'hook_theta_y']
        msg.position = [float(theta_x), float(theta_y)]
        msg.velocity = [float(omega_x), float(omega_y)]
        # No torque to report; JointState.effort is reused for the state
        # variances so consumers can reject an over-uncertain estimate.
        msg.effort = [float(self._Sigma[0, 0]), float(self._Sigma[2, 2])]

        self._hook_swing_pub.publish(msg)

    def _update(self):
        y:np.ndarray[float] = self._last_meas - self._Cd @ self._mu_bar

        S:np.ndarray[float]     = self._Cd @ self._Sigma_bar @ self._Cd.T + self._R
        S_inv:np.ndarray[float] = np.linalg.inv(S)
        K:np.ndarray[float]     = self._Sigma_bar @ self._Cd.T @ S_inv

        d2:float = float(y.T @ S_inv @ y)
        if d2 > self._mahalanobis_thr:
            self._node.get_logger().warn(f'Hook detection rejected as outlier:{d2} > {self._mahalanobis_thr}')
            self._mu = self._mu_bar
            self._Sigma = self._Sigma_bar
            return 

        self._last_accepted_meas_time = self._node.get_clock().now()
        self._mu = self._mu_bar + K @ (self._last_meas - self._Cd @ self._mu_bar)

        term = np.eye(len(self._mu)) - K @ self._Cd
        sigma_update:np.ndarray[float] = term @ self._Sigma_bar @ term.T + K @ self._R @ K.T
        self._Sigma = (sigma_update + sigma_update.T) / 2

        if (np.trace(self._Sigma_bar) - np.trace(self._Sigma)<0):
            self._node.get_logger().warning(f'Uncertainty INCREASED after the update')