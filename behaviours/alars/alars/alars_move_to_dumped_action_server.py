import numpy   as np
import control as ct 

import rclpy
from rclpy.node         import Node
from rclpy.executors    import MultiThreadedExecutor
from rclpy.qos          import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from geographic_msgs.msg import GeoPoint
from geometry_msgs.msg   import PoseStamped, TwistStamped, Vector3Stamped
from nav_msgs.msg        import Odometry
from sensor_msgs.msg     import JointState


from smarc_action_base.gentler_action_server import GentlerActionServer

from alars.alars_common import DroneState

from dji_msgs.msg import Topics as DJITopics
from dji_msgs.msg import Links  as DJILinks

import traceback
import time
import os
import yaml

from sway_controller import PathParametrizer, ZVD, LQR, save_mission_plots

G = 9.81

class MoveToDumpedAction():
    def __init__(self, node:Node):
        self._node:Node = node 

        self._get_node_parameters()
        self._create_publishers()

        self.BASE_FLAT_FRAME : str = self._robot_name + '/' + DJILinks.BASE_FLAT
        self._drone_state = DroneState(node, self._robot_name)

        # The LQR is built per mission in _on_goal_received, from the L/xi that
        # estimate_length_and_damping identified - not from the node parameters,
        # which are only a fallback.
        self._lqr:None|LQR = None
        self._lqr_stabilize:None|LQR = None
        self._swing_state:None|JointState = None
        self._drone_velocity_base_flat:None|np.ndarray = None
        self._create_subscriptions()

        # Defined here too, not only in _on_goal_received: a cancel can arrive
        # before any goal has been accepted, and _save_mission_plots runs there.
        self._phase:str = 'MOVING'
        self._mission_samples:list = []
        self._goal_received_time:float = 0.0

        self._goal_in_map:None|PoseStamped = None
        self._distance_from_goal:None|float = None

        self._as = GentlerActionServer(
                    node,
                    'move_to_dumped',
                    self._on_goal_received,
                    self._on_cancel_received,
                    self._prepare_loop,
                    self._loop_inner,
                    self._give_feedback,
                    loop_frequency = 50
                )        

    def _build_transfer_function(self, path:str):
        d = np.load(path)

        self._tf_x = ct.tf(
        d['b__FLU_axes_0__to__FLUvelocity_ground_fused_x'],
        d['a__FLU_axes_0__to__FLUvelocity_ground_fused_x']
        )
        self._tf_y = ct.tf(
            d['b__FLU_axes_1__to__FLUvelocity_ground_fused_y'],
            d['a__FLU_axes_1__to__FLUvelocity_ground_fused_y']
        )
        self._tf_z = ct.tf(
            d['b__FLU_axes_2__to__FLUvelocity_ground_fused_z'],
            d['a__FLU_axes_2__to__FLUvelocity_ground_fused_z']
        )

    def _declare_parameters(self):
        int_desc = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER)
        double_desc = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        string_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        bool_desc = ParameterDescriptor(type=ParameterType.PARAMETER_BOOL)

        node = self._node
        node.declare_parameter('robot_name', 'M350', string_desc)
        node.declare_parameter('continuous_model_path', '', string_desc)
        node.declare_parameter('discrete_model_path', '', string_desc)
        node.declare_parameter('goal_tolerance', 0.2, double_desc)
        node.declare_parameter('max_speed', 1.0, double_desc)
        node.declare_parameter('max_acceleration', 2.0, double_desc)
        node.declare_parameter('sattle_extra', 20.0, double_desc)
        # Fallbacks only - the identified values from estimate_length_and_damping
        # are preferred, see _load_identified_pendulum_params.
        node.declare_parameter('rope_length', 10.0, double_desc)
        node.declare_parameter('xi', 0.1, double_desc)
        # Set False to fly the ZVD feedforward open-loop exactly as before, which
        # is the A/B comparison for whether the feedback is actually helping.
        node.declare_parameter('enable_lqg', True, bool_desc)
        # 10.0 measured: at rho=1 the trim saturated the 0.5 m/s cap and the loop
        # limit-cycled at ~2.2s (nothing like the 4.6s pendulum). At rho=10 the
        # trim stays ~0.1 m/s, never saturates, and the swing decays INTO the
        # settle band during the move. rho enters the gain as ~sqrt(rho), so it
        # must move by decades to matter.
        node.declare_parameter('lqg_rho', 10.0, double_desc)
        node.declare_parameter('lqg_theta_max', 0.3, double_desc)
        node.declare_parameter('lqg_position_max', 0.3, double_desc)
        # Beyond this the swing estimate is treated as unusable and the loop
        # falls back to pure feedforward rather than correcting on stale data.
        node.declare_parameter('max_estimate_age', 0.5, double_desc)
        # Hard cap on the feedback trim alone, separate from the total-command
        # saturation. Saturating the TOTAL still lets a huge trim swamp the
        # feedforward; capping the trim keeps the plan in charge and keeps the
        # loop inside the region where the linear LQR design actually holds.
        node.declare_parameter('max_trim_speed', 0.5, double_desc)
        # Above this swing the linearised model (sin(theta) ~ theta) is invalid
        # and the gain demands far more authority than exists, so feedback is
        # dropped rather than applied wrongly.
        node.declare_parameter('max_theta_for_lqg', 0.35, double_desc)

        # Damp the payload before departing. This matters because
        # estimate_length_and_damping deliberately EXCITES a swing to identify
        # the pendulum, so the hook is usually still moving when a goal arrives -
        # and ZVD only avoids exciting NEW swing, it cannot cancel existing swing.
        node.declare_parameter('stabilize_before_mission', True, bool_desc)
        node.declare_parameter('stabilize_theta_tol', 0.02, double_desc)    # rad
        node.declare_parameter('stabilize_omega_tol', 0.05, double_desc)    # rad/s
        node.declare_parameter('stabilize_settle_time', 1.0, double_desc)   # s within tol
        # 20s was measured too short: from the ~0.15rad the identification
        # leaves behind, the loop damps to ~0.05rad in 20s (vs ~0.089 from
        # natural decay alone - so it IS working, just not that fast). Reaching
        # the 0.02rad tolerance needs longer. NOTE the plant simulation
        # over-predicts this badly - trust the flight logs, not the model.
        node.declare_parameter('stabilize_timeout', 45.0, double_desc)      # s
        # Cost weights for the stabilising gain, Bryson style (weight =
        # 1/max_deviation^2). Deliberately NOT the mission weights: here the
        # payload is what matters, so the swing tolerance is tight and the
        # position tolerance is loose - we accept drifting a couple of metres if
        # that is what it takes to kill the swing.
        # Its OWN control penalty, separate from lqg_rho. Stabilising needs more
        # authority than trimming: the mission gain only nudges an already-good
        # feedforward, whereas here the command IS the whole control action and
        # it has to actually kill the swing before departure. rho=10 (good for
        # the mission) was measured too soft to settle 0.07rad inside the
        # timeout; rho=2 settles it in ~3.4s and still does not saturate.
        node.declare_parameter('stabilize_rho', 2.0, double_desc)
        node.declare_parameter('stabilize_theta_max', 0.3, double_desc)     # rad
        node.declare_parameter('stabilize_position_max', 2.0, double_desc)  # m

        # Plots are written when a mission ENDS (success, failure or cancel),
        # not on node shutdown as hook_kalman_filter_node does - one set per
        # mission, and they survive the node staying up for the next goal.
        node.declare_parameter('plot_missions', True, bool_desc)
        node.declare_parameter('plot_output_dir', '/home/aleba/move_to_dumped_plots', string_desc)

    def _get_node_parameters(self):
        self._declare_parameters()

        self._robot_name = self._node.get_parameter('robot_name').get_parameter_value().string_value
        _continuous_model_path = self._node.get_parameter('continuous_model_path').get_parameter_value().string_value
        _discrete_model_path = self._node.get_parameter('discrete_model_path').get_parameter_value().string_value

        self._build_transfer_function(_continuous_model_path)

        num_array_x = self._tf_x.num[0][0]  
        den_array_x = self._tf_x.den[0][0]
        self._k_x = num_array_x[0]
        self._tau_x = den_array_x[1]

        num_array_y = self._tf_y.num[0][0]  
        den_array_y = self._tf_y.den[0][0]
        self._k_y = num_array_y[0]
        self._tau_y = den_array_y[1]

        num_array_z = self._tf_z.num[0][0]  
        den_array_z = self._tf_z.den[0][0]
        self._k_z = num_array_z[0]
        self._tau_z = den_array_z[1]

        self._default_goal_tolerance = self._node.get_parameter('goal_tolerance').get_parameter_value().double_value
        self._max_speed = self._node.get_parameter('max_speed').get_parameter_value().double_value
        self._max_acceleration = self._node.get_parameter('max_acceleration').get_parameter_value().double_value
        self._sattle_extra = self._node.get_parameter('sattle_extra').get_parameter_value().double_value

        self.rope_length = self._node.get_parameter('rope_length').get_parameter_value().double_value
        self.xi = self._node.get_parameter('xi').get_parameter_value().double_value
        self.wn = np.sqrt(G/self.rope_length)
        self.dwn = 2 * self.xi * self.wn

        self._enable_lqg = self._node.get_parameter('enable_lqg').get_parameter_value().bool_value
        self._lqg_rho = self._node.get_parameter('lqg_rho').get_parameter_value().double_value
        self._lqg_theta_max = self._node.get_parameter('lqg_theta_max').get_parameter_value().double_value
        self._lqg_position_max = self._node.get_parameter('lqg_position_max').get_parameter_value().double_value
        self._max_estimate_age = self._node.get_parameter('max_estimate_age').get_parameter_value().double_value
        self._max_trim_speed = self._node.get_parameter('max_trim_speed').get_parameter_value().double_value
        self._max_theta_for_lqg = self._node.get_parameter('max_theta_for_lqg').get_parameter_value().double_value

        self._stabilize_before_mission = self._node.get_parameter('stabilize_before_mission').get_parameter_value().bool_value
        self._stabilize_theta_tol = self._node.get_parameter('stabilize_theta_tol').get_parameter_value().double_value
        self._stabilize_omega_tol = self._node.get_parameter('stabilize_omega_tol').get_parameter_value().double_value
        self._stabilize_settle_time = self._node.get_parameter('stabilize_settle_time').get_parameter_value().double_value
        self._stabilize_timeout = self._node.get_parameter('stabilize_timeout').get_parameter_value().double_value
        self._stabilize_rho = self._node.get_parameter('stabilize_rho').get_parameter_value().double_value
        self._stabilize_theta_max = self._node.get_parameter('stabilize_theta_max').get_parameter_value().double_value
        self._stabilize_position_max = self._node.get_parameter('stabilize_position_max').get_parameter_value().double_value
        self._plot_missions = self._node.get_parameter('plot_missions').get_parameter_value().bool_value
        self._plot_output_dir = self._node.get_parameter('plot_output_dir').get_parameter_value().string_value

    def _refresh_tuning_parameters(self):
        """Re-read the LQG tuning knobs from the parameter server.

        Called per goal, not once at startup: the controllers are BUILT per
        mission, so their weights should be read per mission too. Without this,
        `ros2 param set ... lqg_rho 10.0` is silently ignored because
        _get_node_parameters() ran once in __init__ - which wasted a flight."""
        g = lambda n: self._node.get_parameter(n).get_parameter_value()
        self._enable_lqg = g('enable_lqg').bool_value
        self._lqg_rho = g('lqg_rho').double_value
        self._lqg_theta_max = g('lqg_theta_max').double_value
        self._lqg_position_max = g('lqg_position_max').double_value
        self._max_trim_speed = g('max_trim_speed').double_value
        self._max_theta_for_lqg = g('max_theta_for_lqg').double_value
        self._stabilize_rho = g('stabilize_rho').double_value
        self._stabilize_theta_max = g('stabilize_theta_max').double_value
        self._stabilize_position_max = g('stabilize_position_max').double_value
        self._max_speed = g('max_speed').double_value
        self.log(f'LQG tuning for this mission: rho={self._lqg_rho:.2f} '
                 f'stab_rho={self._stabilize_rho:.2f} '
                 f'theta_max={self._lqg_theta_max:.3f} '
                 f'stab_theta_max={self._stabilize_theta_max:.3f} '
                 f'max_trim={self._max_trim_speed:.2f} v_max={self._max_speed:.2f}')

    def _load_identified_pendulum_params(self) -> tuple[float, float]:
        """L/xi as identified by estimate_length_and_damping, falling back to the
        node parameters. Read per mission rather than once at startup so a fresh
        identification is picked up without restarting this node - and because
        the rope can physically change between missions."""
        path = os.path.expanduser(f'~/.ros/hook_pendulum_params_{self._robot_name}.yaml')
        try:
            with open(path) as f:
                fitted = yaml.safe_load(f)
            L = float(fitted['length'])
            xi = float(fitted['damping'])
            # ZVD needs 0 < xi < 1 and LQR asserts the same; a sysid that fits
            # xi = 0 exactly (it used to) would otherwise raise mid-mission.
            if not (L > 0.0):
                raise ValueError(f'non-positive length {L}')
            if not (0.0 < xi < 1.0):
                self.log(f'Identified xi={xi} outside (0,1), clamping for the controllers')
                xi = min(max(xi, 1e-3), 0.99)
            self.log(f'Using identified pendulum params from {path}: L={L:.3f}m, xi={xi:.4f}')
            return L, xi
        except Exception as e:
            self.log(f'Could not use {path} ({e}); falling back to node parameters '
                     f'L={self.rope_length}, xi={self.xi}')
            return self.rope_length, self.xi

    def _create_subscriptions(self):
        qos_best_effort10 = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                                        durability=QoSDurabilityPolicy.VOLATILE)

        # BARE relative names: this node is launched with namespace=robot_name
        # (see alars_move_to_dumped_server_launch.py), so 'hook_swing_state'
        # already resolves to /<robot>/hook_swing_state. Prefixing robot_name
        # here as sway_controller's own nodes do - they are NOT namespaced -
        # would give /<robot>/<robot>/... and silently receive nothing.
        # Published BEST_EFFORT by HookKalmanFilter; a RELIABLE subscriber would
        # also silently receive nothing.
        self._node.create_subscription(
            JointState, 'hook_swing_state',
            self._swing_state_callback, qos_best_effort10
        )
        self._node.create_subscription(
            Odometry, 'smarc/odom',
            self._odom_callback, 10
        )

    def _swing_state_callback(self, msg:JointState):
        self._swing_state = msg

    def _odom_callback(self, msg:Odometry):
        # This twist is in the ODOM frame, not the child frame the message type
        # implies: dji_captain fills it from _velocity_ground, which it stamps
        # ODOM_FRAME. The LQR state is in base_flat_link, which carries the
        # drone's yaw, so it has to be rotated - at a 90deg heading the axes are
        # otherwise completely swapped. (Same trap as in HookKalmanFilter.)
        vel_in = Vector3Stamped()
        vel_in.header.stamp = msg.header.stamp
        vel_in.header.frame_id = msg.header.frame_id
        vel_in.vector = msg.twist.twist.linear

        vel_bf = self._drone_state.vector_stamped_in_base_flat(vel_in)
        if vel_bf is None:
            return
        self._drone_velocity_base_flat = np.array(
            [vel_bf.vector.x, vel_bf.vector.y, vel_bf.vector.z], float
        )

    def _create_publishers(self):
        qos_best_effort10 = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
        self.ref_publisher             = self._node.create_publisher(TwistStamped, 
                                                                     DJITopics.VELOCITY_SETPOINT_TOPIC, 
                                                                     qos_profile=qos_best_effort10)

        self.drone_frame_ref_publisher = self._node.create_publisher(Vector3Stamped, 
                                                                     'cmd_vel_drone_frame', 
                                                                     qos_profile=qos_best_effort10)


    @property
    def now_stamp(self):
        return self._node.get_clock().now().to_msg()
    
    @property
    def now_time(self):
        return self.now_stamp.sec + self.now_stamp.nanosec * 1e-9
    
    def log(self, msg: str):
        self._node.get_logger().info(msg)

    def _have_position(self, timeout:float = 5.0) -> bool:
        start_time = self.now_time
        flag = False

        while (self.now_time - start_time) < timeout:
            if self._drone_state._drone_in_map is not None:
                flag = True
                break
            self._node.get_logger().info('Waiting for drone in map position', throttle_duration_sec=2.0)
            time.sleep(0.5)
        
        if not flag:
            self._node.get_logger().error('Error: no trone in map position')

        return flag 

    def _on_goal_received(self, goal_request:dict) -> bool:
        try:
            gp : GeoPoint = GeoPoint()
            gp.latitude = goal_request['waypoint']['latitude']
            gp.longitude = goal_request['waypoint']['longitude']
            gp.altitude = goal_request['waypoint']['altitude']

            self._goal_in_map = self._drone_state.geopoint_to_pose_stamped_map(gp)
            if self._goal_in_map is None:
                self._node.get_logger().error('Failed to transform goal from latlon to map frame')
                return False
            self._goal_in_base_flat = self._drone_state.pose_stamped_in_base_flat(self._goal_in_map)
            if self._goal_in_base_flat is None:
                self._node.get_logger().error('Failed to transform goal from map frame to base_flat frame')
                return False
            
            self._goal_tolerance = float(goal_request['waypoint']['tolerance']) if 'tolerance' in goal_request['waypoint'] else self._default_goal_tolerance

            pos = self._goal_in_map.pose.position

            self.log(
                f'Received goal in map: [{pos.x:.2f},{pos.y:.2f},{pos.z:.2f}], tolerance: {self._goal_tolerance}\n This is an altitude-constant mission, only x and y coordinates will be considere'
            )

            pos_base_flat = self._goal_in_base_flat.pose.position
            self.log(
                f'Same goal in base_flat: [{pos_base_flat.x:.2f},{pos_base_flat.y:.2f},{pos_base_flat.z:.2f}]'
            )

            if not self._have_position():  
                self._node.get_logger().error('Error: no trone in map position')
                return False 
            
            drone_pose = self._drone_state._drone_in_map.pose.position
            drone_position = np.asarray([drone_pose.x, drone_pose.y], float)
            target_position = np.array([pos.x, pos.y], float)
            self._path_parametrizer = PathParametrizer(drone_position, target_position,
                                                         self._max_speed, self._max_acceleration)
            
            # Both the shaper and the controller are built from the SAME
            # identified pendulum, so a bad identification degrades them
            # together rather than making them disagree with each other.
            self._refresh_tuning_parameters()
            L, xi = self._load_identified_pendulum_params()
            self._mission_L, self._mission_xi = L, xi

            self.zvd = ZVD(L, xi)
            self.Ai, self.Ti = self.zvd.computeZVDMatrix()

            if self._enable_lqg:
                self._lqr = LQR(
                    L=L, xi=xi,
                    k_x=self._k_x, tau_x=self._tau_x,
                    k_y=self._k_y, tau_y=self._tau_y,
                    k_z=self._k_z, tau_z=self._tau_z,
                    v_max=self._max_speed,
                    rho=self._lqg_rho,
                    p_max=self._lqg_position_max,
                    pz_max=self._lqg_position_max,
                    theta_max=self._lqg_theta_max,
                )
                slowest = max(e.real for e in self._lqr.closed_loop_eigenvalues)
                self.log(f'LQG enabled: closed-loop slowest pole {slowest:+.3f} '
                         f'(open-loop swing decay was {-xi*np.sqrt(G/L):+.4f})')

                # Same plant, different priorities: tight on swing, loose on
                # position. See the parameter declarations for why.
                self._lqr_stabilize = LQR(
                    L=L, xi=xi,
                    k_x=self._k_x, tau_x=self._tau_x,
                    k_y=self._k_y, tau_y=self._tau_y,
                    k_z=self._k_z, tau_z=self._tau_z,
                    v_max=self._max_speed,
                    rho=self._stabilize_rho,
                    p_max=self._stabilize_position_max,
                    pz_max=self._stabilize_position_max,
                    theta_max=self._stabilize_theta_max,
                )
            else:
                self._lqr = None
                self._lqr_stabilize = None
                self.log('LQG disabled (enable_lqg=False) - flying the ZVD feedforward open loop')

            # Stabilise first, then fly. Skipped when there is no controller to
            # do it with, or when explicitly disabled.
            self._phase = ('STABILIZING'
                           if (self._stabilize_before_mission and self._lqr_stabilize is not None)
                           else 'MOVING')
            self._stabilize_started = self.now_time
            self._goal_received_time = self.now_time
            self._mission_samples = []
            self._within_tol_since:None|float = None
            self._hold_position_map:None|np.ndarray = None

            shaper_duration = self.Ti[-1]
            self._start_mission_time = self.now_time
            self._t_end = self._start_mission_time + self._path_parametrizer._missionTime + shaper_duration + self._sattle_extra
            

            return True

        except:
            self._node.get_logger().error('Failed to parse goal request')
            traceback.print_exc()
            return False
    
    def _on_cancel_received(self) -> bool:
        self.log('Cancel requested, stopping...')
        self._save_mission_plots('CANCELLED')
        self._goal_in_map = None
        return True
    
    def _prepare_loop(self) -> None:
        goal_error = np.array([
            self._goal_in_base_flat.pose.position.x,
            self._goal_in_base_flat.pose.position.y
        ])
        self._distance_from_goal = np.linalg.norm(goal_error)
        self.log(f'Goal error: [{goal_error[0]:.2f}, {goal_error[1]:.2f}] \n Total distance: {self._distance_from_goal:.2f}')
        return
    
    def _give_feedback(self) -> str:
        if self._distance_from_goal is not None:
            return f'Distance remaining: {self._distance_from_goal:.2f} (tolerance: {self._goal_tolerance:.2f}m)'
        else:
            return 'No distance remaining info'

    def _lqg_correction(self, position_reference_map, velocity_reference_base_flat):
        """The feedback trim added to the shaped feedforward, or None when the
        loop must stay open (LQG off, no estimate, or a stale one).

        Everything is assembled in base_flat_link, which is where the LQR model
        and the swing estimate both live. Position enters as an ERROR vector
        (drone minus plan) rotated into base_flat, so the reference passed to
        controlAction is zero for those states - expressing an absolute position
        in a body frame would be meaningless."""
        if self._lqr is None:
            return None

        if self._swing_state is None:
            self._node.get_logger().warning(
                'No hook_swing_state yet - flying feedforward only. Is '
                'hook_kalman_filter_node running?', throttle_duration_sec=5.0
            )
            return None

        stamp = self._swing_state.header.stamp
        age = self.now_time - (stamp.sec + stamp.nanosec * 1e-9)
        if age > self._max_estimate_age:
            self._node.get_logger().warning(
                f'hook_swing_state is {age:.2f}s old (limit {self._max_estimate_age:.2f}s) '
                f'- flying feedforward only', throttle_duration_sec=5.0
            )
            return None

        if self._drone_velocity_base_flat is None:
            self._node.get_logger().warning(
                'No drone velocity yet - flying feedforward only', throttle_duration_sec=5.0
            )
            return None

        drone_in_map = self._drone_state.drone_in_map_numpy
        if drone_in_map is None:
            return None

        # Position error as a VECTOR in map, then rotated into base_flat.
        err_map = Vector3Stamped()
        err_map.header.stamp = self.now_stamp
        err_map.header.frame_id = self._drone_state.MAP_FRAME
        err_map.vector.x = float(drone_in_map[0] - position_reference_map[0])
        err_map.vector.y = float(drone_in_map[1] - position_reference_map[1])
        err_map.vector.z = 0.0   # altitude-constant mission; z is held elsewhere
        err_bf = self._drone_state.vector_stamped_in_base_flat(err_map)
        if err_bf is None:
            return None

        i = LQR.IDX
        x = np.zeros(LQR.N_STATES)
        r = np.zeros(LQR.N_STATES)

        x[i['p_x']] = err_bf.vector.x
        x[i['p_y']] = err_bf.vector.y
        x[i['p_z']] = 0.0

        x[i['v_x']] = self._drone_velocity_base_flat[0]
        x[i['v_y']] = self._drone_velocity_base_flat[1]
        x[i['v_z']] = self._drone_velocity_base_flat[2]
        r[i['v_x']] = velocity_reference_base_flat.vector.x
        r[i['v_y']] = velocity_reference_base_flat.vector.y
        r[i['v_z']] = 0.0

        # The swing we want is none, so these references stay zero.
        x[i['theta_x']] = self._swing_state.position[0]
        x[i['theta_y']] = self._swing_state.position[1]
        x[i['omega_x']] = self._swing_state.velocity[0]
        x[i['omega_y']] = self._swing_state.velocity[1]

        # Large-angle guard: past this the linearised model (sin th ~ th) does
        # not hold and the gain asks for authority that does not exist, so the
        # feedback would be actively wrong rather than merely weak.
        theta_mag = max(abs(x[i['theta_x']]), abs(x[i['theta_y']]))
        if theta_mag > self._max_theta_for_lqg:
            self._node.get_logger().warning(
                f'|theta| {theta_mag:.3f} rad exceeds {self._max_theta_for_lqg:.3f} - '
                f'outside the linear model, dropping feedback this tick',
                throttle_duration_sec=2.0
            )
            return None

        u_fb = self._lqr.controlAction(x, r)

        # Cap the TRIM itself, not just the total. Saturating only the total
        # still lets a huge trim swamp the feedforward and turn the loop
        # bang-bang, which is exactly how the first flight went unstable.
        trim_speed = float(np.linalg.norm(u_fb[:2]))
        if trim_speed > self._max_trim_speed:
            u_fb = u_fb * (self._max_trim_speed / trim_speed)

        # A persistently large trim means the plan and the plant disagree (wrong
        # L/xi, or a real disturbance) - the feedforward is supposed to be doing
        # nearly all the work, so this is the cheap online health check.
        # est_age is the measurement that decides whether the delay hypothesis
        # for the 2026-07-26 instability holds: simulated, this loop tolerates
        # ~0.5s of feedback delay and starts pumping the swing around 0.8s
        # (63deg of phase at the 4.6s pendulum period). This is only the
        # transport/consumption half - add the KF's own "meas age" (logged by
        # hook_kalman_filter_node) for the total sensing delay.
        self._node.get_logger().info(
            f'[lqg] theta=[{x[i["theta_x"]]:+.3f},{x[i["theta_y"]]:+.3f}]rad '
            f'pos_err=[{x[i["p_x"]]:+.2f},{x[i["p_y"]]:+.2f}]m '
            f'trim=[{u_fb[0]:+.3f},{u_fb[1]:+.3f}]m/s '
            f'est_age={age*1000:.0f}ms',
            throttle_duration_sec=2.0
        )
        return u_fb

    def _stabilize_tick(self) -> bool|None:
        """Hold station and damp the payload before departing.

        References are exactly what was asked for: theta_x = theta_y = 0, and
        the drone held at wherever it was when this phase began. There is no
        feedforward here - the whole command IS the correction.

        Returns None while still stabilising; flips the phase to MOVING (and
        restarts the mission clock) once settled or timed out.
        """
        if self._hold_position_map is None:
            here = self._drone_state.drone_in_map_numpy
            if here is None:
                return None
            self._hold_position_map = here.copy()
            self.log(f'Stabilising payload before departure (tol {self._stabilize_theta_tol:.3f} rad '
                     f'for {self._stabilize_settle_time:.1f}s, timeout {self._stabilize_timeout:.1f}s)')

        now = self.now_time
        elapsed = now - self._stabilize_started

        # Reuse the mission correction machinery, but against the hold point and
        # with the payload-weighted gain.
        # During stabilise there is NO feedforward to protect, so the trim cap
        # sized for "small correction on top of a plan" only throws authority
        # away. It matters: at 0.15rad the hook's own horizontal speed is
        # L*omega ~ 0.79 m/s, so a 0.5 m/s cap cannot chase the payload and the
        # damping ends up barely better than letting it decay on its own. Give
        # the whole speed budget to the trim here.
        mission_lqr, self._lqr = self._lqr, self._lqr_stabilize
        mission_cap, self._max_trim_speed = self._max_trim_speed, self._max_speed
        hold_velocity = Vector3Stamped()
        hold_velocity.header.stamp = self.now_stamp
        hold_velocity.header.frame_id = self.BASE_FLAT_FRAME   # zero, frame irrelevant
        u = self._lqg_correction(self._hold_position_map[:2], hold_velocity)
        self._lqr = mission_lqr
        self._max_trim_speed = mission_cap

        if u is None:
            # No usable swing estimate - stabilising blind would be worse than
            # not stabilising, so hold still and let the timeout move us on.
            if elapsed > self._stabilize_timeout:
                self._node.get_logger().warning(
                    'Could not stabilise (no usable swing estimate) - starting the mission anyway'
                )
                self._begin_mission()
            self._record_sample('STABILIZING', self._hold_position_map[:2],
                                (0.0, 0.0), None, np.zeros(2))
            self._publish_velocity(np.zeros(2))
            return None

        theta = np.array([self._swing_state.position[0], self._swing_state.position[1]])
        omega = np.array([self._swing_state.velocity[0], self._swing_state.velocity[1]])
        settled = (np.max(np.abs(theta)) < self._stabilize_theta_tol and
                   np.max(np.abs(omega)) < self._stabilize_omega_tol)

        if settled:
            if self._within_tol_since is None:
                self._within_tol_since = now
            if (now - self._within_tol_since) >= self._stabilize_settle_time:
                self.log(f'Payload stable after {elapsed:.1f}s '
                         f'(|theta| {np.max(np.abs(theta)):.4f} rad) - starting the mission')
                self._begin_mission()
                return None
        else:
            self._within_tol_since = None

        if elapsed > self._stabilize_timeout:
            self._node.get_logger().warning(
                f'Payload did not settle within {self._stabilize_timeout:.1f}s '
                f'(|theta| {np.max(np.abs(theta)):.4f} rad, tol {self._stabilize_theta_tol:.3f}) '
                f'- starting the mission anyway'
            )
            self._begin_mission()
            return None

        self._node.get_logger().info(
            f'[stabilize] t={elapsed:.1f}s/{self._stabilize_timeout:.0f}s '
            f'|theta|={np.max(np.abs(theta)):.4f}/{self._stabilize_theta_tol:.3f}rad '
            f'theta=[{theta[0]:+.4f},{theta[1]:+.4f}]rad '
            f'omega=[{omega[0]:+.4f},{omega[1]:+.4f}]rad/s u=[{u[0]:+.3f},{u[1]:+.3f}]m/s',
            throttle_duration_sec=1.0
        )
        self._record_sample('STABILIZING', self._hold_position_map[:2],
                            (0.0, 0.0), u, u[:2])
        self._publish_velocity(u[:2])
        return None

    def _record_sample(self, phase:str, p_ref_map, v_ff_bf, trim, u_published):
        """One row per control tick, for the end-of-mission plots. Recorded even
        when there is no swing estimate (NaN), so a gap in the plot is visible
        rather than silently interpolated over."""
        if not self._plot_missions:
            return

        drone = self._drone_state.drone_in_map_numpy
        nan = float('nan')
        th = self._swing_state.position if self._swing_state is not None else (nan, nan)
        om = self._swing_state.velocity if self._swing_state is not None else (nan, nan)

        self._mission_samples.append({
            't': self.now_time - self._goal_received_time,
            'phase': phase,
            'theta_x': float(th[0]), 'theta_y': float(th[1]),
            'omega_x': float(om[0]), 'omega_y': float(om[1]),
            'p_x': float(drone[0]) if drone is not None else nan,
            'p_y': float(drone[1]) if drone is not None else nan,
            'p_ref_x': float(p_ref_map[0]), 'p_ref_y': float(p_ref_map[1]),
            'v_ff_x': float(v_ff_bf[0]), 'v_ff_y': float(v_ff_bf[1]),
            'trim_x': float(trim[0]) if trim is not None else nan,
            'trim_y': float(trim[1]) if trim is not None else nan,
            'u_x': float(u_published[0]), 'u_y': float(u_published[1]),
        })

    def _save_mission_plots(self, outcome:str):
        if not self._plot_missions or not self._mission_samples:
            return
        try:
            ok, message = save_mission_plots(
                self._mission_samples, self._plot_output_dir, self._robot_name,
                theta_tol=self._stabilize_theta_tol,
            )
            self.log(f'[{outcome}] {message}') if ok else self._node.get_logger().warning(message)
        except Exception as e:
            # Never let plotting take down a mission result.
            self._node.get_logger().warning(f'Failed to save mission plots: {e}')
        finally:
            self._mission_samples = []

    def _begin_mission(self):
        """Restart the mission clock. The path is parametrised from t=0, so the
        time spent stabilising must NOT count against it - otherwise the plan
        starts part-way through and the drone jumps."""
        self._phase = 'MOVING'
        self._start_mission_time = self.now_time
        self._t_end = (self._start_mission_time + self._path_parametrizer._missionTime
                       + self.Ti[-1] + self._sattle_extra)

    def _publish_velocity(self, u_xy:np.ndarray) -> np.ndarray:
        """Publishes and returns what was ACTUALLY sent, so the plots show the
        real command rather than a pre-saturation value that never existed."""
        speed = float(np.linalg.norm(u_xy))
        if speed > self._max_speed:
            u_xy = u_xy * (self._max_speed / speed)
        setpoint = TwistStamped()
        setpoint.header.stamp = self.now_stamp
        setpoint.header.frame_id = self.BASE_FLAT_FRAME
        setpoint.twist.linear.x = float(u_xy[0])
        setpoint.twist.linear.y = float(u_xy[1])
        self.ref_publisher.publish(setpoint)

        # ALSO tell the estimator what we just commanded. HookKalmanFilter
        # subscribes to cmd_vel_drone_frame and uses it as the input u in
        #     a_drone = -tau*v + k*u
        # which is the term that forces the pendulum. Nothing was publishing
        # this topic, so the filter had u = 0 permanently: it saw the drone's
        # velocity but not the command driving it, and therefore mis-predicted
        # the swing induced by our own control action. Harmless open loop (the
        # error just sits in the estimate), but in closed loop the estimator is
        # systematically wrong about the effect of the very command the
        # controller is applying - which is a textbook way to destabilise.
        drone_frame_cmd = Vector3Stamped()
        drone_frame_cmd.header.stamp = setpoint.header.stamp
        drone_frame_cmd.header.frame_id = self.BASE_FLAT_FRAME
        drone_frame_cmd.vector.x = float(u_xy[0])
        drone_frame_cmd.vector.y = float(u_xy[1])
        self.drone_frame_ref_publisher.publish(drone_frame_cmd)
        return u_xy

    def _loop_inner(self) -> bool|None:
        if self._phase == 'STABILIZING':
            return self._stabilize_tick()

        goal_in_base_flat_now = self._drone_state.pose_stamped_in_base_flat(self._goal_in_map)
        if goal_in_base_flat_now is None:
            self.log("Failed to transform goal into current base_flat frame, skipping this tick.")
            return None

        goal_error = np.array([
            goal_in_base_flat_now.pose.position.x,
            goal_in_base_flat_now.pose.position.y
        ])
        self._distance_from_goal = np.linalg.norm(goal_error)

        if self._distance_from_goal < self._goal_tolerance:
            self.log(f'Goal reaced withing tolerance {self._goal_tolerance}, return SUCCESS')
            self._save_mission_plots('SUCCESS')
            return True
        #self.log(f'Distance remaining: {self._distance_from_goal:.2f}')

        now = self.now_time
        elapsed_time = now - self._start_mission_time

        if now > self._t_end:
            self._node.get_logger().error('Goal not reached within mission time, return FAILURE')
            self._save_mission_plots('FAILURE')
            return False

        position_references, velocity_references = self.zvd.shapeReferences(
            self.Ai, self.Ti, self._path_parametrizer, elapsed_time
        )

        time_remaining = self._path_parametrizer._missionTime - elapsed_time
        self._node.get_logger().info(
            f'[debug] t_remaining_in_plan: {time_remaining:.2f}s '
            f'(elapsed: {elapsed_time:.2f}s / missionTime: {self._path_parametrizer._missionTime:.2f}s), '
            f'v_map: [{velocity_references[0]:+.3f}, {velocity_references[1]:+.3f}]',
            throttle_duration_sec=2.0
        )

        vel_in_map = Vector3Stamped()
        vel_in_map.header.stamp = self.now_stamp
        vel_in_map.header.frame_id = self._drone_state.MAP_FRAME
        vel_in_map.vector.x = float(velocity_references[0])
        vel_in_map.vector.y = float(velocity_references[1])

        vel_in_base_flat = self._drone_state.vector_stamped_in_base_flat(vel_in_map)
        if vel_in_base_flat is None:
            self.log("Failed to transform velocity reference into base_flat frame, skipping this tick.")
            return None

        # Feedforward: the ZVD-shaped plan, which is what actually flies the
        # mission. The LQR only trims it - see LQG.py.
        u = np.array([vel_in_base_flat.vector.x, vel_in_base_flat.vector.y], float)

        correction = self._lqg_correction(position_references, vel_in_base_flat)
        if correction is not None:
            u = u + correction[:2]

        # Saturates the TOTAL command: the feedforward already respects max_speed
        # by construction, the correction does not.
        u_published = self._publish_velocity(u)
        self._record_sample('MOVING', position_references,
                            (vel_in_base_flat.vector.x, vel_in_base_flat.vector.y),
                            correction, u_published)

        self._node.get_logger().info(
            f'[debug] v_base_flat published (pre-saturation): [{u[0]:+.3f}, {u[1]:+.3f}]',
            throttle_duration_sec=2.0
        )

        return None

def main(args=None):
    rclpy.init(args=args)
    node = Node("alars_move_to_dumped_action_server")
    move_to_dumped_action_server = MoveToDumpedAction(node)
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()