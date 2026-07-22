import numpy   as np
import control as ct 

import rclpy
from rclpy.node      import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos       import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from geographic_msgs.msg import GeoPoint
from geometry_msgs.msg   import PoseStamped, TwistStamped, Vector3Stamped


from smarc_action_base.gentler_action_server import GentlerActionServer

from alars.alars_common import DroneState

from dji_msgs.msg import Topics as DJITopics
from dji_msgs.msg import Links  as DJILinks

import traceback
import time

from sway_controller import PathParametrizer, ZVD

G = 9.81

class MoveToDumpedAction():
    def __init__(self, node:Node):
        self._node:Node = node 

        self._get_node_parameters()
        self._create_state_space()
        self._create_publishers()

        self.BASE_FLAT_FRAME : str = self._robot_name + '/' + DJILinks.BASE_FLAT
        self._drone_state = DroneState(node, self._robot_name)

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
        node.declare_parameter('max_speed', 3.0, double_desc)
        node.declare_parameter('max_acceleration', 2.0, double_desc)
        node.declare_parameter('sattle_extra', 20.0, double_desc)
        node.declare_parameter('rope_length', 10.0, double_desc)
        node.declare_parameter('xi', 0.1, double_desc)

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

    def _create_publishers(self):
        qos_best_effort10 = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
        self.ref_publisher = self._node.create_publisher(TwistStamped, DJITopics.VELOCITY_SETPOINT_TOPIC, qos_profile=qos_best_effort10)


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
            
            self.zvd = ZVD(self.rope_length, self.xi)
            self.Ai, self.Ti = self.zvd.computeZVDMatrix()

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

    def _create_state_space(self):
        cx, cy = self._tau_x / self.rope_length, self._tau_y / self.rope_length      
        dx, dy = self._k_x / self.rope_length,  self._k_y / self.rope_length        

        self.A = np.array([
        [0,0,0, 1,            0,             0,             0,           0,         0,           0        ],
        [0,0,0, 0,            1,             0,             0,           0,         0,           0        ],
        [0,0,0, 0,            0,             1,             0,           0,         0,           0        ],
        [0,0,0,-self._tau_x,  0,             0,             0,           0,         0,           0        ],
        [0,0,0, 0,            -self._tau_y,  0,             0,           0,         0,           0        ],
        [0,0,0, 0,            0,             -self._tau_z,  0,           0,         0,           0        ],
        [0,0,0, 0,            0,             0,             0,           1,         0,           0        ],
        [0,0,0, cx,           0,             0,             -self.wn**2, -self.dwn, 0,           0        ],
        [0,0,0, 0,            0,             0,             0,           0,         0,           1        ],
        [0,0,0, 0,            cy,            0,             0,           0,         -self.wn**2, -self.dwn]])

        self.B = np.array([
               [0,0,0],         [0,0,0],           [0,0,0],
               [self._k_x,0,0], [0,self._k_y,0],   [0,0,self._k_z],
               [0,0,0],         [-dx,0,0],[0,0,0], [0,-dy,0]])
        
        self.C = np.zeros((14, 10))
        for r, s in [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,8),(8,7),(9,9)]:
            self.C[r, s] = 1.0
            
        self.C[10,6] = self.rope_length; self.C[11,8] = self.rope_length; self.C[12,7] = self.rope_length; self.C[13,9] = self.rope_length

        self.D = np.zeros((14, 3))

        self.state_space = ct.ss(self.A, self.B, self.C, self.D)

    def _loop_inner(self) -> bool|None:
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
            return True
        #self.log(f'Distance remaining: {self._distance_from_goal:.2f}')

        now = self.now_time
        elapsed_time = now - self._start_mission_time

        if now > self._t_end:
            self._node.get_logger().error('Goal not reached within mission time, return FAILURE')
            return False

        _, velocity_references = self.zvd.shapeReferences(self.Ai, self.Ti, self._path_parametrizer, elapsed_time)

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

        setpoint = TwistStamped()
        setpoint.header.stamp = self.now_stamp
        setpoint.header.frame_id = self.BASE_FLAT_FRAME
        setpoint.twist.linear.x = vel_in_base_flat.vector.x
        setpoint.twist.linear.y = vel_in_base_flat.vector.y
        self.ref_publisher.publish(setpoint)

        self._node.get_logger().info(
            f'[debug] v_base_flat published: [{setpoint.twist.linear.x:+.3f}, {setpoint.twist.linear.y:+.3f}]',
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