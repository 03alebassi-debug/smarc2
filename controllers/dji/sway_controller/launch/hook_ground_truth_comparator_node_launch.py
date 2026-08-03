from launch                import LaunchDescription
from launch.actions        import DeclareLaunchArgument
from launch.substitutions  import LaunchConfiguration
from launch_ros.actions    import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='M350',
        description='Namespace for the robot'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use simulation clock instead of wall clock'
    )
    ground_truth_topic_arg = DeclareLaunchArgument(
        'ground_truth_topic',
        default_value='hook_ground_truth',
        description="Unity's raw ground truth, in the unity_origin frame"
    )
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='hook_ground_truth_base_flat',
        description='Where the transformed ground truth is republished'
    )
    velocity_smoothing_window_arg = DeclareLaunchArgument(
        'velocity_smoothing_window',
        default_value='1',
        description='Samples averaged when differentiating position into velocity. '
                    'Defaults to 1 (no smoothing) on purpose: this is the REFERENCE '
                    'signal, and averaging lag costs more error than the noise does.'
    )

    robot_name = LaunchConfiguration('robot_name')
    use_sim_time = LaunchConfiguration('use_sim_time')

    node = Node(
        package='sway_controller',
        executable='hook_ground_truth_comparator_node',
        name='hook_ground_truth_comparator_node',
        namespace=robot_name,
        output='screen',
        parameters=[{
            'robot_name': robot_name,
            'use_sim_time': use_sim_time,
            'ground_truth_topic': LaunchConfiguration('ground_truth_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
            'velocity_smoothing_window': ParameterValue(
                LaunchConfiguration('velocity_smoothing_window'), value_type=int),
        }]
    )

    return LaunchDescription([
        robot_name_arg,
        use_sim_time_arg,
        ground_truth_topic_arg,
        output_topic_arg,
        velocity_smoothing_window_arg,
        node
    ])
