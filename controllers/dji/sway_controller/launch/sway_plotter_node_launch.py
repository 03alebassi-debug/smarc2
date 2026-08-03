import os

from ament_index_python.packages import get_package_share_directory
from launch.substitutions        import PathJoinSubstitution, LaunchConfiguration
from launch                      import LaunchDescription
from launch.actions              import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions          import Node

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
    # The tunables below default to '' (empty) rather than to a value, so that
    # anything not passed on the command line falls through to the config yaml.
    plot_output_dir_arg = DeclareLaunchArgument(
        'plot_output_dir',
        default_value='',
        description='Where the plots are written (unset -> yaml)'
    )
    ground_truth_topic_arg = DeclareLaunchArgument(
        'ground_truth_topic',
        default_value='',
        description='Ground truth topic to compare the estimate against (unset -> yaml)'
    )
    theta_tol_arg = DeclareLaunchArgument(
        'theta_tol',
        default_value='',
        description='Settle tolerance band drawn on the swing plot, rad (unset -> yaml)'
    )

    robot_name = LaunchConfiguration('robot_name')
    use_sim_time = LaunchConfiguration('use_sim_time')

    config_file_name = 'sway_plotter_node_config.yaml'
    config_dir = os.path.join(get_package_share_directory('sway_controller'), 'config')
    config_file = PathJoinSubstitution([config_dir, config_file_name])

    def make_node(context, *args, **kwargs):
        # A LaunchConfiguration can only be read inside a context, which is why
        # this lives in an OpaqueFunction: an argument becomes a ROS parameter
        # only if it was actually typed, otherwise the yaml value stands.
        overrides = {}
        for name, cast in (('plot_output_dir', str),
                           ('ground_truth_topic', str),
                           ('theta_tol', float)):
            value = LaunchConfiguration(name).perform(context)
            if value != '':
                overrides[name] = cast(value)

        node = Node(
            package='sway_controller',
            executable='sway_plotter_node',
            name='sway_plotter_node',
            namespace=robot_name,
            output='screen',
            # Later entries win: yaml < robot_name/use_sim_time < command line
            parameters=[
                config_file,
                {
                    'robot_name':robot_name,
                    'use_sim_time':use_sim_time
                },
                overrides
            ]
        )
        return [node]

    return LaunchDescription([
        robot_name_arg,
        use_sim_time_arg,
        plot_output_dir_arg,
        ground_truth_topic_arg,
        theta_tol_arg,
        OpaqueFunction(function=make_node)
    ])
