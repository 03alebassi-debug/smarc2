#!/usr/bin/env python3

import os

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sway_controller.EstimateLengthAndDamping import EstimateLengthAndDamping


def main():
    rclpy.init()

    node = Node("estimate_length_and_damping_node")

    node.declare_parameter("robot_name", "M350")
    node.declare_parameter("output_path", "")

    robot_name = node.get_parameter("robot_name").value
    output_path = node.get_parameter("output_path").value

    if not output_path:
        output_path = os.path.expanduser(f"~/.ros/hook_pendulum_params_{robot_name}.yaml")

    EstimateLengthAndDamping(
        node,
        robot_name=robot_name,
        output_path=output_path,
    )

    node.get_logger().info(f"Identified L/xi will be saved to {output_path}")

    # GentlerActionServer's execution loop uses node.create_rate(...)/rate.sleep(),
    # which needs the executor to keep processing concurrently with the goal
    # execution callback to wake it back up - a single-threaded executor (the
    # default for plain rclpy.spin()) would deadlock the whole node the moment
    # a goal starts executing. Same reason gimbal_action.py uses this too.
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
