#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sway_controller.HookGroundTruthComparator import HookGroundTruthComparator


def main():
    rclpy.init()

    node = Node("hook_ground_truth_comparator_node")

    node.declare_parameter("robot_name", "M350")
    node.declare_parameter("ground_truth_topic", "hook_ground_truth")
    node.declare_parameter("output_topic", "hook_ground_truth_base_flat")
    node.declare_parameter("velocity_smoothing_window", 1)

    HookGroundTruthComparator(
        node,
        robot_name=node.get_parameter("robot_name").value,
        ground_truth_topic=node.get_parameter("ground_truth_topic").value,
        output_topic=node.get_parameter("output_topic").value,
        velocity_smoothing_window=node.get_parameter("velocity_smoothing_window").value,
    )

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
