#!/usr/bin/env bash
# ROS 2 <-> Gazebo topic bridge.
#
# Relays the two command topics into the simulator and odometry back out, and
# bridges /clock so that both nodes can run on simulated time. Without /clock,
# a node started with use_sim_time:=true never advances its timer and appears
# frozen with no error message.

set -e
ros2 run ros_gz_bridge parameter_bridge \
  /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
  /drone_cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
  /odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry \
  '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
