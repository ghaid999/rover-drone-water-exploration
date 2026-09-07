import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Float32MultiArray
import time

# Make the Stage 4 scoring pipeline importable
sys.path.append("/home/majed/rover_drone_project/ai_model")
from water_potential_scorer import load_model, get_water_potential_score_averaged


class DroneNode(Node):
    def __init__(self):
        super().__init__("drone_node")

        self.model = load_model()
        self.get_logger().info("Water-potential model loaded.")

        self.cmd_vel_pub = self.create_publisher(Twist, "/drone_cmd_vel", 10)
        self.result_pub = self.create_publisher(Float32MultiArray, "/drone_survey_result", 10)

        self.survey_sub = self.create_subscription(
            Point, "/survey_request", self.handle_survey_request, 10
        )

        self.get_logger().info("Drone node ready, waiting for survey requests on /survey_request")

    def handle_survey_request(self, msg: Point):
        target_x, target_y = msg.x, msg.y
        self.get_logger().info(f"Received survey request for ({target_x:.2f}, {target_y:.2f})")

        self._move_to(target_x, target_y)

        score, std = get_water_potential_score_averaged(target_x, target_y, self.model, n_samples=10)
        self.get_logger().info(f"Survey complete: score={score:.3f} (std={std:.3f})")

        result = Float32MultiArray()
        result.data = [target_x, target_y, score, std]
        self.result_pub.publish(result)

    def _move_to(self, x, y):
        """Simplified kinematic movement: send a brief forward velocity command
        as a visual/simulation cue, then stop. Position for scoring purposes is
        taken directly from the requested target, consistent with the drone's
        documented kinematic simplification (Day 2)."""
        move_cmd = Twist()
        move_cmd.linear.x = 1.0
        self.cmd_vel_pub.publish(move_cmd)
        time.sleep(1.0)

        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DroneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
