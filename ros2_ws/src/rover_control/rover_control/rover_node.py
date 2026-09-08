import random
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Float32MultiArray

# Mission states
EXPLORING = "EXPLORING"
SURVEYING = "SURVEYING"
INVESTIGATING = "INVESTIGATING"
DONE = "DONE"

# World bounds (matching Day 2's 50x50m world, hidden_water_locations.json bounds)
WORLD_MIN = -24.0
WORLD_MAX = 24.0
N_CANDIDATES = 5


class RoverNode(Node):
    def __init__(self):
        super().__init__("rover_node")

        self.declare_parameter("exploration_duration_sec", 200.0)
        self.exploration_duration = self.get_parameter("exploration_duration_sec").value

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.survey_request_pub = self.create_publisher(Point, "/survey_request", 10)
        self.survey_result_sub = self.create_subscription(
            Float32MultiArray, "/drone_survey_result", self.handle_survey_result, 10
        )

        self.state = EXPLORING
        self.elapsed = 0.0
        self.candidates = []
        self.results = []
        self.pending_candidate_index = None

        self.timer_period = 1.0
        self.timer = self.create_timer(self.timer_period, self.tick)

        self.get_logger().info(
            f"Rover mission started. Exploring for {self.exploration_duration:.0f}s before deploying drone."
        )

    def tick(self):
        if self.state == EXPLORING:
            self._do_explore()
        elif self.state == SURVEYING:
            self._do_survey_step()
        elif self.state == INVESTIGATING:
            self._do_investigate()
        elif self.state == DONE:
            pass

    def _do_explore(self):
        move_cmd = Twist()
        move_cmd.linear.x = 0.5
        self.cmd_vel_pub.publish(move_cmd)

        self.elapsed += self.timer_period
        if self.elapsed >= self.exploration_duration:
            self.get_logger().info(
                f"Exploration window complete ({self.elapsed:.0f}s). No independent ground-based "
                f"water detection available (documented simplification) -- deploying drone."
            )
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)

            self.candidates = self._generate_candidates()
            self.get_logger().info(f"Generated {len(self.candidates)} candidate survey points.")
            self.state = SURVEYING
            self.pending_candidate_index = 0
            self._request_next_survey()

    def _generate_candidates(self):
        rng = random.Random(42)
        return [
            (rng.uniform(WORLD_MIN, WORLD_MAX), rng.uniform(WORLD_MIN, WORLD_MAX))
            for _ in range(N_CANDIDATES)
        ]

    def _request_next_survey(self):
        if self.pending_candidate_index >= len(self.candidates):
            self._rank_and_select()
            return
        x, y = self.candidates[self.pending_candidate_index]
        self.get_logger().info(
            f"Requesting survey {self.pending_candidate_index + 1}/{len(self.candidates)} at ({x:.2f}, {y:.2f})"
        )
        msg = Point()
        msg.x, msg.y, msg.z = x, y, 0.0
        self.survey_request_pub.publish(msg)

    def _do_survey_step(self):
        pass  # waiting for handle_survey_result; nothing to do per-tick here

    def handle_survey_result(self, msg: Float32MultiArray):
        if self.state != SURVEYING:
            return
        x, y, score, std = msg.data
        self.get_logger().info(f"Survey result received: ({x:.2f}, {y:.2f}) -> score={score:.3f}")
        self.results.append((x, y, score, std))
        self.pending_candidate_index += 1
        self._request_next_survey()

    def _rank_and_select(self):
        self.results.sort(key=lambda r: r[2], reverse=True)
        best = self.results[0]
        self.get_logger().info(
            f"Ranking complete. Best candidate: ({best[0]:.2f}, {best[1]:.2f}) with score={best[2]:.3f}"
        )
        for i, r in enumerate(self.results):
            self.get_logger().info(f"  #{i+1}: ({r[0]:.2f}, {r[1]:.2f}) score={r[2]:.3f}")

        self.target = best
        self.investigate_ticks = 0
        self.state = INVESTIGATING

    def _do_investigate(self):
        move_cmd = Twist()
        move_cmd.linear.x = 0.5
        self.cmd_vel_pub.publish(move_cmd)
        self.investigate_ticks += 1

        if self.investigate_ticks >= 10:
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            self.get_logger().info(
                f"Mission complete. Investigated target ({self.target[0]:.2f}, {self.target[1]:.2f}), "
                f"final score={self.target[2]:.3f}"
            )
            self.state = DONE


def main(args=None):
    rclpy.init(args=args)
    node = RoverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
