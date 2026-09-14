import math
import random
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

EXPLORING = "EXPLORING"
SURVEYING = "SURVEYING"
INVESTIGATING = "INVESTIGATING"
DONE = "DONE"

WORLD_MIN = -24.0
WORLD_MAX = 24.0
N_CANDIDATES = 5

ARRIVAL_DISTANCE = 1.0       # metres; close enough counts as "arrived"
MAX_LINEAR_SPEED = 0.5
MAX_ANGULAR_SPEED = 0.8
HEADING_TOLERANCE = 0.15     # radians (~8.6 deg); turn in place if off by more than this
INVESTIGATE_TIMEOUT_TICKS = 120  # safety cap so a bad path can't run forever


def yaw_from_quaternion(q):
    # Standard quaternion-to-yaw conversion (rotation about Z only, which is
    # all that matters for a ground rover)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


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
        self.odom_sub = self.create_subscription(
            Odometry, "/odometry", self.handle_odometry, 10
        )

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.odom_received = False

        self.state = EXPLORING
        self.elapsed = 0.0
        self.candidates = []
        self.results = []
        self.pending_candidate_index = None
        self.investigate_ticks = 0

        self.timer_period = 1.0
        self.timer = self.create_timer(self.timer_period, self.tick)

        self.get_logger().info(
            f"Rover mission started. Exploring for {self.exploration_duration:.0f}s before deploying drone."
        )

    def handle_odometry(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.odom_received = True

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
            self.cmd_vel_pub.publish(Twist())  # stop

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
        pass

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
        self.get_logger().info(
            f"Navigating from current position ({self.current_x:.2f}, {self.current_y:.2f}) "
            f"to target ({self.target[0]:.2f}, {self.target[1]:.2f})"
        )

    def _do_investigate(self):
        self.investigate_ticks += 1

        target_x, target_y = self.target[0], self.target[1]
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance = math.hypot(dx, dy)

        if distance <= ARRIVAL_DISTANCE or self.investigate_ticks >= INVESTIGATE_TIMEOUT_TICKS:
            self.cmd_vel_pub.publish(Twist())
            reason = "arrived" if distance <= ARRIVAL_DISTANCE else "timeout (safety cap reached)"
            self.get_logger().info(
                f"Mission complete ({reason}). Final position ({self.current_x:.2f}, {self.current_y:.2f}), "
                f"distance to target: {distance:.2f}m. Investigated target "
                f"({target_x:.2f}, {target_y:.2f}), final score={self.target[2]:.3f}"
            )
            self.state = DONE
            return

        desired_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(desired_heading - self.current_yaw)

        move_cmd = Twist()
        if abs(heading_error) > HEADING_TOLERANCE:
            # Turn in place toward the target first
            move_cmd.angular.z = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, heading_error))
            move_cmd.linear.x = 0.0
        else:
            # Facing roughly the right way -- drive forward, with a little
            # ongoing steering correction
            move_cmd.linear.x = MAX_LINEAR_SPEED
            move_cmd.angular.z = max(-0.3, min(0.3, heading_error))

        self.cmd_vel_pub.publish(move_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RoverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
