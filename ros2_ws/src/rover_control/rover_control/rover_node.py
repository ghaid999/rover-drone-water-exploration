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

# --- Candidate generation ---
WORLD_MIN = -24.0
WORLD_MAX = 24.0
N_CANDIDATES = 20          # raised from 5: evaluation showed the ranking model
                           # contributes least at small candidate counts.

# --- Exploration sweep geometry ---
# Boustrophedon ("lawnmower") sweep. Lane spacing is twice the ground-sensor
# radius, so adjacent lanes tile the swept area without gaps.
#   5 lanes x 40 m of lane          = 200 m
#   4 lane transitions x 10 m       =  40 m
#   total sweep path                = 240 m
#   at CRUISE_SPEED 0.5 m/s         = 480 s of driving
# The time cap below adds margin for the initial approach and for turning.
SWEEP_MIN = -20.0
SWEEP_MAX = 20.0
SENSOR_RADIUS = 5.0
LANE_SPACING = 2.0 * SENSOR_RADIUS

# --- Motion ---
CRUISE_SPEED = 0.5
MAX_ANGULAR_SPEED = 0.8
HEADING_TOLERANCE = 0.15     # rad; turn in place above this
WAYPOINT_TOLERANCE = 1.5     # m; sweep waypoints need not be hit precisely
ARRIVAL_DISTANCE = 1.0       # m; final target tolerance
INVESTIGATE_TIMEOUT_TICKS = 300


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def generate_sweep_waypoints():
    """Boustrophedon sweep: alternate left-to-right and right-to-left lanes."""
    wps = []
    y = SWEEP_MIN
    going_right = True
    while y <= SWEEP_MAX + 1e-6:
        if going_right:
            wps.append((SWEEP_MIN, y))
            wps.append((SWEEP_MAX, y))
        else:
            wps.append((SWEEP_MAX, y))
            wps.append((SWEEP_MIN, y))
        going_right = not going_right
        y += LANE_SPACING
    return wps


class RoverNode(Node):
    def __init__(self):
        super().__init__("rover_node")

        # Safety cap on exploration, not the primary termination condition.
        # Exploration normally ends when the sweep pattern is complete.
        self.declare_parameter("exploration_duration_sec", 700.0)
        self.exploration_cap = self.get_parameter("exploration_duration_sec").value

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.survey_request_pub = self.create_publisher(Point, "/survey_request", 10)
        self.create_subscription(Float32MultiArray, "/drone_survey_result",
                                 self.handle_survey_result, 10)
        self.create_subscription(Odometry, "/odometry", self.handle_odometry, 10)

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.odom_received = False
        self.odom_warned = False

        self.state = EXPLORING
        self.elapsed = 0.0
        self.waypoints = generate_sweep_waypoints()
        self.waypoint_index = 0
        self.candidates = []
        self.results = []
        self.pending_candidate_index = None
        self.investigate_ticks = 0

        self.timer_period = 1.0
        self.create_timer(self.timer_period, self.tick)

        sweep_len = self._sweep_path_length()
        self.get_logger().info(
            f"Rover mission started. Sweeping {len(self.waypoints)} waypoints "
            f"({sweep_len:.0f} m of path, ~{sweep_len / CRUISE_SPEED:.0f} s at "
            f"{CRUISE_SPEED} m/s). Safety cap {self.exploration_cap:.0f} s."
        )

    def _sweep_path_length(self):
        total = 0.0
        for i in range(len(self.waypoints) - 1):
            ax, ay = self.waypoints[i]
            bx, by = self.waypoints[i + 1]
            total += math.hypot(bx - ax, by - ay)
        return total

    def handle_odometry(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.odom_received = True

    def tick(self):
        if self.state == EXPLORING:
            self._do_explore()
        elif self.state == INVESTIGATING:
            self._do_investigate()
        # SURVEYING is event-driven; DONE does nothing.

    def _drive_toward(self, tx, ty):
        """Publish one control command toward (tx, ty). Returns distance remaining."""
        dx, dy = tx - self.current_x, ty - self.current_y
        distance = math.hypot(dx, dy)
        heading_error = normalize_angle(math.atan2(dy, dx) - self.current_yaw)

        cmd = Twist()
        if abs(heading_error) > HEADING_TOLERANCE:
            cmd.angular.z = max(-MAX_ANGULAR_SPEED,
                                min(MAX_ANGULAR_SPEED, 2.0 * heading_error))
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = CRUISE_SPEED
            cmd.angular.z = max(-0.3, min(0.3, heading_error))
        self.cmd_vel_pub.publish(cmd)
        return distance

    def _do_explore(self):
        self.elapsed += self.timer_period

        if not self.odom_received:
            if self.elapsed > 5.0 and not self.odom_warned:
                self.get_logger().warn(
                    "No /odometry received. Is the ros_gz_bridge running with "
                    "/odometry bridged? Exploration cannot navigate without it."
                )
                self.odom_warned = True
            return

        if self.waypoint_index >= len(self.waypoints):
            self._finish_exploration("sweep pattern completed")
            return

        if self.elapsed >= self.exploration_cap:
            self._finish_exploration(
                f"safety time cap reached at {self.elapsed:.0f} s "
                f"({self.waypoint_index}/{len(self.waypoints)} waypoints)"
            )
            return

        wx, wy = self.waypoints[self.waypoint_index]
        if self._drive_toward(wx, wy) <= WAYPOINT_TOLERANCE:
            self.waypoint_index += 1
            self.get_logger().info(
                f"Sweep waypoint {self.waypoint_index}/{len(self.waypoints)} "
                f"reached ({wx:.1f}, {wy:.1f})"
            )

    def _finish_exploration(self, reason):
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info(
            f"Exploration ended: {reason}. No independent ground-based water "
            f"detection available (documented simplification) -- deploying drone."
        )
        self.candidates = self._generate_candidates()
        self.get_logger().info(
            f"Generated {len(self.candidates)} candidate survey points."
        )
        self.state = SURVEYING
        self.pending_candidate_index = 0
        self._request_next_survey()

    def _generate_candidates(self):
        rng = random.Random(42)
        return [(rng.uniform(WORLD_MIN, WORLD_MAX),
                 rng.uniform(WORLD_MIN, WORLD_MAX)) for _ in range(N_CANDIDATES)]

    def _request_next_survey(self):
        if self.pending_candidate_index >= len(self.candidates):
            self._rank_and_select()
            return
        x, y = self.candidates[self.pending_candidate_index]
        self.get_logger().info(
            f"Requesting survey {self.pending_candidate_index + 1}/"
            f"{len(self.candidates)} at ({x:.2f}, {y:.2f})"
        )
        msg = Point()
        msg.x, msg.y, msg.z = x, y, 0.0
        self.survey_request_pub.publish(msg)

    def handle_survey_result(self, msg: Float32MultiArray):
        if self.state != SURVEYING:
            return
        x, y, score, std = msg.data
        self.get_logger().info(
            f"Survey result: ({x:.2f}, {y:.2f}) -> score={score:.3f}"
        )
        self.results.append((x, y, score, std))
        self.pending_candidate_index += 1
        self._request_next_survey()

    def _rank_and_select(self):
        self.results.sort(key=lambda r: r[2], reverse=True)
        best = self.results[0]
        self.get_logger().info(
            f"Ranking complete over {len(self.results)} candidates. "
            f"Best: ({best[0]:.2f}, {best[1]:.2f}) score={best[2]:.3f}"
        )
        for i, r in enumerate(self.results[:5]):
            self.get_logger().info(
                f"  #{i + 1}: ({r[0]:.2f}, {r[1]:.2f}) score={r[2]:.3f}"
            )
        if len(self.results) > 5:
            self.get_logger().info(f"  ... {len(self.results) - 5} lower-ranked omitted")

        self.target = best
        self.investigate_ticks = 0
        self.state = INVESTIGATING
        self.get_logger().info(
            f"Navigating from ({self.current_x:.2f}, {self.current_y:.2f}) "
            f"to target ({self.target[0]:.2f}, {self.target[1]:.2f})"
        )

    def _do_investigate(self):
        self.investigate_ticks += 1
        tx, ty = self.target[0], self.target[1]
        distance = self._drive_toward(tx, ty)

        if distance <= ARRIVAL_DISTANCE or self.investigate_ticks >= INVESTIGATE_TIMEOUT_TICKS:
            self.cmd_vel_pub.publish(Twist())
            reason = "arrived" if distance <= ARRIVAL_DISTANCE else "timeout"
            self.get_logger().info(
                f"Mission complete ({reason}). Final position "
                f"({self.current_x:.2f}, {self.current_y:.2f}), "
                f"{distance:.2f} m from target ({tx:.2f}, {ty:.2f}), "
                f"score={self.target[2]:.3f}"
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
