"""
Implementasi lengkap Google Cartographer SLAM untuk Webots
Referensi: W. Hess, D. Kohler, H. Rapp, and D. Andor,
"Real-Time Loop Closure in 2D LIDAR SLAM," in ICRA, 2016.
"""

from controller import Robot, Motor, Lidar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import math
from collections import deque
from scipy.optimize import least_squares
import threading
from queue import Queue
import time

class Submap:
    def __init__(self, size, resolution):
        """Initialize submap dengan probability grid"""
        self.size = size
        self.resolution = resolution
        self.origin_x = size // 2
        self.origin_y = size // 2
        self.probability_grid = np.ones((size, size)) * 0.5  # Unknown state
        self.pose = np.eye(3)  # Transformation matrix
        self.finished = False
        self.num_range_data = 0
        self.update_timestamps = []
        
    def local_to_global(self, local_x, local_y):
        """Convert local submap coordinates to global coordinates"""
        point = np.array([local_x, local_y, 1])
        global_point = self.pose @ point
        return global_point[0], global_point[1]
    
    def global_to_local(self, global_x, global_y):
        """Convert global coordinates to local submap coordinates"""
        point = np.array([global_x, global_y, 1])
        inv_pose = np.linalg.inv(self.pose)
        local_point = inv_pose @ point
        return local_point[0], local_point[1]
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        grid_x = int(x / self.resolution) + self.origin_x
        grid_y = int(y / self.resolution) + self.origin_y
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates"""
        world_x = (grid_x - self.origin_x) * self.resolution
        world_y = (grid_y - self.origin_y) * self.resolution
        return world_x, world_y
    
    def update_grid(self, scan_points, robot_pose):
        """Update probability grid using scan data"""
        if self.finished:
            return
            
        # Transform scan points to submap frame
        robot_in_submap = np.linalg.inv(self.pose) @ robot_pose
        
        for point in scan_points:
            # Transform point to submap frame
            point_global = robot_pose @ np.array([point[0], point[1], 1])
            point_local = np.linalg.inv(self.pose) @ point_global
            
            # Convert to grid coordinates
            grid_x, grid_y = self.world_to_grid(point_local[0], point_local[1])
            
            if 0 <= grid_x < self.size and 0 <= grid_y < self.size:
                # Update probability using log odds
                log_odds_update = 0.6  # Hit
                current_log_odds = np.log(self.probability_grid[grid_y, grid_x] / 
                                        (1 - self.probability_grid[grid_y, grid_x]))
                new_log_odds = current_log_odds + log_odds_update
                self.probability_grid[grid_y, grid_x] = 1 - 1/(1 + np.exp(new_log_odds))
                
                # Ray tracing for free space
                robot_grid_x, robot_grid_y = self.world_to_grid(robot_in_submap[0], 
                                                              robot_in_submap[1])
                ray_points = bresenham_line(robot_grid_x, robot_grid_y, grid_x, grid_y)
                
                for ray_x, ray_y in ray_points[:-1]:
                    if 0 <= ray_x < self.size and 0 <= ray_y < self.size:
                        # Update free space probability
                        log_odds_update = -0.4  # Miss
                        current_log_odds = np.log(self.probability_grid[ray_y, ray_x] / 
                                                (1 - self.probability_grid[ray_y, ray_x]))
                        new_log_odds = current_log_odds + log_odds_update
                        self.probability_grid[ray_y, ray_x] = 1 - 1/(1 + np.exp(new_log_odds))
        
        self.num_range_data += 1
        self.update_timestamps.append(time.time())
        
        # Check if submap should be finished
        if self.num_range_data >= 100:  # Adjust based on your needs
            self.finished = True

class PoseGraphOptimizer:
    def __init__(self):
        """Initialize pose graph optimizer"""
        self.nodes = []  # Pose graph nodes
        self.constraints = []  # Pose graph constraints
        self.loop_closure_queue = Queue()
        self.optimization_thread = None
        self.running = True
        
    def add_node(self, pose, node_type="scan"):
        """Add node to pose graph"""
        node = {
            'id': len(self.nodes),
            'pose': pose.copy(),
            'type': node_type
        }
        self.nodes.append(node)
        return node['id']
    
    def add_constraint(self, source_id, target_id, relative_pose, information_matrix):
        """Add constraint between nodes"""
        constraint = {
            'source_id': source_id,
            'target_id': target_id,
            'relative_pose': relative_pose.copy(),
            'information_matrix': information_matrix.copy()
        }
        self.constraints.append(constraint)
    
    def optimize(self):
        """Perform pose graph optimization"""
        if len(self.nodes) < 2:
            return
            
        # Prepare optimization variables
        poses = [node['pose'] for node in self.nodes]
        initial_guess = np.concatenate(poses)
        
        # Define error function for optimization
        def error_function(x):
            errors = []
            for constraint in self.constraints:
                source_pose = x[constraint['source_id']*3:constraint['source_id']*3+3]
                target_pose = x[constraint['target_id']*3:constraint['target_id']*3+3]
                
                # Calculate relative transform
                relative_x = target_pose[0] - source_pose[0]
                relative_y = target_pose[1] - source_pose[1]
                relative_theta = target_pose[2] - source_pose[2]
                
                # Compare with constraint
                error = np.array([
                    relative_x - constraint['relative_pose'][0],
                    relative_y - constraint['relative_pose'][1],
                    angle_diff(relative_theta, constraint['relative_pose'][2])
                ])
                
                # Weight error by information matrix
                weighted_error = constraint['information_matrix'] @ error
                errors.extend(weighted_error)
            
            return np.array(errors)
        
        # Optimize
        result = least_squares(error_function, initial_guess, method='dogbox')
        
        # Update poses
        optimized_poses = result.x.reshape(-1, 3)
        for i, node in enumerate(self.nodes):
            node['pose'] = optimized_poses[i]
    
    def start_optimization_thread(self):
        """Start background optimization thread"""
        def optimization_loop():
            while self.running:
                if not self.loop_closure_queue.empty():
                    self.optimize()
                time.sleep(1.0)  # Adjust optimization frequency
                
        self.optimization_thread = threading.Thread(target=optimization_loop)
        self.optimization_thread.start()
    
    def stop(self):
        """Stop optimization thread"""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join()

class CartographerSLAM:
    def __init__(self):
        # Initialize robot and devices
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Setup motors
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Setup LiDAR
        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(self.timestep)
        self.lidar_horizontal_res = self.lidar.getHorizontalResolution()
        
        # Robot parameters
        self.wheel_radius = 0.033  # meters
        self.wheel_base = 0.160    # meters
        self.max_speed = 2.84      # rad/s
        self.min_obstacle_dist = 0.25  # meters
        
        # SLAM parameters
        self.submap_size = 100
        self.resolution = 0.05
        self.active_submaps = []
        self.finished_submaps = []
        self.pose_graph = PoseGraphOptimizer()
        
        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.pose_matrix = np.eye(3)
        
        # Setup visualization
        self.setup_visualization()
        
        # Start pose graph optimization thread
        self.pose_graph.start_optimization_thread()
        
    def setup_visualization(self):
        """Setup real-time visualization"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title('Cartographer SLAM')
        self.ax.grid(True)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        plt.draw()
        
    def update_visualization(self):
        """Update visualization with submaps and robot pose"""
        self.ax.clear()
        
        # Draw submaps
        for submap in self.active_submaps + self.finished_submaps:
            # Convert probability grid to RGB
            rgb_grid = np.zeros((submap.size, submap.size, 3))
            rgb_grid[submap.probability_grid > 0.5] = [0, 0, 0]  # Occupied
            rgb_grid[submap.probability_grid < 0.5] = [1, 1, 1]  # Free
            rgb_grid[submap.probability_grid == 0.5] = [0.5, 0.5, 0.5]  # Unknown
            
            # Transform submap corners to global frame
            corners = np.array([
                [-submap.size/2, -submap.size/2],
                [submap.size/2, -submap.size/2],
                [submap.size/2, submap.size/2],
                [-submap.size/2, submap.size/2]
            ]) * submap.resolution
            
            global_corners = []
            for corner in corners:
                global_point = submap.pose @ np.array([corner[0], corner[1], 1])
                global_corners.append([global_point[0], global_point[1]])
            
            # Display submap
            self.ax.imshow(rgb_grid, extent=[
                global_corners[0][0], global_corners[1][0],
                global_corners[0][1], global_corners[2][1]
            ], origin='lower')
        
        # Draw robot
        robot_circle = Circle((self.x, self.y), 0.15, color='red', alpha=0.7)
        self.ax.add_patch(robot_circle)
        
        # Draw robot heading
        heading_len = 0.3
        dx = heading_len * math.cos(self.theta)
        dy = heading_len * math.sin(self.theta)
        self.ax.arrow(self.x, self.y, dx, dy, 
                     head_width=0.1, head_length=0.1, 
                     fc='red', ec='red')
        
        # Set axes limits
        self.ax.set_xlim(self.x - 10, self.x + 10)
        self.ax.set_ylim(self.y - 10, self.y + 10)
        
        plt.draw()
        self.fig.canvas.flush_events()
    
    def process_scan(self, scan_data):
        """Process LiDAR scan data"""
        if not scan_data:
            return None
            
        points = []
        for i, distance in enumerate(scan_data):
            if distance < self.lidar.getMaxRange():
                angle = i * 2 * math.pi / self.lidar_horizontal_res + self.theta
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                points.append([x, y])
        
        return np.array(points)
    
    def scan_matching(self, scan_points):
        """Perform scan-to-submap matching"""
        if len(self.active_submaps) == 0:
            return np.eye(3)
            
        def compute_score(pose_estimate):
            """Compute matching score for given pose estimate"""
            score = 0
            transform = pose_matrix_from_params(pose_estimate)
            
            # Transform scan points to estimated pose
            transformed_points = []
            for point in scan_points:
                point_h = np.array([point[0], point[1], 1])
                transformed_point = transform @ point_h
                transformed_points.append(transformed_point[:2])
            
            # Compare with active submaps
            for submap in self.active_submaps:
                for point in transformed_points:
                    grid_x, grid_y = submap.world_to_grid(point[0], point[1])
                    if 0 <= grid_x < submap.size and 0 <= grid_y < submap.size:
                        score += submap.probability_grid[grid_y, grid_x]
            
            return -score  # Negative because we want to maximize score
        
        # Initial guess using current pose
        initial_guess = np.array([self.x, self.y, self.theta])
        
        # Optimize using branch and bound
        result = least_squares(compute_score, initial_guess, 
                             method='dogbox',
                             bounds=([initial_guess[0]-0.5, initial_guess[1]-0.5, initial_guess[2]-0.2],
                                    [initial_guess[0]+0.5, initial_guess[1]+0.5, initial_guess[2]+0.2]))
        
        return pose_matrix_from_params(result.x)
    
    def create_new_submap(self):
        """Create new submap and add it to active submaps"""
        submap = Submap(self.submap_size, self.resolution)
        # Set initial pose relative to robot's current position
        submap.pose = np.array([
            [math.cos(self.theta), -math.sin(self.theta), self.x],
            [math.sin(self.theta), math.cos(self.theta), self.y],
            [0, 0, 1]
        ])
        self.active_submaps.append(submap)
        
        # Add submap to pose graph
        self.pose_graph.add_node(submap.pose, node_type="submap")
    
    def detect_loop_closure(self, scan_points):
        """Detect loop closures using scan matching against finished submaps"""
        if len(self.finished_submaps) < 2:
            return False
            
        best_score = float('inf')
        best_submap = None
        best_transform = None
        
        # Try to match current scan against finished submaps
        for submap in self.finished_submaps[:-1]:  # Skip most recent finished submap
            def compute_match_score(pose_params):
                transform = pose_matrix_from_params(pose_params)
                score = 0
                
                # Transform scan points
                for point in scan_points:
                    point_h = np.array([point[0], point[1], 1])
                    transformed_point = transform @ point_h
                    grid_x, grid_y = submap.world_to_grid(transformed_point[0], 
                                                        transformed_point[1])
                    
                    if 0 <= grid_x < submap.size and 0 <= grid_y < submap.size:
                        prob = submap.probability_grid[grid_y, grid_x]
                        score += (prob - 0.5) ** 2
                
                return -score
            
            # Initial guess using relative transform
            initial_guess = np.array([0, 0, 0])  # Relative to submap
            
            # Optimize
            result = least_squares(compute_match_score, initial_guess,
                                 method='dogbox',
                                 bounds=([-2, -2, -math.pi], [2, 2, math.pi]))
            
            score = -compute_match_score(result.x)
            
            if score < best_score and score < 0.1:  # Threshold for good match
                best_score = score
                best_submap = submap
                best_transform = pose_matrix_from_params(result.x)
        
        if best_submap is not None:
            # Add loop closure constraint to pose graph
            source_id = self.pose_graph.nodes.index(next(
                node for node in self.pose_graph.nodes 
                if np.array_equal(node['pose'], best_submap.pose)
            ))
            target_id = len(self.pose_graph.nodes) - 1
            
            # Information matrix (confidence in loop closure)
            information_matrix = np.eye(3) * (1.0 / best_score)
            
            self.pose_graph.add_constraint(source_id, target_id, 
                                         transform_to_params(best_transform),
                                         information_matrix)
            
            # Add to optimization queue
            self.pose_graph.loop_closure_queue.put((source_id, target_id))
            return True
        
        return False
    
    def update_pose(self, left_speed, right_speed):
        """Update robot pose based on wheel odometry"""
        dt = self.timestep / 1000.0
        
        # Calculate robot movement
        v = (right_speed + left_speed) * self.wheel_radius / 2
        omega = (right_speed - left_speed) * self.wheel_radius / self.wheel_base
        
        # Update pose
        self.theta += omega * dt
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        
        # Update pose matrix
        self.pose_matrix = np.array([
            [math.cos(self.theta), -math.sin(self.theta), self.x],
            [math.sin(self.theta), math.cos(self.theta), self.y],
            [0, 0, 1]
        ])
    
    def navigate(self, lidar_data):
        """Navigation with obstacle avoidance"""
        front_sector_start = 5 * self.lidar_horizontal_res // 12
        front_sector_end = 7 * self.lidar_horizontal_res // 12
        
        min_front_dist = float('inf')
        for i in range(front_sector_start, front_sector_end):
            if lidar_data[i] < min_front_dist:
                min_front_dist = lidar_data[i]
        
        base_speed = self.max_speed * 0.5
        
        if min_front_dist < self.min_obstacle_dist:
            turn_speed = self.max_speed * 0.4
            left_dist = sum(lidar_data[:self.lidar_horizontal_res//4])
            right_dist = sum(lidar_data[3*self.lidar_horizontal_res//4:])
            
            if left_dist > right_dist:
                left_speed = -turn_speed
                right_speed = turn_speed
            else:
                left_speed = turn_speed
                right_speed = -turn_speed
        else:
            left_speed = base_speed
            right_speed = base_speed
        
        return left_speed, right_speed
    
    def run(self):
        """Main control loop"""
        update_counter = 0
        
        # Create initial submap
        self.create_new_submap()
        
        while self.robot.step(self.timestep) != -1:
            # Get LiDAR data
            lidar_data = self.lidar.getRangeImage()
            if not lidar_data:
                continue
            
            # Process scan data
            scan_points = self.process_scan(lidar_data)
            if scan_points is None:
                continue
            
            # Scan matching
            matched_pose = self.scan_matching(scan_points)
            self.pose_matrix = matched_pose
            self.x = matched_pose[0, 2]
            self.y = matched_pose[1, 2]
            self.theta = math.atan2(matched_pose[1, 0], matched_pose[0, 0])
            
            # Update active submaps
            for submap in self.active_submaps:
                submap.update_grid(scan_points, self.pose_matrix)
                
                # Check if submap should be finished
                if submap.finished:
                    self.finished_submaps.append(submap)
                    self.active_submaps.remove(submap)
            
            # Create new submap if needed
            if len(self.active_submaps) == 0:
                self.create_new_submap()
            
            # Loop closure detection
            update_counter += 1
            if update_counter >= 50:  # Check every 50 updates
                if self.detect_loop_closure(scan_points):
                    print("Loop closure detected!")
                update_counter = 0
            
            # Navigation
            left_speed, right_speed = self.navigate(lidar_data)
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
            
            # Update pose from odometry
            self.update_pose(left_speed, right_speed)
            
            # Visualization update
            if self.robot.getTime() % 0.5 < self.timestep/1000.0:
                self.update_visualization()
        
        # Cleanup
        self.pose_graph.stop()

def bresenham_line(x0, y0, x1, y1):
    """Bresenham's line algorithm for ray tracing"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
            
    points.append((x, y))
    return points

def pose_matrix_from_params(params):
    """Convert [x, y, theta] to transformation matrix"""
    x, y, theta = params
    return np.array([
        [math.cos(theta), -math.sin(theta), x],
        [math.sin(theta), math.cos(theta), y],
        [0, 0, 1]
    ])

def transform_to_params(transform):
    """Convert transformation matrix to [x, y, theta]"""
    return np.array([
        transform[0, 2],
        transform[1, 2],
        math.atan2(transform[1, 0], transform[0, 0])
    ])

def angle_diff(a1, a2):
    """Compute smallest angle difference"""
    diff = a1 - a2
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff

# Main program
if __name__ == "__main__":
    controller = CartographerSLAM()
    controller.run()