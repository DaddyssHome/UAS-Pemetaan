"""
Implementasi RTAB-Map SLAM menggunakan simulasi Webots
dengan mempertahankan visualisasi yang sama
"""

from controller import Robot, Motor, Lidar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
from collections import deque
from scipy.optimize import least_squares

class RTABMapSLAM:
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
        self.max_speed = 12       
        self.min_obstacle_dist = 0.30
        
        # SLAM parameters
        self.map_size = 400
        self.resolution = 0.05
        self.occupancy_grid = np.zeros((self.map_size, self.map_size))
        self.map_origin = self.map_size // 2
        
        # RTAB-Map specific parameters
        self.local_map = deque(maxlen=50)  # Local map for loop closure
        self.keyframes = deque(maxlen=100)  # Store keyframes
        self.loop_closures = []  # Store detected loop closures
        self.previous_scan = None
        self.min_loop_closure_score = 0.7
        
        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.pose_graph = []  # Store poses for graph optimization
        
        # Store detected obstacles
        self.obstacle_patches = []
        
        # Setup visualization
        self.setup_visualization()
    
    def setup_visualization(self):
        """Setup real-time visualization with larger plot size"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.map_img = self.ax.imshow(self.occupancy_grid, 
                                    cmap='gray_r', 
                                    extent=[-10, 10, -10, 10],
                                    origin='lower')
        self.robot_marker = Circle((0, 0), 0.15, color='red', alpha=0.7)
        self.ax.add_patch(self.robot_marker)
        self.ax.set_title('RTAB-Map SLAM')
        self.ax.grid(True)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        plt.draw()
    
    def update_visualization(self):
        """Update visualization dengan obstacle detection"""
        for patch in self.ax.patches:
            if patch != self.robot_marker:
                patch.remove()
                
        normalized_grid = (self.occupancy_grid - np.min(self.occupancy_grid)) / (
            np.max(self.occupancy_grid) - np.min(self.occupancy_grid) + 1e-10)
        self.map_img.set_data(normalized_grid)
        
        self.robot_marker.center = (self.x, self.y)
        
        scan = self.lidar.getRangeImage()
        if scan:
            angle_increment = 2 * math.pi / self.lidar_horizontal_res
            for i, distance in enumerate(scan):
                if distance < self.lidar.getMaxRange():
                    angle = i * angle_increment + self.theta
                    obs_x = self.x + distance * math.cos(angle)
                    obs_y = self.y + distance * math.sin(angle)
                    
                    obstacle = Circle((obs_x, obs_y), 0.05, color='black', alpha=0.7)
                    self.ax.add_patch(obstacle)
        
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])
        
        plt.draw()
        self.fig.canvas.flush_events()
    
    def scan_matching(self, current_scan):
        """RTAB-Map scan matching dengan ICP"""
        if self.previous_scan is None:
            self.previous_scan = current_scan
            return 0, 0, 0
        
        # Convert scans to point clouds
        prev_points = []
        curr_points = []
        
        angle_increment = 2 * math.pi / self.lidar_horizontal_res
        
        for i, (prev_r, curr_r) in enumerate(zip(self.previous_scan, current_scan)):
            if prev_r < self.lidar.getMaxRange() and curr_r < self.lidar.getMaxRange():
                angle = i * angle_increment
                # Previous scan points
                prev_x = prev_r * math.cos(angle)
                prev_y = prev_r * math.sin(angle)
                prev_points.append([prev_x, prev_y])
                # Current scan points
                curr_x = curr_r * math.cos(angle)
                curr_y = curr_r * math.sin(angle)
                curr_points.append([curr_x, curr_y])
        
        if len(prev_points) < 10:
            return 0, 0, 0
            
        prev_points = np.array(prev_points)
        curr_points = np.array(curr_points)
        
        # ICP
        def objective(params):
            dx, dy, dtheta = params
            # Create transformation matrix
            cos_theta = math.cos(dtheta)
            sin_theta = math.sin(dtheta)
            R = np.array([[cos_theta, -sin_theta],
                         [sin_theta, cos_theta]])
            t = np.array([dx, dy])
            
            # Transform current points
            transformed = (R @ curr_points.T).T + t
            
            # Calculate distances to closest points
            distances = np.sqrt(np.sum((prev_points[:, np.newaxis] - transformed) ** 2, axis=2))
            min_distances = np.min(distances, axis=1)
            return min_distances
        
        # Optimize
        result = least_squares(objective, [0, 0, 0], 
                             method='lm',
                             xtol=1e-5)
        
        self.previous_scan = current_scan
        return result.x
    
    def detect_loop_closure(self):
        """Detect loop closures using scan similarity"""
        if len(self.keyframes) < 10:
            return False
            
        current_scan = self.previous_scan
        if current_scan is None:
            return False
            
        best_score = 0
        best_match = None
        
        # Compare with previous keyframes
        for i, (kf_scan, kf_pose) in enumerate(self.keyframes):
            if i > len(self.keyframes) - 10:  # Skip recent frames
                continue
                
            score = self.compute_scan_similarity(current_scan, kf_scan)
            
            if score > best_score and score > self.min_loop_closure_score:
                best_score = score
                best_match = i
        
        if best_match is not None:
            self.loop_closures.append((len(self.pose_graph)-1, best_match))
            self.optimize_pose_graph()
            return True
            
        return False
    
    def compute_scan_similarity(self, scan1, scan2):
        """Compute similarity score between two scans"""
        if scan1 is None or scan2 is None:
            return 0
            
        # Simple correlation
        correlation = np.correlate(scan1, scan2)
        score = np.max(correlation) / (np.linalg.norm(scan1) * np.linalg.norm(scan2))
        return score
    
    def optimize_pose_graph(self):
        """Optimize pose graph after loop closure"""
        if len(self.loop_closures) == 0:
            return
            
        # Simple pose averaging for demonstration
        for loop_closure in self.loop_closures:
            current_idx, matched_idx = loop_closure
            if current_idx < len(self.pose_graph) and matched_idx < len(self.pose_graph):
                current_pose = self.pose_graph[current_idx]
                matched_pose = self.pose_graph[matched_idx]
                
                # Average the poses
                avg_x = (current_pose[0] + matched_pose[0]) / 2
                avg_y = (current_pose[1] + matched_pose[1]) / 2
                avg_theta = (current_pose[2] + matched_pose[2]) / 2
                
                # Update current pose
                self.pose_graph[current_idx] = (avg_x, avg_y, avg_theta)
                
                # Update robot pose if this is the latest loop closure
                if current_idx == len(self.pose_graph) - 1:
                    self.x = avg_x
                    self.y = avg_y
                    self.theta = avg_theta
    
    def navigate(self, lidar_data):
        """Enhanced navigation with higher speeds"""
        front_sector_start = 5 * self.lidar_horizontal_res // 12
        front_sector_end = 7 * self.lidar_horizontal_res // 12
        
        min_front_dist = float('inf')
        for i in range(front_sector_start, front_sector_end):
            if lidar_data[i] < min_front_dist:
                min_front_dist = lidar_data[i]
        
        base_speed = self.max_speed * 0.85
        
        if min_front_dist < self.min_obstacle_dist:
            turn_speed = self.max_speed * 0.75
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
    
    def update_pose(self, left_speed, right_speed):
        """Update robot pose with scan matching correction"""
        dt = self.timestep / 1000.0
        
        # Odometry update
        slip_factor = 0.9
        v = (right_speed + left_speed) * self.wheel_radius / 2
        omega = (right_speed - left_speed) * self.wheel_radius / self.wheel_base
        
        # Update pose
        self.theta += omega * dt * slip_factor
        self.x += v * math.cos(self.theta) * dt * slip_factor
        self.y += v * math.sin(self.theta) * dt * slip_factor
        
        # Normalize theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Store pose in pose graph
        self.pose_graph.append((self.x, self.y, self.theta))
        
        # Store keyframe if moved enough
        if len(self.keyframes) == 0 or \
           math.sqrt((self.x - self.keyframes[-1][1][0])**2 + 
                    (self.y - self.keyframes[-1][1][1])**2) > 0.5:  # 50cm threshold
            self.keyframes.append((self.previous_scan, (self.x, self.y, self.theta)))
    
    def world_to_map(self, x, y):
        """Convert world coordinates to map indices"""
        map_x = int((x * (1.0 / self.resolution)) + self.map_origin)
        map_y = int((y * (1.0 / self.resolution)) + self.map_origin)
        return map_x, map_y
    
    def process_lidar_data(self):
        """Process LiDAR data dengan RTAB-Map approach"""
        lidar_data = self.lidar.getRangeImage()
        if not lidar_data:
            return False
            
        angle_increment = 2 * math.pi / self.lidar_horizontal_res
        
        # Clear area around robot
        robot_x, robot_y = self.world_to_map(self.x, self.y)
        clear_radius = int(0.2 / self.resolution)
        y_indices, x_indices = np.ogrid[-clear_radius:clear_radius+1, -clear_radius:clear_radius+1]
        circle_mask = x_indices**2 + y_indices**2 <= clear_radius**2
        
        for dy in range(-clear_radius, clear_radius + 1):
            for dx in range(-clear_radius, clear_radius + 1):
                map_x, map_y = robot_x + dx, robot_y + dy
                if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                    if circle_mask[dy+clear_radius, dx+clear_radius]:
                        self.occupancy_grid[map_y, map_x] = 0.1
        
        # Update map with scan data
        for i, distance in enumerate(lidar_data):
            if distance < self.lidar.getMaxRange():
                angle = i * angle_increment + self.theta
                point_x = self.x + distance * math.cos(angle)
                point_y = self.y + distance * math.sin(angle)
                
                map_x, map_y = self.world_to_map(point_x, point_y)
                
                if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                    self.occupancy_grid[map_y, map_x] = 1.0
                    
                    # Ray tracing
                    robot_x, robot_y = self.world_to_map(self.x, self.y)
                    points = self.bresenham_line(robot_x, robot_y, map_x, map_y)
                    for px, py in points[:-1]:
                        if 0 <= px < self.map_size and 0 <= py < self.map_size:
                            self.occupancy_grid[py, px] = 0.1
        
        return lidar_data
    
    def bresenham_line(self, x0, y0, x1, y1):
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
    
    def run(self):
        """Main control loop dengan RTAB-Map processing"""
        update_counter = 0
        
        while self.robot.step(self.timestep) != -1:
            # Get and process LiDAR data
            lidar_data = self.process_lidar_data()
            if not lidar_data:
                continue
            
            # Scan matching untuk estimasi pose
            if update_counter % 5 == 0:  # Perform scan matching every 5 steps
                dx, dy, dtheta = self.scan_matching(lidar_data)
                # Apply scan matching correction
                self.x += dx
                self.y += dy
                self.theta += dtheta
            
            # Loop closure detection
            if update_counter % 50 == 0:  # Check for loop closures every 50 steps
                if self.detect_loop_closure():
                    print("Loop closure detected!")
            
            # Navigation
            left_speed, right_speed = self.navigate(lidar_data)
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
            
            # Update pose
            self.update_pose(left_speed, right_speed)
            
            # Update visualization
            if self.robot.getTime() % 0.05 < self.timestep/1000.0:
                self.update_visualization()
            
            update_counter += 1

# Main program
controller = RTABMapSLAM()
controller.run()