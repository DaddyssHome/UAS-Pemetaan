"""
Implementasi Google Cartographer SLAM menggunakan simulasi webots
"""

from controller import Robot, Motor, Lidar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
from collections import deque

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
        
        # Robot parameters - Higher speed settings  
        self.wheel_radius = 0.033  # meters
        self.wheel_base = 0.160    # meters
        self.max_speed = 12       # Significantly increased speed
        self.min_obstacle_dist = 0.30  # Slightly reduced for more aggressive movement
        
        # SLAM parameters
        self.map_size = 400
        self.resolution = 0.05
        self.occupancy_grid = np.zeros((self.map_size, self.map_size))
        self.map_origin = self.map_size // 2
        
        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
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
        self.ax.set_title('Cartographer SLAM Map')
        self.ax.grid(True)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        plt.draw()
    
    def update_visualization(self):
        """Update visualization dengan obstacle detection yang lebih baik"""
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
    
    def navigate(self, lidar_data):
        """Enhanced navigation with higher speeds"""
        front_sector_start = 5 * self.lidar_horizontal_res // 12
        front_sector_end = 7 * self.lidar_horizontal_res // 12
        
        # Check front sector for obstacles
        min_front_dist = float('inf')
        for i in range(front_sector_start, front_sector_end):
            if lidar_data[i] < min_front_dist:
                min_front_dist = lidar_data[i]
        
        # Enhanced speed control
        base_speed = self.max_speed * 0.85  # Increased base speed to 85% of max
        
        if min_front_dist < self.min_obstacle_dist:
            # Obstacle detected - quick turn with higher speed
            turn_speed = self.max_speed * 0.75
            # Check which side has more space
            left_dist = sum(lidar_data[:self.lidar_horizontal_res//4])
            right_dist = sum(lidar_data[3*self.lidar_horizontal_res//4:])
            
            if left_dist > right_dist:
                left_speed = -turn_speed
                right_speed = turn_speed
            else:
                left_speed = turn_speed
                right_speed = -turn_speed
        else:
            # No immediate obstacle - move at high speed
            left_speed = base_speed
            right_speed = base_speed
        
        return left_speed, right_speed
    
    def update_pose(self, left_speed, right_speed):
        """Update robot pose with improved accuracy"""
        dt = self.timestep / 1000.0
        
        # Calculate robot movement with slip compensation
        slip_factor = 0.9  # Standard slip compensation
        v = (right_speed + left_speed) * self.wheel_radius / 2
        omega = (right_speed - left_speed) * self.wheel_radius / self.wheel_base
        
        # Update pose
        self.theta += omega * dt * slip_factor
        self.x += v * math.cos(self.theta) * dt * slip_factor
        self.y += v * math.sin(self.theta) * dt * slip_factor
        
        # Normalize theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
    
    def world_to_map(self, x, y):
        """Convert world coordinates to map indices"""
        map_x = int((x * (1.0 / self.resolution)) + self.map_origin)
        map_y = int((y * (1.0 / self.resolution)) + self.map_origin)
        return map_x, map_y
    
    def process_lidar_data(self):
        """Process LiDAR data dengan deteksi obstacle yang lebih baik"""
        lidar_data = self.lidar.getRangeImage()
        if not lidar_data:
            return False
            
        angle_increment = 2 * math.pi / self.lidar_horizontal_res
        
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
        
        for i, distance in enumerate(lidar_data):
            if distance < self.lidar.getMaxRange():
                angle = i * angle_increment + self.theta
                point_x = self.x + distance * math.cos(angle)
                point_y = self.y + distance * math.sin(angle)
                
                map_x, map_y = self.world_to_map(point_x, point_y)
                
                if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                    self.occupancy_grid[map_y, map_x] = 1.0
                    
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
        """Main control loop dengan update visualisasi yang lebih sering"""
        while self.robot.step(self.timestep) != -1:
            lidar_data = self.process_lidar_data()
            if not lidar_data:
                continue
            
            left_speed, right_speed = self.navigate(lidar_data)
            
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
            
            self.update_pose(left_speed, right_speed)
            
            # Update visualization more frequently
            if self.robot.getTime() % 0.05 < self.timestep/1000.0:
                self.update_visualization()

# Main program
controller = CartographerSLAM()
controller.run()