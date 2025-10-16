#!/usr/bin/env python3
"""
Real-time Slither.io Data Visualizer using PyGame

This script connects to the data collector server and visualizes 
the polar grid data in real-time as it's being collected.
"""

import pygame
import numpy as np
import asyncio
importjson
import time
import colorsys
from threading import Thread
import requests
from typing import Dict, Optional, Tuple
import math

# ===========================================
# CONFIGURATION
# ===========================================
CONFIG = {
    'WINDOW_SIZE': 800,
    'GRID_SIZE': 600,
    'FPS': 30,
    'SERVER_URL': 'http://127.0.0.1:5055',
    'UPDATE_INTERVAL_MS': 100,  # Poll server every 100ms
    
    # Grid configuration (should match userscript)
    'ANGULAR_BINS': 64,
    'RADIAL_BINS': 24,
    'CHANNELS': 4,
    
    # Channel names and colors
    'CHANNEL_NAMES': ['Food', 'Enemy Body', 'My Body', 'Enemy Heads'],
    'CHANNEL_COLORS': [
        (0, 255, 0),    # Green for food
        (255, 0, 0),    # Red for enemy bodies
        (0, 0, 255),    # Blue for my body
        (255, 255, 0),  # Yellow for enemy heads
    ]
}

class PolarGridVisualizer:
    def __init__(self):
        pygame.init()
        
        # Window setup
        self.screen = pygame.display.set_mode((CONFIG['WINDOW_SIZE'], CONFIG['WINDOW_SIZE']))
        pygame.display.set_caption("Slither.io Polar Grid Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Data storage
        self.latest_data = None
        self.current_channel = 0
        self.last_update_time = 0
        self.stats = {
            'frames_received': 0,
            'last_timestamp': 0,
            'fps': 0
        }
        
        # Grid geometry
        self.center_x = CONFIG['WINDOW_SIZE'] // 2
        self.center_y = CONFIG['WINDOW_SIZE'] // 2
        self.max_radius = CONFIG['GRID_SIZE'] // 2
        
        # Precompute polar grid coordinates
        self._precompute_grid_coordinates()
        
        # Colors
        self.background_color = (20, 20, 30)
        self.grid_color = (60, 60, 70)
        
        print(f"Visualizer initialized. Window size: {CONFIG['WINDOW_SIZE']}x{CONFIG['WINDOW_SIZE']}")
        
    def _precompute_grid_coordinates(self):
        """Precompute the pixel coordinates for each polar grid cell using the same math as userscript"""
        self.grid_coords = []
        
        # Constants from userscript
        ALPHA_WARP = 6.0
        R_MIN = 60
        R_MAX = 3200
        
        def angular_warp(phi, alpha):
            """Same angular warp function as in userscript"""
            sign = 1 if phi >= 0 else -1
            x = abs(phi) / math.pi
            y = math.log1p(alpha * x) / math.log1p(alpha)
            return sign * y * math.pi
        
        def inverse_angular_warp(warped_phi, alpha):
            """Inverse of angular warp to get original phi from warped phi"""
            sign = 1 if warped_phi >= 0 else -1
            y = abs(warped_phi) / math.pi
            x = (math.exp(y * math.log1p(alpha)) - 1) / alpha
            return sign * x * math.pi
        
        def inverse_log_radius(j, M, rmin, rmax):
            """Inverse of logarithmic radius mapping"""
            log_factor = j / M
            return rmin * (rmax / rmin) ** log_factor
        
        # Create grid coordinates
        for j in range(CONFIG['RADIAL_BINS']):
            for k in range(CONFIG['ANGULAR_BINS']):
                # Calculate radius bounds using inverse logarithmic mapping
                r_inner = inverse_log_radius(j, CONFIG['RADIAL_BINS'], R_MIN, R_MAX)
                r_outer = inverse_log_radius(j + 1, CONFIG['RADIAL_BINS'], R_MIN, R_MAX)
                
                # Scale to screen coordinates
                screen_r_inner = (r_inner / R_MAX) * self.max_radius
                screen_r_outer = (r_outer / R_MAX) * self.max_radius
                
                # Calculate angular bounds using inverse warp
                theta_start = k / CONFIG['ANGULAR_BINS']
                theta_end = (k + 1) / CONFIG['ANGULAR_BINS']
                
                # Convert from [0,1] to [-π, π] then inverse warp
                warped_phi_start = (theta_start * 2 * math.pi) - math.pi
                warped_phi_end = (theta_end * 2 * math.pi) - math.pi
                
                phi_start = inverse_angular_warp(warped_phi_start, ALPHA_WARP)
                phi_end = inverse_angular_warp(warped_phi_end, ALPHA_WARP)
                
                # Convert to screen angle (rotate by 90 degrees to match game orientation)
                angle_start = phi_start - math.pi/2
                angle_end = phi_end - math.pi/2
                
                # Create points for the polar cell
                points = []
                num_segments = 8  # More segments for smoother curves
                
                # Inner arc
                for i in range(num_segments + 1):
                    t = i / num_segments
                    angle = angle_start + (angle_end - angle_start) * t
                    x = self.center_x + screen_r_inner * math.cos(angle)
                    y = self.center_y + screen_r_inner * math.sin(angle)
                    points.append((x, y))
                
                # Outer arc (reversed)
                for i in range(num_segments, -1, -1):
                    t = i / num_segments
                    angle = angle_start + (angle_end - angle_start) * t
                    x = self.center_x + screen_r_outer * math.cos(angle)
                    y = self.center_y + screen_r_outer * math.sin(angle)
                    points.append((x, y))
                
                self.grid_coords.append(points)
    
    def fetch_latest_data(self) -> Optional[Dict]:
        """Fetch the latest data from the server"""
        try:
            response = requests.get(f"{CONFIG['SERVER_URL']}/latest", timeout=0.5)
            if response.status_code == 200:
                data = response.json()
                if data and 'grid' in data:
                    self.stats['frames_received'] += 1
                    self.stats['last_timestamp'] = data.get('timestamp', 0)
                    return data
        except Exception as e:
            if self.stats['frames_received'] == 0:  # Only log if we haven't received any data yet
                print(f"Error fetching data: {e}")
        return None
    
    def update_data(self):
        """Update the data from server"""
        new_data = self.fetch_latest_data()
        if new_data:
            self.latest_data = new_data
            current_time = time.time()
            if self.last_update_time > 0:
                dt = current_time - self.last_update_time
                self.stats['fps'] = 1.0 / dt if dt > 0 else 0
            self.last_update_time = current_time
    
    def draw_grid_outline(self):
        """Draw the grid outline for reference"""
        # Constants from userscript
        ALPHA_WARP = 6.0
        R_MIN = 60
        R_MAX = 3200
        
        def angular_warp(phi, alpha):
            """Same angular warp function as in userscript"""
            sign = 1 if phi >= 0 else -1
            x = abs(phi) / math.pi
            y = math.log1p(alpha * x) / math.log1p(alpha)
            return sign * y * math.pi
        
        def inverse_angular_warp(warped_phi, alpha):
            """Inverse of angular warp to get original phi from warped phi"""
            sign = 1 if warped_phi >= 0 else -1
            y = abs(warped_phi) / math.pi
            x = (math.exp(y * math.log1p(alpha)) - 1) / alpha
            return sign * x * math.pi
        
        def inverse_log_radius(j, M, rmin, rmax):
            """Inverse of logarithmic radius mapping"""
            log_factor = j / M
            return rmin * (rmax / rmin) ** log_factor
        
        # Draw radial lines (every 8th line for clarity)
        for k in range(0, CONFIG['ANGULAR_BINS'], 8):
            theta = k / CONFIG['ANGULAR_BINS']
            warped_phi = (theta * 2 * math.pi) - math.pi
            phi = inverse_angular_warp(warped_phi, ALPHA_WARP)
            angle = phi - math.pi/2  # Rotate to match game orientation
            
            end_x = self.center_x + self.max_radius * math.cos(angle)
            end_y = self.center_y + self.max_radius * math.sin(angle)
            pygame.draw.line(self.screen, self.grid_color, 
                           (self.center_x, self.center_y), (end_x, end_y), 1)
        
        # Draw concentric circles (every 4th circle for clarity)
        for j in range(0, CONFIG['RADIAL_BINS'], 4):
            radius = inverse_log_radius(j + 1, CONFIG['RADIAL_BINS'], R_MIN, R_MAX)
            screen_radius = (radius / R_MAX) * self.max_radius
            pygame.draw.circle(self.screen, self.grid_color, 
                             (self.center_x, self.center_y), int(screen_radius), 1)
    
    def draw_polar_grid(self):
        """Draw the polar grid with current channel data"""
        if not self.latest_data or 'grid' not in self.latest_data:
            return
        
        grid_data = np.array(self.latest_data['grid'])
        
        # The grid is stored as 1D array with indexing: (j * ANGULAR_BINS + k) * CHANNELS + channel
        # where j=radial_bin, k=angular_bin, channel=data_channel
        
        # Extract current channel data and find max for normalization
        max_val = 0.0
        channel_values = []
        
        for j in range(CONFIG['RADIAL_BINS']):
            for k in range(CONFIG['ANGULAR_BINS']):
                # Use same indexing formula as userscript: splatToGrid function
                idx = (j * CONFIG['ANGULAR_BINS'] + k) * CONFIG['CHANNELS'] + self.current_channel
                value = grid_data[idx] if idx < len(grid_data) else 0.0
                channel_values.append(value)
                max_val = max(max_val, value)
        
        # Normalize for visualization
        if max_val <= 0:
            max_val = 1.0
        
        # Draw each cell
        base_color = CONFIG['CHANNEL_COLORS'][self.current_channel]
        
        for j in range(CONFIG['RADIAL_BINS']):
            for k in range(CONFIG['ANGULAR_BINS']):
                # Get the value for this cell
                value_idx = j * CONFIG['ANGULAR_BINS'] + k
                intensity = channel_values[value_idx] / max_val
                
                if intensity > 0.01:  # Only draw cells with some data
                    # Calculate color based on intensity
                    color = tuple(int(base_color[i] * intensity) for i in range(3))
                    
                    # Get the precomputed polygon points
                    # Grid coords are stored as [j][k] but we stored them linearly
                    cell_idx = j * CONFIG['ANGULAR_BINS'] + k
                    points = self.grid_coords[cell_idx]
                    
                    # Draw filled polygon
                    pygame.draw.polygon(self.screen, color, points)
    
    def draw_metadata(self):
        """Draw metadata and stats"""
        if not self.latest_data:
            return
        
        y_offset = 10
        line_height = 25
        
        # Current channel
        channel_text = f"Channel: {CONFIG['CHANNEL_NAMES'][self.current_channel]} [{self.current_channel+1}/4]"
        color = CONFIG['CHANNEL_COLORS'][self.current_channel]
        text = self.font.render(channel_text, True, color)
        self.screen.blit(text, (10, y_offset))
        y_offset += line_height
        
        # Controls
        controls_text = "Controls: 1-4 = Change Channel, ESC = Quit"
        text = self.small_font.render(controls_text, True, (200, 200, 200))
        self.screen.blit(text, (10, y_offset))
        y_offset += line_height
        
        # Metadata from game
        if 'metadata' in self.latest_data:
            meta = self.latest_data['metadata']
            
            metadata_lines = [
                f"Heading: {meta.get('heading', 0):.2f} rad",
                f"Velocity: {meta.get('velocity', 0):.1f}",
                f"Boost: {'ON' if meta.get('boost', False) else 'OFF'}",
                f"Distance to Border: {meta.get('distanceToBorder', 0):.0f}",
                f"Snake Length: {meta.get('snakeLength', 0)}"
            ]
            
            for line in metadata_lines:
                text = self.small_font.render(line, True, (180, 180, 180))
                self.screen.blit(text, (10, y_offset))
                y_offset += line_height * 0.8
        
        # Debug info from game
        if 'debug' in self.latest_data:
            debug = self.latest_data['debug']
            y_offset += 10
            
            debug_lines = [
                f"Food Count: {debug.get('foodCount', 0)}",
                f"Enemy Segments: {debug.get('enemySegments', 0)}",
                f"My Segments: {debug.get('mySegments', 0)}",
                f"Enemy Heads: {debug.get('enemyHeads', 0)}"
            ]
            
            for line in debug_lines:
                text = self.small_font.render(line, True, (150, 150, 150))
                self.screen.blit(text, (10, y_offset))
                y_offset += line_height * 0.8
        
        # Stats
        y_offset = CONFIG['WINDOW_SIZE'] - 80
        stats_lines = [
            f"Frames Received: {self.stats['frames_received']}",
            f"Update FPS: {self.stats['fps']:.1f}",
            f"Session: {self.latest_data.get('sessionId', 'Unknown')[:8]}..."
        ]
        
        for line in stats_lines:
            text = self.small_font.render(line, True, (120, 120, 120))
            self.screen.blit(text, (10, y_offset))
            y_offset += line_height * 0.8
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key >= pygame.K_1 and event.key <= pygame.K_4:
                    # Switch channel
                    self.current_channel = event.key - pygame.K_1
                    print(f"Switched to channel: {CONFIG['CHANNEL_NAMES'][self.current_channel]}")
        return True
    
    def run(self):
        """Main visualization loop"""
        print("Starting visualization...")
        print("Controls:")
        print("  1-4: Switch between channels (Food, Enemy Body, My Body, Enemy Heads)")
        print("  ESC: Quit")
        print(f"Connecting to server at {CONFIG['SERVER_URL']}")
        
        running = True
        last_data_update = 0
        
        while running:
            current_time = time.time()
            
            # Handle events
            running = self.handle_events()
            
            # Update data periodically
            if current_time - last_data_update > CONFIG['UPDATE_INTERVAL_MS'] / 1000.0:
                self.update_data()
                last_data_update = current_time
            
            # Clear screen
            self.screen.fill(self.background_color)
            
            # Draw grid outline
            self.draw_grid_outline()
            
            # Draw polar grid data
            self.draw_polar_grid()
            
            # Draw metadata and controls
            self.draw_metadata()
            
            # If no data yet, show waiting message
            if not self.latest_data:
                waiting_text = "Waiting for data from Slither.io..."
                text = self.font.render(waiting_text, True, (255, 255, 0))
                text_rect = text.get_rect(center=(CONFIG['WINDOW_SIZE']//2, CONFIG['WINDOW_SIZE']//2))
                self.screen.blit(text, text_rect)
            
            # Update display
            pygame.display.flip()
            self.clock.tick(CONFIG['FPS'])
        
        pygame.quit()
        print("Visualizer stopped.")

def main():
    """Main entry point"""
    visualizer = PolarGridVisualizer()
    
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()