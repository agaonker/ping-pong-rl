import pygame
import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque

class PongEnv:
    def __init__(self, width=800, height=600):
        # Game dimensions
        self.width = width
        self.height = height
        
        # Paddle properties
        self.paddle_width = 15
        self.paddle_height = 100
        self.paddle_speed = 8
        
        # Ball properties
        self.ball_radius = 8
        self.ball_speed = 7
        self.max_ball_angle = 45  # Maximum angle in degrees
        
        # AI trainee movement properties
        self.ai_speed = 5  # Reduced base speed
        self.ai_inertia = 0.8  # Inertia factor (0-1)
        self.ai_target_buffer = 20  # Buffer zone around target
        
        # Player movement properties
        self.player_speed = 5  # Reduced base speed
        self.player_inertia = 0.8  # Inertia factor (0-1)
        
        # Initialize game state
        self.reset()
        
        # Learning parameters
        self.gamma = 0.95    # discount rate for future rewards
        self.epsilon = 1.0   # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995  # decay rate for exploration
        self.learning_rate = 0.001  # learning rate for optimizer
        
        self.memory = deque(maxlen=10000)  # Experience replay buffer size
        
        # Initialize scores
        self.player_score = 0
        self.ai_trainee_score = 0
        
        # Initialize velocities
        self.ai_velocity = 0
        self.player_velocity = 0
        
    def reset(self, last_scorer="player"):
        # Reset player paddle
        self.player_y = self.height // 2 - self.paddle_height // 2
        self.player_velocity = 0
        
        # Reset AI trainee paddle
        self.ai_trainee_y = self.height // 2 - self.paddle_height // 2
        self.ai_velocity = 0
        
        # Reset ball position based on who scored last
        if last_scorer == "player":
            # Start from AI trainee's side (since player scored)
            self.ball_x = self.width - self.paddle_width - self.ball_radius
            self.ball_dx = -self.ball_speed  # Move left
        else:
            # Start from player's side (since AI trainee scored)
            self.ball_x = self.paddle_width + self.ball_radius
            self.ball_dx = self.ball_speed  # Move right
        
        # Ball starts at paddle height
        self.ball_y = self.height // 2
        
        # Random initial vertical direction with controlled angle
        angle = random.uniform(-self.max_ball_angle, self.max_ball_angle)
        angle_rad = np.radians(angle)
        self.ball_dy = self.ball_speed * np.sin(angle_rad)
        
        # Ensure ball starts moving
        while abs(self.ball_dy) < 0.1:  # Prevent nearly vertical trajectories
            angle = random.uniform(-self.max_ball_angle, self.max_ball_angle)
            angle_rad = np.radians(angle)
            self.ball_dy = self.ball_speed * np.sin(angle_rad)
        
        return self._get_state()
    
    def _get_state(self):
        # Normalize all values to [0, 1] range
        return np.array([
            self.player_y / self.height,
            self.ai_trainee_y / self.height,
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_dx / self.ball_speed,
            self.ball_dy / self.ball_speed
        ])
    
    def step(self, action):
        # 0: Stay, 1: Up, 2: Down
        if action == 1:  # Move up
            desired_velocity = -self.player_speed
        elif action == 2:  # Move down
            desired_velocity = self.player_speed
        else:  # Stay
            desired_velocity = 0
        
        # Apply inertia to player velocity
        self.player_velocity = self.player_velocity * self.player_inertia + desired_velocity * (1 - self.player_inertia)
        # Update player position with velocity
        self.player_y = max(min(self.player_y + self.player_velocity, 
                              self.height - self.paddle_height), 0)
        
        # Smooth AI trainee movement with inertia
        target_y = self.ball_y - self.paddle_height/2
        current_y = self.ai_trainee_y + self.paddle_height/2
        
        # Only move if outside the buffer zone
        if abs(current_y - target_y) > self.ai_target_buffer:
            # Calculate desired velocity
            desired_velocity = (target_y - current_y) * 0.1
            # Apply inertia to velocity
            self.ai_velocity = self.ai_velocity * self.ai_inertia + desired_velocity * (1 - self.ai_inertia)
            # Limit maximum velocity
            self.ai_velocity = max(min(self.ai_velocity, self.ai_speed), -self.ai_speed)
            # Update position
            self.ai_trainee_y = max(min(self.ai_trainee_y + self.ai_velocity, 
                                      self.height - self.paddle_height), 0)
        
        # Update ball position with sub-pixel movement
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball collision with top and bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.height:
            self.ball_dy = -self.ball_dy
            # Ensure ball stays within bounds
            self.ball_y = max(min(self.ball_y, self.height), 0)
        
        # Ball collision with paddles
        reward = 0
        done = False
        last_scorer = None
        
        # Player paddle collision with improved hit detection
        if (self.ball_x - self.ball_radius <= self.paddle_width and 
            self.player_y <= self.ball_y <= self.player_y + self.paddle_height):
            self.ball_dx = -self.ball_dx
            # Add slight randomness to ball direction based on hit position
            hit_position = (self.ball_y - self.player_y) / self.paddle_height
            self.ball_dy += (hit_position - 0.5) * 2  # -1 to 1 based on hit position
            reward = 1
        
        # AI trainee paddle collision with improved hit detection
        elif (self.ball_x + self.ball_radius >= self.width - self.paddle_width and 
              self.ai_trainee_y <= self.ball_y <= self.ai_trainee_y + self.paddle_height):
            self.ball_dx = -self.ball_dx
            hit_position = (self.ball_y - self.ai_trainee_y) / self.paddle_height
            self.ball_dy += (hit_position - 0.5) * 2
        
        # Ball out of bounds
        if self.ball_x <= 0:
            reward = -1
            done = True
            last_scorer = "ai_trainee"
            self.ai_trainee_score += 1
        elif self.ball_x >= self.width:
            reward = 1
            done = True
            last_scorer = "player"
            self.player_score += 1
        
        # Normalize ball speed
        speed = np.sqrt(self.ball_dx**2 + self.ball_dy**2)
        if speed > 0:  # Prevent division by zero
            self.ball_dx = (self.ball_dx / speed) * self.ball_speed
            self.ball_dy = (self.ball_dy / speed) * self.ball_speed
        
        return self._get_state(), reward, done, last_scorer
    
    def render(self, screen):
        # Clear screen with a dark background
        screen.fill((20, 20, 20))
        
        # Draw paddles with smooth edges
        pygame.draw.rect(screen, (255, 255, 255),
                        pygame.Rect(0, self.player_y, 
                                  self.paddle_width, self.paddle_height),
                        border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255),
                        pygame.Rect(self.width - self.paddle_width, self.ai_trainee_y,
                                  self.paddle_width, self.paddle_height),
                        border_radius=5)
        
        # Draw ball with smooth edges
        pygame.draw.circle(screen, (255, 255, 255),
                         (int(self.ball_x), int(self.ball_y)), self.ball_radius)
        
        # Draw center line with better visibility
        for i in range(0, self.height, 30):  # Increased spacing
            pygame.draw.rect(screen, (100, 100, 100),  # Lighter color
                           pygame.Rect(self.width//2 - 2, i, 4, 15))  # Thicker line
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Player: {self.player_score} - AI Trainee: {self.ai_trainee_score}", 
                               True, (200, 200, 200))
        screen.blit(score_text, (self.width//2 - 100, 10))
        
        # Update display
        pygame.display.flip() 