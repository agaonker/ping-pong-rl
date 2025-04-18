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
        
        # Initialize game state
        self.reset()
        
        # Learning parameters
        self.gamma = 0.95    # discount rate for future rewards
        self.epsilon = 1.0   # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995  # decay rate for exploration
        self.learning_rate = 0.001  # learning rate for optimizer
        
        self.memory = deque(maxlen=10000)  # Experience replay buffer size
        
    def reset(self):
        # Reset player paddle
        self.player_y = self.height // 2 - self.paddle_height // 2
        
        # Reset opponent paddle (AI controlled)
        self.opponent_y = self.height // 2 - self.paddle_height // 2
        
        # Reset ball position and velocity
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        
        # Random initial direction with controlled angle
        angle = random.uniform(-self.max_ball_angle, self.max_ball_angle)
        angle_rad = np.radians(angle)
        self.ball_dx = self.ball_speed * np.cos(angle_rad)
        self.ball_dy = self.ball_speed * np.sin(angle_rad)
        
        # Randomly choose direction
        self.ball_dx *= random.choice([-1, 1])
        
        return self._get_state()
    
    def _get_state(self):
        # Normalize all values to [0, 1] range
        return np.array([
            self.player_y / self.height,
            self.opponent_y / self.height,
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_dx / self.ball_speed,
            self.ball_dy / self.ball_speed
        ])
    
    def step(self, action):
        # 0: Stay, 1: Up, 2: Down
        if action == 1:  # Move up
            self.player_y = max(self.player_y - self.paddle_speed, 0)
        elif action == 2:  # Move down
            self.player_y = min(self.player_y + self.paddle_speed, 
                              self.height - self.paddle_height)
        
        # Simple opponent AI - follows ball
        if self.opponent_y + self.paddle_height/2 < self.ball_y:
            self.opponent_y = min(self.opponent_y + self.paddle_speed,
                                self.height - self.paddle_height)
        elif self.opponent_y + self.paddle_height/2 > self.ball_y:
            self.opponent_y = max(self.opponent_y - self.paddle_speed, 0)
        
        # Update ball position
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball collision with top and bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.height:
            self.ball_dy = -self.ball_dy
        
        # Ball collision with paddles
        reward = 0
        done = False
        
        # Player paddle collision
        if (self.ball_x <= self.paddle_width and 
            self.player_y <= self.ball_y <= self.player_y + self.paddle_height):
            self.ball_dx = -self.ball_dx
            # Add slight randomness to ball direction
            self.ball_dy += random.uniform(-1, 1)
            reward = 1
        
        # Opponent paddle collision
        elif (self.ball_x >= self.width - self.paddle_width and 
              self.opponent_y <= self.ball_y <= self.opponent_y + self.paddle_height):
            self.ball_dx = -self.ball_dx
            self.ball_dy += random.uniform(-1, 1)
        
        # Ball out of bounds
        if self.ball_x <= 0:
            reward = -1
            done = True
        elif self.ball_x >= self.width:
            reward = 1
            done = True
        
        # Normalize ball speed
        speed = np.sqrt(self.ball_dx**2 + self.ball_dy**2)
        self.ball_dx = (self.ball_dx / speed) * self.ball_speed
        self.ball_dy = (self.ball_dy / speed) * self.ball_speed
        
        return self._get_state(), reward, done
    
    def render(self, screen):
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw paddles
        pygame.draw.rect(screen, (255, 255, 255),
                        pygame.Rect(0, self.player_y, 
                                  self.paddle_width, self.paddle_height))
        pygame.draw.rect(screen, (255, 255, 255),
                        pygame.Rect(self.width - self.paddle_width, self.opponent_y,
                                  self.paddle_width, self.paddle_height))
        
        # Draw ball
        pygame.draw.circle(screen, (255, 255, 255),
                         (int(self.ball_x), int(self.ball_y)), self.ball_radius)
        
        # Draw center line
        for i in range(0, self.height, 20):
            pygame.draw.rect(screen, (128, 128, 128),
                           pygame.Rect(self.width//2 - 2, i, 4, 10)) 