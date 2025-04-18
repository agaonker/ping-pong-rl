import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from pong_env import PongEnv
import pygame
import math
from torch.optim.lr_scheduler import StepLR
import os
import json
import sys

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None
            
        priorities = self.priorities[:len(self.memory)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class DQNAgent:
    def __init__(self, state_size, action_size, model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.update_target_model()

    def save_model(self, path, episode=None, rewards_history=None):
        """Save the model and training state"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory,
        }, path)
        
        # Save training metadata if provided
        if episode is not None and rewards_history is not None:
            metadata = {
                'episode': episode,
                'rewards_history': rewards_history,
                'epsilon': self.epsilon,
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
            with open(f"{path}.meta.json", 'w') as f:
                json.dump(metadata, f)
        
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load the model and training state"""
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return False
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory = checkpoint['memory']
        
        # Try to load metadata
        meta_path = f"{path}.meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                print(f"Loaded model from episode {metadata['episode']}")
                print(f"Last average reward: {np.mean(metadata['rewards_history'][-100:]):.2f}")
        
        print(f"Model loaded from {path}")
        return True

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size):
        if len(self.memory.memory) < batch_size:
            return
            
        samples, indices, weights = self.memory.sample(batch_size)
        if samples is None:
            return
            
        # Ensure we have enough samples for batch normalization
        if len(samples) < 2:
            return
            
        states = torch.FloatTensor([i[0] for i in samples]).to(self.device)
        actions = torch.LongTensor([i[1] for i in samples]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in samples]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in samples]).to(self.device)
        dones = torch.FloatTensor([i[4] for i in samples]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute TD errors for prioritized replay
        td_errors = torch.abs(current_q_values.squeeze().detach() - target_q_values.detach()).cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        loss = (weights * nn.MSELoss(reduction='none')(current_q_values.squeeze(), target_q_values)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()  # Update learning rate
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

def train(model_path=None, load_model=False):
    try:
        # Create models directory if it doesn't exist
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Initialize environment
        env = PongEnv()
        state_size = 6
        action_size = 3
        
        # Initialize agent
        agent = DQNAgent(state_size, action_size, 
                        model_path if load_model else None)
        
        # Initialize pygame
        pygame.init()
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption('Pong RL Training')
        clock = pygame.time.Clock()
        
        # Training parameters
        episodes = 2000
        batch_size = 64
        target_update = 10
        save_interval = 100  # Save model every 100 episodes
        
        # Training statistics
        rewards_history = []
        epsilons_history = []
        
        for episode in range(episodes):
            try:
                state = env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            # Save model before quitting
                            if model_path:
                                agent.save_model(model_path, episode, rewards_history)
                            pygame.quit()
                            return
                    
                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay(batch_size)
                    
                    state = next_state
                    total_reward += reward
                    
                    env.render(screen)
                    pygame.display.flip()
                    clock.tick(60)
                
                if episode % target_update == 0:
                    agent.update_target_model()
                
                rewards_history.append(total_reward)
                epsilons_history.append(agent.epsilon)
                
                # Save model periodically
                if model_path and (episode + 1) % save_interval == 0:
                    agent.save_model(model_path, episode, rewards_history)
                
                # Print training statistics
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, "
                      f"Avg Reward (last 100): {avg_reward:.2f}, "
                      f"Epsilon: {agent.epsilon:.2f}, "
                      f"LR: {agent.scheduler.get_last_lr()[0]:.6f}")
            
            except Exception as e:
                print(f"Error in episode {episode + 1}: {str(e)}")
                continue
        
        # Save final model
        if model_path:
            agent.save_model(model_path, episodes - 1, rewards_history)
        
        pygame.quit()
    
    except Exception as e:
        print(f"Training error: {str(e)}")
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Train and save model
        train(model_path="models/pong_model.pt")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1) 