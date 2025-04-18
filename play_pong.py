import pygame
import torch
from pong_env import PongEnv
from train_pong import DQNAgent
import sys
import os

def play_against_model(model_path):
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please train the model first using train_pong.py")
            return
        
        # Initialize environment
        env = PongEnv()
        state_size = 6
        action_size = 3
        
        # Initialize agent
        agent = DQNAgent(state_size, action_size, model_path)
        agent.epsilon = 0.0  # Disable exploration
        
        # Initialize pygame
        pygame.init()
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption('Play Against AI')
        clock = pygame.time.Clock()
        
        # Game variables
        player_score = 0
        ai_score = 0
        font = pygame.font.Font(None, 36)
        
        while True:
            try:
                state = env.reset()
                done = False
                total_reward = 0
                
                while not done:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                    
                    # Get AI action
                    ai_action = agent.act(state, training=False)
                    
                    # Get player action from keyboard
                    keys = pygame.key.get_pressed()
                    player_action = 1 if keys[pygame.K_UP] else 2 if keys[pygame.K_DOWN] else 0
                    
                    # Take actions
                    next_state, reward, done = env.step(ai_action)
                    
                    # Update scores
                    if reward == 1:  # AI scored
                        ai_score += 1
                    elif reward == -1:  # Player scored
                        player_score += 1
                    
                    state = next_state
                    total_reward += reward
                    
                    # Render game
                    env.render(screen)
                    
                    # Display scores
                    score_text = font.render(f"AI: {ai_score} - Player: {player_score}", True, (255, 255, 255))
                    screen.blit(score_text, (env.width//2 - 100, 10))
                    
                    pygame.display.flip()
                    clock.tick(60)
                
                # Display game over message
                game_over_text = font.render("Game Over! Press SPACE to continue or ESC to quit", True, (255, 255, 255))
                screen.blit(game_over_text, (env.width//2 - 300, env.height//2))
                pygame.display.flip()
                
                # Wait for player input
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                waiting = False
                            elif event.key == pygame.K_ESCAPE:
                                pygame.quit()
                                sys.exit()
            
            except Exception as e:
                print(f"Game error: {str(e)}")
                pygame.quit()
                return
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Default model path
        model_path = "models/pong_model.pt"
        
        # Check if models directory exists
        if not os.path.exists("models"):
            print("Error: 'models' directory not found")
            print("Please train the model first using train_pong.py")
            sys.exit(1)
        
        # Use command line argument if provided
        if len(sys.argv) > 1:
            model_path = sys.argv[1]
        
        print(f"Loading model from {model_path}")
        play_against_model(model_path)
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1) 