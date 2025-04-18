# Pong Reinforcement Learning

A Pong game implementation with Reinforcement Learning using PyTorch. The AI agent learns to play Pong through Deep Q-Learning (DQN).

![Pong Game Screenshot](docs/images/pong_game.png)

## Features

- Deep Q-Learning (DQN) implementation
- Prioritized Experience Replay
- Smooth paddle movement with inertia
- Hardware-accelerated rendering
- Real-time score display
- Training progress tracking

## Current Score
As shown in the screenshot, the AI Trainee has learned to play effectively:
- Player: 40
- AI Trainee: 173

## Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd pong-rl

# Install dependencies
pip install -e .
```

## Training

To train the AI:

```bash
python train_pong.py
```

The model will be saved in the `models` directory.

## Project Structure

- `pong_env.py`: Pong game environment
- `train_pong.py`: Training script with DQN implementation
- `pyproject.toml`: Project dependencies and configuration

## Dependencies

- PyTorch
- Pygame
- NumPy
- OpenCV (for video creation)
- Pillow (for image processing)

## License

[Your chosen license] 