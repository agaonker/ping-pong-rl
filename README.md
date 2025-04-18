# Pong Reinforcement Learning

A Pong game environment for reinforcement learning using PyTorch and Pygame.

## Setup

1. Install `uv` (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -e .
```

## Project Structure

- `pong_env.py`: The Pong game environment
- `train_pong.py`: Training script using DQN
- `pyproject.toml`: Project configuration and dependencies

## Usage

To train the agent:
```bash
python train_pong.py
```

## Features

- Custom Pong environment with realistic physics
- Deep Q-Network (DQN) implementation
- Visual rendering using Pygame
- Continuous state space with normalized values
- Three actions: stay, up, down

## Requirements

- Python >= 3.8
- PyTorch
- NumPy
- Pygame 