# Super Mario Bros. Reinforcement Learning Agent

This project uses **Reinforcement Learning** (RL) to train an agent to play the original NES game **Super Mario Bros.**. The agent is trained using the **Proximal Policy Optimization (PPO)** algorithm and the **gym-super-mario-bros** environment, built upon OpenAI's Gym.

## Project Overview

This project sets up an RL environment for **Super Mario Bros.** using the `gym-super-mario-bros` environment. The agent observes the game screen as grayscale frames, with a stack of 4 frames at a time, and makes decisions based on a simplified set of movements (left, right, jump). The RL model is trained to maximize the reward by progressing through the levels.

### Key Components:

- **Environment**: Gym-based environment that interacts with Super Mario Bros.
- **RL Algorithm**: PPO (Proximal Policy Optimization) from the stable-baselines3 library.
- **Observations**: Grayscale images of the game with stacked frames for temporal awareness.
- **Actions**: Simplified movements to control Mario (left, right, jump).

## Installation

To set up the environment and dependencies, run:

```bash
pip install gym gym-super-mario-bros stable-baselines3[extra] opencv-python
```

## Usage

1. Train the Agent:

```python
import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Create the Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = gym.wrappers.ActionWrapper(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4)

# Initialize the PPO agent
model = PPO('CnnPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=1_000_000)

# Save the model
model.save("mario_ppo")
```

2. Evaluate and Watch the Agent:

```python
from stable_baselines3.common.evaluation import evaluate_policy

# Load the saved model
model = PPO.load("mario_ppo")

# Evaluate the model
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f'Mean reward: {mean_reward}')

# Render and watch the agent play
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
```

## Project Structure

- `main.py`: Script to set up, train, and evaluate the model.
- `README.md`: Project overview and instructions.
- `requirements.txt`: List of dependencies to run the project.
- `mario_ppo.zip`: Pretrained model.

## To-Do List and Enhancements

1. Reward Shaping: Modify the reward function to incentivize faster completion times to make the agent "speedrun" levels.
2. Advanced Action Space: Expand the action space to include more complex moves (e.g., running, ducking).
3. Different Levels: Train the agent on different levels of the game and implement multi-level training.
4. Transfer Learning: Implement transfer learning to use a pretrained model and adapt it to new levels or objectives.
5. Checkpoint Saving: Save checkpoints during training to resume from certain steps.
6. Visualization Tools: Add tools to visualize training performance (reward over time, etc.).
7. Multiplayer Agent: Explore multi-agent training or co-op gameplay with other agents.

## Future Enhancements

- Hyperparameter Tuning: Experiment with different hyperparameters to improve performance and training speed.
- AI Model Exploration: Test different RL algorithms (e.g., DQN, A2C) and compare their performance against PPO.
- Custom Mario Modifications: Integrate with custom Mario levels or modified environments for a new challenge.
- Optimizing for Speedrunning: Tweak the agent's policy and environment interaction to mimic human-like speedrunning strategies.
