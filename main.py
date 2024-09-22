import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import os
import json

torch.backends.cudnn.benchmark = True
from wrappers import apply_wrappers
from agent import Agent

ENV_NAME = "SuperMarioBros-1-1-v3"
NUM_OF_EPISODES = 2000
MODEL_SAVE_PATH = "./models/mario_agent"  # Directory to save/load models
CHECKPOINT_PATH = "./models/checkpoint-2.pth"  # Path for the latest model

# Create the model directory if it doesn't exist
os.makedirs("./models", exist_ok=True)

# Initialize the environment
env = gym_super_mario_bros.make(
    ENV_NAME, render_mode="None", apply_api_compatibility=True
)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# Initialize the agent
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Load model if a checkpoint exists
start_episode = 0
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    agent.online_network.load_state_dict(checkpoint["model_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.learn_step_counter = checkpoint["learn_step_counter"]
    start_episode = checkpoint["episode"]
    agent.epsilon = checkpoint["epsilon"]
    print(
        f"Resuming training from episode {start_episode + 1}, epsilon: {agent.epsilon}"
    )

# Start training loop
for i in range(start_episode, NUM_OF_EPISODES):
    done = False
    state, _ = env.reset()
    episode_reward = 0
    episode_learning_data = []

    agent.learning_data = []
    while not done:
        action = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)
        agent.store_in_memory(state, action, reward, new_state, done)
        agent.learn()
        state = new_state
        episode_reward += reward

    
    print(
        f"episode {i+1} finished with epsilon = {agent.epsilon}, reward = {episode_reward}, loss = {agent.curr_loss}, steps = {agent.learn_step_counter}"
    )
    if (i + 1) % 10 == 0:
        with open(f"./vis/learning_data.json", "w") as f:
            json.dump(agent.get_learning_data(), f)

        torch.save(
            {
                "episode": i + 1,
                "model_state_dict": agent.online_network.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "learn_step_counter": agent.learn_step_counter,
                "epsilon": agent.epsilon,
            },
            CHECKPOINT_PATH,
        )
        print(f"Model saved for episode {i+1}")


with open("./vis/final_learning_data.json", w) as f:
    json.dump(agent.get_learning_data(), f)

env.close()
