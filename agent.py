import torch
import numpy as np
from agent_nn import AgentNN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data import PrioritizedReplayBuffer


class Agent:
    def __init__(self, input_dims, num_actions):
        self.num_actions = num_actions
        self.learn_step_counter = 0

        self.lr = 0.001
        self.gamma = 0.93
        self.epsilon = 1.0
        self.eps_decay = 0.99995
        self.eps_min = 0.15
        self.batch_size = 64
        self.sync_network_rate = 1000

        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()
        self.curr_loss = 0
        self.curr_reward = 0
        self.replay_buffer_capacity = 100_000
        storage = LazyMemmapStorage(self.replay_buffer_capacity)
        self.replay_buffer = PrioritizedReplayBuffer(
            storage=storage, alpha=0.6, beta=0.4
        )

        self.learning_data = []

    # def clear_memory(self):
    #     storage = LazyMemmapStorage(self.replay_buffer_capacity)
    #     self.replay_buffer = TensorDictReplayBuffer(storage=storage)
    #     self.learning_data = []
    #     print("Memory cleared. Replay buffer size:", len(self.replay_buffer))

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        observation = (
            torch.tensor(np.array(observation), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.online_network.device)
        )
        with torch.no_grad():
            q_values = self.online_network(observation)

        return torch.argmax(q_values).item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        # If the replay buffer is at capacity, remove the oldest entry
        if len(self.replay_buffer) >= self.replay_buffer_capacity:
            self.replay_buffer.pop(0)  # This removes the oldest entry

        self.replay_buffer.add(
            TensorDict(
                {
                    "state": torch.tensor(np.array(state), dtype=torch.float32),
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "next_state": torch.tensor(
                        np.array(next_state), dtype=torch.float32
                    ),
                    "done": torch.tensor(done),
                },
                batch_size=[],
            )
        )

    def sync_networks(self):
        if (
            self.learn_step_counter % self.sync_network_rate == 0
            and self.learn_step_counter > 0
        ):
            self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        self.sync_networks()
        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size).to(
            self.online_network.device
        )

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_state, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[
            np.arange(self.batch_size), actions.squeeze()
        ]

        target_q_values = self.target_network(next_state).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10)
        self.optimizer.step()
        self.curr_loss = loss
        self.curr_reward = rewards.mean().item()
        self.learn_step_counter += 1
        self.decay_epsilon()
        # Save the model and optimizer state after every step
        # self.learning_data.append(
        #     {
        #         "reward": rewards.mean().item(),
        #         "loss": self.curr_loss.item(),
        #         "epsilon": self.epsilon,
        #     }
        # )

    def track_learning(self, reward):
        if type(self.curr_loss) is not int:
            self.curr_loss = self.curr_loss.item()
        self.learning_data.append(
            {
                "reward": self.curr_reward,
                "loss": self.curr_loss,
                "epsilon": self.epsilon,
            }
        )

    def get_learning_data(self):
        return self.learning_data
