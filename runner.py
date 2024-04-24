from tqdm import tqdm
import numpy as np
import torch

from common.utils import get_args, make_env
from common.replay_buffer import ReplayBuffer
from maddpg import MADDPG
from common.plot import save_plot


class Runner(object):
    """A class representing a runner for training a multi-agent reinforcement learning algorithm.

        Args:
            args: An object containing various configuration parameters.
            env: The environment in which the agents interact.

        Attributes:
            args: Configuration parameters for the runner.
            env: The environment in which the agents interact.
            buffer: A replay buffer for storing and sampling agent experiences.
            device: The device (GPU or CPU) on which the computations are performed.
            use_rnn: A flag indicating whether a recurrent neural network is used in the network architecture.
            maddpg: The MADDPG algorithm controller.

        """

    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.buffer = ReplayBuffer(self.args.buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_rnn = True if self.args.net_arch == "rnn" else False

        state_dims = [self.env.observation_space(self.env.agents[i]).shape[0] for i in range(self.env.num_agents)]
        action_dims = [self.env.action_space(self.env.agents[i]).n for i in range(self.env.num_agents)]
        critic_dim = sum(state_dims) + sum(action_dims)
        # create algorithm controller
        self.maddpg = MADDPG(state_dims=state_dims,
                             action_dims=action_dims,
                             critic_dim=critic_dim,
                             hidden_dim=self.args.hidden_dim,
                             net_arch=self.args.net_arch,
                             actor_lr=self.args.actor_lr,
                             critic_lr=self.args.critic_lr,
                             device=self.device,
                             gamma=self.args.gamma,
                             tau=self.args.tau)

    def train(self):
        return_list = []
        total_step = 0

        for episode in tqdm(range(self.args.num_episodes)):
            states_dict, _ = self.env.reset()

            states = [state for state in states_dict.values()]
            hidden_states = [torch.zeros(1, 1, self.args.hidden_dim) for _ in range(self.env.num_agents)]

            for episode_step in range(self.args.len_episodes):
                if self.use_rnn:
                    actions, hidden_states = self.maddpg.take_action(states, explore=True, hidden_states=hidden_states)
                else:
                    actions = self.maddpg.take_action(states, explore=True)
                actions_SN = [np.argmax(onehot) for onehot in actions]
                actions_dict = {self.env.agents[i]: actions_SN[i] for i in range(self.env.max_num_agents)}

                next_states_dict, rewards_dict, terminations_dict, truncations_dict, _ = self.env.step(actions_dict)
                rewards = [reward for reward in rewards_dict.values()]
                next_states = [next_state for next_state in next_states_dict.values()]
                terminations = [termination for termination in terminations_dict.values()]
                # hidden_states = [hidden_state.detach().numpy() for hidden_state in hidden_states]
                self.buffer.add(states, actions, rewards, next_states, terminations, hidden_states)

                states = next_states
                total_step += 1

                # When the replay buffer reaches a certain size and the number of steps reaches the specified update interval
                # 1. Sample from the replay buffer.
                # 2. Update Actors and Critics.
                # 3. Update Target networks' parameters.
                def sample_rearrange(x, device):
                    """Rearrange the transition in the sample."""
                    rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
                    return [torch.FloatTensor(np.vstack(attribute)).to(device) for attribute in rearranged]

                if len(self.buffer) >= self.args.minimal_size and total_step % self.args.update_interval == 0:
                    sample = self.buffer.sample(self.args.batch_size)
                    sample = [sample_rearrange(x, self.device) for x in sample]
                    # Update Actors and Critics
                    for agent_idx in range(self.env.max_num_agents):
                        self.maddpg.update(sample, agent_idx)
                    # Update Target Parameters
                    self.maddpg.update_all_targets_params()

            if (episode + 1) % self.args.evaluate_interval == 0:
                episodes_returns = self.evaluate()
                return_list.append(episodes_returns)
                # print(f"Episode: {episode + 1}, {episodes_returns}")

        self.env.close()
        return_array = np.array(return_list)
        return return_array

    def evaluate(self, num_episode=100, len_episode=25):
        """Evaluate the strategies for learning, so no exploration is undertaken at this time.

            Args:
                num_episode: The number of episodes.
                len_episode: The length of each episode.
            """
        env = make_env(self.args.env)
        returns = np.zeros((num_episode, env.max_num_agents))  # store the cumulative returns for each agent.
        for episode in range(num_episode):
            states_dict, rewards_dict = env.reset()
            states = [state for state in states_dict.values()]
            hidden_states = [torch.zeros(1, 1, self.args.hidden_dim) for _ in range(env.num_agents)]
            for episode_step in range(len_episode):
                if self.use_rnn:
                    actions, hidden_states = self.maddpg.take_action(states, explore=False, hidden_states=hidden_states)
                else:
                    actions = self.maddpg.take_action(states, explore=False)

                actions_SN = [np.argmax(onehot) for onehot in actions]
                actions_dict = {env.agents[i]: actions_SN[i] for i in range(env.max_num_agents)}
                next_states_dict, rewards_dict, terminations_dict, truncations_dict, _ = env.step(actions_dict)
                rewards = [reward for reward in rewards_dict.values()]
                next_states = [next_state for next_state in next_states_dict.values()]
                states = next_states
                rewards = np.array(rewards)
                returns[episode] += rewards
        env.close()
        return returns.tolist()


if __name__ == "__main__":
    args = get_args('mpe_config.yaml')
    env = make_env(args.env)
    runner = Runner(args, env)
    return_array = runner.train()
    save_plot(return_array, env, args.data_save_dir)
