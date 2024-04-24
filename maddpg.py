from ddpg import DDPG
import torch
import torch.nn as nn
import numpy as np
from common.gumbel_softmax import trans2onehot, gumbel_softmax


class MADDPG:
    """The Multi-Agent DDPG Algorithm.

    Attributes:
        agents:     A list of DDPG instances, which are corresponded with the agents in the environment one by one.
        device:     The device to compute.
        gamma:      The gamma parameter in TD target.
        tau:        The tau parameter for soft update.
        critic_criterion: The loss function for the Critic networks.
        use_rnn: A flag indicating whether a recurrent neural network is used in the network architecture.
    """

    def __init__(self, state_dims, action_dims, critic_dim, hidden_dim, net_arch, actor_lr, critic_lr, device, gamma,
                 tau):
        self.agents = [
            DDPG(state_dim=state_dim,
                 action_dim=action_dim,
                 critic_dim=critic_dim,
                 hidden_dim=hidden_dim,
                 n_agents=len(state_dims),
                 net_arch=net_arch,
                 actor_lr=actor_lr,
                 critic_lr=critic_lr,
                 device=device)
            for state_dim, action_dim in zip(state_dims, action_dims)
        ]
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = nn.MSELoss()
        self.use_rnn = True if net_arch == "rnn" else False

    @property
    def policies(self):
        return [agent.actor for agent in self.agents]

    @property
    def target_policies(self):
        return [agent.target_actor for agent in self.agents]

    def take_action(self, states, explore, hidden_states=None):
        if self.use_rnn:
            states = [torch.tensor(np.array([state]), dtype=torch.float, device=self.device) for state in states]
            results = [agent.take_action(state, explore, hidden_state) for agent, state, hidden_state in
                       zip(self.agents, states, hidden_states)]

            return [result[0] for result in results], [result[1] for result in results]
        else:
            states = [torch.tensor(np.array([state]), dtype=torch.float, device=self.device) for state in states]
            return [agent.take_action(state, explore) for agent, state in
                    zip(self.agents, states)]

    def update(self, sample, agent_idx):
        states, actions, rewards, next_states, dones, hidden_states = sample

        current_agent = self.agents[agent_idx]
        current_agent.critic_optimizer.zero_grad()

        # Use the double DQN strategy, choose actions from the original network
        if self.use_rnn:
            target_action = [
                trans2onehot(policy(_next_obs, hidden_state)[0])
                for policy, _next_obs, hidden_state in zip(self.policies, next_states, hidden_states)
            ]
            target_critic_input = torch.cat((*next_states, *target_action), dim=1)
            target_critic_value = (rewards[agent_idx].view(-1, 1) + self.gamma *
                                   current_agent.target_critic(target_critic_input, torch.cat(hidden_states, dim=1))[0]
                                   * (1 - dones[agent_idx].view(-1, 1)))
            critic_input = torch.cat((*states, *actions), dim=1)
            critic_value = current_agent.critic(critic_input, torch.cat(hidden_states, dim=1))[0]
        else:
            target_action = [
                trans2onehot(policy(_next_obs))
                for policy, _next_obs in zip(self.policies, next_states)
            ]
            target_critic_input = torch.cat((*next_states, *target_action), dim=1)
            target_critic_value = (rewards[agent_idx].view(-1, 1) +
                                   self.gamma * current_agent.target_critic(target_critic_input) *
                                   (1 - dones[agent_idx].view(-1, 1)))
            critic_input = torch.cat((*states, *actions), dim=1)
            critic_value = current_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        current_agent.critic_optimizer.step()
        current_agent.actor_optimizer.zero_grad()

        if self.use_rnn:
            current_actor_action_value = current_agent.actor(states[agent_idx], hidden_states[agent_idx])[0]
            current_actor_action_onehot = gumbel_softmax(current_actor_action_value)
            all_actor_actions = [
                current_actor_action_onehot if i == agent_idx else trans2onehot(_policy(_state, hidden_state)[0])
                for i, (_policy, _state, hidden_state) in enumerate(zip(self.policies, states, hidden_states))
            ]
            current_critic_input = torch.cat((*states, *all_actor_actions), dim=1)
            actor_loss = (-current_agent.critic(current_critic_input, torch.cat(hidden_states, dim=1))[0].mean() +
                          (current_actor_action_value ** 2).mean() * 1e-3)
        else:
            current_actor_action_value = current_agent.actor(states[agent_idx])
            current_actor_action_onehot = gumbel_softmax(current_actor_action_value)
            all_actor_actions = [
                current_actor_action_onehot if i == agent_idx else trans2onehot(_policy(_state))
                for i, (_policy, _state) in enumerate(zip(self.policies, states))
            ]
            current_critic_input = torch.cat((*states, *all_actor_actions), dim=1)
            actor_loss = (-current_agent.critic(current_critic_input).mean() +
                          (current_actor_action_value ** 2).mean() * 1e-3)
        actor_loss.backward()
        current_agent.actor_optimizer.step()

    def update_all_targets_params(self):
        for agent in self.agents:
            agent.soft_update(agent.actor, agent.target_actor, self.tau)
            agent.soft_update(agent.critic, agent.target_critic, self.tau)
