from common.network import MLP, RNN
import torch.optim as optim
from common.gumbel_softmax import trans2onehot, gumbel_softmax


class DDPG:
    """The DDPG Algorithm.

    Attributes:
        actor:              The Actor (Policy Network).
        target_actor:       The Target Actor (Target Policy Network).
        critic:             The Critic (value Network).
        target_critic:      The Target Critic (Target Value Network).
        actor_optimizer:    The optimizer of Actor.
        critic_optimizer:   The optimizer of Critic.
        use_rnn:    A flag indicating whether a recurrent neural network is used in the network architecture.
    """

    def __init__(self, state_dim, action_dim, critic_dim, hidden_dim, n_agents, net_arch, actor_lr, critic_lr, device):
        self.use_rnn = True if net_arch == "rnn" else False
        if net_arch == "mlp":
            self.actor = MLP(state_dim, action_dim, hidden_dim).to(device)
            self.target_actor = MLP(state_dim, action_dim, hidden_dim).to(device)

            self.critic = MLP(critic_dim, 1, hidden_dim).to(device)
            self.target_critic = MLP(critic_dim, 1, hidden_dim).to(device)

        elif net_arch == "rnn":
            self.actor = RNN(state_dim, action_dim, hidden_dim).to(device)
            self.target_actor = RNN(state_dim, action_dim, hidden_dim).to(device)

            self.critic = RNN(critic_dim, 1, hidden_dim * n_agents).to(device)
            self.target_critic = RNN(critic_dim, 1, hidden_dim * n_agents).to(device)
        else:
            Exception("Must specify a net arch from mlp or rnn.")

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, state, explore=False, hidden_state=None):
        if self.use_rnn:
            action, hidden_state = self.actor(state, hidden_state)
            if explore:
                action = gumbel_softmax(action)
            else:
                action = trans2onehot(action)
            return action.detach().cpu().numpy()[0], hidden_state.squeeze(0).detach()

        else:
            action = self.actor(state)
            if explore:
                action = gumbel_softmax(action)
            else:
                action = trans2onehot(action)

            return action.detach().cpu().numpy()[0]

    @staticmethod
    def soft_update(net, target_net, tau):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
