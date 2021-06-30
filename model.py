import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0")

class ActorCritic(nn.Module):
    def __init__(self, N, state_dim, action_dim, action_std):
        h1 = N
        h2 = N

        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, 1)
        ).to(device)
        self.action_var = torch.full((action_dim,), 1, device=device, dtype=torch.double)

        self.std = action_std

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_probabilities = self.actor(state)

        value = self.critic(state)

        dist = Categorical(action_probabilities)

        action = dist.sample()

        action_logprob = dist.log_prob(action)
        memory.append_memory_as(action.detach(), state.detach(), action_logprob.detach(), value.detach())

        return action.detach()

    def evaluate(self, state, action):
        action_probabilities = self.actor(state)

        dist = Categorical(action_probabilities)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy