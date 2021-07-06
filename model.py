import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.functional import F

device = torch.device("cuda:0")

class Actor(nn.Module):
    def __init__(self, state_dim, h1, h2, output_dim):
        super(Actor, self).__init__()

        self.fc_1 = nn.Linear(state_dim, h1)
        self.fc_2 = nn.Linear(h1, h2)
        self.fc_3 = nn.Linear(h2, output_dim)

    def forward(self, input):
        x = F.relu(self.fc_1(input))
        x = F.relu(self.fc_2(x))
        x = torch.softmax(self.fc_3(x), dim=1)
        return x

class ActorCritic(nn.Module):
    def __init__(self, N, state_dim, action_dim, eps_threshold):
        h1 = N
        h2 = N
        self.eps_threshold = eps_threshold

        super(ActorCritic, self).__init__()

        self.actor = Actor(state_dim, h1, h2, action_dim).to(device)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        ).to(device)
        self.action_var = torch.full((action_dim,), 1, device=device, dtype=torch.double)


    def forward(self):
        raise NotImplementedError

    def act(self, state, memory, evaluation=False):

        action_probabilities = self.actor(state)

        value = self.critic(state)

        dist = Categorical(action_probabilities)

        # if in evaluation mode, execute the most probable action
        # else make an action distribution based on probabilities
        if evaluation:
            action = torch.argmax(action_probabilities, dim=1)
        else:
            action = dist.sample()

        action_logprob = dist.log_prob(action)
        if not evaluation:
            memory.append_memory_as(action.detach(), state.detach(), action_logprob.detach(), value.detach())

        return action.detach()

    def evaluate(self, state, action):
        action_probabilities = self.actor(state)

        dist = Categorical(action_probabilities)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy