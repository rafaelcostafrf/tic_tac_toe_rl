import sys
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from model import ActorCritic


torch.set_printoptions(threshold=10_000)
## HYPERPARAMETERS - CHANGE IF NECESSARY ##
lr_ac = 0.0001
lr_ct = 0.0005

K_epochs = 5
critic_epochs = 5

eps_clip = 0.2
gamma = 0.2
betas = (0.9, 0.999)
DEBUG = 0
BATCH_SIZE = 2048*2


log_interval = 10

def plot_returns(returns, values, terminals):
    f, ax = plt.subplots()
    ax.plot(returns)
    ax.plot(values)
    for i, value in enumerate(terminals):
        if value:
            ax.vlines(i, min(returns), max(returns))
    return f


class PPO:
    def __init__(self, input_dim, action_dim, evaluation = False):
        self.evaluation = evaluation
        if not self.evaluation:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()
            eps = 0.0
        else:
            eps = 0
        self.device = torch.device("cuda:0")
        self.policy = ActorCritic(N=2048, state_dim=input_dim, action_dim=action_dim, eps_threshold=eps)
        self.optimizer_ac = torch.optim.Adam(self.policy.actor.parameters(), lr=lr_ac, betas=betas)
        self.optimizer_ct = torch.optim.Adam(self.policy.critic.parameters(), lr=lr_ct, betas=betas)

        self.policy_old = ActorCritic(N=2048, state_dim=input_dim, action_dim=action_dim, eps_threshold=eps)
        self.policy_old.load_state_dict(self.policy.state_dict())

        try:
            self.policy.load_state_dict(torch.load('./policy/tic_tac_toe.pth', map_location=self.device))
            self.policy_old.load_state_dict(
                torch.load('./policy/tic_tac_toe_old.pth', map_location=self.device))
            print('Saved Policy loaded')
        except:
            torch.save(self.policy.state_dict(), './policy/tic_tac_toe.pth')
            torch.save(self.policy_old.state_dict(), './policy/tic_tac_toe_old.pth')
            print('New Policy generated')

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        evaluation = self.evaluation
        state = torch.Tensor(state).to(self.device).detach()
        network_input = state.unsqueeze(0)
        out = self.policy_old.act(network_input, memory, evaluation).cpu().numpy()
        return out

    def split_tensors(self, list_of_tensors):
        splitted_list = []
        for tensor in list_of_tensors:
            splitted_list.append(torch.split(tensor, BATCH_SIZE))
        return splitted_list

    def optimizer_step(self, old_states, old_actions, old_logprobs, returns, advantages):

        for _ in range(K_epochs):
            random_index = torch.randperm(old_states.size()[0])
            old_states = old_states[random_index].detach()
            old_actions = old_actions[random_index].detach()
            old_logprobs = old_logprobs[random_index].detach()
            advantages = advantages[random_index].detach()
            split_list = old_states, old_actions, old_logprobs, advantages

            old_states_list, old_actions_list, old_logprobs_list, advantages_list = \
                self.split_tensors(split_list)

            for old_states_sp, old_actions_sp, old_logprobs_sp, advantages_sp in zip(old_states_list,
                                                                                     old_actions_list,
                                                                                     old_logprobs_list,
                                                                                     advantages_list):
                self.optimizer_ac.zero_grad()

                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_sp, old_actions_sp)
                ratios = (logprobs - old_logprobs_sp).exp()
                surr1 = ratios * advantages_sp.flatten()
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages_sp.flatten()
                actor_loss = - torch.min(surr1, surr2)
                entropy_loss = - dist_entropy
                loss_ac = actor_loss.mean() + 0.01 * entropy_loss.mean()
                loss_ac.mean().backward()
                self.optimizer_ac.step()
        return loss_ac.mean().detach().cpu().numpy()

    def critic_optimizer_step(self, old_states, old_actions, old_logprobs, returns, advantages):
        for _ in range(critic_epochs):
            random_index = torch.randperm(old_states.size()[0])
            old_states = old_states[random_index].detach()
            old_actions = old_actions[random_index].detach()
            old_logprobs = old_logprobs[random_index].detach()
            returns = returns[random_index].detach()
            advantages = advantages[random_index].detach()
            split_list = old_states, old_actions, old_logprobs, advantages, returns

            old_states_list, old_actions_list, old_logprobs_list, advantages_list, returns_list = \
                self.split_tensors(split_list)

            for old_states_sp, old_actions_sp, old_logprobs_sp, advantages_sp, returns_sp in zip(
            old_states_list, old_actions_list, old_logprobs_list, advantages_list, returns_list):

                self.optimizer_ct.zero_grad()

                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_sp, old_actions_sp)
                critic_loss = torch.square(state_values - returns_sp).mean()

                critic_loss.backward()
                self.optimizer_ct.step()
        return critic_loss.mean().detach().cpu().numpy()

    def get_advantages(self, masks, rewards):
        """
        Calculates the advantage using the GAE - Generalized Advantage Estimation
        """
        returns = torch.empty(*rewards.size(), dtype=torch.float).to(self.device)
        gmma = gamma

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                returns[i] = rewards[i]
            else:
                returns[i] = rewards[i] + gmma * returns[i + 1] * masks[i]
        # print(masks)
        # print(rewards)
        # print(returns)

        return returns

    def update(self, memory, episode):
        # convert list to tensor
        old_states = torch.stack(memory.states).detach().to(self.device, dtype=torch.float)
        old_actions = torch.stack(memory.actions).detach().to(self.device, dtype=torch.float)
        old_logprobs = torch.stack(memory.logprobs).detach().to(self.device, dtype=torch.float)
        state_values = torch.stack(memory.values).detach().to(self.device, dtype=torch.float)

        returns = self.get_advantages(np.logical_not(memory.is_terminals), torch.tensor(memory.rewards).to(self.device))

        advantages = returns - state_values.detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8).detach()


        actor_loss = self.optimizer_step(old_states, old_actions, old_logprobs, returns, advantages)
        critic_loss = self.critic_optimizer_step(old_states, old_actions, old_logprobs, returns, advantages)

        if (not self.evaluation) and episode % log_interval == 0:
            self.writer.add_scalar("Critic loss", critic_loss, episode)
            self.writer.add_scalar("Actor loss", actor_loss, episode)
            self.writer.add_scalar("Returns Mean", returns.mean(), episode)
            self.writer.add_figure("Returns",
                                   plot_returns(returns.cpu().numpy()[0:10], state_values.flatten().cpu().numpy()[0:10],
                                                memory.is_terminals[0:10]),
                                   episode)

        self.critic_epoch_loss = []
        self.actor_epoch_loss = []
        self.policy_old.load_state_dict(self.policy.state_dict())
        torch.save(self.policy.state_dict(), './policy/tic_tac_toe.pth')
        torch.save(self.policy_old.state_dict(), './policy/tic_tac_toe_old.pth')
