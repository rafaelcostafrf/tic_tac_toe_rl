from q_learning import Agent
from game import game
from collections import deque
from matplotlib import pyplot as plt
import numpy as np
import torch
from train import random_play, action2place, board2list, first_player, second_player

agent = Agent(state_size=9*2, action_size=9, seed=0)
try:
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    print('Saved policy Loaded')
except:
    print('New policy Created')

env = game()
def dqn(n_episodes=50000, max_t=1000, eps_start=1.0, eps_end=0.05,
        eps_decay=0.9999):
    """Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon

    """
    scores = []  # list containing score from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        board = env.reset()
        state, board_list = board2list(board, main=first_player, opponent=second_player)
        score = 0
        for t in range(max_t):

            action = agent.act(state, eps)

            place = action2place(action)
            board, reward, done = env.step(mark=first_player, place=place)
            next_state, board_list = board2list(board, main=first_player, opponent=second_player)
            if not done:
                random_action = random_play(next_state, board_list, ppo_opponent=False)
                board, reward, done = env.step(mark=second_player, place=random_action)
                next_state, board = board2list(board, main=first_player, opponent=second_player)
                reward = -reward
            agent.step(state, action, reward, next_state, done)
            print(state)
            ## above step decides whether we will train(learn) the network
            ## actor (local_qnetwork) or we will fill the replay buffer
            ## if len replay buffer is equal to the batch size then we will
            ## train the network or otherwise we will add experience tuple in our
            ## replay buffer.
            state = next_state
            score += reward
            scores_window.append(score)  ## save the most recent score
            eps = max(eps * eps_decay, eps_end)  ## decrease the epsilon
            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode, np.mean(scores_window)))

            if np.mean(scores_window) >= 1.5:
                print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode - 100,
                                                                                            np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
            if done:
                scores.append(np.mean(scores_window))  ## sae the most recent score
                print(env.winner)
                break
    return scores

torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Epsiode #')
plt.show()