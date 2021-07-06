from q_learning import Agent
from game import game
from train import board2list, first_player, second_player, action2place, random_play
import torch
from matplotlib import pyplot as plt

agent = Agent(state_size=9*3, action_size=9, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
env = game()

for i in range(10):
    board = env.reset()
    state, board_list = board2list(board, main=first_player, opponent=second_player)
    score = 0
    for t in range(10):
        print(state)
        action = agent.act(state)
        print(action)
        place = action2place(action)
        board, reward, done = env.step(mark=first_player, place=place)
        next_state, board_list = board2list(board, main=first_player, opponent=second_player)
        agent.step(state, action, reward, next_state, done)
        if not done:
            random_action = random_play(next_state, board_list, ppo_opponent=False)
            board, reward, done = env.step(mark=second_player, place=random_action)
            next_state, board = board2list(board, main=first_player, opponent=second_player)
            reward = -reward
        state = next_state
        score += reward
        if done:
            print(env.winner)
            break