import torch
import numpy as np
from game import game
from ppo import PPO
from memory import memory

# Change o and x here. Note - First player should always be the bot (but it does not mean it starts first)
first_player = 'o'  # ALWAYS THE BOT
second_player = 'x'

# Convert the 3x3 board to a linear list 1x9
def board2list(board):
    board_list = []
    x_array = np.zeros(9)
    empty_array = np.ones(9)
    o_array = np.zeros(9)
    for row in board:
        board_list += row
    for i in range(9):
        if board_list[i] == second_player:
            o_array[i] = 1
            empty_array[i] = 0
        if board_list[i] == first_player:
            x_array[i] = 1
            empty_array[i] = 0
    input_array = np.concatenate((o_array, x_array, empty_array))
    return input_array, board_list


# Converts the action in 1x9 to 3x3
def action2place(action):
    place = [int(action / 3), action[0] % 3]
    return place


# Random operator
def random_play(board_list):
    possible_play = []
    for i, entry in enumerate(board_list):
        if entry == '':
            possible_play.append(i)
    random_play = np.random.choice(possible_play).reshape(-1)
    random_play = action2place(random_play)
    return random_play


# Run the training
if __name__ == '__main__':
    BATCH_SIZE = 2048
    winner_dict = {first_player: 0, second_player: 0}

    memory_game = memory()
    tic_tac_toe = game()
    policy = PPO(9 * 3, 9)

    samples = 0
    episode = 0
    while True:
        board = tic_tac_toe.reset()
        done = False
        input_array, board_list = board2list(board)
        random_first = np.random.random()
        if random_first > 0.5:
            possible_play = random_play(board_list)
            board, _, done = tic_tac_toe.step(second_player, possible_play)
            input_array, board_list = board2list(board)
        while not done:
            action = policy.select_action(input_array, memory_game)
            action = action2place(action)
            board, reward, done = tic_tac_toe.step(first_player, action)
            input_array, board_list = board2list(board)
            if not done:
                possible_play = random_play(board_list)
                board, reward, done = tic_tac_toe.step(second_player, possible_play)
                input_array, board_list = board2list(board)
                reward = -reward
            memory_game.append_memory_rt(reward, done)
            samples += 1
        if samples > BATCH_SIZE:
            policy.update(memory_game, episode)
            memory_game.clear_memory()
            samples = 0
            print('Win Rate: {:.2%}'.format(winner_dict[first_player] / episode))
        episode += 1
        winner = tic_tac_toe.winner
        if winner is not None:
            winner_dict[winner] += 1
