import torch
import numpy as np
from game import game
from ppo import PPO
from memory import memory
from train import board2list, action2place, random_play
# Change o and x here. Note - First player should always be the bot (but it does not mean it starts first)
first_player = 'x'  # ALWAYS THE BOT
second_player = 'o'

# Run the evaluate
if __name__ == '__main__':
    BATCH_SIZE = 2048*2
    winner_dict = {first_player: 0, second_player: 0}

    memory_game = memory()
    tic_tac_toe = game()
    policy = PPO(9 * 2, 9, evaluation = True)

    samples = 0
    episode = 0
    while True:
        board = tic_tac_toe.reset()
        done = False
        input_array, board_list = board2list(board, main=first_player, opponent=second_player)
        random_first = np.random.random()
        if random_first > 0.5:
            possible_play = random_play(input_array, board_list, ppo_opponent=False)
            board, _, done, _ = tic_tac_toe.step(second_player, possible_play)
            input_array, board_list = board2list(board, main=first_player, opponent=second_player)
        while not done:
            action = policy.select_action(input_array, memory_game)
            action = action2place(action)
            board, reward, done, _ = tic_tac_toe.step(first_player, action)
            input_array, board_list = board2list(board, main=first_player, opponent=second_player)
            if not done:
                possible_play = random_play(input_array, board_list, ppo_opponent=False)
                board, reward, done, _ = tic_tac_toe.step(second_player, possible_play)
                input_array, board_list = board2list(board, main=first_player, opponent=second_player)
                reward = -reward
            memory_game.append_memory_rt(reward, done)
            samples += 1
        if samples > BATCH_SIZE:
            memory_game.clear_memory()
            print('Win Rate: {:.2%}'.format(winner_dict[first_player] / episode))
            winner_dict = {first_player: 0, second_player: 0}
            samples = 0
            episode = 0
            break
        episode += 1
        winner = tic_tac_toe.winner
        if winner is not None:
            winner_dict[winner] += 1
