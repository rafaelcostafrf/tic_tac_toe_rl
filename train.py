import torch
import numpy as np
from game import game
from ppo import PPO
from memory import memory

# Change o and x here. Note - First player should always be the bot (but it does not mean it starts first)
first_player = 'x'  # ALWAYS THE BOT
second_player = 'o'

# Updates the adversarial network over x steps
# The adversarial network is used when ppo_opponent is set to True
# The adversarial network is an older copy of the current ppo network.
adversarial_update = 5

def ppo_move(board):
    input_array, board_list = board2list(board)
    action = old_policy.select_action(input_array, memory_game)
    action = action2place(action)
    return action

# Convert the 3x3 board to a linear list 1x9
def board2list(board, main, opponent):
    board_list = []
    x_array = np.zeros(9)
    empty_array = np.ones(9)
    o_array = np.zeros(9)
    for row in board:
        board_list += row
    for i in range(9):
        if board_list[i] == main:
            o_array[i] = 1
            empty_array[i] = 0
        if board_list[i] == opponent:
            x_array[i] = 1
            empty_array[i] = 0
    input_array = np.concatenate((o_array, x_array))
    return input_array, board_list


# Converts the action in 1x9 to 3x3
def action2place(action):
    place = [int(action / 3), action[0] % 3]
    return place


# Random operator
def random_play(input_array, board_list, ppo_opponent):
    if ppo_opponent:
        action = old_policy.select_action(input_array, memory_opponent)
        random_play = action2place(action)
    else:
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
    tic_tac_toe = game()
    # Current policy
    policy = PPO(9 * 2, 9, evaluation = False)
    memory_game = memory()

    # Adversarial network
    old_policy = PPO(9 * 2, 9, evaluation = True)
    memory_opponent = memory()

    samples = 0
    episode = 0
    train_episode = 0

    # Runs the training
    while True:
        # Adversarial or random player
        # ppo_opponent = False if np.random.random() > 0.5 else True
        ppo_opponent = False

        # Set up the board
        board = tic_tac_toe.reset()
        done = False

        # Checks if random or adversarial goes first
        random_first = np.random.random()
        if random_first > 0.5:
            input_array, board_list = board2list(board, main=second_player, opponent=first_player)
            possible_play = random_play(input_array, board_list, ppo_opponent)
            board, _, _, _ = tic_tac_toe.step(second_player, possible_play)

        # While the game is not done
        while not done:
            input_array, board_list = board2list(board, main=first_player, opponent=second_player)
            action = policy.select_action(input_array, memory_game)
            action = action2place(action)
            board, reward, done, reason = tic_tac_toe.step(first_player, action)

            # If the game is not done, make a random or adversarial play
            if not done:
                input_array, board_list = board2list(board, main=second_player, opponent=first_player)
                possible_play = random_play(input_array, board_list, ppo_opponent)
                board, reward, done, reason = tic_tac_toe.step(second_player, possible_play)
                reward = -2*reward

            # Checks if the game ended up in a tie, then checks the winner and give appropriate rewards
            if reason == 'tie':
                if tic_tac_toe.winner == first_player:
                    reward = 0.5
                else:
                    reward = -1

            # Appends the reward and the terminals to transition
            memory_game.append_memory_rt(reward, done)

            # Sums the sample of current batch
            samples += 1

        # If there are enough samples, train
        if samples > BATCH_SIZE:
            policy.update(memory_game, train_episode)
            memory_game.clear_memory()
            memory_opponent.clear_memory()
            samples = 0
            print('Win Rate: {:.2%}'.format(winner_dict[first_player] / episode))
            train_episode += 1
            episode = 0
            winner_dict[first_player] = 0
            if train_episode % adversarial_update == 0:
                print('Updated adversarial network')
                old_policy = PPO(9 * 2, 9, evaluation = True)
        episode += 1
        winner = tic_tac_toe.winner
        if winner is not None:
            winner_dict[winner] += 1
