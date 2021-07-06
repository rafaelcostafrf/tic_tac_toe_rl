import torch
from ppo import PPO
from train import board2list, action2place, random_play, first_player, second_player
from memory import memory

memory_game = memory()
policy = PPO(9*2, 9, evaluation=True)

def get_move(board, marker):
    input_array, board_list = board2list(board, main=first_player, opponent=second_player)
    action = policy.select_action(input_array, memory_game)
    action = action2place(action)
    board[action[0]][action[1]] = marker
    return board

def print_game(board):
    print('-'*10)
    for row in board:
        print('|', end='')
        for i, entry in enumerate(row):
            if i == 0:
                prefix = ''
            else:
                prefix = '\t'
            if entry == '':
                print(prefix+' '.format(entry), end='')
            else:
                print(prefix+'{}'.format(entry), end='')
        print('|')
    print('-' * 10)

if __name__=='__main__':
    board = [[''] * 3 for _ in range(3)]
    for i in range(4):
        board = get_move(board, 'x')
        board_array, board_list = board2list(board, main=first_player, opponent=second_player)
        possible_play = random_play(board_array, board_list, ppo_opponent=False)
        board[possible_play[0]][possible_play[1]] = 'o'
        print_game(board)