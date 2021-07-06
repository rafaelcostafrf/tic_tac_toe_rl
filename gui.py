import tkinter
import numpy as np
import time
from game import game
from train import action2place, board2list, random_play, first_player, second_player
from memory import memory
from ppo import PPO

root = tkinter.Tk()
root.title('Tic Tac Toe')

memory_game = memory()
policy = PPO(9*2, 9, evaluation=True)
tic_tac_toe = game()
board = tic_tac_toe.reset()

# If demo is true, play with random operator
demo = False


def start_check(board):
    """
    Checks if the ppo agent or the player starts first, returns the updated board
    :param board: list of lists (3x3)
    :return: board
    """
    start = np.random.random()
    if start > 0.5:
        input_array, board_list = board2list(board, main=first_player, opponent=second_player)
        action = policy.select_action(input_array, memory_game)
        action = action2place(action)
        board, _, done, _ = tic_tac_toe.step(first_player, action)
    return board

def button_click(i):
    """
    Checks if a button on the board has been clicked, then runs a step on the algorithm
    :param i:
    :return:
    """
    input_array, board_list = board2list(tic_tac_toe.board, main=first_player, opponent=second_player)
    if demo:
        place = random_play(input_array, board_list, ppo_opponent=False)
    else:
        place = action2place(np.array([i]))
    board, _, done, _ = tic_tac_toe.step(second_player, place)
    input_array, board_list = board2list(tic_tac_toe.board, main=first_player, opponent=second_player)
    if not done:
        action = policy.select_action(input_array, memory_game)
        action = action2place(action)
        board, _, done, _ = tic_tac_toe.step(first_player, action)
        input_array, board_list = board2list(tic_tac_toe.board, main=first_player, opponent=second_player)
    if done:
        for i, entry in enumerate(board_list):
            buttons_list[i].config(text=entry, bg='green')
        if tic_tac_toe.winner is not None:
            tkinter.messagebox.showinfo('Winner', 'The winner is {}'.format(tic_tac_toe.winner))
        board = tic_tac_toe.reset()
        board = start_check(board)
    input_array, board_list = board2list(tic_tac_toe.board, main=first_player, opponent=second_player)
    for i, entry in enumerate(board_list):
        buttons_list[i].config(text=entry, bg='gray80')


# Buttons
buttons_list = []
for i in range(9):
    buttons_list.append(tkinter.Button(root, text='', font=('Helvetica', 20), height=3, width=6,
                            command=lambda i=i: button_click(i), bg='gray80'))
    buttons_list[i].grid(row=int(i/3), column=i%3)

board = start_check(board)
input_array, board_list = board2list(board, main=first_player, opponent=second_player)
for i, entry in enumerate(board_list):
    buttons_list[i].config(text=entry, bg='gray80')
root.mainloop()