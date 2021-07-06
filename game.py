import numpy as np

class game():
    def __init__(self):
        self.board = [['']*3 for _ in range(3)]
        self.occupied_error = 'Place is Already occupied, choose another place'
        self.wrong_place_error = 'Place is not on the board, choose numbers from 0 to 2'
        self.winner = None

    def reset(self):
        # Resets the board for the next game
        self.board = [[''] * 3 for _ in range(3)]
        self.winner = None
        return self.board

    def step(self, mark, place):
        # Goes a step forward on the board, checks if terminal and its rewards
        self.current_player = mark.lower()
        reward = 0
        done = False
        wrong_place = True if (place[0] < 0 or place[0] > 2 or place[1] < 0 or place[1] > 2) else False
        if wrong_place:
            occupied = True
        else:
            occupied = True if self.board[place[0]][place[1]] != '' else False

        if not occupied and not wrong_place:
            self.board[place[0]][place[1]] = mark.lower()
            reward, done, self.winner, reason = self.get_reward()
        else:
            done = True
            reward = -5
            reason = 'wrong_move'
        return self.board, reward, done, reason

    def get_reward(self):
        # Checks terminal states and current reward
        reward = 0
        done = False
        winner = None
        reason = ''
        if self.check_horizontal():
            reward = 3
            winner = self.current_player
            reason = 'horizontal'
            done = True
        elif self.check_vertical():
            reward = 1
            winner = self.current_player
            reason = 'vertical'
            done = True
        elif self.check_diagonal():
            reward = 1
            winner = self.current_player
            reason = 'diagonal'
            done = True
        elif self.check_top_marks():
            reward = 0.5
            winner = self.top_marks_winner
            reason = 'tie'
            done = True
        return reward, done, winner, reason

    def check_horizontal(self):
        # Horizontal win terminal state
        for row in self.board:
            counter = {'x': 0, 'o': 0}
            for column in row:
                if column != '':
                    counter[column] += 1
                    if counter[column] > 2:
                        return True
        return False


    def check_diagonal(self):
        # Diagonal win terminal state
        counter = {'x': 0, 'o': 0}
        anti_counter = {'x': 0, 'o': 0}
        for i in range(3):
            if self.board[i][i] != '':
                counter[self.board[i][i]] += 1
                if counter[self.board[i][i]] > 2:
                    return True
            if self.board[2-i][i] != '':
                anti_counter[self.board[2-i][i]] += 1
                if anti_counter[self.board[2-i][i]] > 2:
                    return True
        return False

    def check_vertical(self):
        # Vertical win terminal state
        for i in range(3):
            counter = {'x': 0, 'o': 0}
            for j in range(3):
                if self.board[j][i] != '':
                    counter[self.board[j][i]] += 1
                    if counter[self.board[j][i]] > 2:
                        return True
        return False

    def check_top_marks(self):
        # Top row terminal state (only when the board is full)
        # If the board is full, checks the top row
        counter = 0
        counter_player = {'x': 0, 'o': 0}
        for row in self.board:
            for entry in row:
                if entry == '':
                    counter += 1
        if counter == 0:
            for entry in self.board[0]:
                counter_player[entry] += 1
                if counter_player[entry] > 1:
                    self.top_marks_winner = entry

                    return True
        return False

if __name__ == '__main__':
    tic_tac_toe = game()
    print(tic_tac_toe.step('o', [0, 2]))
    print(tic_tac_toe.step('x', [0, 0]))
    print(tic_tac_toe.step('o', [2, 2]))
    print(tic_tac_toe.step('x', [1, 2]))
    print(tic_tac_toe.step('o', [1, 2]))