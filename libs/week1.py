import numpy as np


def evaluate(row_strategy: np.array, column_strategy: np.array, matrix: np.array, ) -> float:
    """Value of the row play when the row and column player use their respective strategies"""
    temp = row_strategy @ matrix

    expected_value = temp @ column_strategy

    return expected_value


def evaluate_pair(row_strategy, col_strategy, matrix1, matrix2) -> tuple:
    """Given mixed strategies of two players and two utility matrices, compute value for each player."""
    # Compute the expected utility for the row player using matrix1
    row_util = evaluate(row_strategy, col_strategy, matrix1)
    # Compute the expected utility for the column player using matrix2
    col_util = evaluate(row_strategy, col_strategy, matrix2)
    
    return row_util, col_util


def evaluate_zero_sum(row_strategy, col_strategy, matrix) -> tuple:
    """Do the same, but for zero-sum games. This time, your function should only take a single matrix (as the other one is implied)"""
    # Compute the expected utility for the row player using matrix
    row_util = evaluate(row_strategy, col_strategy, matrix)
    # Compute the expected utility for the column player using -matrix
    col_util = evaluate(row_strategy, col_strategy, -matrix)
    
    return row_util, col_util


def best_response_value_row(matrix: np.array, row_strategy: np.array) -> float:
    """Value of the row player when facing a best-responding column player"""
    # expected payoff for each column action given the row strategy
    payoffs = row_strategy @ matrix 
    # column player is trying to minimize the row player's payoff
    best_response_value_row = np.min(payoffs)
    return best_response_value_row


def best_response_value_column(matrix: np.array, column_strategy: np.array) -> float:
    """Value of the column player when facing a best-responding row player"""
    payoffs = matrix @ column_strategy
    best_response_value_column = -np.max(payoffs)
    return best_response_value_column


def evaluate_row_against_best_response(row_strategy: np.array, matrix: np.array):
    """Utility for row player when playing against a best-responding column player"""
    row_util = best_response_value_row(matrix, row_strategy)
    # Since it's zero-sum, column utility is -1 * row utility
    col_util = -row_util
    return row_util, col_util


def evaluate_col_against_best_response(col_strategy: np.array, matrix: np.array) -> tuple:
    """Utility for column player when playing against a best-responding row player"""
    col_util = best_response_value_column(matrix, col_strategy)
    # Since it's zero-sum, row utility is -1 * column utility
    row_util = -col_util
    return row_util, col_util


def best_response_strategy_row(matrix: np.array, col_strategy: np.array) -> np.array:
    """Given a column strategy, compute the best response for the row player"""
    payoffs = matrix @ col_strategy
    best_action = np.argmax(payoffs)
    best_response = np.zeros(matrix.shape[0])
    best_response[best_action] = 1.0
    return best_response


def best_response_strategy_column(matrix: np.array, row_strategy: np.array) -> np.array:
    """Given a row strategy, compute the best response for the column player"""
    payoffs = row_strategy @ matrix
    best_action = np.argmin(payoffs)
    best_response = np.zeros(matrix.shape[1])
    best_response[best_action] = 1.0
    return best_response


def find_dominated_actions(matrix: np.array, player: str) -> list:
    """Find dominated actions for a player (row or column)"""
    dominated = []
    if player == 'row':
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                # Check if i-th row is dominated by j-th row, i.e. check that action i yields a payoff less than or equal to action j in every column and strictly loew payoff in at least one column
                if i != j and np.all(matrix[i, :] <= matrix[j, :]) and np.any(matrix[i, :] < matrix[j, :]):
                    dominated.append(i)
                    break
    elif player == 'column':
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[1]):
                if i != j and np.all(matrix[:, i] <= matrix[:, j]) and np.any(matrix[:, i] < matrix[:, j]):
                    dominated.append(i)
                    break
    return dominated


def iterated_removal_of_dominated_actions(matrix1: np.array, matrix2: np.array) -> tuple:
    """Iteratively remove dominated actions for both players"""
    actions1 = list(range(matrix1.shape[0]))
    actions2 = list(range(matrix1.shape[1]))

    while True:
        dominated_row_actions = find_dominated_actions(matrix1, 'row')
        if dominated_row_actions:
            matrix1 = np.delete(matrix1, dominated_row_actions, axis=0)
            matrix2 = np.delete(matrix2, dominated_row_actions, axis=0)
            # Update of actions list
            actions1 = [a for i, a in enumerate(actions1) if i not in dominated_row_actions]
        
        dominated_col_actions = find_dominated_actions(matrix2, 'column')
        if dominated_col_actions:
            matrix1 = np.delete(matrix1, dominated_col_actions, axis=1)
            matrix2 = np.delete(matrix2, dominated_col_actions, axis=1)
            actions2 = [a for i, a in enumerate(actions2) if i not in dominated_col_actions]
        
        if not dominated_row_actions and not dominated_col_actions:
            break

    return matrix1, matrix2, actions1, actions2


