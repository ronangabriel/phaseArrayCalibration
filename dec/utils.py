import copy
import random
import math
import random
import numpy as np
import torch

def conflict(row1, col1, row2, col2):
    """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
    return (row1 == row2 or  # same row
            col1 == col2 or  # same column
            row1 - col1 == row2 - col2 or  # same \ diagonal
            row1 + col1 == row2 + col2)  # same / diagonal

def h_queens(state):
    """Return number of conflicting queens for a given node.
    state: a list of length 'n' for n-queens problem,
    where the c-th element of the list holds the value for the row number of the queen in column c.
    """
    num_conflicts = 0
    N = len(state)
    for c1 in range(N):
        for c2 in range(c1+1, N):
            num_conflicts += conflict(state[c1], c1, state[c2], c2)
    return num_conflicts

def h2(state):
    return np.sum(abs(state - np.ones(np.shape(state))))

def h_dec(state):
    state = torch.asarray(state).float()
    N = np.shape(state)[0]

    c1 = torch.reshape(torch.linspace(0, N - 1, N), (N, 1))
    c2 = torch.reshape(torch.linspace(0, N - 1, N), (1, N))

    c1 = N - torch.abs(c1 - (N - 1) / 2)
    c2 = N - torch.abs(c2 - (N - 1) / 2)

    c = c1 + c2

    state = torch.cos((state * (2 * N - 1) - c) * torch.pi / c)

    return torch.sum(state)

def h_dec_phase(state):
    N = np.shape(state)[0]
    avg = torch.mean(state)
    result = torch.cos((avg - 1) * torch.pi)
    return result

def h_dec_std(state):
    '''
    input:
        state - input grid, values in range [0, 1]
    
    returns:
        h function, has maximum for grid of ones (max(h(x)) = 100), but has many local maxima
    '''
    state = torch.asarray(state).float()

    mean = torch.mean(state)
    std = torch.std(state) # Reduces over all dimensions
    mu = 0.01
    scale = 1 / (std + mu)
    return mean * scale # for maximum for grid of ones, but has a lot of local maxima