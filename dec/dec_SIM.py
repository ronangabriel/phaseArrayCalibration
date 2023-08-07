import utils
import numpy as np
import time

'''
input:
  maxItr: max number of obj fn calls
  state: initial state
  runRuns: number of runs

returns:
  convInfo: (n x 2) matrix of minimum conflicts and number of objective function calls

'''
class SIM():
  def __init__(self, h_func, max_calls, initial_state, tunneling_prob=0, num_loops=1):
    self.h_func = h_func
    self.initial_state = initial_state
    self.state = initial_state.copy()
    self.max_calls = max_calls
    self.tunneling_prob = tunneling_prob
    self.num_loops = num_loops                      

    self.logger_loop = []
    self.logger = {
        'max_h': [],
        'calls': [],
        'delta_t': time.time_ns(),
    }

    self.max_h = 0
    self.num_calls = 0

  def run(self):
    for i in range(self.num_loops):
      self.max_h = 0
      self.num_calls = 0
      self.logger['max_h'].append(self.max_h)
      self.logger['calls'].append(self.num_calls)

      self.simulated_annealing()

      self.logger_loop.append(self.logger)

    return self.logger_loop
  
  def simulated_annealing(self):
    M, N = np.shape(self.state)
    self.max_h = self.h_func(self.state)
    curr_h = self.max_h

    for i in range(self.max_calls):
      T = (1 - (i + 1)/self.max_calls)

      # with probability tunnelingProb, generate next state very far away from current
      if np.random.uniform(low=0, high=1) < self.tunneling_prob:
        next = self.state.copy()
        numRand = int(np.ceil(T*N))
        inds = np.random.choice(np.linspace(0, N - 1, N).astype(int), size=(numRand,), replace=False)
        for ind in inds:
          val = np.random.randint(low=0, high=2)
          next[ind] = val
          
      else:
        ind_x = np.random.randint(low=0, high=M)
        ind_y = np.random.randint(low=0, high=N)
        val = np.random.randint(low=0, high=64) / 64
        next = self.state.copy()
        next[ind_x, ind_y] = val

      next_h = self.h_func(next)
      self.num_calls += 1

      deltaE = next_h - curr_h
      if T != 0:
        P = 1 / (1 + np.exp(-deltaE / T))
      else:
        P = 0

      if deltaE > 0:
        self.state = next.copy()
        curr_h = next_h
      elif np.random.uniform(low=0, high=1) < P:
        self.state = next.copy()
        curr_h = next_h
              
      if curr_h > self.max_h:
        self.max_h = curr_h 
        print('Run: {}, h: {}'.format(self.num_calls, self.max_h))