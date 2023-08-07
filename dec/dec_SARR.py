import utils
import numpy as np
import time

'''
The main steepest ascent algorithm. This function is called after every restart.

input: 
  max_calls                 max number of calls to environment
  initial_state             initial environment state
  num_restarts              number of times to restart after finding a maximum
  num_loops                 number of runs to average together 

returns:
  nothing, query data with logger variable
'''
class SARR():
    def __init__(self, h_func, max_calls, initial_state, num_restarts=100, num_loops=1):
        self.h_func = h_func
        self.initial_state = initial_state
        self.state = initial_state.copy()
        self.max_calls = max_calls
        self.num_loops = num_loops
        self.num_restarts = num_restarts
        self.logger_loop = []
        self.logger = {
            'max_h': [],
            'calls': [],
            'delta_t': time.time_ns(),
        }

        self.max_h = 0
        self.num_calls = 0
        self.curr_h = 0

    def run(self):
        for i in range(self.num_loops):
            self.state = self.initial_state.copy()
            self.max_h = 0
            self.num_calls = 0
            self.logger['max_h'].append(self.max_h)
            self.logger['calls'].append(self.num_calls)

            self.steepest_ascent_random_restart()

            self.logger_loop.append(self.logger)
        
        return self.logger_loop

    def steepest_ascent_random_restart(self):
        self.max_h = self.h_func(self.state)
        self.curr_h = self.max_h
    
        for i in range(self.num_restarts):
            self.steepest_ascent()

            # check if the problem is solved
            if self.curr_h == 0:
                break
            else:
                # randomize the state
                self.state = np.random.randint(low=0, high=64, size=np.shape(self.state)) / 64

            if self.logger['calls'][-1] >= self.max_calls:
                break

    # steepest ascent applied to 2D grid
    def steepest_ascent(self):
        M, N = np.shape(self.state)
        current = self.state.copy()

        while True:
            neighbor = current.copy()
            neighborH = self.h_func(current)

            #find best neighbor
            for i in range(M):
                for j in range(N):
                    temp = current.copy()

                    # move down
                    while (temp[i, j] >= 0):
                        temp[i, j] -= 1 / 64
                        if self.h_func(temp) > neighborH:
                            neighbor = temp.copy()
                            neighborH = self.h_func(temp)
                            self.num_calls += 1

                    # move up
                    while (temp[i, j] <= 1):
                        temp[i, j] += 1 / 64
                        if self.h_func(temp) > neighborH:
                            neighbor = temp.copy()
                            neighborH = self.h_func(temp)
                            self.num_calls += 1

            if neighborH <= self.h_func(current):
                return
            
            current = neighbor.copy()

            if self.h_func(current) > self.max_h:
                self.max_h = self.h_func(current)
                self.logger['max_h'].append(self.max_h)
                self.logger['calls'].append(self.num_calls)
                print('Calls: {}, h: {}'.format(self.num_calls, self.max_h))