import numpy as np
import utils
import time

class GA():
    def __init__(self, h_func, initial_population, population_size, mutation_rate, num_runs=10000, num_loops=1):
        self.h_func = h_func
        self.population = initial_population            # initial population
        self.population_size = population_size          # number of agents in each generation
        self.mutation_rate = mutation_rate              # rate of mutation
        self.num_runs = num_runs                        # number of runs per loop
        self.num_loops = num_loops                      # number of loops to average results

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

            self.genetic_algorithm()

            self.logger_loop.append(self.logger)

        return self.logger_loop
    
    def genetic_algorithm(self):

        M, N = np.shape(self.population[0])

        self.max_h = self.h_func(self.population[0])

        for k in range(self.num_runs):
            weights = np.zeros((self.population_size,))

            # find selection probabilities
            for i in range(self.population_size):
                weights[i] = self.h_func(self.population[i])
            
                self.num_calls += 1

            self.logger['max_h'].append(self.max_h)
            self.logger['calls'].append(self.num_calls)

            max_val = np.max(weights)
            max_ind = np.argmax(weights)
            super_child = self.population[max_ind]

            if max_val > self.max_h:
                self.max_h = max_val

            print('Run: {}, h: {}'.format(self.num_calls, self.max_h))

            if max_val >= M * N - 1:
                print(super_child)
                return
            weights = np.power(weights, 3)
            weights = weights / np.linalg.norm(weights, ord=1)

            children = [None] * self.population_size

            for i in range(self.population_size):
                # create a child
                inds = np.random.choice(np.linspace(0, self.population_size - 1, self.population_size).astype(int), p=weights, size=(2,), replace=False)
                parent1 = self.population[inds[0]]
                parent2 = self.population[inds[1]]
                child = self.reproduce(parent1, parent2)

                mutationRate = (M * N - max_val) / (M * N)
                # mutate this child
                for j in range(M):
                    for k in range(N):
                        if (np.random.uniform(low=0, high=1) < mutationRate):
                            if (np.random.uniform(low=0, high=1)) < 0.5:
                                if child[j, k] >= 0:
                                    child[j, k] -= 1 / 64
                            else:
                                if child[j, k] <= 1:
                                    child[j, k] += 1 / 64

                children[i] = child
            #children[-1] = super_child

            self.population = children.copy()
        
    
    def reproduce(self, parent1, parent2):
        M, N = np.shape(parent1)
        flat1 = parent1.flatten()
        flat2 = parent2.flatten()
        n = len(flat1)
        inds = np.random.choice(np.linspace(0, n - 1, n).astype(int), size=(int(n / 2),), replace=False)
        for ind in inds:
            flat2[ind] = flat1[ind]
        return np.reshape(flat2, (M, N))