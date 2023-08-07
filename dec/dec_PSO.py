import numpy as np
import random as rand
from matplotlib import pyplot as plt
import math
from collections import Counter
from numpy import nonzero
import time

class PSO():
    def __init__(self, h_func, initial_state, num_particles, num_runs, num_loops):
        self.h_func = h_func
        self.initial_state = initial_state
        self.M, self.N = np.shape(initial_state)
        self.num_particles = num_particles
        self.num_runs = num_runs
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

            self.particle_swarm_optimization()

            self.logger_loop.append(self.logger)

        return self.logger_loop
    
    # https://en.wikipedia.org/wiki/Particle_swarm_optimization
    def particle_swarm_optimization(self):
        w = 0.1 # inertia
        phi_p = 2.3 # cognitive coefficient
        phi_g = 2.3 # social coefficient

        g_best = np.random.rand(self.M, self.N)
        g_best_score = -np.inf

        particles = [{'state': None,
                'p_best': None,
                'p_best_score': None,
                'velocity': None} for _ in range(self.num_particles)]
        
        for particle in particles:
            particle['state'] = np.random.rand(self.M, self.N)
            particle['p_best'] = particle['state'].copy()
            particle['p_best_score'] = self.h_func(particle['p_best'])
            self.num_calls += 1

            if particle['p_best_score'] > g_best_score:
                self.max_h = particle['p_best_score']
                self.logger['max_h'].append(self.max_h)
                self.logger['calls'].append(self.num_calls) 

                g_best = particle['p_best'].copy()
                g_best_score = particle['p_best_score']
                
            particle['velocity'] = 2 * np.random.rand(self.M, self.N) - 1

        while self.num_calls < self.num_runs:
            for particle in particles:
                r_p, r_g = np.random.rand(self.M, self.N), np.random.rand(self.M, self.N)
                particle['velocity'] = w * particle['velocity'] + \
                    phi_p * np.multiply(r_p, (particle['p_best'] - particle['state'])) + \
                    phi_g * np.multiply(r_g, (g_best - particle['state']))
                particle['state'] = particle['state'] + particle['velocity']
                # enforce upper and lower bounds
                particle['state'][particle['state'] > 1] = 1
                particle['state'][particle['state'] < 0] = 0

                self.num_calls += 1
                if self.h_func(particle['state']) > particle['p_best_score']:
                    particle['p_best'] = particle['state'].copy()
                    particle['p_best_score'] = self.h_func(particle['state'])

                    if particle['p_best_score'] > g_best_score:
                        self.max_h = particle['p_best_score']
                        self.logger['max_h'].append(self.max_h)
                        self.logger['calls'].append(self.num_calls)

                        g_best = particle['p_best'].copy()
                        g_best_score = particle['p_best_score']

                        print('Run: {}, h: {}'.format(self.num_calls, self.max_h))