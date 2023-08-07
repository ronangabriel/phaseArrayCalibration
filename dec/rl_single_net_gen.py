import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from collections import deque 
from random import shuffle
import utils
import copy
import multiprocessing as mp
import os
import warnings

def model(x,unpacked_params):
    l1,b1,l2,b2,l3,b3 = unpacked_params #A
    y = F.linear(x,l1,b1) #B
    y = torch.relu(y) #C
    y = F.linear(y,l2,b2)
    y = torch.relu(y)
    y = F.linear(y,l3,b3)
    y = torch.sigmoid(y) #D
    return y 

def unpack_params(params, layers=[(64,256),(64,64),(256,64)]): #A
    unpacked_params = [] #B
    e = 0
    for i,l in enumerate(layers): #C
        s,e = e,e+np.prod(l)
        weights = params[s:e].view(l) #D
        s,e = e,e+l[0]
        bias = params[s:e]
        unpacked_params.extend([weights,bias]) #E
    return unpacked_params

def spawn_population(N=50,size=407): #A
    pop = []
    for i in range(N):
        vec = torch.randn(size) / 2.0 #B
        fit = 0
        p = {'params':vec, 'fitness':fit, 'id':i} #C
        pop.append(p)
    return pop

def recombine(x1,x2): #A
    x1 = x1['params'] #B
    x2 = x2['params']
    l = x1.shape[0]
    split_pt = np.random.randint(l) #C
    child1 = torch.zeros(l)
    child2 = torch.zeros(l)
    child1[0:split_pt] = x1[0:split_pt] #D
    child1[split_pt:] = x2[split_pt:]
    child2[0:split_pt] = x2[0:split_pt]
    child2[split_pt:] = x1[split_pt:]
    c1 = {'params':child1, 'fitness': 0.0} #E
    c2 = {'params':child2, 'fitness': 0.0}
    return c1, c2


def mutate(x, rate=0.01): #A
    x_ = x['params']
    num_to_change = int(rate * x_.shape[0]) #B
    idx = np.random.randint(low=0,high=x_.shape[0],size=(num_to_change,))
    x_[idx] = torch.randn(num_to_change) / 10.0 #C
    x['params'] = x_
    return x

def init_grid(size, sz):
    vals = np.linspace(0, sz - 1, sz)
    grid = np.random.choice(vals, size, replace=True)
    grid = torch.from_numpy(grid)
    return grid

# TODO: Change to square env
def test_model(agent):
    N = 16
    size = (N, N)
    sz = 64
    score = 0
    attempts = 100
    for i in range(attempts):
        state = init_grid(size, sz) / 64
        #score_temp = torch.reshape(utils.h_dec(state), (1, 1, 1)).float()
        state = torch.reshape(state, (1, 1, N * N)).float()
        #state = torch.cat((state, score_temp), dim=2)
        params = unpack_params(agent['params'])
        state = torch.reshape(model(state, params), (N, N))
        score += utils.h_dec(state)
    return score / attempts


def evaluate_population(pop):
    tot_fit = 0 #A
    lp = len(pop)

    
    with mp.Pool(mp.cpu_count()) as p:
        scores = p.map(test_model, pop)
        for agent, score in zip(pop, scores):
           agent['fitness'] = score
        avg_fit = sum(scores) / lp
    
    '''
    for agent in pop:
        score = test_model(agent)
        agent['fitness'] = score
        tot_fit += score
    avg_fit = tot_fit / lp
    '''
    
    return pop, avg_fit


# Use chromosomal algo: treat layers separately to prevent corruption?
def next_generation(pop,mut_rate=0.001,tournament_size=0.2):
    new_pop = []
    lp = len(pop)
    while len(new_pop) < len(pop): #A
        rids = np.random.randint(low=0,high=lp,size=(int(tournament_size*lp))) #B
        batch = np.array([[i,x['fitness']] for (i,x) in enumerate(pop) if i in rids]) #C
        scores = batch[batch[:, 1].argsort()] #D
        i0, i1 = int(scores[-1][0]),int(scores[-2][0]) #E
        parent0,parent1 = pop[i0],pop[i1]
        offspring_ = recombine(parent0,parent1) #F
        child1 = mutate(offspring_[0], rate=mut_rate) #G
        child2 = mutate(offspring_[1], rate=mut_rate)
        offspring = [child1, child2]
        new_pop.extend(offspring)
    return new_pop


def running_mean(x,n=5):
    conv = np.ones(n)
    y = np.zeros(x.shape[0]-n)
    for i in range(x.shape[0]-n):
        y[i] = (conv @ x[i:i+n]) / n
    return y


def main():
    print(os.cpu_count())
    num_generations = 400 #A
    population_size = 4800 #B
    mutation_rate = 0.01
    size = 256 * 64 + 64 + 64 * 64 + 64 + 64 * 256 + 256
    pop_fit = []
    #pop_fit = torch.load('dec/pop_fit_MP_popsize100_layer64.txt')
    pop = spawn_population(N=population_size,size=size) #C
    #pop = torch.load('dec/population_MP_popsize100_layer64.txt')
    for i in range(num_generations):
        pop, avg_fit = evaluate_population(pop) #D
        pop_fit.append(avg_fit)
        pop = next_generation(pop, mut_rate=mutation_rate,tournament_size=0.2) #E
        print(i)

    plt.figure(figsize=(12,7))
    plt.xlabel("Generations",fontsize=22)
    plt.ylabel("Score",fontsize=22)
    plt.title("num_generations={}, population_size={}".format(num_generations, population_size) )
    plt.plot(running_mean(np.array(pop_fit),3))
    plt.show()

    torch.save(pop_fit, 'dec/pop_fit_MP_popsize4800_layer64.txt')
    torch.save(pop, 'dec/population_MP_popsize100_layer64.txt')

    wait = input("Press Enter to continue.")

if __name__ == '__main__':
    mp.freeze_support()
    main()
