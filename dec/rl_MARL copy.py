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


def init_grid(vals, size=(16, 16)):
    grid = np.random.choice(vals, size, True)
    grid = torch.from_numpy(grid)
    return grid

def transform_grid(grid, j):
    N = grid.size()[0]
    n = int(np.ceil(N / 2) - 1)
    x, y = get_coords(grid, j)
    grid_repeat = grid.repeat(2, 2)

    if x <= n:
        x += N
    if y <= n:
        y += N
    
    return grid_repeat[x - n : x + n + 1, y - n : y + n + 1]

def softmax_policy(qvals,temp=0.9): 
    soft = torch.exp(qvals/temp) / torch.sum(torch.exp(qvals/temp)) 
    action = torch.multinomial(soft,1)
    return action

def get_coords(grid,j):
    x = int(np.floor(j / grid.shape[0]))
    y = int(j - x * grid.shape[0])
    return x,y

def reward(grid_new, grid_old):
    return utils.h_dec_phase(grid_new) - utils.h_dec_phase(grid_old)

def gen_params(N,size):
    ret = []
    for i in range(N):
        vec = torch.randn(size) / 10.
        vec.requires_grad = True
        ret.append(vec)
    return ret

def get_substate(b): #A
    s = torch.zeros(64) 
    s[int(b)] = 1
    return s

def mean_action(grid, j):
    M, N = np.shape(grid)
    x, y = get_coords(grid, j)
    action_mean = torch.zeros(64)
    for m in range(M):
        for n in range(N):
            if m != x and n != y:
               action_mean[int(grid[m, n])] += 1
    action_mean /= action_mean.sum()
    return action_mean

def get_reward_2d(action, action_mean):
    r = (action*(action_mean-action/2)).sum()/action.sum()
    return torch.tanh(5 * r) 


def qfunc(s,theta,layers=[(4,20),(20,20),(20,2)],afn=torch.tanh):
    l1n = layers[0]
    l1s = np.prod(l1n) #A
    theta_1 = theta[0:l1s].reshape(l1n) #B
    l2n = layers[1]
    l2s = np.prod(l2n)
    theta_2 = theta[l1s:l2s+l1s].reshape(l2n)
    l3n = layers[2]
    l3s = np.prod(l3n)
    theta_3 = theta[l2s+l1s:l3s+l2s+l1s].reshape(l3n)
    bias1 = torch.ones((1,theta_1.shape[1]))
    bias2 = torch.ones((1,theta_2.shape[1]))

    l1 = s @ theta_1 + bias1 #C
    l1 = torch.nn.functional.elu(l1)

    l2 = l1 @ theta_2 + bias2
    l2 = torch.nn.functional.elu(l2)

    l3 = afn(l2 @ theta_3) #D
    return l3.flatten()

N = 16
Nt = int(2 * np.ceil(N / 2) - 1)
size = (N, N)
vals = torch.linspace(0, 63, 64)
J = np.prod(size)
hid_layer1 = 64
hid_layer2 = 64
layers = [(64, hid_layer1), (hid_layer1, hid_layer2), (hid_layer1, 64)]
params = gen_params(1, 64*hid_layer1 + hid_layer1*hid_layer2 + hid_layer2*64)
grid = init_grid(vals, size=size)
grid_ = grid.clone()
grid__ = grid.clone()
plt.imshow(grid)
print(grid.sum())

epochs = 100
lr = 0.001
num_iter = 3
losses = [ [] for i in range(size[0])]
replay_size = 50
replay = deque(maxlen=replay_size)
batch_size = 10
gamma = 0.9
losses = [[] for i in range(J)]

for i in range(epochs):
    print(i)
    act_means = torch.zeros((J,64)) #G
    q_next = torch.zeros(J) #H
    for m in range(num_iter): #I
        for j in range(J): #J
            action_mean = mean_action(grid_,j).detach()
            act_means[j] = action_mean.clone()
            qvals = qfunc(action_mean.detach(),params[0],layers=layers)
            action = softmax_policy(qvals.detach(),temp=0.5)
            grid__[get_coords(grid_,j)] = action
            q_next[j] = torch.max(qvals).detach()
        grid_.data = grid__.data
    grid.data = grid_.data
    actions = torch.stack([get_substate(a.item()) for a in grid.flatten()])
    rewards = torch.stack([get_reward_2d(actions[j],act_means[j]) for j in range(J)])
    exp = (actions,rewards,act_means,q_next) #K
    replay.append(exp)
    shuffle(replay)
    if len(replay) > batch_size: #L
        ids = np.random.randint(low=0,high=len(replay),size=batch_size) #M
        exps = [replay[idx] for idx in ids]
        for j in range(J):
            jacts = torch.stack([ex[0][j] for ex in exps]).detach()
            jrewards = torch.stack([ex[1][j] for ex in exps]).detach()
            jmeans = torch.stack([ex[2][j] for ex in exps]).detach()
            vs = torch.stack([ex[3][j] for ex in exps]).detach()
            qvals = torch.stack([ qfunc(jmeans[h].detach(),params[0],layers=layers) \
                                 for h in range(batch_size)])
            target = qvals.clone().detach()
            target[:,torch.argmax(jacts,dim=1)] = jrewards + gamma * vs
            loss = torch.sum(torch.pow(qvals - target.detach(),2))
            losses[j].append(loss.item())
            loss.backward()
            with torch.no_grad():
                params[0] = params[0] - lr * params[0].grad
            params[0].requires_grad = True

fig,ax = plt.subplots(2,1)
fig.set_size_inches(10,10)
ax[0].plot(np.array(losses).mean(axis=0))
ax[1].imshow(grid)
fig.show()
end = utils.h_dec_phase(grid)
print(end)

print('done')
