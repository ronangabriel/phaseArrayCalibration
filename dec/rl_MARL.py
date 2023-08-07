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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2, 2), stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(2, 2), stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(2, 2), stride=2, padding=1)
        self.linear1 = nn.LazyLinear(128)
        self.linear2 = nn.LazyLinear(64) 

    def forward(self, x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = y.flatten(start_dim=1)
        y = F.relu(self.linear1(y))
        y = 2*F.tanh(self.linear2(y)) # restrict output to (-2, 2)

        return y

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

N = 16
Nt = int(2 * np.ceil(N / 2) - 1)
size = (N, N)
vals = torch.linspace(0, 1, 64)
J = np.prod(size)
J = 1
grid = init_grid(vals, size)
grid[0, 0] = 0
grid_ = grid.clone()
grid__ = grid.clone()
reward_grid = grid.clone()
start_grid = copy.deepcopy(grid)
plt.imshow(grid)
print(grid.sum())
start = utils.h_dec_phase(grid)
print(start)

epochs = 5000
lr = 0.001
num_iter = 3
losses = [ [] for i in range(size[0])]
replay_size = 50
replay = deque(maxlen=replay_size)
batch_size = 10
gamma = 0.9
sync_freq = 50
losses = [[] for i in range(J)]

Qnets = [Model() for _ in range(J)]
Qnets2 = copy.deepcopy(Qnets)
for j in range(J):
    Qnets2[j].load_state_dict(Qnets[j].state_dict())
optimizers = [torch.optim.Adam(Qnets[j].parameters(), lr=lr) for j in range(J)]

for i in range(epochs):
    print(i)
    act_means = torch.zeros((J,2))
    grids = torch.zeros((1, 1, Nt, Nt, J))
    q_next = torch.zeros(J)
    rewards = torch.zeros(J)
    actions = torch.zeros(J)
    for m in range(num_iter):
        for j in range(J):
            grids[:, :, :, :, j] = torch.reshape(transform_grid(grid_, j), (1, 1, Nt, Nt)).detach()
            qvals = Qnets[j](grids[:, :, :, :, j])
            action = softmax_policy(qvals.detach(),temp=0.5)
            actions[j] = action
            val = vals[action]
            temp = reward_grid[get_coords(grid_,j)]
            reward_grid[get_coords(grid_,j)] = val
            grid__[get_coords(grid_,j)] = val
            rewards[j] = reward(reward_grid, grid_)
            ind = torch.argmax(Qnets[j](torch.reshape(transform_grid(reward_grid, j), (1, 1, Nt, Nt))))
            q_next[j] = Qnets2[j](torch.reshape(transform_grid(reward_grid, j), (1, 1, Nt, Nt)))[0, ind]
            reward_grid[get_coords(grid_,j)] = temp
        grid_[:, :] = grid__
    grid[:, :] = grid_
    exp = (actions,rewards,grids,q_next)
    replay.append(exp)
    shuffle(replay)
    if len(replay) > batch_size:
        ids = np.random.randint(low=0,high=len(replay),size=batch_size)
        exps = [replay[idx] for idx in ids]
        for j in range(J):
            jacts = torch.stack([ex[0][j] for ex in exps]).detach()
            jrewards = torch.stack([ex[1][j] for ex in exps]).detach()
            jgrids = torch.stack([ex[2][:, :, :, :, j] for ex in exps]).detach()
            vs = torch.stack([ex[3][j] for ex in exps]).detach()
            qvals = torch.stack([Qnets[j](jgrids[h, :, :, :, :]) for h in range(batch_size)])
            target = qvals.clone().detach()
            target[:, 0, jacts.int()] = jrewards + gamma * vs
            loss = torch.sum(torch.pow(qvals - target.detach(),2))
            losses[j].append(loss.item())
            loss.backward()
            optimizers[j].step()

            if i % sync_freq == 0:
                Qnets2[j].load_state_dict(Qnets[j].state_dict())

fig,ax = plt.subplots(2,1)
fig.set_size_inches(10,10)
ax[0].plot(np.array(losses).mean(axis=0))
ax[1].imshow(grid)
test = grid - start_grid
fig.show()
end = utils.h_dec_phase(grid)
print(end)
print(end - start)
test2 = Qnets2[0](torch.reshape(transform_grid(grid, 0), (1, 1, Nt, Nt)))
print(test2)

print('done')
