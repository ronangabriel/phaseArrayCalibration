import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import torch.multiprocessing as mp
import utils
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2, 2), stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(2, 2), stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(2, 2), stride=2, padding=1)
        self.linear1 = nn.LazyLinear(256)
        self.linear2 = nn.LazyLinear(512)
        self.linear3 = nn.LazyLinear(1)

    def forward(self, x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = y.flatten(start_dim=1)
        y = F.relu(self.linear1(y))
        actor = self.linear2(y)
        critic = self.linear3(y)

        return actor, critic
    
def run_episode(curr_state, worker_model, history_performance, avg_reward, num_itrs, N_steps=1):
    N = 16
    values, logprobs, rewards = [],[],[]
    done = False
    j=0
    G=torch.Tensor([0])
    while (j < N_steps and done == False): 
        j+=1
        curr_state_in = torch.reshape(curr_state, (1, 1, N, N)).detach()
        policy, value = worker_model(curr_state_in)
        values.append(value)
        pred = torch.squeeze(policy)
        mu = pred[:N**2]
        sigma = torch.exp(pred[N**2:])
        #mu = pred
        #sigma = 0.05
        action = torch.normal(mean=mu, std=sigma)
        prev_state = curr_state
        curr_state = prev_state + torch.reshape(action, (N, N))
        curr_state[curr_state > 1] = 1
        curr_state[curr_state < 0] = 0
        prev_score = utils.h_dec(prev_state.detach().numpy())
        current_score = utils.h_dec(curr_state.detach().numpy())
        reward = current_score - prev_score
        avg_reward = (avg_reward * num_itrs + reward) / (num_itrs + 1)
        reward -= avg_reward
        num_itrs += 1

        prob = torch.exp(-0.5 *((action - mu) / (sigma))**2) * 1 / (sigma * np.sqrt(2 * np.pi))
        logprob_ = torch.log(prob + 1e-5)
        logprobs.append(logprob_)
    
        if current_score > 255:
            reward = 10000
            curr_state = torch.ones((N, N)) / 2
            done = True
        else:
            G = value.detach()
        rewards.append(reward)
        history_performance.append(current_score)
    return curr_state, values, logprobs, rewards, G, avg_reward, num_itrs

def update_params(worker_opt,values,logprobs,rewards,G,clc=0.1,gamma=0.95):
        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
        logprobs = torch.stack(logprobs)
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        Returns = []
        ret_ = G
        for r in range(rewards.shape[0]):
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns,dim=0)
        Returns = Returns.unsqueeze(1).repeat(1, 256)
        values = values.unsqueeze(1).repeat(1, 256)
        actor_loss = torch.mul(-1*logprobs, (Returns - values.detach()))
        critic_loss = torch.pow(values - Returns,2)
        loss = actor_loss.sum() + clc*critic_loss.sum()
        loss.backward()
        worker_opt.step()
        return actor_loss, critic_loss, len(rewards)
    
def worker(t, worker_model, counter, params, history_performance,history_actor_loss,history_critic_loss):
    N = 16
    avg_reward = 0
    num_itrs = 0
    curr_state = torch.ones((N, N)) / 2
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()
    
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        curr_state, values, logprobs, rewards, G, avg_reward, num_itrs = run_episode(curr_state, worker_model, history_performance, avg_reward, num_itrs)
        actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards, G)
        history_actor_loss.append(actor_loss)
        history_critic_loss.append(critic_loss)
        counter.value = counter.value + 1
        print(i)

    plt.plot(history_performance)
    plt.show()

    plt.plot(history_actor_loss)
    plt.show()

    plt.plot(history_critic_loss)
    plt.show()

def main():
    N = 16
    print(mp.cpu_count()) # cpu count is 16 for ASUS Tuff Gaming F17
    MasterNode = ActorCritic()
    raw_state = torch.ones((N, N)) / 2
    raw_state_in = torch.reshape(raw_state, (1, 1, N, N))
    MasterNode(raw_state_in)
    MasterNode.share_memory()

    history_performance = []
    history_actor_loss = []
    history_critic_loss = []

    processes = [] 
    params = {
        'epochs':150000,
        'n_workers':1,
    }
    counter = mp.Value('i',0) 
    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i,MasterNode,counter,params,history_performance,history_actor_loss,history_critic_loss)) 
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes: 
        p.terminate()
    
    print(counter.value,processes[1].exitcode)

if __name__ == "__main__":
    main()