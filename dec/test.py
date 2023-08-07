import numpy as np
import dec_SARR
import dec_SIM
import dec_GAv2
import dec_PSO
import utils
import torch

def h(arr):
    N = np.shape(arr)[0]

    c1 = np.reshape(np.linspace(0, N - 1, N), (N, 1))
    c2 = np.reshape(np.linspace(0, N - 1, N), (1, N))

    c1 = N - np.abs(c1 - (N - 1) / 2)
    c2 = N - np.abs(c2 - (N - 1) / 2)

    c = c1 + c2

    new_arr = np.cos((arr * (2 * N - 1) - c) * np.pi / c)

    return np.sum(new_arr)

def main():
    N = 16
    population_size = 160
    mut_rate = 0.15
    num_runs = 50_000
    num_loops = 2
    arr_ones = np.ones((N, N)) / 2
    population = [arr_ones for _ in range(population_size)]

    # 8.5k
    #SARR = dec_SARR.SARR(utils.h2, max_calls=num_runs, initial_state=arr_ones, num_restarts=100, num_loops=num_loops)
    #SARR.run()

    # very big
    #GA = dec_GAv2.GA(utils.h2, population, population_size, mut_rate, num_runs, num_loops)
    #GA.run()

    # 24k
    #PSO = dec_PSO.PSO(utils.h2, initial_state=arr_ones, num_particles=1200, num_runs=num_runs, num_loops=2)
    #PSO.run()

    # 5k
    SIM = dec_SIM.SIM(utils.h2, max_calls=5000, initial_state=arr_ones, tunneling_prob=0.01, num_loops=num_loops)
    SIM.run()

    # reinforcement learning
    pop = torch.load('dec/population_MP_popsize4800_layer64.txt')

    wait = input("Press Enter to continue.")

if __name__ == "__main__":
    main()