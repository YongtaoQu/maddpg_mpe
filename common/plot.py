from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd


def save_plot(return_array, env, data_save_dir):
    """Save the return data and plot."""
    if not os.path.exists(data_save_dir):
        os.mkdir(data_save_dir)

    num_agents = return_array.shape[2]
    agents = env.possible_agents
    df = pd.DataFrame(return_array, columns=agents)
    df.to_excel(data_save_dir + env.__str__() + '_data.xlsx', index=False)

    plt.figure(figsize=(10, 5))
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title(env.__str__())
    plt.grid(True)

    for agent in range(num_agents):
        rewards = return_array[:, :, agent]
        episodes = range(0, len(rewards) * 100, 100)
        avg_rewards = np.mean(rewards, axis=1)
        plt.plot(episodes, avg_rewards, label=agents[agent])

        stds = [np.std(reward) for reward in rewards]
        errors = [1.96 * std / np.sqrt(100) for std in stds]
        lower = [x - error for x, error in zip(avg_rewards, errors)]
        upper = [x + error for x, error in zip(avg_rewards, errors)]
        plt.fill_between(episodes, lower, upper, alpha=0.1)

    plt.legend()
    plt.savefig(data_save_dir + env.__str__() + '.png')
    plt.show()


def comparison_plot(return_arrays, env, data_save_dir):
    """Compare two returns and plot."""
    if not os.path.exists(data_save_dir):
        os.mkdir(data_save_dir)

    plt.figure(figsize=(10, 5))
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title(env.__str__())
    plt.grid(True)
    nets = ["mlp", "rnn"]

    for return_array, net in zip(return_arrays, nets):
        num_agents = return_array.shape[2]
        agents = env.possible_agents
        df = pd.DataFrame(np.mean(return_array, axis=1), columns=agents)
        df.to_excel(data_save_dir + env.__str__() + '_' + net + '_data.xlsx', index=False)

        for agent in range(num_agents):
            if agents[agent] in ["adversary_0", "adversary_1", "adversary_2"]:
                continue
            rewards = return_array[:, :, agent]
            episodes = range(0, len(rewards) * 100, 100)
            avg_rewards = np.mean(rewards, axis=1)
            plt.plot(episodes, avg_rewards, label=agents[agent] + "_" + net)

            stds = [np.std(reward) for reward in rewards]
            errors = [1.96 * std / np.sqrt(100) for std in stds]
            lower = [x - error for x, error in zip(avg_rewards, errors)]
            upper = [x + error for x, error in zip(avg_rewards, errors)]
            plt.fill_between(episodes, lower, upper, alpha=0.1)
    plt.yticks(range(-200, 21, 20))
    plt.legend()
    plt.savefig(data_save_dir + env.__str__() + '.png')
    plt.show()
