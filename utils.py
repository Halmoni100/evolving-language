import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_rewards(filepath):
    env_name = filepath.split("results/")[1].split(".")[0]
    rewards = []
    with open(filepath, "r") as file:
        for line in file:
            r = line.strip().split(":")[1]
            rewards.append(float(r))
    rewards = pd.Series(rewards)
    rewards.plot()
    plt.title(env_name + " rewards by episode")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()
    return rewards


def plot_rewards_compare(fp, fp_copier):
    env_name = fp.split("/results/")[1].split(".")[0]
    rewards = []
    with open(fp, "r") as file:
        for line in file:
            r = line.strip().split(":")[1]
            rewards.append(float(r))
    rewards_cp = []
    with open(fp_copier, "r") as file:
        for line in file:
            r = line.strip().split(":")[1]
            rewards_cp.append(float(r))

    rewards = pd.Series(rewards)
    rewards_cp = pd.Series(rewards_cp)

    plt.figure()
    rewards.plot(label="no copier")
    rewards_cp.plot(label="copier")
    plt.title(env_name + " rewards by episode")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.show()
    return


def plot_rewards_odds_only(filepath):
    rewards = []
    with open(filepath, "r") as file:
        i = 0
        for line in file:
            if i % 2 == 0:
                r = line.strip().split(":")[1]
                rewards.append(float(r))
            i += 1
    rewards = pd.Series(rewards)
    plt.figure()
    rewards.plot()
    plt.title("Rewards by episode")
    plt.show()
    return

