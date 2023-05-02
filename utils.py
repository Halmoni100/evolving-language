import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

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


def plot_rewards_compare(fp1, fp2, label1, label2, title=None, footnote=None):
    env_name = fp1.split("/results/")[1].split(".")[0]
    rewards = []
    with open(fp1, "r") as file:
        for line in file:
            r = line.strip().split(":")[1]
            rewards.append(float(r))
    rewards_cp = []
    with open(fp2, "r") as file:
        for line in file:
            r = line.strip().split(":")[1]
            rewards_cp.append(float(r))

    rewards = pd.Series(rewards)
    rewards_cp = pd.Series(rewards_cp)

    plt.figure()
    rewards.plot(label=label1, alpha=0.8)
    rewards_cp.plot(label=label2, alpha=0.8)
    if title == None:
        plt.title(env_name + " rewards by episode")
    else:
        plt.title(title + " rewards by episode")
    plt.xlabel("episode")
    if footnote != None:
        plt.text(0.5, -0.2, "note: "+footnote, ha='center', va='center', transform=plt.gca().transAxes)
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

