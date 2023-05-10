import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def plot_rewards(filepath, has_entropy:bool, title=None, footnote=None):
    env_name = filepath.split("results/")[1].split(".")[0]
    rewards = []
    entropy = []
    with open(filepath, "r") as file:
        for line in file:
            if not has_entropy:
                r = line.strip().split(":")[1]
            else:
                r = line.strip().split("Reward:")[1].split(", Entropy")[0]
                entropy.append(float(line.strip().split("Entropy mean:")[1].split(", Entropy std")[0]))
            rewards.append(float(r))
    rewards = pd.Series(rewards)
    entropy = pd.Series(entropy)
    fig, ax1 = plt.subplots()
    ax1.plot(rewards.index, rewards, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylabel('reward', color='tab:blue')
    ax1.set_xlabel("episode")

    ax2 = ax1.twinx()
    ax2.plot(entropy.index, entropy, color='tab:red')
    ax2.set_ylabel('entropy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout() 
    if title == None:
        plt.title(env_name + " rewards by episode")
    else:
        plt.title(title + " rewards by episode")
    if footnote != None:
        plt.text(0.5, -0.2, "note: "+footnote, ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()
    return rewards


def plot_rewards_compare(path_ls, label_ls, title=None, footnote=None, alpha=0.8):
    rewards = []
    fp0 = path_ls[0]
    env_name = fp0.split("/results/")[1].split(".")[0]

    for i in range(len(path_ls)):
        fp = path_ls[i]
        r = []
        with open(fp, "r") as file:
            for line in file:
                r_value = line.strip().split("Reward:")[1].split(", Entropy")[0]
                r.append(float(r_value))
        r = pd.Series(r)
        rewards.append(r)

    plt.figure()
    for i in range(len(path_ls)):
        r = rewards[i]
        label = label_ls[i]
        r.plot(label=label, alpha=alpha)

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


def plot_entropy_compare(path_ls, label_ls, title=None, footnote=None, alpha=0.8):
    entropy = []
    fp0 = path_ls[0]
    env_name = fp0.split("/results/")[1].split(".")[0]

    for i in range(len(path_ls)):
        fp = path_ls[i]
        r = []
        with open(fp, "r") as file:
            for line in file:
                r_value = line.strip().split("Entropy mean:")[1].split(", Entropy std")[0]
                r.append(float(r_value))
        r = pd.Series(r)
        entropy.append(r)

    plt.figure()
    for i in range(len(path_ls)):
        r = entropy[i]
        label = label_ls[i]
        r.plot(label=label, alpha=alpha)

    if title == None:
        plt.title(env_name + " entropy by episode")
    else:
        plt.title(title + " entropy by episode")
    plt.xlabel("episode")
    if footnote != None:
        plt.text(0.5, -0.2, "note: "+footnote, ha='center', va='center', transform=plt.gca().transAxes)
    plt.ylabel("entropy")
    plt.legend()
    plt.show()
    return

def plot_rewards_compare2(path_ls, label_ls, title=None, footnote=None, alpha=0.8):
    reward_ls = []
    entropy_ls = []
    fp0 = path_ls[0]
    #env_name = fp0.split("/results/")[1].split(".")[0]
    env_name = path_ls[0].split("results/")[1].split(".")[0]

    
    for i in range(len(path_ls)):
        fp = path_ls[i]
        r = []
        entropy = []
        with open(fp, "r") as file:
            for line in file:
                #r_value = line.strip().split(":")[1]
                r_value = line.strip().split("Reward:")[1].split(", Entropy")[0]
                entropy.append(float(line.strip().split("Entropy mean:")[1].split(", Entropy std")[0]))
                r.append(float(r_value))
        r = pd.Series(r)
        entropy = pd.Series(entropy)
        reward_ls.append(r)
        entropy_ls.append(entropy)

    """plt.figure()
    for i in range(len(path_ls)):
        r = rewards[i]
        label = label_ls[i]
        r.plot(label=label, alpha=alpha)
    """

    fig, ax1 = plt.subplots()
    ax1.plot(reward_ls[0].index, reward_ls[0], color='tab:blue', label='reward '+label_ls[0], alpha=alpha)
    ax1.plot(reward_ls[1].index, reward_ls[1], color='tab:cyan', label='reward '+label_ls[1], alpha=alpha)

    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylabel('reward', color='tab:blue')
    ax1.set_xlabel("episode")

    ax2 = ax1.twinx()
    ax2.plot(entropy_ls[0].index, entropy_ls[0], color='tab:red', label='entropy '+label_ls[0], alpha=alpha)
    ax2.plot(entropy_ls[1].index, entropy_ls[1], color='tab:orange', label='entropy '+label_ls[1], alpha=alpha)

    ax2.set_ylabel('entropy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    #fig.tight_layout() 

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

