import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fp = r'/Users/eleanorye/Documents/GitHub/evolving-language/results.txt'

def plot_rewards(filepath):
    rewards = []
    with open(filepath, "r") as file:
        for line in file:
            r = line.strip().split(":")[1]
            rewards.append(float(r))
    rewards = pd.Series(rewards)
    plt.figure()
    rewards.plot()
    plt.title("Rewards by episode")
    plt.show()

    return


