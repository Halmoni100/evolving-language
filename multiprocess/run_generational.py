#!/usr/bin/env python

import sys
sys.path.append("..")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from multiprocessing import Process, Lock, Condition
import uuid
import shutil

import yaml
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from taxi import taxi_observation_transform
from suppress_output import RedirectStdStreams
from progress_bar import ProgressBar

g_tmp_dir = "/tmp/evolve"

g_sync_lock = Lock()
g_generation_done = Condition(g_sync_lock)
g_sync_done = Condition(g_sync_lock)
g_num_agents_done_filepath = os.path.join(g_tmp_dir, "num_agents_done")
g_copier_filepath = os.path.join(g_tmp_dir, "curr_copier")

def read_num_agents_done():
    with open(g_num_agents_done_filepath) as f: 
        num_agents_done = int(f.read().strip())
    return num_agents_done

def write_num_agents_done(new_value):
    with open(g_num_agents_done_filepath, 'w') as f:
        f.write(str(new_value))

def save_buffer(buffer, prefix, suffix):
    buffer_np = np.array(buffer)
    filename = prefix + suffix + ".npy"
    filepath = os.path.join(g_tmp_dir, filename)
    np.save(filepath, buffer_np)

def get_copier_embedding(copier, observation, num_actions):
    from tensorflow import keras

    if copier is None:
        copier_embedding = np.zeros(num_actions)
    else:
        copier_prediction = copier.predict(observation, verbose=0)
        copier_embedding = keras.utils.to_categorical(copier_prediction, num_classes=num_actions)
    return copier_embedding

def train_agent(run_id, idx, dqn_config, dqn_misc, num_episodes, copier, buffer_filename_prefix, num_agents_per_generation, observation_transform):
    from agents.dqn_model import Agent
    from tensorflow import keras

    observation_buffer = list()
    action_buffer = list()
    reward_buffer = list()
    env = gym.make('Taxi-v3')
    observation, info = env.reset()
    observation = observation_transform(observation)
    num_actions = 6 # taxi
    obs_dim = 4 + 5 + 5 + 5 # taxi
    assert(len(observation) == obs_dim)
    dqn_agent = Agent(id=idx,
                      input_dims=obs_dim + num_actions,
                      n_actions=num_actions,
                      **dqn_config)

    if idx == 0:
        pb = ProgressBar(num_episodes, length=50)
        pb.start(front_msg="episodes ")

    for episode in range(num_episodes):
        if idx == 0:
            pb.update(front_msg="episodes ")

        observation, info = env.reset()
        observation = observation_transform(observation)
        curr_observation = observation
        curr_reward = None
        curr_termination = False
        curr_truncation = False
        curr_info = info
        episode_reward = 0

        # run episode
        while True:
            if curr_termination or curr_truncation:
                break

            copier_embedding = get_copier_embedding(copier, curr_observation, num_actions)
            curr_observation_with_copier_embedding = np.concatenate((curr_observation, copier_embedding))
            curr_action, dqn_command, entropy = dqn_agent.choose_action(curr_observation_with_copier_embedding, verbose=0)
            next_observation, next_reward, next_termination, next_truncation, next_info = env.step(curr_action)
            next_observation = observation_transform(next_observation)

            observation_buffer.append(curr_observation)
            action_buffer.append(curr_action)
            episode_reward += next_reward

            next_copier_embedding = get_copier_embedding(copier, next_observation, num_actions)
            next_observation_with_copier_embedding = np.concatenate((next_observation, next_copier_embedding))
            dqn_agent.store_transition(curr_observation_with_copier_embedding, curr_action, next_reward, next_observation_with_copier_embedding, False)

            if episode > dqn_misc["episodes_until_learn"]:
                devnull = open(os.devnull, 'w')
                with RedirectStdStreams(stdout=devnull, stderr=devnull): # hack to suppress output
                    dqn_agent.learn()
                devnull.close()

            curr_observation = next_observation
            curr_reward = next_reward
            curr_termination = next_termination
            curr_truncation = next_truncation
            curr_info = next_info
        
        reward_buffer.append(episode_reward)
        keras.backend.clear_session()

    if idx == 0:
        pb.reset()


    save_buffer(observation_buffer, buffer_filename_prefix, "_obs")
    save_buffer(action_buffer, buffer_filename_prefix, "_act")
    save_buffer(reward_buffer, buffer_filename_prefix, "_rew")

    with g_sync_lock:
        num_agents_done = read_num_agents_done()
        num_agents_done += 1
        write_num_agents_done(num_agents_done)
        if num_agents_done == num_agents_per_generation: 
            g_generation_done.notify()

def get_generation_data(generation, num_agents_per_generation, num_episodes_per_agent):
    observations = dict()
    actions = dict()
    rewards = dict()
    for element in os.listdir("/tmp/evolve"):
        root_ext = os.path.splitext(element)
        if root_ext[1] != ".npy":
            continue
        root_split = root_ext[0].split('_')
        if (len(root_split) != 5
                or root_split[0] != "gen"
                or root_split[2] != "agent"
                or (root_split[4] != "obs" 
                    and root_split[4] != "act" 
                    and root_split[4] != "rew")):
            continue
        file_generation = int(root_split[1])
        agent_idx = int(root_split[3])
        suffix = root_split[4]
        filepath = os.path.join("/tmp/evolve", element)
        if file_generation != generation:
            continue
        data = np.load(filepath)
        if suffix == "obs":
            assert(agent_idx not in observations.keys())
            observations[agent_idx] = data
        elif suffix == "act":
            assert(agent_idx not in actions.keys())
            actions[agent_idx] = data
        else:
            assert(agent_idx not in rewards.keys())
            rewards[agent_idx] = data

    assert(len(observations.keys()) == num_agents_per_generation)
    assert(len(actions.keys()) == num_agents_per_generation)
    assert(len(rewards.keys()) == num_agents_per_generation)
    for idx in range(num_agents_per_generation):
        assert(idx in observations.keys())
        assert(idx in actions.keys())
        assert(idx in rewards.keys())

    return observations, actions, rewards

def delete_generation_data(generation, num_agents_per_generation):
    for agent_idx in range(num_agents_per_generation):
        prefix = "gen_" + str(generation) + "_agent_" + str(agent_idx)
        observations_filename = prefix + "_obs.npy"
        observations_filepath = os.path.join("/tmp/evolve", observations_filename)
        os.remove(observations_filepath)
        actions_filename = prefix + "_act.npy"
        actions_filepath = os.path.join("/tmp/evolve", actions_filename)
        os.remove(actions_filepath)
        rewards_filename = prefix + "_rew.npy"
        rewards_filepath = os.path.join("/tmp/evolve", rewards_filename)
        os.remove(rewards_filepath)

def create_buffers(observations, actions, rewards, num_agents_per_generation, num_episodes_per_agent):
    total_timepoints = 0
    for agent_idx in range(num_agents_per_generation):
        agent_observations = observations[agent_idx]
        agent_actions = actions[agent_idx]
        agent_rewards = rewards[agent_idx]
        assert(len(agent_observations) == len(agent_actions))
        assert(len(agent_rewards) == num_episodes_per_agent)
        agent_timepoints = len(agent_observations)
        total_timepoints += agent_timepoints
    observation_dim = observations[0].shape[1]
    observation_dtype = observations[0][0].dtype
    observation_buffer = np.zeros((total_timepoints, observation_dim), dtype=observation_dtype)
    action_buffer = np.zeros(total_timepoints)
    reward_buffer = np.zeros(num_episodes_per_agent * num_agents_per_generation)
    curr_timepoint = 0
    curr_reward_idx = 0
    for agent_idx in range(num_agents_per_generation):
        agent_observations = observations[agent_idx]
        agent_actions = actions[agent_idx]
        agent_rewards = rewards[agent_idx]
        agent_timepoints = len(agent_observations)
        next_timepoint = curr_timepoint + agent_timepoints
        next_reward_idx = curr_reward_idx + num_episodes_per_agent
        observation_buffer[curr_timepoint:next_timepoint, :] = agent_observations
        action_buffer[curr_timepoint:next_timepoint] = agent_actions
        reward_buffer[curr_reward_idx:next_reward_idx] = agent_rewards
        curr_timepoint = next_timepoint
        curr_reward_idx = next_reward_idx

    return observation_buffer, action_buffer, reward_buffer

def plot_rewards(generation_rewards, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    plt.clf()
    fig, ax = plt.subplots()
    ax.violinplot(generation_rewards)
    plot_path = os.path.join(plot_dir, "rewards.png")
    plt.savefig(plot_path)

def synchronize(run_id, config):
    from copier import Copier

    run_log_dir = os.path.join("logs", run_id)
    os.makedirs(run_log_dir, exist_ok=True)

    num_generations = config["num_generations"]
    num_agents_per_generation = config["num_agents_per_generation"]
    num_episodes_per_agent = config["num_episodes_per_agent"]
    generation_rewards = list()
    for generation in range(num_generations):
        with g_sync_lock:
            g_generation_done.wait()
        num_agents = read_num_agents_done()
        assert(num_agents == num_agents_per_generation)
        observations, actions, rewards = get_generation_data(generation, num_agents_per_generation, num_episodes_per_agent)
        observation_buffer, action_buffer, reward_buffer = create_buffers(observations, actions, rewards, num_agents_per_generation, num_episodes_per_agent)

        reward_mean = np.mean(reward_buffer)
        reward_std = np.std(reward_buffer)
        print(f"Reward mean: {reward_mean}, std_dev: {reward_std}")
        generation_rewards.append(reward_buffer)
        plot_rewards(generation_rewards, "results")

        copier = Copier(config["copier"])
        fit_dir = os.path.join(run_log_dir, f"gen{generation}")
        copier.train(observation_buffer, action_buffer, verbose=0, tensorboard_log_dir=fit_dir)
        copier.model.save(g_copier_filepath)
        delete_generation_data(generation, num_agents_per_generation)

        with g_sync_lock:
            write_num_agents_done(0)
            g_sync_done.notify_all()

def run_train_agent(generation, run_id, agent_idx, config):
    from tensorflow import keras
    from copier import Copier

    num_episodes = config["num_episodes_per_agent"]
    num_agents_per_generation = config["num_agents_per_generation"]

    prefix = "gen_" + str(generation) + "_agent_" + str(agent_idx)

    if generation == 0:
        copier = None
    else:
        copier = Copier(config["copier"])
        copier.dqn_model = keras.models.load_model(g_copier_filepath)
    train_agent(run_id, agent_idx, config["dqn_config"], config["dqn_misc"], num_episodes, copier, prefix, num_agents_per_generation, taxi_observation_transform)

def agent_process(run_id, agent_idx, config):
    num_generations = config["num_generations"]
    for generation in range(num_generations):
        p = Process(target=run_train_agent, args=(generation, run_id, agent_idx, config))
        p.start()
        p.join()
        with g_sync_lock:
            g_sync_done.wait()

def main():
    run_id = str(uuid.uuid4())

    if os.path.isdir(g_tmp_dir):
        shutil.rmtree(g_tmp_dir)
    os.makedirs(g_tmp_dir, exist_ok=True)
    with g_sync_lock:
        write_num_agents_done(0)

    with open("generational_config.yml") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    num_agents_per_generation = config["num_agents_per_generation"]

    processes = list()
    p = Process(target=synchronize, args=(run_id, config))
    processes.append(p)
    p.start()
    for idx in range(num_agents_per_generation):
        p = Process(target=agent_process, args=(run_id, idx, config))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
