#!/usr/bin/env python

import sys
sys.path.append("..")
import os
from multiprocessing import Process, Lock, Condition

import yaml
import tensorflow as tf
import gymnasium as gym
from tensorflow.keras.utils import to_categorical
import numpy as np

from agents.dqn_model import Agent
from copier import Copier

g_sync_lock = Lock()
g_generation_done = Condition(g_sync_lock)
g_num_agents_done_filepath = "/tmp/evolve/num_agents_done"

def read_num_agents_done():
    with open(g_num_agents_done_filepath) as f: 
        num_agents_done = int(f.read().strip())
    return num_agents_done

def write_num_agents_done(new_value):
    with open(g_num_agents_done_filepath, 'w') as f:
        f.write(str(new_value))

def save_buffer(buffer, dirpath, prefix, suffix):
    buffer_np = np.array(buffer)
    filename = prefix + suffix + ".npy"
    filepath = os.path.join(dirpath, filename)
    np.save(buffer_np, filepath)

def get_copier_embedding(copier, observation, num_actions):
    if copier is None:
        copier_embedding = np.zeros(num_actions)
    else:
        copier_prediction = copier.predict(observation)
        copier_embedding = to_categorical(copier_prediction, num_classes=num_actions)
    return copier_embedding

def train_agent(idx, dqn_config, num_episodes, copier, buffer_file_dir, buffer_filename_prefix, agent_done_cond, total_num_agents_per_generation):
    env = gym.make('Taxi-v3')
    observation, _ = env.reset()
    num_actions = 6 # taxi
    obs_dim = 500 # taxi
    assert(len(observation) == obs_dim)
    dqn_agent = Agent(id=idx,
                      input_dims=obs_dim + num_actions,
                      n_actions=num_actions,
                      **dqn_config)
    observation_buffer = list()
    action_buffer = list()
    for episode in range(num_episodes):
        observation, reward, termination, truncation, info = env.last() 
        if termination or truncation:
            break

        copier_embedding = get_copier_embedding(copier, observation, num_actions)
        observation_with_copier_embedding = np.concatenate((observation, copier_embedding))
        action, dqn_command, entropy = dqn_agent.choose_action(observation_with_copier_embedding)
        env.step(action_i)

        observation_buffer.append(observation)
        action_buffer.append(action)

        new_observation, reward, termination, truncation, info = env.last()
        new_copier_embedding = get_copier_embedding(copier, new_observation, num_actions)
        new_observation_with_copier_embedding = np.concatenate((new_observation, new_copier_embedding))
        dqn_agent.store_transition(observation_with_copier_embedding, action, reward, new_observation_with_copier_embedding, False)

        if episode > episodes_until_dqn_learn:
            dqn_agent.learn()

    save_buffer(observation_buffer, buffer_file_dir, buffer_filename_prefix, "_obs")
    save_buffer(action_buffer, buffer_file_dir, buffer_filename_prefix, "_act")

    with g_sync_lock:
        num_agents_done = read_num_agents_done()
        num_agents_done += 1
        write_num_agents_done(num_agents_done)
        if num_agents_done == total_num_agents_per_generation: 
            g_generation_done.notify()

def get_generation_data(generation, total_num_agents_per_generation):
    observations = dict()
    actions = dict()
    for element in os.listdir("/tmp/evolve"):
        root_ext = os.path.splitext(element)
        if root_ext[1] != ".npy"
            continue
        root_split = root_ext[0].split('_')
        if len(root_split) != 5 
                or root_split[0] != "gen" 
                or root_split[2] != "agent" 
                or (root_split[4] != "obs" and root_split[4] != "act"):
            continue
        file_generation = int(root_split[1])
        agent_idx = int(root_split[3])
        is_obs = root_split[4] == "obs"
        filepath = os.path.join("/tmp/evolve", element)
        if file_generation != generation:
            continue
        data = np.load(filepath)
        if is_obs:
            assert(agent_idx not in observations.keys())
            observations[agent_idx] = data
        else:
            assert(agent_idx not in actions.keys())
            actions[agent_idx] = data

    assert(len(observations.keys()) == total_num_agents_per_generation)
    assert(len(actions.keys()) == total_num_agents_per_generation)
    for idx in range(total_num_agents_per_generation):
        assert(idx in observations.keys())
        assert(idx in actions.keys())

    return observations, actions

def delete_generation_data(generation, total_num_agents_per_generation):
    for agent_idx in range(total_num_agents_per_generation):
        pass

def create_copier_buffer(observations, actions, total_num_agents_per_generation):
    total_timepoints = 0
    for agent_idx in range(total_num_agents_per_generation):
        agent_observations = observations[agent_idx]
        agent_actions = actions[agent_idx]
        assert(len(agent_observations) == len(agent_actions))
        agent_timepoints = len(agent_observations)
        total_timepoints += agent_timepoints
    observation_dim = len(observations[0])
    observation_buffer = np.zeros((observation_dim, total_timepoints))
    action_buffer = np.zeros(total_timepoints)
    curr_timepoint = 0
    for agent_idx in range(total_num_agents_per_generation):
        agent_observations = observations[agent_idx]
        agent_actions = actions[agent_idx]
        agent_timepoints = len(agent_observations)
        next_timepoint = curr_timepoint + agent_timepoints
        observations[curr_timepoint:next_timepoint, :] = agent_observations
        actions[curr_timepoint:next_timepoint] = agent_actions
        curr_timepoint = next_timepoint
    return observation_buffer, action_buffer

def synchronize(config, num_generations, total_num_agents_per_generation, buffer_file_dir):
    for generation in range(num_generations):
        g_generation_done.wait()
        num_agents = read_num_agents_done()
        assert(num_agents == total_num_agents_per_generation)
        observations, actions = get_generation_data(generation, total_num_agents_per_generation)
        observation_buffer, action_buffer = create_copier_buffer(observations, actions, total_num_agents_per_generation)
        copier = Copier(config["copier"])
        copier.train(observation_buffer, action_buffer)

def agent_process(agent_idx, config, num_generations):
    num_episodes = config["num_episodes_per_agent"]
    for generation in range(num_generations):
        prefix = "gen_" + str(generation) + "_agent_" + str(agent_idx)
        train_agent(agent_idx, config["dqn_config"], num_episodes, copier, "/tmp/evolve", prefix)
        with g_generation_done:
            g_generation_done.wait_for()

def main():
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    num_agents_per_generation = 4

    processes = list()
    for idx in range(num_agents):
        p = Process(target=agent, args=(idx, config["dqn_config"]))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
