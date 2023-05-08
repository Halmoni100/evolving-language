#!/usr/bin/env python

import sys
sys.path.append("..")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from multiprocessing import Process, Lock, Condition

import yaml
import tensorflow as tf
import gymnasium as gym
from tensorflow import keras
import numpy as np

from agents.dqn_model import Agent
from copier import Copier

class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

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
    if copier is None:
        copier_embedding = np.zeros(num_actions)
    else:
        copier_prediction = copier.predict(observation, verbose=0)
        copier_embedding = keras.utils.to_categorical(copier_prediction, num_classes=num_actions)
    return copier_embedding

def taxi_observation_transform(observation):
    observation_left = observation

    destination = observation_left % 4
    destination_one_hot = keras.utils.to_categorical(destination, num_classes=4)
    observation_left = observation_left // 4

    passenger_location = observation_left % 5
    passenger_location_one_hot = keras.utils.to_categorical(passenger_location, num_classes=5)
    observation_left = observation_left // 5

    taxi_col = observation_left % 5
    taxi_col_one_hot = keras.utils.to_categorical(taxi_col, num_classes=5)
    observation_left = observation_left // 5

    taxi_row = observation_left
    taxi_row_one_hot = keras.utils.to_categorical(taxi_row, num_classes=5)

    transformed_observation = np.concatenate((destination_one_hot, passenger_location_one_hot, taxi_col_one_hot, taxi_row_one_hot))
    return transformed_observation

def train_agent(idx, dqn_config, dqn_misc, num_episodes, copier, buffer_filename_prefix, num_agents_per_generation, observation_transform):
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
    observation_buffer = list()
    action_buffer = list()
    reward_buffer = list()
    curr_observation = observation
    curr_reward = None
    curr_termination = False
    curr_truncation = False
    curr_info = info
    for episode in range(num_episodes):
        if curr_termination or curr_truncation:
            break

        copier_embedding = get_copier_embedding(copier, curr_observation, num_actions)
        curr_observation_with_copier_embedding = np.concatenate((curr_observation, copier_embedding))
        curr_action, dqn_command, entropy = dqn_agent.choose_action(curr_observation_with_copier_embedding, verbose=0)
        next_observation, next_reward, next_termination, next_truncation, next_info = env.step(curr_action)
        next_observation = observation_transform(next_observation)

        observation_buffer.append(curr_observation)
        action_buffer.append(curr_action)
        reward_buffer.append(next_reward)

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

    save_buffer(observation_buffer, buffer_filename_prefix, "_obs")
    save_buffer(action_buffer, buffer_filename_prefix, "_act")
    save_buffer(reward_buffer, buffer_filename_prefix, "_rew")

    with g_sync_lock:
        num_agents_done = read_num_agents_done()
        num_agents_done += 1
        write_num_agents_done(num_agents_done)
        if num_agents_done == num_agents_per_generation: 
            g_generation_done.notify()

def get_generation_data(generation, num_agents_per_generation):
    observations = dict()
    actions = dict()
    for element in os.listdir("/tmp/evolve"):
        root_ext = os.path.splitext(element)
        if root_ext[1] != ".npy":
            continue
        root_split = root_ext[0].split('_')
        if (len(root_split) != 5
                or root_split[0] != "gen"
                or root_split[2] != "agent"
                or (root_split[4] != "obs" and root_split[4] != "act")):
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

    assert(len(observations.keys()) == num_agents_per_generation)
    assert(len(actions.keys()) == num_agents_per_generation)
    for idx in range(num_agents_per_generation):
        assert(idx in observations.keys())
        assert(idx in actions.keys())

    return observations, actions

def delete_generation_data(generation, num_agents_per_generation):
    for agent_idx in range(num_agents_per_generation):
        prefix = "gen_" + str(generation) + "_agent_" + str(agent_idx)
        observations_filename = prefix + "_obs.npy"
        observations_filepath = os.path.join("/tmp/evolve", observations_filename)
        os.remove(observations_filepath)
        actions_filename = prefix + "_act.npy"
        actions_filepath = os.path.join("/tmp/evolve", actions_filename)
        os.remove(actions_filepath)

def create_copier_buffer(observations, actions, num_agents_per_generation):
    total_timepoints = 0
    for agent_idx in range(num_agents_per_generation):
        agent_observations = observations[agent_idx]
        agent_actions = actions[agent_idx]
        assert(len(agent_observations) == len(agent_actions))
        agent_timepoints = len(agent_observations)
        total_timepoints += agent_timepoints
    observation_dim = observations[0].shape[1]
    observation_dtype = observations[0][0].dtype
    observation_buffer = np.zeros((total_timepoints, observation_dim), dtype=observation_dtype)
    action_buffer = np.zeros(total_timepoints)
    curr_timepoint = 0
    for agent_idx in range(num_agents_per_generation):
        agent_observations = observations[agent_idx]
        agent_actions = actions[agent_idx]
        agent_timepoints = len(agent_observations)
        next_timepoint = curr_timepoint + agent_timepoints
        observation_buffer[curr_timepoint:next_timepoint, :] = agent_observations
        action_buffer[curr_timepoint:next_timepoint] = agent_actions
        curr_timepoint = next_timepoint
    return observation_buffer, action_buffer

def synchronize(config):
    num_generations = config["num_generations"]
    num_agents_per_generation = config["num_agents_per_generation"]
    for generation in range(num_generations):
        with g_sync_lock:
            g_generation_done.wait()
        num_agents = read_num_agents_done()
        assert(num_agents == num_agents_per_generation)
        observations, actions = get_generation_data(generation, num_agents_per_generation)
        observation_buffer, action_buffer = create_copier_buffer(observations, actions, num_agents_per_generation)
        copier = Copier(config["copier"])
        copier.train(observation_buffer, action_buffer, verbose=0)
        copier.model.save(g_copier_filepath)
        delete_generation_data(generation, num_agents_per_generation)
        with g_sync_lock:
            write_num_agents_done(0)
            g_sync_done.notify_all()


def agent_process(agent_idx, config):
    num_episodes = config["num_episodes_per_agent"]
    num_agents_per_generation = config["num_agents_per_generation"]
    num_generations = config["num_generations"]
    for generation in range(num_generations):
        prefix = "gen_" + str(generation) + "_agent_" + str(agent_idx)
        if generation == 0:
            copier = None
        else:
            copier = Copier(config["copier"])
            copier.dqn_model = keras.models.load_model(g_copier_filepath)
        train_agent(agent_idx, config["dqn_config"], config["dqn_misc"], num_episodes, copier, prefix, num_agents_per_generation, taxi_observation_transform)
        with g_sync_lock:
            g_sync_done.wait()

def main():
    os.makedirs(g_tmp_dir, exist_ok=True)
    with g_sync_lock:
        write_num_agents_done(0)

    with open("generational_config.yml") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    num_agents_per_generation = config["num_agents_per_generation"]

    processes = list()
    p = Process(target=synchronize, args=(config,))
    processes.append(p)
    p.start()
    for idx in range(num_agents_per_generation):
        p = Process(target=agent_process, args=(idx, config))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
