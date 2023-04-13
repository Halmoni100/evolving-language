import argparse
from agents.dqn_model import Agent

import gym

from ma_gym.wrappers import Monitor

if __name__ == '__main__':

    #env = gym.make(args.env)
    env = gym.make('ma_gym:Lumberjacks-v1')
    # env = Monitor(env, directory='recordings/' + args.env, force=True)

    lr = 0.001
    episodes = 100

    agent_list = []
    for i in env.n_agents:
        agent = Agent(gamma=0.998, epsilon=1.0, lr=lr, input_dims=env._obs_len ** 2, n_actions=5, mem_size=50000, batch_size=64, epsilon_dec=1, epsilon_end=0.001, fname="dqn_model_23jul.h5")
        agent_list.append(agent)


    for ep_i in range(episodes):
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        env.seed(ep_i)
        obs_n = env.reset()
        # env.render()

        while not all(done_n):
            
            action_n = [None]*env.n_agents
            for i in range(env.n_agents):
                action_n[i] = agent_list[i].choose_action(obs_n[i])

            new_obs_n, reward_n, done_n, info = env.step(action_n)
            ep_reward += sum(reward_n)
            
            for i in range(env.n_agents):
                agent_list[i].store_transition(obs_n[i], action_n[i], reward_n[i], new_obs_n[i], done_n[i])

            obs_n = new_obs_n

            for i in range(env.n_agents):
                agent_list[i].learn()

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    
    env.close()
