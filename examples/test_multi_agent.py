import gym
from ma_gym.envs.lumberjacks import Lumberjacks

env = Lumberjacks()
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)

print(ep_reward)