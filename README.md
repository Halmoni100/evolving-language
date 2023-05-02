# evolving-language

Roadmap/Scope:
    - Environment: {simple_spread, simple_comm, "single agent env"} representing MA env, MA env with comm, and single agent env
    - For each of the environments, the following needs to be done (keeping almost everything the same so we can compare apple to apple):
        1. Agent model: DQN
        2. Copier model: start with the "global ex-self" copier which copies what everyone else does; vary later
        3. Evaluation: reward and entropy of distribution on predicted actions
        3. Test on 1-agent (no copier) setup to make sure reward and entropy converge 
        4. Increase number of agents to 2, 3, 5, 10, 20 (run no copier and with copier)


TODO:
    - add entropy metric in addition to reward
    - remove agent_i itself from copier buffer - i.e. the copier for each agent does not involve itself
    - vary agent's learning scheme: learn from 64 steps after every step, or learn from the entire buffer after every episode

notes:

- DQN structure: 2 layers (32 &16) with leaky relu and L2 regularization
- learning rate set to 0.001 and batch_size 64 (for agent replay buffer)


- fixed bugs:
    - epsilon should decay every episode and not every step
    - convergence: switched to leaky relu with L2, learning rate 0.001

