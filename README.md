# evolving-language

notes

- DQN is able to learn; leaky relu with L2 regularization converges a lot faster and stabler than tanh
- learning rate set to 0.001 and batch_size 64 (for agent replay buffer)

- maxcycle param needs tuning? 
- reduce overfitting/policy oscillation: try different architecture/activation/learning rate

- plots to generate:
    - no copier vs with copier (tune lookback param?)
    - number of agents: 1, 2, 3, 5, 10, 20
    - different environments (currently using simple_spread)

- fixed bugs:
    - epsilon should decay every episode and not every step
    - reduce overfitting/policy oscillation: switched to leaky relu with L2, learning rate 0.001
