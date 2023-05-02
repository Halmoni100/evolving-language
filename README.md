# evolving-language

notes

- DQN is able to learn - see simple and simple_spread_1agent_maxcycle300
- however, reward plots for 2 agents & 3 agents show cycles and no convergence (probably stuck in suboptimal policies and unable to explore)
- maxcycle param needs tuning? 
- reduce overfitting/policy oscillation: try different architecture/activation/learning rate

- plots to generate:
    - no copier vs with copier (episode lookback 20, 50, 100)
    - number of agents: 1, 2, 3, 5, 10, 20
    - different environments (currently using simple_spread)

- fixed bugs:
    - epsilon should decay every episode and not every step
    - batch_size (how many steps the agent samples to learn) too small (previously 64)