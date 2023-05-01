from run_simple_spread import *

if __name__ == '__main__':

    #main(NUM_EPISODES=400, NUM_AGENTS=5, MAX_CYCLES_PER_EP=300, use_copier=False, continue_from_checkpoint=False)
    #main(NUM_EPISODES=400, NUM_AGENTS=5, MAX_CYCLES_PER_EP=300, use_copier=True, continue_from_checkpoint=False, copier_ep_lookback=50)

    main(NUM_EPISODES=200, NUM_AGENTS=2, MAX_CYCLES_PER_EP=300, use_copier=False, continue_from_checkpoint=True, checkpoint_ep=200)
    main(NUM_EPISODES=200, NUM_AGENTS=2, MAX_CYCLES_PER_EP=300, use_copier=True, continue_from_checkpoint=True, copier_ep_lookback=50, checkpoint_ep=200)
    