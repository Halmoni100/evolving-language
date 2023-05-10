from run_simple_spread import *

if __name__ == '__main__':
    main(300,3,200,False,True,checkpoint_ep=200)
    main(300,3,200,True,True, copier_ep_lookback=1000, checkpoint_ep=200)
    main(300,5,200,False,True,checkpoint_ep=200)
    main(300,5,200,True,True, copier_ep_lookback=1000, checkpoint_ep=200)
    main(300,10,200,False,True,checkpoint_ep=200)
    main(300,10,200,True,True, copier_ep_lookback=1000, checkpoint_ep=200)