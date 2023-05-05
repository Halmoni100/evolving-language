from run_simple_spread import *

if __name__ == '__main__':

    main(500,2,200,True,False, copier_ep_lookback=2000)
    main(500,3,200,False,False)
    main(500,3,200,True,False, copier_ep_lookback=2000)