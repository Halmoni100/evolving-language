from run_simple_spread import *

if __name__ == '__main__':
    #main(200,2,200,False,False)
    main(200,2,200,True,False, copier_ep_lookback=1000)
    #main(200,3,200,False,False)
    main(200,3,200,True,False, copier_ep_lookback=1000)
    main(200,5,200,False,False)
    main(200,5,200,True,False, copier_ep_lookback=1000)
    main(200,10,200,False,False)
    main(200,10,200,True,False, copier_ep_lookback=1000)