import numpy as np

def numToss(start, goal, trials):
    alist=[]
    for i in range (trials):
        toss=0
        balance=start
        while (balance<goal and balance>0):
            toss+=1
            if (balance>0 and (balance*2)<=goal):
                if (np.random.uniform(0,1) < 0.5):
                    balance*2
                else:
                    balance -= balance
            elif (start>0 and (balance*2)>goal):
                if (np.random.uniform(0,1) < 0.5):
                    a=goal-balance
                    balance+=a
            else:
                return 0
        alist.append(toss)
        mean= np.mean(alist)
    print "Mean # of coin tosses was",mean,"to reach $",goal," or go broke." 
