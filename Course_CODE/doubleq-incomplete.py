import numpy as np
import matplotlib.pyplot as plt

class MDP(object):

    def __init__(self, mu=-0.1, sigma=1, k=5):
        self.mu = mu
        self.sigma = sigma
        self.s = None
        self.k = k

    def reset(self):
        self.s = 0
        return self.s

    def step(self, a):
        assert(self.s!=2)
        # s==0 means circle A, s==1 means circle B, s==2 means terminal
        # a==0 means left, a==1 means right
        if self.s == 0 and a == 1:
            self.s = 2
            r = 0
            t = True
        elif self.s == 0 and a == 0:
            self.s = 1
            r = 0
            t = False
        else:
            self.s = 2
            r = np.random.normal(self.mu, self.sigma)
            t = True

        return (r, self.s, t)

k=5
env = MDP(k=5)
a_big_number = 100
epsilon = 0.1
alpha = 0.1
gamma = 0.99

left_count = 0

Q = np.zeros([3, k+2])
Q[0,2:k+2] = -a_big_number
Q[1,0:2] = -a_big_number
print(Q)

for i in range(300):
    s = env.reset()
    t = False
    while(not t):
    # TODO: Q-learning Algorithm
        rand = np.random.rand()
        if (rand>epsilon):
            max_a = np.argmax(Q[s])
        elif (s==0):
            max_a = np.random.randint(0,2)
        elif (s==1):
            max_a = np.random.randint(2,k+2)
        r, ts, t = env.step(max_a)
        max_Q = np.max(Q[ts])
        Q[s, max_a] +=  alpha*(r+gamma*max_Q-Q[s,max_a])
        s = ts
#     	max_a = np.argmax(Q[s])
#     	r, ts, t = env.step(max_a)
#     	max_Q = 0
#         for i in range(k+2):
#             if (max_Q < Q[ts, i]):
#                 max_Q = Q[ts, i]
# 	    Q[s,max_a] += alpha*(r+gamma*max_Q-Q[s,max_a]
# 	    if ts==2:
# 	        t = True
# 	    s = ts

print(Q)
