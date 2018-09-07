import numpy as np 
import matplotlib.pyplot as plt
p = 0.4

v = np.zeros(101)
theta = 1e-4
gamma = 0.9
astar = np.zeros(101)

while(True):
    delta = 0
    for i in range(1,100):
        tv = v[i]
        maxv = 0
        for j in range(0,min(i,100-i)+1):
            ts1 = i - j
            r1 = (ts1==100)
            ts2 = i + j
            r2 = (ts2==100)

            e = p*(r1+gamma*v[ts1])+(1-p)*(r2+gamma*v[ts2])
            if maxv<e:
                maxv=e 
                astar[i]=j
        
        v[i] = maxv
        delta = max(delta, abs(tv-v[i]))

    if delta<theta:
        break

print(astar)
plt.plot(range(101),v)
plt.show()