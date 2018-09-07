import numpy as np

mu = np.array([6,2,5,3.5,4,1,1.6,2,7,11])

def generator(a):
    d = (np.random.normal(0., 10, 10)+mu)
    return d[a]

def At(Qt):
    temp_max = np.argmax(Qt)
    return temp_max

def greedy(ep, Qt, t, iteration):
    rd = np.random.rand()
    if (iteration == 0):
        for i in range(len(mu)):
            Qt[i] = generator(i)
            t[i] = 1
    elif (rd>=ep):
        max_ind = At(Qt)
        Qt[max_ind] = Qt[max_ind]+(generator(max_ind)-Qt[max_ind])/t[max_ind]
        t[max_ind] += 1
    elif (rd<ep):
        max_ind = At(Qt)
        ran_ind = np.random.randint(len(mu))
        while (max_ind == ran_ind):
            ran_ind = np.random.randint(len(mu))

        Qt[ran_ind] = Qt[ran_ind]+(generator(ran_ind)-Qt[ran_ind])/t[ran_ind]
        t[ran_ind] += 1

    iteration += 1

    return Qt, t, iteration


Qt = [0,0,0,0,0,0,0,0,0,0]
t = [0,0,0,0,0,0,0,0,0,0]
ep = 0.6
iteration = 0

while iteration<1e+5:
    Qt, t, iteration = greedy(ep, Qt, t, iteration)
print(Qt)
print(iteration)
