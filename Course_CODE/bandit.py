import numpy as np 

def generator(a):
    mu = np.array([3, 4, 0, 1, 2, 3.5, 2.5, 3, 1.5, 0.5])
    d = (np.random.normal(0., 1, 10)+mu)
    return d[a]

def greedy(q, t):
    epsilon = 0.1
    p = np.random.random()
    if p < epsilon:
        a = np.random.randint(0,10)
    else:
        a = np.argmax(q)
    return a

def ucb(q, t):
    c = 0.1
    delta = 1e-4
    if np.sum(t) < 1:
        p = q
    else:
        p = q + c*np.sqrt(np.log(np.sum(t)))/(t+delta)
        # print(p)
    a = np.argmax(p)
    return a

# epsilon-greedy
q = np.zeros(10)
t = np.zeros(10)
R = 0.
for i in range(500):
    # a = greedy(q, t)
    a = ucb(q, t)
    r = generator(a)
    R = R + r
    t[a] = t[a] + 1
    q[a] = q[a] + (r-q[a])/t[a]

print(q)
print(R)