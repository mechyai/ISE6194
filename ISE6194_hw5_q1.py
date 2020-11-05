import matplotlib.pyplot as plt
import numpy as np
import random
import math


def get_action() -> int:
    # return random walk action
    return random.choice(np.arange(-100, 101))


def get_state(state: int, action: int) -> int:
    # return s' or declare termination with 0 (left) or 1001 (right)
    if state <= 100 and action <= -state: return 0;
    if state >= 901 and action > 1000 - state: return 1001;
    return state + action


def get_reward(state: int) -> int:
    if state == 0: return -1;
    elif state == 1001: return 1;
    return 0


# ------------------------------ TRUE VALUE FUNCTION (every-visit MC prediction) ------------------------------
value = np.zeros(1002)  # index = state location
avg_cnt = np.ones(1002)  # n for incremental avg

episodes = 100000
gamma = 1

for i in range(episodes):
    state_list = []
    trajectory = []  # track s,r pairs
    s = 500  # initial state
    # episode generation
    while True:
        s_ = s
        a = get_action()
        s = get_state(s_, a)
        r = get_reward(s)
        # print(f'State:{s_}, Action: {a}, State:{s}, Reward:{r}, ')
        trajectory.append((s_, r))  # ! store (s_t, r_t+1) pairs
        state_list.append(s_)
        if r != 0:
            break  # episode termination

    # value estimation, every-visit MC
    G = 0
    for pair in reversed(trajectory):
        G = gamma*G + pair[1]
        state = state_list.pop()
        # incremental avg return
        G_avg0 = value[state]
        value[state] = G_avg0 + (1 / avg_cnt[state])*(G - G_avg0)
        # print(f'Value: {value[state]}')
        avg_cnt[state] += 1

# plt.plot(range(1000), value[1:-1], label='\'True\' Value (MC Every Visit)')


# ------------------------------ LINEAR FUNCTION APPROXIMATION ------------------------------

def x_feat1(state: int) -> np.ndarray:
    # STATE AGGREGATION - produce group one hot encoding feature vector
    x = np.zeros(10)
    group = (state-1)//100
    x[group] = 1
    return x


def x_feat2(state: int, n: int = 5, k: int = 1) -> np.ndarray:
    # POLYNOMIAL BASIS
    norm_s = (state - 1) / (1000 - 1)
    x = np.zeros(n + k)
    for i in range(n + k):
        x[i] = norm_s ** i
    return x


def x_feat3(state: int, n: int = 5, k: int = 1) -> np.ndarray:
    # FOURIER BASIS
    norm_s = (state - 1) / (1000 - 1)
    x = np.zeros(n + k)
    for i in range(n + k):
        x[i] = math.cos(math.pi * norm_s * i)
    return x


# initialize weight vector
w1 = np.zeros(10)
w2, w3 = np.zeros(6), np.zeros(6)

# params
episodes = 5000
alpha1 = 0.00002
alpha2 = 0.0001
alpha3 = 0.00005

for i in range(episodes):
    state_list = []
    trajectory = []  # track s,r pairs
    returns = []
    s = 500  # initial state
    # episode generation
    while True:
        s_ = s
        a = get_action()
        s = get_state(s_, a)
        r = get_reward(s)
        # print(f'State:{s_}, Action: {a}, State:{s}, Reward:{r}, ')
        trajectory.append((s_, r))  # ! store (s_t, r_t+1) pairs
        state_list.append(s_)
        if r != 0:
            # print(f'Termination: State:{s_}, Reward:{r}')
            break  # episode termination

    # value function approximation, state aggregation - gradient MC

    # get cumulative return list
    G = 0
    for pair in reversed(trajectory):
        G += pair[1]  # accumulate reward to return
        returns.append(G)
    returns.reverse()  # in order t=0 -> t=T

    # weight gradient decent
    i = 0  # incrementer of return
    for pair in trajectory:  # in order t=0 -> t=T
        s = pair[0]
        G = returns[i]
        # feature vector
        x1 = x_feat1(s)
        x2 = x_feat2(s)
        x3 = x_feat3(s)
        # value
        v1 = np.dot(w1, x1)
        v2 = np.dot(w2, x2)
        v3 = np.dot(w3, x3)
        # gradient
        grad_w1 = x1
        grad_w2 = x2
        grad_w3 = x3
        # weight update
        dw1 = alpha1 * (G - v1) * grad_w1
        w1 = w1 + dw1
        dw2 = alpha2 * (G - v2) * grad_w2
        w2 = w2 + dw2
        dw3 = alpha3 * (G - v3) * grad_w3
        w3 = w3 + dw3
        # iterator
        i += 1

value1 = np.zeros(1000)
# STATE AGGREGATION value function
for group in range(10):
    v1 = w1[group]
    print(f'Group {group} - State Agr. Val: {v1}')
    for s in range(len(value1)):
        if s // 100 == group:
            value1[s] = v1
        else:
            continue

# POLYNOMIAL BASIS value function
value2 = np.zeros(1000)
for s in range(len(value2)):
    x2 = x_feat2(s + 1)
    v2 = np.dot(w2, x2)
    value2[s] = v2

# FOURIER BASIS value function
value3 = np.zeros(1000)
for s in range(len(value3)):
    x3 = x_feat3(s + 1)
    v3 = np.dot(w3, x3)
    value3[s] = v3

states = np.arange(1000) + 1
plt.plot(range(1000), value[1:-1], label='\'True\' Value (MC Every-Visit)')
plt.plot(states, value1, label='State Aggregation')
plt.plot(states, value2, label='Polynomial Basis')
plt.plot(states, value3, label='Fourier Basis')
plt.xlabel('State Position')
plt.ylabel('State Value')
plt.title('Value Function Approximation Comparison')
plt.legend()
plt.show()

