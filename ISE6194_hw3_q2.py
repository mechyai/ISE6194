import random
import numpy as np
import matplotlib.pyplot as plt


def init_val(mean: float, variance: float):
    return np.random.normal(mean, variance)


def action_list(state: int):
    max_bid = min(state, 100-state)
    return list(range(max_bid+1))


def get_reward(state: int, goal: int):
    # return +1 if goal reached, 0 otherwise
    if state is goal:
        return 1
    return 0


# CONSTANTS
goal = 100
ph = 0.55  # prob(heads)
gamma = 1  # discount factor

# ----------------------------------------------- VALUE ITERATION 2(a) -----------------------------------------------
# INITIALIZATION
theta = 0.0001  # threshold
val_mean, val_var = 0, 0
state_val = {}
optimal_action = {}
# create state value table (dictionary)
for i in range(goal+1):
    if i is 0 or i is goal:
        state_val[i] = float(0)
    else:
        state_val[i] = init_val(val_mean, val_var)

delta = theta + 1
loop_cnt = 0
while delta > theta:
    # loop thru all states
    delta = 0
    for s in range(len(state_val)-1):
        if s is not 0 or s is not goal:
            v_s = state_val[s]
            max_val = -1000
            # loop thru all actions to find max action
            actions = action_list(s)  # action set for a given state
            for a in actions:
                # find maximal action value
                next_win_state = s + a
                next_lose_state = s - a
                reward_win = get_reward(next_win_state, goal)
                reward_lose = get_reward(next_lose_state, goal)
                val = ph*(reward_win + gamma * state_val[next_win_state]) + \
                      (1 - ph)*(reward_lose + gamma * state_val[next_lose_state])
                if val > max_val:
                    max_val = val
                    max_action = a

            optimal_action[s] = max_action  # keep track of best action per state per sweep
            state_val[s] = max_val
            val_dif = abs(v_s-state_val[s])
            delta = max(delta, val_dif)
        print(f'Delta Val: {delta} at loop #: {loop_cnt}')
    loop_cnt += 1


# clean up dictionaries for plotting
optimal_action.pop(0)
state_val.pop(0)
state_val.pop(100)
# plotting
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle(f'Coin Toss Results \n (ph={ph}, theta={theta})', fontsize=22)
plt.xlabel('Capital ($)')
ax1.plot(*zip(*sorted(state_val.items())), label='Value Iteration')
ax1.set_title('Bidding Value Estimates', fontsize=18)
ax1.set(ylabel='Value Estimates')
ax2.plot(*zip(*sorted(optimal_action.items())))
ax2.set_title('Final Policy', fontsize=18)
ax2.set(ylabel='Stake ($)')
plt.show

# ----------------------------------------------- MONTE CARLO 2(a) -----------------------------------------------

class CoinFlipEnv:

    def __init__(self, goal: int, p_heads: float):
        self.state_space = np.arange(0, goal+1)
        self.state = random.choice(self.state_space[1:-1])  # initial random starts from 1-99
        self.goal = goal
        self.ph = p_heads
        self.st_reward = 0  # state transition reward
        self.g_reward = 1  # goal reward

    def coin_flip(self):
        """
        Coin flip function, returns True if flip results in heads, returns False if tails

        :param prob_heads: (float) probability of coin landing on heads (not necessarily 50/50 coin)
        :return: (boolean) result of coin flip, True = heads, False = tails
        """
        flip = random.uniform(0, 1)
        if flip < self.ph:
            return True
        else:
            return False

    def get_state(self, action):
        if self.coin_flip():
            # win
            self.state += action
        else:
            # lose
            self.state -= action
        return self.state

    def get_reward(self):
        if self.state == self.goal:
            return self.g_reward
        return self.st_reward

    def terminate(self):
        # episode termination, win/lose
        if self.state == 0 or self.state == self.goal:
            return True
        else:
            return False


class CoinFlipAgent:

    def __init__(self, goal):
        self.goal = goal
        self.value = self.init_val()
        self.returns = self.init_returns()

    def init_val(self):
        val_mean, val_var = 0, 0
        state_val = {}
        # create state value table (dictionary)
        for i in range(self.goal + 1):
            if i is 0 or i is self.goal:
                state_val[i] = float(0)
            else:
                state_val[i] = np.random.normal(val_mean, val_var)
        return state_val

    def init_returns(self):
        returns = {}
        # create returns table (dictionary) of empty lists
        for i in range(self.goal + 1):
            if i is 0 or i is self.goal:
                returns[i] = [0]
            else:
                returns[i] = []  # empty list to append return
        return returns

    def get_action(self, optimal_actions, state):
        # optimal policy found from Value Iteration
        return optimal_actions[state]

episodes = 500
agent = CoinFlipAgent(goal)
for i in range(episodes):
    state_list = []
    reward_list = []
    # init
    env = CoinFlipEnv(goal, ph)
    # run episode
    print(f'Start State:{env.state}')
    while not env.terminate():
        # random policy
        action = agent.get_action(optimal_action, env.state)
        new_state = env.get_state(action)
        reward = env.get_reward()
        # memory
        state_list.append(new_state)
        reward_list.append(reward)
    # after episode
    G = 0
    while len(state_list) > 0:
        # in reverse, remove as encounter
        G = gamma*G + reward_list.pop()
        state = state_list.pop()
        if state not in state_list:
            agent.returns[state].append(G)
            # incremental average
            n = len(agent.returns[state])
            v_0 = agent.value[state]
            v_1 = v_0 + (1/n)*(G - v_0)
            agent.value[state] = v_1

ax1.plot(*zip(*sorted(agent.value.items())), label=f'First-Visit Monte Carlo, {episodes} Episodes')
leg = ax1.legend(loc="lower right", ncol=1, shadow=True, fancybox=True)
plt.show




