import numpy as np

# import matplotlib.pyplot as plt
import math

# from scipy.optimize import minimize
# import matlab.engine
import random
import os


def output_process(input_cul, output_cul, output_time, pre_output_process=0):
    # result is the number of bits that outputs in output_time and arrives at each time prior to the output time
    result = np.zeros([output_time + 1])
    if output_time == 0:
        result[output_time] = output_cul[output_time]
    else:
        for input_time in range(output_time + 1):
            if input_time == 0:
                result[input_time] = np.min(
                    [output_cul[output_time], input_cul[input_time]]
                ) - np.sum(pre_output_process[input_time, :])
            else:
                # arrives at time_index, leave at output_time
                result[input_time] = np.min(
                    [
                        np.max(
                            [output_cul[output_time] - input_cul[input_time - 1], 0]
                        ),
                        input_cul[input_time] - input_cul[input_time - 1],
                    ]
                ) - np.sum(pre_output_process[input_time, :])
    return result


# RANDOMLY SELECT WHETHER TO OFFLOAD AND WHICH edge TO OFFLOAD
class RandomSelction:
    def __init__(self, episode, env):
        self.n_actions = env.n_actions
        self.n_features = env.n_features
        self.reward_store = np.zeros([episode, env.n_time, env.n_iot])
        self.delay_store = np.zeros([episode, env.n_time, env.n_iot])
        self.action_store = -np.ones([episode, env.n_time, env.n_iot])

    def random_choose_action(self, observation, lstm_state, iot):
        if observation[0] != 0:
            action = np.random.randint(0, self.n_actions)
        else:
            action = 0

        return action

    def do_store_reward(self, episode, time, iot, reward):
        self.reward_store[episode, time, iot] = reward

    def do_store_delay(self, episode, time, iot, delay):
        self.delay_store[episode, time, iot] = delay

    def do_store_action(self, episode, time, iot, action):
        self.action_store[episode, time, iot] = action


# ALWAYS NOT OFFLOAD
class NotOffload:
    def __init__(self, episode, env):
        self.n_actions = env.n_actions
        self.n_features = env.n_features
        self.reward_store = np.zeros([episode, env.n_time, env.n_iot])
        self.delay_store = np.zeros([episode, env.n_time, env.n_iot])
        self.action_store = -np.ones([episode, env.n_time, env.n_iot])

    def random_choose_action(self, observation, lstm_state, iot):
        # 0 means not offload
        action = 0

        return action

    def do_store_reward(self, episode, time, iot, reward):
        self.reward_store[episode, time, iot] = reward

    def do_store_delay(self, episode, time, iot, delay):
        self.delay_store[episode, time, iot] = delay

    def do_store_action(self, episode, time, iot, action):
        self.action_store[episode, time, iot] = action


# ALWAYS OFFLOAD
class RandomOffload:
    def __init__(self, episode, env):
        self.n_actions = env.n_actions
        self.n_features = env.n_features
        self.reward_store = np.zeros([episode, env.n_time, env.n_iot])
        self.delay_store = np.zeros([episode, env.n_time, env.n_iot])
        self.action_store = -np.ones([episode, env.n_time, env.n_iot])

    def random_choose_action(self, observation, lstm_state, iot):
        # Suppose n_actions = 4; [0 1 0 0], so randomly among the later two
        if observation[0] != 0:
            action = np.random.randint(1, self.n_actions)
        else:
            action = 0

        return action

    def do_store_reward(self, episode, time, iot, reward):
        self.reward_store[episode, time, iot] = reward

    def do_store_delay(self, episode, time, iot, delay):
        self.delay_store[episode, time, iot] = delay

    def do_store_action(self, episode, time, iot, action):
        self.action_store[episode, time, iot] = action


# POTENTIAL GAME, SELECT THE STRATEGY THAT OPTIMIZE THE PAYOFF GIVEN THE PREVIOUS STRATEGIES
class PotentialGame:
    def __init__(self, episode, env):
        self.n_actions = env.n_actions
        self.n_features = env.n_features

        self.epsilon = 0

        self.past_action = np.zeros(env.n_iot)

        self.reward_store = np.zeros([episode, env.n_time, env.n_iot])
        self.delay_store = np.zeros([episode, env.n_time, env.n_iot])
        self.action_store = -np.ones([episode, env.n_time, env.n_iot])
        self.count = 0
        self.process_duration = np.zeros([env.n_iot, env.n_actions])
        self.process_duration_average = np.zeros([env.n_iot, env.n_actions])
        self.action_duration_belief = np.zeros([env.n_iot, env.n_actions])

    def random_choose_action(self, env, iot):
        self.action_duration_belief[iot, 0] = self.process_duration_average[iot, 0]
        for i in range(env.n_edge):
            self.action_duration_belief[iot, i + 1] = (
                self.process_duration_average[iot, i + 1] * env.queue_length_edge[i]
            )

        if np.random.uniform() < self.epsilon:
            action = np.argmin(self.action_duration_belief[iot, :])

        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def do_store_reward(self, episode, time, iot, reward):
        self.reward_store[episode, time, iot] = reward

    def do_store_transition(self, env):
        edge = env.action - 1
        self.count += 1
        if env.action == 0 or env.queue_length_edge[edge] == 0:
            self.process_duration[env.current_iot][env.action] += env.current_duration
        else:
            self.process_duration[env.current_iot][env.action] += (
                env.current_duration / env.queue_length_edge[edge]
            )
        self.process_duration_average[env.current_iot][env.action] = (
            self.process_duration[env.current_iot][env.action] / self.count
        )

    def do_store_action(self, episode, time, iot, action):
        self.action_store[episode, time, iot] = action
