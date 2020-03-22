import numpy as np
import random

import emab_model
from feedbal_algorithm import check_dropout_cond
from user_arrival_simulator import DROPOUT_PROB, DROPOUT_WEEK
from user import User

"""
This script is responsible for assigning one epsilon greedy agent to each week and running the M-epsilon algorithm.
"""


class EpsilonAgent:
    C = 0.6
    D = 0.1

    def __init__(self, arm_list):
        self.action_num_played_arr = np.zeros(len(arm_list))
        self.is_action_initialized_arr = np.zeros(len(arm_list))
        self.num_rounds = 0
        self.action_rewards = np.zeros(len(arm_list))
        self.action_list = arm_list

    def get_best_action(self):
        if 0 in self.is_action_initialized_arr:
            not_initialized_action = random.choice(np.where(self.is_action_initialized_arr == 0)[0])
            return not_initialized_action
        epsilon = min(1.0, EpsilonAgent.C * len(self.action_list) / (EpsilonAgent.D ** 2 * self.num_rounds))
        if np.random.binomial(1, 1 - epsilon) == 1:
            mean_term_list = []
            for i in range(len(self.action_list)):
                num_times_played = self.action_num_played_arr[i]
                mean_term_list.append(self.action_rewards[i] / num_times_played)
            return self.action_list[mean_term_list.index(max(mean_term_list))]
        else:
            return random.choice(self.action_list)

    def update_rewards(self, action_index, reward):
        self.action_num_played_arr[action_index] += 1
        self.num_rounds += 1
        self.is_action_initialized_arr[action_index] = 1
        self.action_rewards[action_index] += reward


def run_m_epsilon(emab_episode: emab_model.AbstractEmabEpisode, num_weeks, user_arrivals):
    user_set = set()
    removal_set = set()
    week = 1
    num_users_added = 0
    num_dropouts = 0
    action_reward_list = []  # this list keeps track of the reward of taking an action

    # list of list of actions taken by each user
    actions_taken_arr_arr = np.zeros((len(user_arrivals), num_weeks - 1, len(emab_episode.action_set)))

    # list of app sessions in a given week (of all users)
    user_group_app_sessions_dict = {}  # Maps user group to list of session counts (per week)
    for i in range(num_weeks):
        user_group_app_sessions_dict[i] = [0] * (num_weeks - 1)

    # Assign agent to each step
    epsilon_agent_arr = []
    for i in range(emab_episode.l_max):
        epsilon_agent_arr.append(EpsilonAgent(emab_episode.action_set))

    while week == 1 or len(user_set) > 0:
        if week <= num_weeks:
            # Observe arrived users
            for _ in range(user_arrivals[week - 1]):
                user_set.add(User(week - 1, num_users_added, week, emab_episode))
                num_users_added += 1

        # Play app for each user
        removal_set.clear()
        action_reward_list.clear()
        for user in user_set:
            # Get agent for this week
            epsilon_agent = epsilon_agent_arr[user.emab_episode.t - 1]
            best_action = epsilon_agent.get_best_action()
            feedback = user.emab_episode.perform_action(best_action)
            usr_episode = user.emab_episode
            if feedback == -1 or usr_episode.t > DROPOUT_WEEK and check_dropout_cond(usr_episode) and np.random.binomial(1, DROPOUT_PROB) == 1:
                if feedback != -1:
                    for i, action in enumerate(user.emab_episode.action_taken_arr):
                        actions_taken_arr_arr[user.group_id, i, action] += 1
                    num_dropouts += 1
                else:
                    for i, action in enumerate(user.emab_episode.action_taken_arr[:-1]):
                        actions_taken_arr_arr[user.group_id, i, action] += 1
                removal_set.add(user)
            else:
                reward = feedback - user.emab_episode.cost_arr[-1]
                action_reward_list.append((best_action, reward))
                user_group_app_sessions_dict[user.group_id][user.emab_episode.t - 2] += feedback / user_arrivals[
                    user.group_id]   #(t-1) comes from the fact that when perform_action is called t is incremented by 1 + we are indexing an array
        # update means
        for action, reward in action_reward_list:
            epsilon_agent.update_rewards(action, reward)
        user_set = set(filter(lambda item: item not in removal_set, user_set))
        week += 1

    output = {
        "actions_taken_arr_arr": actions_taken_arr_arr,
        "num_dropouts": num_dropouts,
        "user_group_app_sessions_dict": user_group_app_sessions_dict}
    return output
