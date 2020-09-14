import numpy as np
import random

import emab_model
from feedbal_algorithm import check_dropout_cond
from user_arrival_simulator import DROPOUT_PROB, DROPOUT_WEEK
from user import User

"""
This script is responsible for assigning one TS agent to each week and running the M-TS algorithm.
"""


class ThompsonAgent:
    def __init__(self, arm_list):
        self.scaling_factor = 1
        self.action_num_played_arr = np.zeros(len(arm_list))
        self.is_action_initialized_arr = np.zeros(len(arm_list))
        self.num_rounds = 0
        self.action_rewards = np.zeros(len(arm_list))
        self.alpha_beta_of_arms = np.ones((2, len(arm_list)))
        self.action_list = arm_list

    def get_best_action(self):
        if 0 in self.is_action_initialized_arr:
            not_initialized_action = random.choice(np.where(self.is_action_initialized_arr == 0)[0])
            return not_initialized_action
        # sample posterior of all arms
        samples = np.random.beta(self.alpha_beta_of_arms[0], self.alpha_beta_of_arms[1])
        return samples.argmax()

    def update_rewards(self, action_index, feedback, cost, cost_scale_factor, t, x):
        cost_scale_factor = 1 if cost_scale_factor == 0 else cost_scale_factor
        reward = feedback / self.scaling_factor - cost / cost_scale_factor
        reward = 0 if reward < 0 or feedback / self.scaling_factor < abs(cost / cost_scale_factor) else reward
        self.scaling_factor = max(self.scaling_factor, feedback)
        if reward > 1:
            reward = 1
        self.action_num_played_arr[action_index] += 1
        self.num_rounds += 1
        self.is_action_initialized_arr[action_index] = 1
        temp = np.random.binomial(1, reward)
        self.alpha_beta_of_arms[0][action_index] += temp
        self.alpha_beta_of_arms[1][action_index] += (1 - temp)


def run_smart_ts(emab_episode: emab_model.AbstractEmabEpisode, num_weeks, user_arrivals, scale_factors=None):
    user_set = set()
    removal_set = set()
    week = 1
    num_users_added = 0
    num_dropouts = 0
    action_feedback_user_list = []  # this list keeps track of the feedback received when taking action with user

    # list of list of actions taken by each user
    actions_taken_arr_arr = np.zeros((len(user_arrivals), num_weeks - 1, len(emab_episode.action_set)))

    # list of app sessions in a given week (of all users)
    user_group_app_sessions_dict = {}  # Maps user group to list of session counts (per week)
    for i in range(num_weeks):
        user_group_app_sessions_dict[i] = [0] * (num_weeks - 1)

    # Assign agent to each step
    ts_agent_arr = []
    for i in range(emab_episode.l_max):
        ts_agent_arr.append(ThompsonAgent(emab_episode.action_set))
        if scale_factors is not None and i + 1 < emab_episode.l_max:
            ts_agent_arr[-1].scaling_factor = scale_factors[i + 1]

    while week == 1 or len(user_set) > 0:
        if week <= num_weeks:
            # Observe arrived users
            for _ in range(user_arrivals[week - 1]):
                user_set.add(User(week - 1, num_users_added, week, emab_episode))
                num_users_added += 1

        # Play app for each user
        removal_set.clear()
        action_feedback_user_list.clear()
        for user in user_set:
            # Get TS agent for this week
            ts_agent = ts_agent_arr[user.emab_episode.t - 1]

            # use prev week's scaling factor as feedback increases monotone
            if user.emab_episode.t > 1 and ts_agent.scaling_factor == 1:
                ts_agent.scaling_factor = ts_agent_arr[user.emab_episode.t - 2].scaling_factor
            best_action = ts_agent.get_best_action()
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
                action_feedback_user_list.append((best_action, feedback, user, ts_agent))
                user_group_app_sessions_dict[user.group_id][user.emab_episode.t - 2] += feedback / user_arrivals[
                    user.group_id]
        for action, feedback, user, ts_agent in action_feedback_user_list:
            ts_agent.update_rewards(action, feedback, user.emab_episode.cost_arr[-1],
                                    user.emab_episode.state_arr[-2] / 2, user.emab_episode.t - 1,
                                    user.emab_episode.state_arr[-2])
        user_set = user_set - removal_set
        week += 1

    output = {
        "actions_taken_arr_arr": actions_taken_arr_arr,
        'num_dropouts': num_dropouts,
        "user_group_app_sessions_dict": user_group_app_sessions_dict}
    return output
