import random

import numpy as np
import pandas as pd
import surprise
from surprise import Reader, Dataset

import emab_model
from feedbal_algorithm import check_dropout_cond
from user import User
from user_arrival_simulator import DROPOUT_PROB, DROPOUT_WEEK

"""
This script has code for the CF-based algorithms (KNN and SVD).
"""

COL_NAMES = ["user", "item", "feedback", "cost", "reward"]
READER = Reader(rating_scale=(0, 1))


class ColabFiltering:
    def __init__(self, arm_list, cf_params=None):
        # To use item-based cosine similarity
        sim_options = {
            "name": "pearson",
            "user_based": False,  # Compute  similarities between items
        }
        if cf_params:
            self.algo = surprise.SVD() if cf_params['algo'] == 'svd' else surprise.KNNBaseline(sim_options=sim_options, verbose=False)

        self.first_train = False
        self.data = []
        self.action_num_played_arr = np.zeros(len(arm_list))
        self.is_action_initialized_arr = np.zeros(len(arm_list))
        self.num_rounds = 0
        self.action_rewards = np.zeros(len(arm_list))
        self.action_list = arm_list

    def get_best_action(self, user_id):
        if 0 in self.is_action_initialized_arr:
            not_initialized_action = random.choice(np.where(self.is_action_initialized_arr == 0)[0])
            return not_initialized_action
        if not self.first_train:
            self.train()
            self.first_train = True

        test = self.algo.predict(user_id, 0)
        if test.details['was_impossible']:
            return random.choice(self.action_list)
        ratings = [self.algo.predict(user_id, x).est for x in self.action_list]
        return np.argmax(ratings)

    def update_rewards(self, user_id, action_index, feedback, cost, cost_scale_factor):
        # cost_scale_factor = 1 if cost_scale_factor == 0 else cost_scale_factor
        # cost = cost / cost_scale_factor
        self.action_num_played_arr[action_index] += 1
        self.num_rounds += 1
        self.is_action_initialized_arr[action_index] = 1
        self.data.append([user_id, action_index, feedback, cost, feedback - cost])

    def train(self):
        if len(self.data) > 0:
            df = pd.DataFrame(self.data, columns=COL_NAMES)
            # max_cost = df["cost"].max()
            # max_cost = 1 if max_cost == 0 else max_cost
            # df["reward"] = df["feedback"] / df["feedback"].max() - df["cost"] / max_cost
            df["reward"] = (df["reward"] - df["reward"].min()) / (df["reward"].max() - df["reward"].min())
            # df["reward"] = df["reward"].clip(lower=0, upper=1)

            # Loads Pandas dataframe
            data = Dataset.load_from_df(df[["user", "item", "reward"]], READER)

            training_set = data.build_full_trainset()
            self.algo.fit(training_set)


def cf(emab_episode: emab_model.AbstractEmabEpisode, num_weeks, user_arrivals, cf_params):
    user_set = set()
    removal_set = set()
    week = 1
    num_users_added = 0
    num_dropouts = 0
    action_feedback_user_list = []  # this list keeps track of the feedback received when taking action with user

    # list of list of actions taken by each user
    actions_taken_arr_arr = np.zeros((len(user_arrivals), num_weeks - 1, len(emab_episode.action_set)))

    # list of app sessions in a given week (of all users)
    user_group_app_sessions_dict = np.zeros(
        (num_weeks, num_weeks - 1))  # Maps user group to list of session counts (per week)
    # for i in range(num_weeks):
    #     user_group_app_sessions_dict[i] = [0] * (num_weeks - 1)

    main_agent = ColabFiltering(emab_episode.action_set, cf_params=cf_params)
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
            best_action = main_agent.get_best_action(user.user_id)
            feedback = user.emab_episode.perform_action(best_action)
            usr_episode = user.emab_episode
            if feedback == -1 or usr_episode.t > DROPOUT_WEEK and check_dropout_cond(
                    usr_episode) and np.random.binomial(1, DROPOUT_PROB) == 1:
                if feedback != -1:
                    for i, action in enumerate(user.emab_episode.action_taken_arr):
                        actions_taken_arr_arr[user.group_id, i, action] += 1
                    num_dropouts += 1
                else:
                    for i, action in enumerate(user.emab_episode.action_taken_arr[:-1]):
                        actions_taken_arr_arr[user.group_id, i, action] += 1
                removal_set.add(user)
            else:
                action_feedback_user_list.append((best_action, feedback, user))
                user_group_app_sessions_dict[user.group_id][user.emab_episode.t - 2] += feedback / user_arrivals[
                    user.group_id]
        for action, feedback, user in action_feedback_user_list:
            main_agent.update_rewards(user.user_id, action, feedback, user.emab_episode.cost_arr[-1],
                                      user.emab_episode.state_arr[-2] / 2)
        user_set = user_set - removal_set
        main_agent.train()
        week += 1

    output = {
        "actions_taken_arr_arr": actions_taken_arr_arr,
        'num_dropouts': num_dropouts,
        "user_group_app_sessions_dict": user_group_app_sessions_dict}
    return output


def smart_cf(emab_episode: emab_model.AbstractEmabEpisode, num_weeks, user_arrivals, cf_params):
    user_set = set()
    removal_set = set()
    week = 1
    num_users_added = 0
    num_dropouts = 0
    agents_to_update = set()
    action_feedback_user_list = []  # this list keeps track of the feedback received when taking action with user

    # list of list of actions taken by each user
    actions_taken_arr_arr = np.zeros((len(user_arrivals), num_weeks - 1, len(emab_episode.action_set)))

    # list of app sessions in a given week (of all users)
    user_group_app_sessions_dict = np.zeros(
        (num_weeks, num_weeks - 1))  # Maps user group to list of session counts (per week)
    # for i in range(num_weeks):
    #     user_group_app_sessions_dict[i] = [0] * (num_weeks - 1)

    # Assign agent to each step
    cf_agent_arr = []
    for i in range(emab_episode.l_max):
        cf_agent_arr.append(ColabFiltering(emab_episode.action_set, cf_params=cf_params))

    while week == 1 or len(user_set) > 0:
        if week <= num_weeks:
            # Observe arrived users
            for _ in range(user_arrivals[week - 1]):
                user_set.add(User(week - 1, num_users_added, week, emab_episode))
                num_users_added += 1

        # Play app for each user
        removal_set.clear()
        agents_to_update.clear()
        action_feedback_user_list.clear()
        for user in user_set:
            # Get TS agent for this week
            cf_agent = cf_agent_arr[user.emab_episode.t - 1]

            best_action = cf_agent.get_best_action(user.user_id)
            feedback = user.emab_episode.perform_action(best_action)
            usr_episode = user.emab_episode
            if feedback == -1 or usr_episode.t > DROPOUT_WEEK and check_dropout_cond(
                    usr_episode) and np.random.binomial(1, DROPOUT_PROB) == 1:
                if feedback != -1:
                    for i, action in enumerate(user.emab_episode.action_taken_arr):
                        actions_taken_arr_arr[user.group_id, i, action] += 1
                    num_dropouts += 1
                else:
                    for i, action in enumerate(user.emab_episode.action_taken_arr[:-1]):
                        actions_taken_arr_arr[user.group_id, i, action] += 1
                removal_set.add(user)
            else:
                action_feedback_user_list.append((best_action, feedback, user, cf_agent))
                user_group_app_sessions_dict[user.group_id][user.emab_episode.t - 2] += feedback / user_arrivals[
                    user.group_id]
        for action, feedback, user, cf_agent in action_feedback_user_list:
            agents_to_update.add(cf_agent)
            cf_agent.update_rewards(user.user_id, action, feedback, user.emab_episode.cost_arr[-1],
                                    user.emab_episode.state_arr[-2] / 2)
        user_set = user_set - removal_set
        for agent in agents_to_update:
            agent.train()
        week += 1

    output = {
        "actions_taken_arr_arr": actions_taken_arr_arr,
        'num_dropouts': num_dropouts,
        "user_group_app_sessions_dict": user_group_app_sessions_dict}
    return output


def run_cf(emab_episode: emab_model.AbstractEmabEpisode, num_weeks, user_arrivals, multi_week, cf_params):
    if multi_week:  # if multi_week then one agent is assigned to each week, like with M-TS and M-epsilon
        # we found multi_week to not work well, so we did not include it in the paper
        return smart_cf(emab_episode, num_weeks, user_arrivals, cf_params)
    else:
        return cf(emab_episode, num_weeks, user_arrivals, cf_params)
