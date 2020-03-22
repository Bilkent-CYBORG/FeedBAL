import numpy as np

import emab_model as emab
from feedbal_algorithm import check_dropout_cond
from user import User
from user_arrival_simulator import DROPOUT_WEEK, DROPOUT_PROB

"""This script runs a benchmark for eMAB, one that picks the action with the maximum gain at each step t.
 """


def get_best_action(emab_episode: emab.AbstractEmabEpisode):
    return max(emab_episode.action_set, key=emab_episode.get_gain_of_action)


def perform_benchmark(emab_episode: emab.AbstractEmabEpisode, num_weeks, user_arrivals):
    user_set = set()
    removal_set = set()
    week = 1
    num_users_added = 0
    max_state = 0
    max_feedback = 0
    num_dropouts = 0

    # list of list of actions taken by each user
    actions_taken_arr_arr = np.zeros((len(user_arrivals), num_weeks - 1, len(emab_episode.action_set)))

    app_sessions = [0] * (num_weeks - 1)
    user_group_app_sessions_dict = {}  # Maps user group to list of session counts (per week)
    for i in range(num_weeks):
        user_group_app_sessions_dict[i] = [0] * (num_weeks - 1)

    while week == 1 or len(user_set) > 0:
        if week <= num_weeks:
            # Observe arrived users
            for _ in range(user_arrivals[week - 1]):
                user_set.add(User(week - 1, num_users_added, week, emab_episode))
                num_users_added += 1

        # Play app for each user
        removal_set.clear()
        for user in user_set:
            best_action = get_best_action(user.emab_episode)
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
                max_feedback = max(max_feedback, feedback)
                user_group_app_sessions_dict[user.group_id][user.emab_episode.t - 2] += feedback
                if user.emab_episode.t == num_weeks:
                    max_state = max(user.emab_episode.x, max_state)
        user_set = set(filter(lambda item: item not in removal_set, user_set))
        week += 1

    output = {
        "actions_taken_arr_arr": actions_taken_arr_arr,
        "user_group_app_sessions_dict": user_group_app_sessions_dict,
        'num_dropouts': num_dropouts,
        "app_sessions": app_sessions,
        "max_state": max_state,
        "max_feedback": max_feedback}
    return output
