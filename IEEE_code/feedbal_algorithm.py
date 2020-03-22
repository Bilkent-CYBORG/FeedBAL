import math as math
import random

import numpy as np

import emab_model as emab
from user import User
from user_arrival_simulator import DROPOUT_PROB, DROPOUT_WEEK

"""This script runs the FeedLAB algorithm in the batch setting."""

default_zero = 0


def calc_confidence_bound(t, x, g_txa_dict, n_txa_dict, a, K, sigma, delta):
    n_txa_value = n_txa_dict.get((t, x, a), default_zero)
    g_txa_value = g_txa_dict.get((t, x, a), 0)
    if n_txa_value == 0:
        return -1
    return g_txa_value + 1 / 8 * np.sqrt((1 + n_txa_value) / (n_txa_value ** 2) *
                                         4 * sigma ** 2 * np.log(K / delta * np.sqrt(1 + n_txa_value)))


def calc_confidence_bound_stop_action(t, x, r_tx_dict, n_tx_dict, k, sigma, delta):
    n_tx_value = n_tx_dict.get((t, x), default_zero)
    if n_tx_value == 0:
        return -1
    return r_tx_dict.get((t, x), 0) + math.sqrt((1 + n_tx_value) / (n_tx_value ** 2) *
                                                4 * sigma ** 2 * math.log(k / delta * math.sqrt(1 + n_tx_value)))


def get_bin_of_x(x):
    """
    Performs discretization of the state and reduces x by a factor of approximately 40.
    :param x: the state to be discretized
    :return: the discretization of x
    """
    if x <= 100:
        return x
    else:
        return int(round(x / 50)) + 101


def run_algorithm(og_emab_episode: emab.AbstractEmabEpisode, delta, num_weeks, user_arrivals):
    k = og_emab_episode.l_max * len(og_emab_episode.state_set) * len(og_emab_episode.action_set)
    feedbacks_t_dict = {}  # maps t to list of feedbacks received in week t
    n_tx_dict = {}  # maps t to N_tx
    n_txa_dict = {}  # maps t to an array of N_txa for different apps
    r_tx_dict = {}  # maps t to value
    g_txa_dict = {}  # maps t to an array of values for apps

    actions_taken_arr_arr = np.zeros((len(user_arrivals), num_weeks - 1, len(og_emab_episode.action_set)))
    u_txa_dict = {}  # maps t,x,a to u
    inft_ucb_action_arr = []  # actions with infinite conf. bounds
    user_set = set()
    removal_set = set()
    week = 1
    num_users_added = 0
    num_random_picks = 0
    max_state = 0
    num_dropouts = 0
    app_sessions = [0] * (num_weeks - 1)  # list of app sessions in a given week (of all users)
    user_group_app_sessions_dict = {}  # Maps user group to list of session counts (per week)
    for i in range(num_weeks):
        user_group_app_sessions_dict[i] = [0] * (num_weeks - 1)
    max_feedback = {}

    while week == 1 or len(user_set) > 0:
        if week <= num_weeks:
            # Observe arrived users
            for _ in range(user_arrivals[week - 1]):
                user_set.add(User(week - 1, num_users_added, week, og_emab_episode))
                num_users_added += 1
        removal_set.clear()
        for user in user_set:
            emab_episode = user.emab_episode
            inft_ucb_action_arr.clear()
            t = emab_episode.t
            x = get_bin_of_x(emab_episode.x)

            # compute sigma based on prev feedbacks
            if t in feedbacks_t_dict and len(feedbacks_t_dict[t]) > 1:
                sigma = np.std(feedbacks_t_dict[t], ddof=1)
            elif t - 1 in feedbacks_t_dict and len(feedbacks_t_dict[t - 1]) > 1:
                sigma = np.std(feedbacks_t_dict[t - 1], ddof=1)
            else:
                sigma = 1
            for a in emab_episode.action_set:

                if a == emab_episode.stop_action:  # this never happens in our simulations
                    u_txa = calc_confidence_bound_stop_action(t, x, r_tx_dict, n_tx_dict, k, sigma, delta)
                else:
                    u_txa = calc_confidence_bound(t, x, g_txa_dict, n_txa_dict, a, k, sigma, delta)
                if u_txa == -1:
                    inft_ucb_action_arr.append(a)
                else:
                    u_txa_dict[(t, x, a)] = u_txa

            if len(inft_ucb_action_arr) > 0:
                num_random_picks += 1
                best_action = random.choice(inft_ucb_action_arr)
            else:
                best_action = emab_episode.action_set[0]
                for action in emab_episode.action_set:
                    if u_txa_dict[(t, x, action)] > u_txa_dict[(t, x, best_action)]:
                        best_action = action
            feedback = emab_episode.perform_action(best_action)
            if feedback == -1 or emab_episode.t > DROPOUT_WEEK and feedback <= emab_episode.x / t and np.random.binomial(
                    1, DROPOUT_PROB) == 1:
                if feedback != -1:
                    for i, action in enumerate(user.emab_episode.action_taken_arr):
                        actions_taken_arr_arr[user.group_id, i, action] += 1
                    num_dropouts += 1
                else:
                    for i, action in enumerate(user.emab_episode.action_taken_arr[:-1]):
                        actions_taken_arr_arr[user.group_id, i, action] += 1
                removal_set.add(user)
            else:
                if user.emab_episode.t == num_weeks:
                    max_state = max(user.emab_episode.x, max_state)
                if t not in feedbacks_t_dict:
                    feedbacks_t_dict[t] = []
                feedbacks_t_dict[t].append(feedback)
                max_feedback[t] = max(feedback, max_feedback.get(t, 0))
                # if user.user_id >= 8000 - 2080:
                # app_sessions[user.emab_episode.t - 2] += feedback  # -2 b/c index and also played action
                user_group_app_sessions_dict[user.group_id][emab_episode.t - 2] += feedback / user_arrivals[
                    user.group_id]

            t_list = []
            x_list = []
            if emab_episode.t == emab_episode.stopping_t:
                index = -1
                t_list = [emab_episode.t - 1, emab_episode.t]
                x_list = [get_bin_of_x(emab_episode.state_arr[-2]), get_bin_of_x(emab_episode.x)]
            else:
                index = -1
                t_list.append(emab_episode.t - 1)
                x_list.append(get_bin_of_x(emab_episode.state_arr[-2]))  # -2 b/c -1 is the most recent x

            for i in range(len(t_list)):
                t_temp = t_list[i]
                x_t_bin = x_list[i]

                # update g_tx stop
                temp_value = (n_tx_dict.get((t_temp, x_t_bin), default_zero) * r_tx_dict.get((t_temp, x_t_bin), 0) +
                              emab_episode.terminal_reward_arr[index]) / (
                                     n_tx_dict.get((t_temp, x_t_bin), default_zero) + 1)
                g_txa_dict[(t_temp, x_t_bin, emab_episode.stop_action)] = r_tx_dict[(t_temp, x_t_bin)] = temp_value

                # update n_tx
                n_tx_dict[(t_temp, x_t_bin)] = n_tx_dict.get((t_temp, x_t_bin), default_zero) + 1

                # update g_txa: since we need the reward of the next t, we will only update g_txa if there are more than
                # two rewards available
                if len(emab_episode.terminal_reward_arr) >= 2:
                    old_t_temp = t_temp - 1
                    old_x_t_bin = get_bin_of_x(emab_episode.state_arr[-3])
                    old_a_t = emab_episode.action_taken_arr[index - 1]

                    temp_value = (n_txa_dict.get((old_t_temp, old_x_t_bin, old_a_t), default_zero) * g_txa_dict.get(
                        (old_t_temp, old_x_t_bin, old_a_t), 0) +
                                  emab_episode.terminal_reward_arr[index] - emab_episode.cost_arr[index - 1]) / (
                                         n_txa_dict.get((old_t_temp, old_x_t_bin, old_a_t), default_zero) + 1)
                    g_txa_dict[(old_t_temp, old_x_t_bin, old_a_t)] = temp_value

                # update n_txa
                if t_temp != emab_episode.stopping_t:
                    a_t = emab_episode.action_taken_arr[index]
                    n_txa_dict[(t_temp, x_t_bin, a_t)] = n_txa_dict.get((t_temp, x_t_bin, a_t), default_zero) + 1

        user_set = set(filter(lambda item: item not in removal_set, user_set))
        week += 1
    output = {
        'num_random_picks': num_random_picks,
        "actions_taken_arr_arr": actions_taken_arr_arr,
        'num_dropouts': num_dropouts,
        "user_group_app_sessions_dict": user_group_app_sessions_dict,
        "app_sessions": app_sessions,
        "max_state": max_state,
        "max_feedback": max_feedback}
    print('FeedBAL sigma: {}'.format([np.std(x, ddof=1) for x in feedbacks_t_dict.values()]))
    return output
