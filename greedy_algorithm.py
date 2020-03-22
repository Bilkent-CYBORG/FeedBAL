import random
import math as math

import numpy as np

import emab_model as emab
from feedback import feedback_mean
from feedbal_algorithm import check_dropout_cond, get_bin_of_x
from user_arrival_simulator import DROPOUT_PROB, DROPOUT_WEEK
from user import User

"""
This script is responsible for running the greedy algorithm. It is essentially a copy-paste of the code for FeedBAL,
with the only difference being that the confidence term is always zero.
"""


def calc_confidence_bound(t, x, g_txa_dict, n_txa_dict, a):
    n_txa_value = n_txa_dict.get((t, x, a), 0)
    g_txa_value = g_txa_dict.get((t, x, a), 0)
    if n_txa_value == 0:
        return False, 0
    return True, g_txa_value


def calc_confidence_bound_stop_action(t, x, r_tx_dict, n_tx_dict):
    n_tx_value = n_tx_dict.get((t, x), 0)
    if n_tx_value == 0:
        return False, 0
    return True, r_tx_dict.get((t, x), 0)



def run_algorithm(og_emab_episode: emab.AbstractEmabEpisode, num_weeks, user_arrivals):
    n_tx_dict = {}  # maps t to N_tx
    n_txa_dict = {}  # maps t to an array of N_txa for different apps
    r_tx_dict = {}  # maps t to value
    g_txa_dict = {}  # maps t to an array of values for apps

    emab_episode_list = []  # list of users' emab episodes that are used to update counters AFTER each week
    actions_taken_arr_arr = np.zeros((len(user_arrivals), num_weeks - 1, len(og_emab_episode.action_set)))
    u_txa_dict = {}  # maps t,x,a to u
    inft_ucb_action_arr = []  # actions with infinite conf. bounds
    user_set = set()
    removal_set = set()
    week = 1
    num_users_added = 0
    num_random_picks = 0
    num_dropouts = 0
    user_group_app_sessions_dict = {}  # Maps user group to list of session counts (per week)
    for i in range(num_weeks):
        user_group_app_sessions_dict[i] = [0] * (num_weeks - 1)

    while week == 1 or len(user_set) > 0:
        if week <= num_weeks:
            # Observe arrived users
            for _ in range(user_arrivals[week - 1]):
                user_set.add(User(week - 1, num_users_added, week, og_emab_episode))
                num_users_added += 1
        removal_set.clear()
        emab_episode_list.clear()
        for user in user_set:
            emab_episode = user.emab_episode
            inft_ucb_action_arr.clear()
            t = emab_episode.t
            x = get_bin_of_x(emab_episode.x)

            for a in emab_episode.action_set:
                if a == emab_episode.stop_action:  # this never happens in our simulations
                    is_played, u_txa = calc_confidence_bound_stop_action(t, x, r_tx_dict, n_tx_dict)
                else:
                    is_played, u_txa = calc_confidence_bound(t, x, g_txa_dict, n_txa_dict, a)
                if not is_played:
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
            if feedback == -1 or emab_episode.t > DROPOUT_WEEK and check_dropout_cond(emab_episode) and np.random.binomial(1, DROPOUT_PROB) == 1:
                if feedback != -1:
                    for i, action in enumerate(user.emab_episode.action_taken_arr):
                        actions_taken_arr_arr[user.group_id, i, action] += 1
                    num_dropouts += 1
                else:
                    for i, action in enumerate(user.emab_episode.action_taken_arr[:-1]):
                        actions_taken_arr_arr[user.group_id, i, action] += 1
                removal_set.add(user)
            else:
                user_group_app_sessions_dict[user.group_id][emab_episode.t - 2] += feedback / user_arrivals[user.group_id]
            emab_episode_list.append(emab_episode)

        for emab_episode in emab_episode_list:
            t_list = []
            x_list = []
            index = -1
            if emab_episode.t == emab_episode.stopping_t:
                t_list = [emab_episode.t - 1, emab_episode.t]
                x_list = [get_bin_of_x(emab_episode.state_arr[-2]), get_bin_of_x(emab_episode.x)]
            else:
                t_list.append(emab_episode.t - 1)
                x_list.append(get_bin_of_x(emab_episode.state_arr[-2]))  # -2 b/c -1 is the most recent x

            for i in range(len(t_list)):
                t_temp = t_list[i]
                x_t_bin = x_list[i]

                # update g_tx stop
                temp_value = (n_tx_dict.get((t_temp, x_t_bin), 0) * r_tx_dict.get((t_temp, x_t_bin), 0) +
                              emab_episode.terminal_reward_arr[index]) / (
                                     n_tx_dict.get((t_temp, x_t_bin), 0) + 1)
                g_txa_dict[(t_temp, x_t_bin, emab_episode.stop_action)] = r_tx_dict[(t_temp, x_t_bin)] = temp_value

                # update n_tx
                n_tx_dict[(t_temp, x_t_bin)] = n_tx_dict.get((t_temp, x_t_bin), 0) + 1

                # update g_txa: since we need the reward of the next t, we will only update g_txa if there are more than
                # two rewards available
                if len(emab_episode.terminal_reward_arr) >= 2:
                    old_t_temp = t_temp - 1
                    old_x_t_bin = get_bin_of_x(emab_episode.state_arr[-3])
                    old_a_t = emab_episode.action_taken_arr[index - 1]

                    temp_value = (n_txa_dict.get((old_t_temp, old_x_t_bin, old_a_t), 0) * g_txa_dict.get(
                        (old_t_temp, old_x_t_bin, old_a_t), 0) +
                                  emab_episode.terminal_reward_arr[index] - emab_episode.cost_arr[index - 1]) / (
                                         n_txa_dict.get((old_t_temp, old_x_t_bin, old_a_t), 0) + 1)
                    g_txa_dict[(old_t_temp, old_x_t_bin, old_a_t)] = temp_value

                # update n_txa
                if t_temp != emab_episode.stopping_t:
                    a_t = emab_episode.action_taken_arr[index]
                    n_txa_dict[(t_temp, x_t_bin, a_t)] = n_txa_dict.get((t_temp, x_t_bin, a_t), 0) + 1

        user_set = set(filter(lambda item: item not in removal_set, user_set))
        week += 1
    output = {
        "actions_taken_arr_arr": actions_taken_arr_arr,
        'num_dropouts': num_dropouts,
        "user_group_app_sessions_dict": user_group_app_sessions_dict}
    return output
