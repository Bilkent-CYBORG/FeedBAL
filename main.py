import multiprocessing
import pickle
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

import benchmark as emab_benchmark
import emab_model as emab
import feedback
import feedbal_algorithm as emab_algorithm
import greedy_algorithm
import m_epsilon_algo
import user_arrival_simulator
from colab_algo import run_cf
from m_ts_algo import run_smart_ts
from user_arrival_simulator import get_user_arrivals

sns.set(style='white')

"""This is the main script for the simulations. It runs the algorithms seen in the paper and produces/saves the plots.

***IMPORTANT NOTE: The variables for user dropouts, dropout chance and week, are in the user_arrival_simulator.py file.
set those variables before running this script ***

Also, make sure to set the following four variables before running the script. 
"""

use_saved_sessions = False
run = True      # whether to run simulations or just plot the figures using data of already saved results file
num_threads_to_use = 1  # number of threads to run the simulation on. When set to -1, will run on all available threads

num_times_sample_users = 2  # the number of times to sample user arrivals and run simulations
num_times_to_run = 2  # number of times to run simulations with each user arrival sample
# note that total number times the simulation is ran is equal to the product of the above two variables

action_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
stop_action = -1
expected_num_users = 200
total_num_weeks = 16
marker_list = ['o', 'v', '^', 'x', 's', 'D']



class DataEmab(emab.AbstractEmabEpisode):
    """
    This is the eMAB class used for all the simulations. This eMAB setting is based on data from IntelliCare.
    """

    def expected_cost_fun(self, t, x, a):
        y_max = 0.5 * x / t
        switcher = {
            0: y_max / (3.77 * total_num_weeks ** 2) * t * (t + 2) ** 2,
            1: y_max * 1.41,
            2: y_max / (4 * total_num_weeks ** 2) * t * (t + 10) ** 2,
            3: y_max / 17 * t ** 2,
            4: y_max / 30 * t,
            5: y_max / 15.5 * t ** 2,
            6: y_max / 150 * t,
            7: y_max / 4.5 * t * (1 - np.exp(-t / total_num_weeks * 2.5)),
            8: y_max / 1.7 * t * (1 - np.exp(-t / total_num_weeks * 2.5)),
            9: y_max / 2.0 * t * (1 - np.exp(-(t - 1) / total_num_weeks * 3)),
            10: 1.08 * y_max * t * (1 - np.exp(-t / total_num_weeks * 3)),
            11: y_max / 37.8 * t * np.exp(-(t - 4) / (3 * total_num_weeks))
        }
        return switcher[a]

    def expected_terminal_reward_fun(self, t, x):
        return x

    def state_transition_fun(self, t, x, a, f):
        return x + f

    def feedback_distribution(self, t, x, a):
        return feedback.feedback(t, x, a)

    def ex_ante_reward_fun(self, t, x, a):
        return x + feedback.feedback_mean(t, x, a)

    def calc_terminal_reward(self, t):
        return self.expected_terminal_reward_arr[t - 1]

    def calc_cost(self, t):
        return self.expected_cost_arr[t - 1]


def run_and_get_regret_arr(user_arrivals):
    """
    This function runs all five algorithms once and returns all of their outputs.
    :param user_arrivals:
    :return:
    """
    # based on previous runs, we found 8000 to be max state (i.e., max terminal reward) that FeedBAL achieves
    max_state = 8000
    state_set = range(emab_algorithm.get_bin_of_x(max_state))

    # create the model object
    emab_model = DataEmab(total_num_weeks, stop_action, action_set, state_set)
    knn_output = run_cf(emab_model, total_num_weeks, user_arrivals, False, {"algo": "knn"})
    svd_output = run_cf(emab_model, total_num_weeks, user_arrivals, False, {"algo": "svd"})
    benchmark_output = emab_benchmark.perform_benchmark(emab_model, total_num_weeks, user_arrivals)
    ts_output = run_smart_ts(emab_model, total_num_weeks, user_arrivals)
    ga_output = greedy_algorithm.run_algorithm(emab_model, total_num_weeks, user_arrivals)
    epsilon_output = m_epsilon_algo.run_m_epsilon(emab_model, total_num_weeks, user_arrivals)
    feedbal_output = emab_algorithm.run_algorithm(emab_model, 0.01, total_num_weeks, user_arrivals)

    output = {'benchmark_output': benchmark_output,
              'epsilon_output': epsilon_output,
              'ga_output': ga_output,
              'feedbal_output': feedbal_output,
              'ts_output': ts_output,
              'knn_output': knn_output,
              'svd_output' : svd_output}
    return output


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def plot_time_spent(time_taken_avg, time_taken_std, algo_names):
    suffix = "po" if user_arrival_simulator.DROPOUT_PROB == 0 else "do"
    plt.figure(figsize=(6, 2.5))
    plt.gca().set_yscale("log")
    for i, name in enumerate(algo_names):
        plt.errorbar(np.arange(1, 2 * total_num_weeks), time_taken_avg[i],
                     label=name, yerr=time_taken_std[i], linestyle="--", capsize=2, marker=marker_list[i])

    plt.legend()
    plt.xlabel("Week")
    plt.ylabel("Time taken (seconds)")
    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.savefig("time_taken_{}{}".format(suffix, ".pdf"), bbox_inches='tight', pad_inches=0.01)
    # plt.show()


def plot_sessions(epsilon_total_user_group_app_sessions_dict, feedbal_total_user_group_app_sessions_dict,
                  ga_total_user_group_app_sessions_dict, ts_total_user_group_app_sessions_dict,
                  knn_total_user_group_app_sessions_dict, svd_total_user_group_app_sessions_dict):
    """
    Plots the cum. session figures seen in the paper.
    """

    epsilon_avg_dict = {}
    feedbal_avg_dict = {}
    ga_avg_dict = {}
    ts_avg_dict = {}
    knn_avg_dict = {}
    svd_avg_dict = {}

    epsilon_std_dict = {}
    feedbal_std_dict = {}
    ga_std_dict = {}
    ts_std_dict = {}
    knn_std_dict = {}
    svd_std_dict = {}

    for i in range(total_num_weeks):
        epsilon_avg_dict[i] = [0] * (total_num_weeks - 1)
        feedbal_avg_dict[i] = [0] * (total_num_weeks - 1)
        ga_avg_dict[i] = [0] * (total_num_weeks - 1)
        ts_avg_dict[i] = [0] * (total_num_weeks - 1)
        knn_avg_dict[i] = [0] * (total_num_weeks - 1)
        svd_avg_dict[i] = [0] * (total_num_weeks - 1)

        epsilon_std_dict[i] = [0] * (total_num_weeks - 1)
        feedbal_std_dict[i] = [0] * (total_num_weeks - 1)
        ga_std_dict[i] = [0] * (total_num_weeks - 1)
        ts_std_dict[i] = [0] * (total_num_weeks - 1)
        knn_std_dict[i] = [0] * (total_num_weeks - 1)
        svd_std_dict[i] = [0] * (total_num_weeks - 1)

        for j in range(total_num_weeks - 1):
            epsilon_std_dict[i][j] = np.std(epsilon_total_user_group_app_sessions_dict[i][j])
            feedbal_std_dict[i][j] = np.std(feedbal_total_user_group_app_sessions_dict[i][j])
            ga_std_dict[i][j] = np.std(ga_total_user_group_app_sessions_dict[i][j])
            ts_std_dict[i][j] = np.std(ts_total_user_group_app_sessions_dict[i][j])
            knn_std_dict[i][j] = np.std(knn_total_user_group_app_sessions_dict[i][j])
            svd_std_dict[i][j] = np.std(svd_total_user_group_app_sessions_dict[i][j])

            epsilon_avg_dict[i][j] = np.mean(epsilon_total_user_group_app_sessions_dict[i][j])
            feedbal_avg_dict[i][j] = np.mean(feedbal_total_user_group_app_sessions_dict[i][j])
            ga_avg_dict[i][j] = np.mean(ga_total_user_group_app_sessions_dict[i][j])
            ts_avg_dict[i][j] = np.mean(ts_total_user_group_app_sessions_dict[i][j])
            knn_avg_dict[i][j] = np.mean(knn_total_user_group_app_sessions_dict[i][j])
            svd_avg_dict[i][j] = np.mean(svd_total_user_group_app_sessions_dict[i][j])

    names = ['M-$\epsilon$-greedy', 'Greedy', 'M-TS', 'KNN', 'SVD']

    suffix = "_po" if user_arrival_simulator.DROPOUT_PROB == 0 else "_do"
    file_names = [x.lower().replace(' ', '_') + suffix for x in names]
    avg_list = [epsilon_avg_dict, ga_avg_dict, ts_avg_dict, knn_avg_dict, svd_avg_dict]
    std_list = [epsilon_std_dict, ga_std_dict, ts_std_dict, knn_std_dict, svd_std_dict]
    for i, name in enumerate(names):
        plt.figure(figsize=(6, 2.5))
        marker_index = 0
        for key in feedbal_user_group_app_sessions_dict.keys():
            if key == 0 or key == int(total_num_weeks / 2) or key == total_num_weeks - 1:
                plt.errorbar(range(1, total_num_weeks), avg_list[i][key],
                             label=name + r" group entering in week {key}".format(key=(key + 1)),
                             yerr=std_list[i][key], linestyle="--", capsize=2, marker=marker_list[marker_index])
                marker_index += 1

        for key in feedbal_user_group_app_sessions_dict.keys():
            # print('max feedbal sessions: {}'.format(feedbal_avg_dict[l_max - 1][-1]))
            if key == 0 or key == int(total_num_weeks / 2) or key == total_num_weeks - 1:
                plt.errorbar(range(1, total_num_weeks), feedbal_avg_dict[key],
                             label="FeedBAL group entering in week {key}".format(key=(key + 1)),
                             yerr=feedbal_std_dict[key], linestyle="--", capsize=2, marker=marker_list[marker_index])
                marker_index += 1

        plt.legend()
        plt.xlabel("Week")
        plt.ylabel("Cum. number of app sessions\nper num. users")
        plt.tight_layout()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.savefig(file_names[i] + ".pdf", bbox_inches='tight', pad_inches=0.01)
        # plt.show()


def plot_actions(bench_actions, epsilon_actions, feedbal_actions, ga_actions, ts_actions, knn_actions, svd_actions,
                 weeks_to_plot=None):
    """
    Plots stacked barchart of the actions recommended in given weeks of all algorithms, including the benchmark.
    :param bench_actions:
    :param epsilon_actions:
    :param feedbal_actions:
    :param ga_actions:
    :param ts_actions:
    :param weeks_to_plot:
    """
    if weeks_to_plot is None:
        weeks_to_plot = [0, -1]
    ind = np.arange(1, total_num_weeks)
    sns.set_palette(sns.color_palette("husl", 12, desat=0.7))

    data_arr = [bench_actions, epsilon_actions, feedbal_actions, ga_actions, ts_actions, knn_actions, svd_actions]
    file_names = ['bench', 'epsilon', 'feedbal', 'ga', 'ts', 'knn', 'svd']
    suffix = "_po" if user_arrival_simulator.DROPOUT_PROB == 0 else "_do"
    file_names = ["{}_actions{}".format(x.lower(), suffix) for x in file_names]
    for i, arr_to_plot in enumerate(data_arr):
        for week_to_plot in weeks_to_plot:
            plt.figure(figsize=(6, 2.5))
            for action in action_set:
                if action == 0:
                    plt.bar(ind, arr_to_plot[week_to_plot, :, action], label=action + 1)
                else:
                    plt.bar(ind, arr_to_plot[week_to_plot, :, action],
                            bottom=arr_to_plot[week_to_plot].cumsum(axis=1)[:, action - 1], label=action)
            plt.ylabel('Number of recommendations')
            plt.xlabel('Week')
            plt.ylim(0, arr_to_plot[week_to_plot, 0, :].sum())
            plt.tight_layout()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.savefig(file_names[i] + '_week_' + str(week_to_plot) + ".pdf", bbox_inches='tight', pad_inches=0.01)
            # plt.show()


if __name__ == '__main__':
    # if sys.argv[1] == "svd":
    #     ColabFiltering.use_svd = True
    #     user_arrival_simulator.DROPOUT_PROB = 0
    # elif sys.argv[1] == "knn":
    #     ColabFiltering.use_svd = False
    #     user_arrival_simulator.DROPOUT_PROB = 0.0000000001

    results_filename = 'results_po_FINAL' if user_arrival_simulator.DROPOUT_PROB == 0 else 'results_do_FINAL'  # filename of the results file to save

    epsilon_total_user_group_app_sessions_dict = {}
    feedbal_total_user_group_app_sessions_dict = {}
    ga_total_user_group_app_sessions_dict = {}
    ts_total_user_group_app_sessions_dict = {}
    bench_total_user_group_app_sessions_dict = {}
    knn_total_user_group_app_sessions_dict = {}
    svd_total_user_group_app_sessions_dict = {}
    for i in range(total_num_weeks):
        epsilon_total_user_group_app_sessions_dict[i] = {}
        feedbal_total_user_group_app_sessions_dict[i] = {}
        ga_total_user_group_app_sessions_dict[i] = {}
        ts_total_user_group_app_sessions_dict[i] = {}
        bench_total_user_group_app_sessions_dict[i] = {}
        knn_total_user_group_app_sessions_dict[i] = {}
        svd_total_user_group_app_sessions_dict[i] = {}
        for j in range(total_num_weeks - 1):  # since its always the stop action in the last week.
            epsilon_total_user_group_app_sessions_dict[i][j] = []
            feedbal_total_user_group_app_sessions_dict[i][j] = []
            ga_total_user_group_app_sessions_dict[i][j] = []
            ts_total_user_group_app_sessions_dict[i][j] = []
            bench_total_user_group_app_sessions_dict[i][j] = []
            knn_total_user_group_app_sessions_dict[i][j] = []
            svd_total_user_group_app_sessions_dict[i][j] = []

    if run:
        starting_time = time()
        # change num_cores to set the number of threads to run on.
        if num_threads_to_use > 0:
            num_cores = num_threads_to_use
        else:
            num_cores = int(multiprocessing.cpu_count())
        dropout_str = ' dropout' if user_arrival_simulator.DROPOUT_PROB > 0 else 'out dropout'
        print("Running with {} users with{} on {thread_count} threads".format(expected_num_users, dropout_str,
                                                                              thread_count=num_cores))
        results = []
        for _ in tqdm(range(num_times_sample_users)):
            user_arrivals = get_user_arrivals(expected_num_users, total_num_weeks)
            results.extend(Parallel(n_jobs=num_cores)(delayed(run_and_get_regret_arr)(user_arrivals)
                                                      for i in tqdm(range(num_times_to_run))))

        print("Time taken: {}s".format(time() - starting_time))
    # Save the results
    if run:
        save_obj(results, results_filename)
    else:
        results = load_obj(results_filename)

    # Process the results
    # time_taken_all = np.zeros((len(algo_name_temp), len(results),
    #                            2 * total_num_weeks - 1))  # element i,j,k is time taken by algo. i during run j in week k+1

    ga_actions = np.zeros(results[0]['ga_output']['actions_taken_arr_arr'].shape)
    epsilon_actions = np.zeros(ga_actions.shape)
    feedbal_actions = np.zeros(ga_actions.shape)
    ts_actions = np.zeros(ga_actions.shape)
    bench_actions = np.zeros(ga_actions.shape)
    knn_actions = np.zeros(ga_actions.shape)
    svd_actions = np.zeros(ga_actions.shape)

    ga_avg_num_dropouts = np.zeros(len(results))
    ep_avg_num_dropouts = np.zeros(len(results))
    ts_avg_num_dropouts = np.zeros(len(results))
    fb_avg_num_dropouts = np.zeros(len(results))
    bench_avg_num_dropouts = np.zeros(len(results))
    knn_avg_num_dropouts = np.zeros(len(results))
    svd_avg_num_dropouts = np.zeros(len(results))

    for i, result in enumerate(results):
        # for algo_idx, algo_name in enumerate(algo_name_temp):
        #     algo_result = result[algo_name]
        #     time_taken_all[algo_idx, i, :] = algo_result['time_taken_arr']
        ga_output = result['ga_output']
        epsilon_output = result['epsilon_output']
        feedbal_output = result['feedbal_output']
        ts_output = result['ts_output']
        bench_output = result['benchmark_output']
        knn_output = result['knn_output']
        svd_output = result['svd_output']

        ga_actions += ga_output['actions_taken_arr_arr']
        epsilon_actions += epsilon_output['actions_taken_arr_arr']
        feedbal_actions += feedbal_output['actions_taken_arr_arr']
        ts_actions += ts_output['actions_taken_arr_arr']
        bench_actions += bench_output['actions_taken_arr_arr']
        knn_actions += knn_output['actions_taken_arr_arr']
        svd_actions += svd_output['actions_taken_arr_arr']

        ga_user_group_app_sessions_dict = ga_output["user_group_app_sessions_dict"]
        epsilon_user_group_app_sessions_dict = epsilon_output["user_group_app_sessions_dict"]
        feedbal_user_group_app_sessions_dict = feedbal_output["user_group_app_sessions_dict"]
        ts_user_group_app_sessions_dict = ts_output["user_group_app_sessions_dict"]
        bench_user_group_app_sessions_dict = bench_output["user_group_app_sessions_dict"]
        knn_user_group_app_sessions_dict = knn_output["user_group_app_sessions_dict"]
        svd_user_group_app_sessions_dict = svd_output["user_group_app_sessions_dict"]

        ga_avg_num_dropouts[i] = ga_output['num_dropouts'] / expected_num_users
        ep_avg_num_dropouts[i] = epsilon_output['num_dropouts'] / expected_num_users
        ts_avg_num_dropouts[i] = ts_output['num_dropouts'] / expected_num_users
        fb_avg_num_dropouts[i] = feedbal_output['num_dropouts'] / expected_num_users
        bench_avg_num_dropouts[i] = bench_output['num_dropouts'] / expected_num_users
        knn_avg_num_dropouts[i] = knn_output['num_dropouts'] / expected_num_users
        svd_avg_num_dropouts[i] = svd_output['num_dropouts'] / expected_num_users

        for key in range(total_num_weeks):
            for i in range(1, len(epsilon_user_group_app_sessions_dict[key])):
                epsilon_user_group_app_sessions_dict[key][i] += epsilon_user_group_app_sessions_dict[key][i - 1]
                feedbal_user_group_app_sessions_dict[key][i] += feedbal_user_group_app_sessions_dict[key][i - 1]
                ga_user_group_app_sessions_dict[key][i] += ga_user_group_app_sessions_dict[key][i - 1]
                ts_user_group_app_sessions_dict[key][i] += ts_user_group_app_sessions_dict[key][i - 1]
                bench_user_group_app_sessions_dict[key][i] += bench_user_group_app_sessions_dict[key][i - 1]
                knn_user_group_app_sessions_dict[key][i] += knn_user_group_app_sessions_dict[key][i - 1]
                svd_user_group_app_sessions_dict[key][i] += svd_user_group_app_sessions_dict[key][i - 1]

        for i in range(total_num_weeks):
            for j in range((len(epsilon_user_group_app_sessions_dict[i]))):
                epsilon_total_user_group_app_sessions_dict[i][j].append(epsilon_user_group_app_sessions_dict[i][j])
                feedbal_total_user_group_app_sessions_dict[i][j].append(feedbal_user_group_app_sessions_dict[i][j])
                ga_total_user_group_app_sessions_dict[i][j].append(ga_user_group_app_sessions_dict[i][j])
                ts_total_user_group_app_sessions_dict[i][j].append(ts_user_group_app_sessions_dict[i][j])
                bench_total_user_group_app_sessions_dict[i][j].append(bench_user_group_app_sessions_dict[i][j])
                knn_total_user_group_app_sessions_dict[i][j].append(knn_user_group_app_sessions_dict[i][j])
                svd_total_user_group_app_sessions_dict[i][j].append(svd_user_group_app_sessions_dict[i][j])

    print('FeedBAL dropout avg: {} std: {}'.format(fb_avg_num_dropouts.mean(), fb_avg_num_dropouts.std()))
    print('Greedy dropout avg: {} std: {}'.format(ga_avg_num_dropouts.mean(), ga_avg_num_dropouts.std()))
    print('Epsilon dropout avg: {} std: {}'.format(ep_avg_num_dropouts.mean(), ep_avg_num_dropouts.std()))
    print('TS dropout avg: {} std: {}'.format(ts_avg_num_dropouts.mean(), ts_avg_num_dropouts.std()))
    print('Benchmark dropout avg: {} std: {}'.format(bench_avg_num_dropouts.mean(), bench_avg_num_dropouts.std()))
    print('KNN dropout avg: {} std: {}'.format(knn_avg_num_dropouts.mean(), knn_avg_num_dropouts.std()))
    print('SVD dropout avg: {} std: {}'.format(svd_avg_num_dropouts.mean(), svd_avg_num_dropouts.std()))

    plot_sessions(epsilon_total_user_group_app_sessions_dict, feedbal_total_user_group_app_sessions_dict,
                  ga_total_user_group_app_sessions_dict, ts_total_user_group_app_sessions_dict,
                  knn_total_user_group_app_sessions_dict, svd_total_user_group_app_sessions_dict)

    plot_actions(bench_actions / len(results), epsilon_actions / len(results), feedbal_actions / len(results),
                 ga_actions / len(results), ts_actions / len(results), knn_actions / len(results), svd_actions / len(results))
    # plot_time_spent(time_taken_all.mean(axis=1), time_taken_all.std(axis=1), algo_names)

