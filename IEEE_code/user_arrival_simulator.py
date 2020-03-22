import numpy as np

DROPOUT_PROB = 0.5  # the chance of dropping out if the condition stated in the paper is met (f_{t,x,a} <= x/t)
DROPOUT_WEEK = 5  # the week AT THE END of which users may drop out


def get_user_arrivals(total_num_users, num_weeks, equally_distributed=False):
    if equally_distributed:
        arrivals = [int(total_num_users / num_weeks)] * num_weeks
        arrivals[-1] += total_num_users - sum(arrivals)
        return arrivals

    # Distribute according to Poisson (each week has mean total_num_users / num_weeks)
    arrivals = np.random.poisson(total_num_users / num_weeks, num_weeks)
    return arrivals
