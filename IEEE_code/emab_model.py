import abc


class AbstractEmabEpisode(object, metaclass=abc.ABCMeta):
    """
    This class represents an abstract eMAB episode object. Sub classes of this class must implement all of the
    functions such as expected_cost_fun, expected_terminal_reward_fun and etc. This class also keeps track of the
    step and state of the episode, and provides a function to perform a given action, observe and record the cost and
    feedback.
    """
    def __init__(self, l_max, stop_action, action_set, state_set):

        self.l_max = l_max
        self.stop_action = stop_action
        self.action_set = action_set
        self.state_set = state_set

        # Initializations
        self.episode_num = 1
        self.stopping_t = -1
        self.t = 1
        self.x = 0
        self.state_arr = [self.x]
        self.action_taken_arr = []
        self.feedback_acquired_arr = []
        self.expected_cost_arr = []
        self.expected_terminal_reward_arr = []
        self.terminal_reward_arr = []
        self.cost_arr = []
        self.is_stopped = False
        self.terminal_reward = 0

    def reinitialize(self):
        self.episode_num = 1
        self.stopping_t = -1
        self.t = 1
        self.x = 0
        self.state_arr = [self.x]
        self.action_taken_arr = []
        self.feedback_acquired_arr = []
        self.expected_cost_arr = []
        self.expected_terminal_reward_arr = []
        self.terminal_reward_arr = []
        self.cost_arr = []
        self.is_stopped = False
        self.terminal_reward = 0

    # Abstract methods that subclass must override
    @abc.abstractmethod
    def expected_cost_fun(self, t, x, a):
        raise NotImplementedError("Should implement")

    @abc.abstractmethod
    def expected_terminal_reward_fun(self, t, x):
        raise NotImplementedError("Should implement")

    @abc.abstractmethod
    def state_transition_fun(self, t, x, a, f):
        raise NotImplementedError("Should implement")

    @abc.abstractmethod
    def feedback_distribution(self, t, x, a):
        raise NotImplementedError("Should implement")

    @abc.abstractmethod
    def ex_ante_reward_fun(self, t, x, a):
        raise NotImplementedError("Should implement")

    # Functions that provide info about an ongoing episode
    def get_feedback_of_action(self, a):
        return self.feedback_distribution(self.t, self.x, a)

    def get_gain_of_action(self, a):
        if a == self.stop_action:
            return self.expected_terminal_reward_fun(self.t, self.x) - self.expected_cost_fun(self.t, self.x, a)
        else:
            return self.ex_ante_reward_fun(self.t, self.x, a) - self.expected_cost_fun(self.t, self.x, a)

    def get_next_state(self, a, f):
        return self.state_transition_fun(self.t, self.x, a, f)

    # Returns the feedback of the taken action (or -1 if episode is stopped)
    def perform_action(self, a):
        if self.is_stopped:
            return -1
        if self.t == self.l_max or a == self.stop_action:
            self.stopping_t = self.t
            self.is_stopped = True

            # Update the arrays
            self.expected_cost_arr.append(self.expected_cost_fun(self.t, self.x, a))
            self.action_taken_arr.append(self.stop_action)
            self.expected_terminal_reward_arr.append(self.expected_terminal_reward_fun(self.t, self.x))
            self.terminal_reward_arr.append(self.calc_terminal_reward(self.t))
            self.terminal_reward = self.calc_terminal_reward(self.stopping_t)
            self.cost_arr.append(self.calc_cost(self.t))
            return -1
        feedback = self.get_feedback_of_action(a)
        next_state = self.get_next_state(a, feedback)

        # Update the arrays
        self.expected_cost_arr.append(self.expected_cost_fun(self.t, self.x, a))
        self.action_taken_arr.append(a)
        self.expected_terminal_reward_arr.append(self.expected_terminal_reward_fun(self.t, self.x))
        self.terminal_reward_arr.append(self.calc_terminal_reward(self.t))
        self.cost_arr.append(self.calc_cost(self.t))
        self.feedback_acquired_arr.append(feedback)
        self.state_arr.append(next_state)

        self.x = next_state
        self.t += 1
        return feedback

    # Functions that provide info after the stop action is taken
    @abc.abstractmethod
    def calc_terminal_reward(self, t):
        raise NotImplementedError("Should implement")

    @abc.abstractmethod
    def calc_cost(self, t):
        raise NotImplementedError("Should implement")
