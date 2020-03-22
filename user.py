import copy

from emab_model import AbstractEmabEpisode

"""
This is a model class for a user that enters the simulation (experiment).
"""


class User:
    def __init__(self, group_id, user_id, arrived_week, emab_episode: AbstractEmabEpisode):
        self.arrived_week = arrived_week
        self.group_id = group_id
        self.emab_episode = copy.deepcopy(emab_episode)
        self.user_id = user_id
        self.emab_episode.reinitialize()

    def __repr__(self):
        return "User(%s)" % self.user_id

    def __eq__(self, other):
        if isinstance(other, User):
            return self.user_id == other.user_id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__repr__())
