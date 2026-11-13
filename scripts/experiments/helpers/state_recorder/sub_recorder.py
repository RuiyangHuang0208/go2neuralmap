from abc import ABC, abstractmethod
from matplotlib.pyplot import Axes


class SubRecorder(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def handle(self, var_dict):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass


class SubPlotter(ABC):

    def __init__(self, ax: Axes):
        pass

    @abstractmethod
    def draw(self, var_dict):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass
