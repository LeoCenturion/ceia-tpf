from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def get_signal(self):
        pass

    @abstractmethod
    def get_order_size(self):
        pass
