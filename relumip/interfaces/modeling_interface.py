from abc import ABCMeta, abstractmethod


class ModelingInterface(metaclass=ABCMeta):
    """Base class for interface to modeling frameworks.

    Warning: This class should not be used directly. Use derived classes instead.
    """

    def __init__(self, name: str):
        self._name = name

    @staticmethod
    def get_variable_bounds(opt_vars: list):
        pass

    @abstractmethod
    def connect_network_input(self, opt_model, input_vars: list):
        pass

    @abstractmethod
    def connect_network_output(self, opt_model, output_vars: list):
        pass

    @abstractmethod
    def embed_network_formulation(self, opt_model, ann_param, mip_formulation: str, bound_tightening_strategy: str,
                                  node_time_limit: float, use_full_model: bool, solver):
        pass

    @staticmethod
    def type_check(obj, type_str):
        pass



