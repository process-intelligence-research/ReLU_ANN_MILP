from .supported_modeling_languages import supported_model_types
from .gurobi_modeling_interface import GurobiModelingInterface
from .pyomo_modeling_interface import PyomoModelingInterface

class InterfaceFactory:
    """Factory class to produce modeling interfaces."""

    def __init__(self):
        self._supported_model_types = supported_model_types

    def create(self, modeling_language: str, name: str):
        if modeling_language in self._supported_model_types:
            if modeling_language == 'GUROBI':
                return GurobiModelingInterface(name)
            elif modeling_language == 'PYOMO':
                return PyomoModelingInterface(name)
        else:
            raise ValueError('Modeling language' + modeling_language + 'is not supported. See '
                                                                       'supported_modeling_languages.py')

