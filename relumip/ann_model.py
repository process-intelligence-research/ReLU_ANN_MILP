import tensorflow as tf
import numpy as np

from .utils.storage import AnnParameters

from .interfaces.interface_factory import InterfaceFactory


class AnnModel:
    """Class for constructing optimization formulations for trained Artificial neural networks (ANNs) with Rectified
    linear unit (ReLU) activation functions. The formulation is embedded in optimization models defined by the user
    in the Pyomo or  Gurobi modelling languages.
    """

    def __init__(self, tf_model: tf.keras.Sequential, modeling_language: str, name: str = "ANN") -> None:
        """An AnnModel is constructed for a specific tensorflow.keras.Sequential model and the desired modelling
        Interface (eg. pyomo). The interface is initialized and weights and biases are extracted from the tensorflow
        model.

        Parameters
        ----------
        tf_model: tensorflow Sequential model with layer type Dense using ReLU activation.
        modeling_language: Optimization modelling language that the model will be written in (eg. 'PYOMO', 'GUROBI').
        name: String describing the ANN.
        """
        self._network_type_check(tf_model)
        self.name = name
        self._modelingInterface = InterfaceFactory().create(modeling_language, name)
        self.networkParam = self._init_network_param(tf_model)
        self._inputIsConnected = False
        self._outputIsConnected = False
        self._paramIsLoaded = False

    def connect_network_input(self, opt_model, input_vars: list) -> None:
        """
        Assigns the optimization model that the ANN is embedded in plus the ANN input variables (as a list).
        Connecting input variables is necessary before embedding the ANN. The input variable bounds will be used for
        bound tightening and should therefore be defined and finite.
        Parameters
        ----------
        opt_model : Parent optimization model
        input_vars : List of input variables for the ANN. Make sure that the variables are listed in the same order as
        during training of the loaded ANN.
        """
        if self._inputIsConnected:
            print("Warning: Overwriting network input.")
        self._modelingInterface.type_check(opt_model, "model")
        for var in input_vars:
            self._modelingInterface.type_check(var, "variable")
        assert (len(input_vars) == self.networkParam.input_dim)
        input_bounds = self._modelingInterface.get_variable_bounds(input_vars)
        self.networkParam.input_bounds = input_bounds
        self._modelingInterface.connect_network_input(opt_model=opt_model, input_vars=input_vars)
        self._inputIsConnected = True

    def connect_network_output(self, opt_model, output_vars: list) -> None:
        """
        Assigns he ANN output variables (as a list).
        Parameters
        ----------
        opt_model: Parent optimization model
        output_vars: List of output variables for the ANN.
        """
        if self._outputIsConnected:
            print("Warning: Overwriting network output.")
        if not self._inputIsConnected:
            raise Exception('Network input has to be connected before network output.')

        for var in output_vars:
            self._modelingInterface.type_check(var, "variable")
        if len(output_vars) != self.networkParam.output_dim:
            raise Exception('The specified output dimension is not equal to the one implied by the currently loaded '
                            'tensorflow model')

        self.networkParam.output_bounds = self._modelingInterface.get_variable_bounds(output_vars)
        # TODO: fbbt_backward_pass(self.network_param, output_bounds) - see Grimstad et. al Appendix

        self._modelingInterface.connect_network_output(opt_model=opt_model, output_vars=output_vars)
        self._outputIsConnected = True

    def embed_network_formulation(self, opt_model=None, input_vars=None, output_vars=None,
                                  mip_formulation: str = 'Big-M',
                                  bound_tightening_strategy: str = 'LP',
                                  node_time_limit: float = 1.,
                                  use_full_model: bool = False,
                                  solver=None):
        """
        Embeds an optimization formulation for the loaded ANN into the parent optimization variables.
        This defines the functional relationship
                        output_variables = f_ANN(input_variables)
        in the parent model, by introducing auxiliary variables and constraints describing the ANN function.
        The auxiliary variables are automatically tightened according to the specified strategy.
        Parameters
        ----------
        opt_model : Parent optimization model
        input_vars : Input variables for the ANN
        output_vars : Output variables for the ANN
        mip_formulation : Encodes the ReLU logic as MIP variables + constraints
        bound_tightening_strategy : Defines the method to compute bounds for the internal ANN variables
        node_time_limit : Time limit for each node during bound tightening
        use_full_model : Set to True, if the full parent model shall be used for bound tightening.
        solver : Solver object needed for some modelling interfaces (eg. Pyomo)
        """
        if input_vars is not None:
            self.connect_network_input(opt_model=opt_model, input_vars=input_vars)
        if output_vars is not None:
            self.connect_network_output(opt_model=opt_model, output_vars=output_vars)
        if not (self._outputIsConnected and self._inputIsConnected):
            raise Exception("Input and output variables must be connected to the ANN model before embedding the "
                            "formulation.")
        if not self._paramIsLoaded:
            self._fbbt_forward_pass()
        self._node_redundancy_check()
        self._modelingInterface.embed_network_formulation(opt_model=opt_model, ann_param=self.networkParam,
                                                          mip_formulation=mip_formulation,
                                                          bound_tightening_strategy=bound_tightening_strategy,
                                                          node_time_limit=node_time_limit,
                                                          use_parent_model=use_full_model,
                                                          solver=solver)

    def _fbbt_forward_pass(self) -> None:
        """Computes feasibility-based bounds for the ANN variables.

        """
        self.networkParam.LB[0] = self.networkParam.input_bounds[:, 0]
        self.networkParam.LB[0].shape = (self.networkParam.nodes_per_layer[0], 1)
        self.networkParam.UB[0] = self.networkParam.input_bounds[:, 1]
        self.networkParam.UB[0].shape = (self.networkParam.nodes_per_layer[0], 1)
        w_plus = [w.clip(min=0) for w in self.networkParam.weights]
        w_minus = [w.clip(max=0) for w in self.networkParam.weights]
        for i in range(1, self.networkParam.n_layers):
            tmp = np.matmul(np.transpose(w_plus[i - 1]),
                            self.networkParam.UB[i - 1]) + np.matmul(np.transpose(w_minus[i - 1]),
                                                                     self.networkParam.LB[i - 1]) + \
                                                                     self.networkParam.bias[i - 1]
            self.networkParam.UB[i] = tmp.clip(min=0)
            self.networkParam.M_plus[i - 1] = tmp
            tmp2 = np.matmul(np.transpose(w_plus[i - 1]), self.networkParam.LB[i - 1]) + np.matmul(
                np.transpose(w_minus[i - 1]), self.networkParam.UB[i - 1]) + self.networkParam.bias[i - 1]
            self.networkParam.M_minus[i - 1] = tmp2
        if type(self.networkParam.output_bounds) == np.ndarray:
            # If the computed bounds improve on the given output bounds, they are updated
            assert (np.shape(self.networkParam.output_bounds) == (self.networkParam.nodes_per_layer[-1], 2))
            for i in range(self.networkParam.nodes_per_layer[-1]):
                if self.networkParam.output_bounds[i, 0] > self.networkParam.M_minus[self.networkParam.n_layers - 2][i]:
                    self.networkParam.M_minus[self.networkParam.n_layers - 2][i] = self.networkParam.output_bounds[i, 0]
                    self.networkParam.LB[-1][i] = self.networkParam.output_bounds[i, 0]
                if self.networkParam.output_bounds[i, 1] < self.networkParam.M_plus[self.networkParam.n_layers - 2][i]:
                    self.networkParam.M_plus[self.networkParam.n_layers - 2][i] = self.networkParam.output_bounds[i, 1]
                    self.networkParam.UB[-1][i] = self.networkParam.output_bounds[i, 1]

    @staticmethod
    def _init_network_param(tf_model: tf.keras.Sequential) -> AnnParameters:
        """Extracts weights and biases from tensorflow model.

        Parameters
        ----------
        tf_model : tensorflow Sequential model with layer type Dense using ReLU activation.

        Returns
        -------
        NetworkParam : AnnParameters object storing all necessary ANN parameters
        """
        nn_params = tf_model.get_weights()
        bias = nn_params[1::2]
        for bi in bias:
            bi.shape = (len(bi), 1)
        weights = nn_params[0::2]
        n_layers = len(bias) + 1
        nodes_per_layer = np.array([np.shape(weights[i])[0] for i in range(len(weights))] + [len(bias[-1])])
        # Big-M values
        m_plus = [np.zeros((len(b), 1)) for b in bias]
        m_minus = [np.zeros((len(b), 1)) for b in bias]
        # Variable bounds
        lb = [np.zeros((n, 1)) for n in nodes_per_layer]
        ub = [np.zeros((n, 1)) for n in nodes_per_layer]

        redundancy_matrix = [np.zeros((n, 1)) for n in nodes_per_layer[1:-1]]

        network_param = AnnParameters(n_layers=n_layers, nodes_per_layer=nodes_per_layer, input_dim=nodes_per_layer[0],
                                      output_dim=nodes_per_layer[-1], weights=weights, bias=bias, M_plus=m_plus,
                                      M_minus=m_minus, LB=lb, UB=ub, redundancy_matrix=redundancy_matrix)
        return network_param

    @staticmethod
    def _network_type_check(tf_model: tf.keras.Sequential) -> None:
        """Checks if tensorflow model has correct specifications

        Parameters
        ----------
        tf_model : tensorflow Sequential model with Dense layers using ReLU activation.
        -------

        """
        if len(tf_model.layers) < 2:
            print('Warning: The tensorflow model does not appear to contain any hidden layers. This will lead to undefined behaviour.')
        for idx, layer in enumerate(tf_model.layers[:-1]):
            if not isinstance(layer, tf.keras.layers.Dense):
                raise Exception('Layer ' + str(idx) + ' of tensorflow model is not dense.')
            if layer.activation.__name__ != 'relu':
                raise Exception('Layer ' + str(idx) + ' of tensorflow model does not have ReLU activation')

    def _node_redundancy_check(self) -> None:
        """
        Determines which nodes in the ANN are redundant based on the current variable bounds. Results are stored in
        networkParam.redundancy_matrix. Always active nodes are marked with +1, always inactive ones with -1,
        non-redundant nodes with 0.
        """
        for i in range(1, self.networkParam.n_layers):
            for j in range(self.networkParam.nodes_per_layer[i]):
                if self.networkParam.M_plus[i - 1][j] <= 0:
                    self.networkParam.redundancy_matrix[i - 1][j] = -1
                elif self.networkParam.M_minus[i - 1][j] >= 0:
                    self.networkParam.redundancy_matrix[i - 1][j] = 1

    def save_param(self, filename: str = 'ann_parameters') -> None:
        """
        Saves the current variable bounds of the ANN as .npy files. Use load_param to load results into an ANN model.
        Parameters
        ----------
        filename: File path where parameters are saved.
        """
        np.save(filename + '.npy', [self.networkParam.M_plus, self.networkParam.M_minus], allow_pickle=True)

    def load_param(self, tf_model: tf.keras.Sequential, filename: str) -> None:
        """
        Loads ANN variable bounds from .npy file generated by save_param.
        Parameters
        ----------
        tf_model: The tensorflow model associated with the save parameters.
        filename: File path of saved parameters.
        """
        self._init_network_param(tf_model)
        data = np.load(filename, allow_pickle=True)
        self.networkParam.M_plus = data[0]
        self.networkParam.M_minus = data[1]
        self.networkParam.UB[1:] = self.networkParam.M_plus
        self._paramIsLoaded = True
