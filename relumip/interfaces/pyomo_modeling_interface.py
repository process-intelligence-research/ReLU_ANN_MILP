import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.core.expr.numvalue import RegisterNumericType

import numpy as np
import tqdm

from ..utils.storage import AnnParameters
from .modeling_interface import ModelingInterface


class PyomoModelingInterface(ModelingInterface):
    """Interface for creating a Pyomo optimization model from ANN parameters. Variables and constraints describing the
    given ANN are inserted in a user defined pyomo model in Block form. Boundaries for these auxiliary variables are
    determined through progressive bound tightening methods."""

    def __init__(self, name: str = ""):
        """Constructor initializing the Pyomo Block describing the ANN.

        Parameters
        ----------
        name : Descriptor
        """
        super().__init__(name)
        self._ann_model = None
        self._input = []
        self._output = []
        self._parent_model = None
        self._bound_tightening_strategy = ""
        self._node_time_limit = None
        self._solver = None
        self._use_parent_model = False

    def connect_network_input(self, input_vars: [pyo.Var], opt_model: pyo.Block):
        """Assigns the input variables to the ANN and the parent model where the ANN is to be embedded into.

        Parameters
        ----------
        input_vars: List of pyomo.Var describing the input vector to the network. Have to be defined in the parent model
        opt_model: Parent optimization model containing the input_vars
        """
        self._ann_model = opt_model
        self._input = input_vars

    def connect_network_output(self, opt_model: pyo.Block, output_vars: [pyo.Var]):
        """Assigns the output variables of the ANN. Must be defined in same Block as the input variables.

        Parameters
        ----------
        opt_model
        output_vars: List of pyo.Var describing the output (vector) of the ANN.
        """
        self._output = output_vars

    def embed_network_formulation(self, opt_model: pyo.Block, ann_param: AnnParameters,
                                  bound_tightening_strategy: str = 'LP', node_time_limit: float = None,
                                  mip_formulation: str = 'Big-M', use_parent_model: bool = False,
                                  solver: pyo.SolverFactory = pyo.SolverFactory('gurobi')) -> None:
        """A pyomo Block describing the ANN function is added to the parent optimization model.
        Bound tightening is performed on the network variables.

        Parameters
        ----------
        opt_model : Parent optimization model
        ann_param : AnnParameters object defining the ANN weights, biases and Big-M values for each node.
        bound_tightening_strategy : String describing the chosen bound tightening strategy.
        node_time_limit : Bound tightening time limit for each node. Not implemented in Pyomo.
        mip_formulation : MIP formulation for each node.
        use_parent_model : Not implemented in Pyomo.
        solver : SolverFactor to use for bound tightening.
        """
        if not (bound_tightening_strategy == 'LP' or bound_tightening_strategy == 'MIP' or
                bound_tightening_strategy == ''):
            raise Exception('The specified bound tightening strategy is not supported for Pyomo.')
        else:
            self._bound_tightening_strategy = bound_tightening_strategy

        if use_parent_model:
            raise Exception('Using the parent model for bound-tightening is not supported in Pyomo.')

        self._solver = solver
        self._node_time_limit = node_time_limit

        if not mip_formulation == 'Big-M':
            raise Exception('The specified MIP formulation is not supported in Pyomo.')
        else:
            self._build_model_bigm(model=self._ann_model, ann_param=ann_param)

    def _build_model_bigm(self, model: pyo.Block, ann_param: AnnParameters):
        """Iterates over the ANN nodes and inserts the optimization formulation for each one,
         after performing bound tightening.

        Parameters
        ----------
        model: Parent optimization model.
        ann_param: AnnParameters
        """
        self._init_vars_bigm(model, ann_param)
        t = tqdm.tqdm(total=np.sum(ann_param.nodes_per_layer[2:-1]))
        for layerIdx in range(1, ann_param.n_layers):
            # TODO: Parallelize at deep levels of network
            if (1 < layerIdx) and (self._bound_tightening_strategy != ''):
                for nodeIdx in range(ann_param.nodes_per_layer[layerIdx]):
                    self._solve_subproblem(layerIdx, nodeIdx, model, ann_param)
                    t.set_description('Evaluating node ({},{})'.format(layerIdx, nodeIdx))
                    t.update(1)
            if layerIdx < ann_param.n_layers - 1:
                for nodeIdx in range(ann_param.nodes_per_layer[layerIdx]):
                    self._add_hidden_node_formulation_bigm(layerIdx, nodeIdx, model, ann_param)
            else:
                self._connect_hidden_nodes(model=model, ann_param=ann_param)

        t.close()
        model.del_component(model.obj)
        if self._bound_tightening_strategy == 'LP':
            for idx in range(1, ann_param.n_layers - 1):
                model.nodes[idx, :].z.domain = pyo.Binary

    def _init_vars_bigm(self, model: pyo.Block, ann_param: AnnParameters) -> None:
        """Defines Block structure of the ANN model as well as the input variables and ConstraintLists.

        Parameters
        ----------
        model: Parent optimization model.
        ann_param: AnnParameters

        Returns
        -------

        """
        model.obj = pyo.Objective()
        RegisterNumericType(np.float64)

        def node_filter(m, layer_idx, node_idx):
            return layer_idx < ann_param.n_layers and node_idx < ann_param.nodes_per_layer[layer_idx]

        max_idx = np.max([np.max(ann_param.nodes_per_layer), ann_param.n_layers])
        self._ann_model.nodeSet = pyo.Set(initialize=pyo.RangeSet(0, max_idx)*pyo.RangeSet(0, max_idx),
                                          filter=node_filter)
        self._ann_model.nodes = pyo.Block(self._ann_model.nodeSet)

        self._ann_model.c1 = pyo.ConstraintList()
        self._ann_model.c2 = pyo.ConstraintList()
        self._ann_model.c3 = pyo.ConstraintList()
        self._ann_model.cR = pyo.ConstraintList()

        for i in range(ann_param.nodes_per_layer[0]):
            self._ann_model.nodes[0, i].x = pyo.Var(within=self._input[i].domain, bounds=(self._input[i].lb,
                                                                                          self._input[i].ub))

    # noinspection PyMethodMayBeStatic
    def _add_hidden_node_formulation_bigm(self, layer_idx: int, node_idx: int, model: pyo.Block,
                                          ann_param: AnnParameters):
        """Adds Big-M MIP formulation for each node. For example node y = relu(x) = max(w*x + b,0):
            y >= w*x + b
            y <= w*x + b - M_(1-z)
            y <= M+*z
           where w, b are weights and bias of the node, x is the output of the previous layers andM_,M+ are the Big-M
           parameters. z is a binary variable determining whether the node is active of not.

        Parameters
        ----------
        layer_idx : Current layer: 0 is input layer, 1 is 1st hidden layer, ...
        node_idx : Current node in current layer.
        model : The ANN optimization model.
        ann_param : AnnParameters
        """
        model.nodes[layer_idx, node_idx].x = pyo.Var(within=pyo.Reals,
                                                     bounds=(ann_param.LB[layer_idx][node_idx],
                                                             ann_param.UB[layer_idx][node_idx]))
        model.nodes[layer_idx, node_idx].z = pyo.Var(within=pyo.Binary)

        linexp = sum(model.nodes[layer_idx - 1, k].x * ann_param.weights[layer_idx - 1][k, node_idx] \
                     for k in range(ann_param.nodes_per_layer[layer_idx-1])) + ann_param.bias[layer_idx - 1][node_idx]

        if ann_param.M_minus[layer_idx - 1][node_idx, 0] >= 0:
            model.cR.add(model.nodes[layer_idx, node_idx].x == linexp)
            model.nodes[layer_idx, node_idx].x.setlb(ann_param.M_minus[layer_idx - 1][node_idx, 0])
            model.nodes[layer_idx, node_idx].z.setlb(1)
            model.nodes[layer_idx, node_idx].z.setub(1)
        elif ann_param.M_plus[layer_idx - 1][node_idx, 0] <= 0:
            model.nodes[layer_idx, node_idx].x.setlb(0)
            model.nodes[layer_idx, node_idx].x.setub(0)
            model.nodes[layer_idx, node_idx].z.setlb(0)
            model.nodes[layer_idx, node_idx].z.setub(0)
        else:
            model.c1.add(model.nodes[layer_idx, node_idx].x >= linexp)
            model.c2.add(model.nodes[layer_idx, node_idx].x <= linexp - ann_param.M_minus[layer_idx - 1][node_idx, 0]
                         * (1 - model.nodes[layer_idx, node_idx].z))

            model.c3.add(model.nodes[layer_idx, node_idx].x <= model.nodes[layer_idx, node_idx].z *
                         ann_param.M_plus[layer_idx - 1][node_idx, 0])

    def _solve_subproblem(self, layer_idx: int, node_idx: int, model: pyo.Block, ann_param: AnnParameters):
        """Computes Big-M values/ variable bounds for specific network node based on bound tightening strategy.
        For example node y = relu(x) = max(0,w*x+b):
        M+ = max w*x+b
        M_ = min w*x+b
        s.t. the network constraints up to the current layer are satisfied. For the 'LP' bound tightening strategy,
        integrality constraints on the z variables are relaxed

        Parameters
        ----------
        layer_idx : Current layer: 0 is input layer, 1 is 1st hidden layer, ...
        node_idx : Current node in current layer.
        model : The ANN optimization model.
        ann_param : AnnParameters

        Returns
        -------

        """
        RegisterNumericType(float)
        w = ann_param.weights[layer_idx - 1][:, node_idx]
        linexp = sum(self._ann_model.nodes[layer_idx - 1, k].x * ann_param.weights[layer_idx - 1][k, node_idx] for k in
                     range(ann_param.nodes_per_layer[layer_idx-1])) + ann_param.bias[layer_idx - 1][node_idx]

        model.del_component(model.obj)
        model.obj = pyo.Objective(expr=linexp, sense=pyo.minimize)
        # TODO: write callback that stops solver as soon as the incumbent is > 0

        if self._bound_tightening_strategy == 'LP':
            for idx in range(1, layer_idx - 1):
                model.nodes[idx, :].z.domain = pyo.Reals

        results = self._solver.solve(model)
        # model.load(results)
        if results.solver.status == SolverStatus.ok:
            if results.solver.termination_condition == TerminationCondition.optimal:
                lb = pyo.value(linexp)
                if lb > ann_param.M_minus[layer_idx - 1][node_idx]:
                    ann_param.M_minus[layer_idx - 1][node_idx] = lb
                if lb >= 0:
                    # This node is redundant
                    return None
            else:
                # No optimal solution found within time limit
                # TODO: Get objective bounds (if they exist)
                pass
        else:
            # Sub-problem terminated with an error
            pass

        model.del_component(model.obj)
        model.obj = pyo.Objective(expr=linexp, sense=pyo.maximize)

        # TODO: write callback that stops solver as soon as the incumbent is < 0
        results = self._solver.solve(model)
        # model.load(results)
        if results.solver.status == SolverStatus.ok:
            if results.solver.termination_condition == TerminationCondition.optimal:
                ub = pyo.value(linexp)
                if ub < ann_param.M_plus[layer_idx - 1][node_idx]:
                    # change ub parameters
                    ann_param.M_plus[layer_idx - 1][node_idx] = ub
                    ann_param.UB[layer_idx][node_idx] = ub
            else:
                # No optimal solution found within time limit
                # TODO: Get objective bounds (if they exist)
                pass
        else:
            # Sub-problem terminated with an error
            return None

    def _connect_hidden_nodes(self, model: pyo.Block, ann_param: AnnParameters):
        """Adds equality constraints connecting the network to the user defined input/ output variables.

        Parameters
        ----------
        model : The ANN optimization model.
        ann_param : AnnParameters
        """

        for node_idx, var in enumerate(self._output):
            linexp = sum(self._ann_model.nodes[ann_param.n_layers - 2, k].x *
                         ann_param.weights[ann_param.n_layers - 2][k, node_idx] for k in
                         range(ann_param.nodes_per_layer[ann_param.n_layers - 2])) + \
                         ann_param.bias[ann_param.n_layers - 2][node_idx]
            model.cR.add(self._output[node_idx] == linexp)
            var.setlb(ann_param.M_minus[-1][node_idx, 0])
            var.setub(ann_param.M_plus[-1][node_idx, 0])

        for node_idx, var in enumerate(self._input):
            self._ann_model.cR.add(
                var == self._ann_model.nodes[0, node_idx].x)

    @staticmethod
    def get_variable_bounds(opt_vars: [pyo.Var]):
        """Queries bounds for Pyomo optimization variables.

        Parameters
        ----------
        opt_vars : List of pyomo Variables

        Returns
        -------
        bounds : numpy ndarray with shape (D,2)
        """
        bounds = np.zeros((len(opt_vars), 2))
        for i, var in enumerate(opt_vars):
            bounds[i, 0] = float(var.lb)
            bounds[i, 1] = float(var.ub)
        return bounds

    @staticmethod
    def type_check(obj, type_str):
        """Checks if given objects are of the correct Pyomo type.

        Parameters
        ----------
        obj : Object to be checked.
        type_str : Type descriptor for check.
        """
        if type_str is "model":
            assert (isinstance(obj, pyo.ConcreteModel) or isinstance(obj, pyo.Block))
        elif type_str is "variable":
            assert (isinstance(obj, pyo.Var))


