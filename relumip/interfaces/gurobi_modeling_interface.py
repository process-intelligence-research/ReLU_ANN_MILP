import numpy as np
import gurobipy as grb
from gurobipy import GRB
import tqdm
import time
from typing import List

from ..utils.storage import AnnParameters
from .modeling_interface import ModelingInterface


class GurobiModelingInterface(ModelingInterface):
    """Interface for creating a Gurobi optimization model from ANN parameters. Variables and constraints describing the
    given ANN are inserted in a user defined Gurobi model. Boundaries for these auxiliary variables are
    determined through progressive bound tightening methods."""

    def __init__(self, name: str = ""):
        """Constructor initializing the Gurobi model describing the ANN.

        Parameters
        ----------
        name : Descriptor
        """
        super().__init__(name)
        self._ann_model = grb.Model(name=name)
        self._input = []
        self._output = []
        self._vars = {}
        self._constr = {}
        self._parent_model = None
        self._bound_tightening_strategy = ""
        self._node_time_limit = None
        self._use_parent_model = False

    def connect_network_input(self, opt_model: grb.Model, input_vars: List[grb.Var]):
        """Assigns the input variables of the ANN and the parent model where the ANN is to be embedded into.

        Parameters
        ----------
        input_vars: List of grb.Var describing the input vector to the network. Have to be defined in the parent model.
        opt_model: Parent optimization model containing the input_vars.
        """
        self._parent_model = opt_model
        self._input = input_vars

    def connect_network_output(self, opt_model: grb.Model, output_vars: List[grb.Var]):
        """Assigns the output variables of the ANN. Must be defined in the same parent model as the input variables.

        Parameters
        ----------
        opt_model: Parent optimization model
        output_vars: List of grb.Var describing the output (vector) of the ANN.
        """
        # rigorous test possible with Gurobi interface?
        assert(opt_model is self._parent_model)
        self._output = output_vars

    def embed_network_formulation(self, opt_model: grb.Model, ann_param: AnnParameters, mip_formulation: str,
                                  bound_tightening_strategy: str, node_time_limit: float,
                                  use_parent_model: bool, solver=None) -> None:
        """Auxiliary variables and constraints describing the ANN function are added to the parent optimization model.
        Bound tightening is performed on the network variables.

        Parameters
        ----------
        opt_model : Parent optimization model
        ann_param : AnnParameters object defining the ANN weights, biases and Big-M values for each node.
        bound_tightening_strategy : String describing the chosen bound tightening strategy.
        node_time_limit : Bound tightening time limit for each node.
        mip_formulation : MIP formulation for each node.
        use_parent_model : Set True if the full parent model shall be used for bound tigtening
        solver : Not needed for the Gurobi Interface
        """
        self._bound_tightening_strategy = bound_tightening_strategy
        self._node_time_limit = node_time_limit
        self._use_parent_model = use_parent_model
        if not self._use_parent_model:
            if mip_formulation == 'Big-M':
                self._build_model_bigm(model=self._ann_model, ann_param=ann_param)
            else:  # TODO: SOS, MC, ...
                pass
            self._insert_network_in_parent(ann_param, mip_formulation)
        else:
            if mip_formulation == 'Big-M':
                self._build_model_bigm(model=self._parent_model, ann_param=ann_param)
            else:
                pass

    def _build_model_bigm(self, model: grb.Model, ann_param: AnnParameters):
        """Iterates over the ANN nodes and inserts the optimization formulation for each one,
         after performing bound tightening.

        Parameters
        ----------
        model: Parent optimization model.
        ann_param: AnnParameters
        """
        model.setParam(GRB.Param.LogToConsole, 0)
        self._init_vars_bigm(model, ann_param)
        t = tqdm.tqdm(total=np.sum(ann_param.nodes_per_layer[2:]))
        for layerIdx in range(1, ann_param.n_layers):
            # TODO: Parallelize at deep levels of network
            if (1 < layerIdx):
                for nodeIdx in range(ann_param.nodes_per_layer[layerIdx]):
                    if (self._bound_tightening_strategy != ''):
                        self._solve_subproblem(layerIdx, nodeIdx, model, ann_param)
                    t.set_description('Evaluating node ({},{})'.format(layerIdx, nodeIdx))
                    t.update(1)
            if layerIdx < ann_param.n_layers - 1:
                for nodeIdx in range(ann_param.nodes_per_layer[layerIdx]):
                    self._add_hidden_node_formulation_bigm(layerIdx, nodeIdx, model, ann_param)
            else:
                self._add_output_nodes(model=model, ann_param=ann_param)

        t.close()
        if self._bound_tightening_strategy == 'LP':
            for z_layer in self._vars['z']:
                for z_node in z_layer:
                    z_node.setAttr(GRB.Attr.VType, GRB.BINARY)

    def _init_vars_bigm(self, model: grb.Model, ann_param: AnnParameters) -> None:
        """Defines Block the input variables of the ANN aswell as a dictionary tp store Variables and Constraints.

        Parameters
        ----------
        model: Parent optimization model.
        ann_param: AnnParameters.

        """
        self._vars['x'] = [[]]
        if model is not self._parent_model:
            for var in self._input:
                self._vars['x'][0].append(model.addVar(lb=var.getAttr(GRB.Attr.LB), ub=var.getAttr(GRB.Attr.UB),
                                                       name=var.getAttr(GRB.Attr.VarName)))
        else:
            self._vars['x'][0] = self._input
        self._vars['z'] = [[]]
        self._constr['c1'] = [[]]
        self._constr['c2'] = [[]]
        self._constr['c3'] = [[]]
        for i in range(1, ann_param.n_layers):
            self._vars['z'] .append([])
            self._vars['x'].append([])
            self._constr['c1'].append([])
            self._constr['c2'].append([])
            self._constr['c3'].append([])

    def _add_hidden_node_formulation_bigm(self, layer_idx: int, node_idx: int, model: grb.Model,
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
        self._vars['x'][layer_idx].append(model.addVar(ann_param.LB[layer_idx][node_idx],
                                                       ann_param.UB[layer_idx][node_idx], obj=0.,
                                                       vtype=grb.GRB.CONTINUOUS,
                                                       name=self._name + '_x_' + str(layer_idx) + str(node_idx)))

        self._vars['z'][layer_idx].append(model.addVar(0, 1, obj=0., vtype=grb.GRB.BINARY,
                                                       name=self._name + '_z_' + str(layer_idx) + str(node_idx)))

        self._constr['c1'][layer_idx].append([])
        self._constr['c2'][layer_idx].append([])
        self._constr['c3'][layer_idx].append([])
        linexp = grb.LinExpr([w for w in ann_param.weights[layer_idx - 1][:, node_idx]],
                             self._vars['x'][layer_idx - 1])
        linexp += ann_param.bias[layer_idx - 1][node_idx]

        if ann_param.M_minus[layer_idx - 1][node_idx, 0] >= 0:
            model.addConstr(self._vars['x'][layer_idx][node_idx] == linexp)
            self._vars['x'][layer_idx][node_idx].setAttr(GRB.Attr.LB, ann_param.M_minus[layer_idx - 1][node_idx, 0])
            self._vars['z'][layer_idx][node_idx].setAttr(GRB.Attr.UB, 1)
            self._vars['z'][layer_idx][node_idx].setAttr(GRB.Attr.LB, 1)
        elif ann_param.M_plus[layer_idx - 1][node_idx, 0] <= 0:
            self._vars['x'][layer_idx][node_idx].setAttr(GRB.Attr.UB, 0)
            self._vars['x'][layer_idx][node_idx].setAttr(GRB.Attr.LB, 0)
            self._vars['z'][layer_idx][node_idx].setAttr(GRB.Attr.UB, 0)
            self._vars['z'][layer_idx][node_idx].setAttr(GRB.Attr.LB, 0)
        else:
            self._constr['c1'][layer_idx][node_idx] = model.addConstr(
                self._vars['x'][layer_idx][node_idx] >= linexp)

            self._constr['c2'][layer_idx][node_idx] = model.addConstr(
                self._vars['x'][layer_idx][node_idx] <=
                linexp - ann_param.M_minus[layer_idx - 1][node_idx, 0] * (
                            1 - self._vars['z'][layer_idx][node_idx]))

            self._constr['c3'][layer_idx][node_idx] = model.addConstr(
                self._vars['x'][layer_idx][node_idx] <=
                ann_param.M_plus[layer_idx - 1][node_idx, 0] * self._vars['z'][layer_idx][node_idx])

    def _solve_subproblem(self, layer_idx: int, node_idx: int, model: grb.Model, ann_param: AnnParameters):
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
        linexp = grb.LinExpr([w for w in ann_param.weights[layer_idx - 1][:, node_idx]],
                             self._vars['x'][layer_idx - 1])
        linexp += ann_param.bias[layer_idx - 1][node_idx]

        model.setParam(GRB.Param.TimeLimit, self._node_time_limit)
        model.setObjective(linexp, GRB.MINIMIZE)
        # TODO: write callback that stops solver as soon as the incumbent is > 0

        if self._bound_tightening_strategy == 'LP':
            for z_layer in self._vars['z']:
                for z_node in z_layer:
                    z_node.setAttr(GRB.Attr.VType, GRB.CONTINUOUS)

        model.optimize()
        if model.getAttr(GRB.Attr.SolCount) != 0:
            if model.getAttr(GRB.Attr.Status) == GRB.OPTIMAL:
                lb = model.getAttr(GRB.Attr.ObjVal)
            else:
                lb = model.getAttr(GRB.Attr.ObjBound)

            if lb > ann_param.M_minus[layer_idx - 1][node_idx]:
                ann_param.M_minus[layer_idx - 1][node_idx] = lb
            if lb >= 0:
                # TODO: heuristic to determine whether to find upper or lower bound first
                #  (maybe whichever is closest to 0)
                return None

        else:
            print('No solution found within time limit')

        model.setObjective(linexp, GRB.MAXIMIZE)
        # TODO: write callback that stops solver as soon as the incumbent is < 0
        model.optimize()
        if model.getAttr(GRB.Attr.SolCount) != 0:
            if model.getAttr(GRB.Attr.Status) == GRB.OPTIMAL:
                ub = model.getAttr(GRB.Attr.ObjVal)
            else:
                ub = model.getAttr(GRB.Attr.ObjBound)

            if ub < ann_param.M_plus[layer_idx - 1][node_idx]:
                # change ub parameters
                ann_param.M_plus[layer_idx - 1][node_idx] = ub
                ann_param.UB[layer_idx][node_idx] = ub
        else:
            print('No solution found within time limit')

    def _add_output_nodes(self, model: grb.Model, ann_param: AnnParameters):
        """Adds variables describing the output layer of the network. No ReLu activation is applied here.

        Parameters
        ----------
        model : The ANN optimization model.
        ann_param : AnnParameters
        """

        if model is self._parent_model:
            self._vars['x'][-1] = self._output
            for idx, var in enumerate(self._vars['x'][-1]):
                if ann_param.M_plus[-1][idx] < var.getAttr(GRB.Attr.UB):
                    var.setAttr(GRB.Attr.UB, ann_param.M_plus[-1][idx])
                if ann_param.M_minus[-1][idx] > var.getAttr(GRB.Attr.LB):
                    var.setAttr(GRB.Attr.LB, ann_param.M_minus[-1][idx])
        else:
            for node_idx in range(ann_param.output_dim):
                self._vars['x'][-1].append(model.addVar(ann_param.M_minus[-1][node_idx, 0],
                                                        ann_param.M_plus[-1][node_idx, 0], obj=0.,
                                                        vtype=grb.GRB.CONTINUOUS,
                                                        name=self._name + '_output_' + str(node_idx)))
        for node_idx in range(ann_param.output_dim):
            linexp = grb.LinExpr([w for w in ann_param.weights[-1][:, node_idx]],
                                 self._vars['x'][-2])
            linexp += ann_param.bias[-1][node_idx]
            model.addConstr(self._vars['x'][-1][node_idx] == linexp)

    def _insert_network_in_parent(self, ann_param: AnnParameters, mip_formulation: str):
        """All network variables are added to the parent model. This function is used when a separate model was used for
        bound tightening (use_full_model = False).

        Parameters
        ----------
        ann_param : AnnParameters
        """
        self._parent_model.setParam(GRB.Param.LogToConsole, 1)
        self._init_vars_bigm(self._parent_model, ann_param)
        for layerIdx in range(1, ann_param.n_layers - 1):
            for nodeIdx in range(ann_param.nodes_per_layer[layerIdx]):
                if mip_formulation == 'Big-M':
                    self._add_hidden_node_formulation_bigm(layerIdx, nodeIdx, self._parent_model, ann_param)
        self._add_output_nodes(self._parent_model, ann_param)


    @staticmethod
    def get_variable_bounds(opt_vars: List[grb.Var]):
        """Queries bounds for Gurobi optimization variables.

        Parameters
        ----------
        opt_vars : List of Gurobi Variables

        Returns
        -------
        bounds : numpy ndarray with shape (D,2)
        """
        bounds = np.zeros((len(opt_vars), 2))
        for i, var in enumerate(opt_vars):
            bounds[i, 0] = float(var.getAttr(GRB.Attr.LB))
            bounds[i, 1] = float(var.getAttr(GRB.Attr.UB))
        return bounds

    @staticmethod
    def type_check(obj, type_str):
        """Checks if given objects are of the correct Pyomo type.

        Parameters
        ----------
        obj : Object to be checked.
        type_str : Type descriptor for check.
        """
    def get_model(self):
        pass
