import tensorflow as tf
import numpy as np
import gurobipy as grb
from relumip.ann_model import AnnModel

from relumip.utils.visualization import plot_results_2d

# Load the trained tensorflow model which will be embedded into the optimization problem
tf_model = tf.keras.models.load_model('data/peaks_3x10.h5')

# Create a pyomo model into which the ANN will be embedded.
opt_model = grb.Model()

# The network input and output variables have to be defined by the user.
# For the network input, finite variable bounds have to be supplied (they can be inferred from the data used to train
# the model, for example).
# Input and output variables are stored in lists to be passed to the AnnModel.
input_vars = [opt_model.addVar(-3, 3, vtype=grb.GRB.CONTINUOUS), opt_model.addVar(-3, 3, vtype=grb.GRB.CONTINUOUS)]
output_vars = [opt_model.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS, name='output')]
opt_model.update()

# Now the AnnModel instance can be created.
ann_model = AnnModel(tf_model=tf_model, modeling_language='GUROBI')

# Input and output variables are connected to the network.
# The parent optimization model has to be passed as well.
ann_model.connect_network_input(opt_model, input_vars)
ann_model.connect_network_output(opt_model, output_vars)

# This call generates the network formulation inside the block.
# The bound tightening strategy has to be specified, for Gurobi the options are 'MIP' or 'LP' (default).
# Additionally, a node time limit can be defined. Each bound tightening sub-problem for a hidden node will be terminated
# after the specified time, and the current bounds on the objective will be used as bounds (default is 1 second).
ann_model.embed_network_formulation(bound_tightening_strategy='LP')

# In this example, no additional model components besides the ANN are considered.
# We choose to minimize the network output and display the solved model.
opt_model.setObjective(output_vars[0], grb.GRB.MINIMIZE)
opt_model.optimize()

# To visualize the computed results, a test data set is generated within the ANN input domain and the tensorflow model
# is evaluated on it. The solution point computed above is extracted and shown on the response surface plot.
sample_input = 6 * np.random.rand(10000, 2) - 3
sample_output = tf_model.predict(sample_input)

sol_point = [input_vars[0].getAttr(grb.GRB.Attr.X),
             input_vars[1].getAttr(grb.GRB.Attr.X),
             output_vars[0].getAttr(grb.GRB.Attr.X)]
plot_results_2d(sample_input, sample_output, sol_point=sol_point)

# The model parameters computed during bound tightening can be saved for future use of the same model. See the
# 'load_precomputed_parameters_example.py' file on more information on how to load precomputed parameters
ann_model.save_param('data/peaks3x10_param')
