import tensorflow as tf
import numpy as np
import pyomo.environ as pyo

from relumip.ann_model import AnnModel
from relumip.utils.visualization import plot_results_2d

# Load the trained tensorflow model which will be embedded into the optimization problem.
tf_model = tf.keras.models.load_model('data/peaks_3x10.h5')

# Create a pyomo model into which the ANN will be embedded.
model = pyo.ConcreteModel()
model.construct()

# All network variables will be added to a user-defined block within the model.
model.ann = pyo.Block()

# The network input and output variables have to be defined by the user.
# For the network input, finite variable bounds have to be supplied (they can be inferred from the data used to train
# the model, for example).
model.ann.Input1 = pyo.Var(within=pyo.Reals, bounds=(-3, 3))
model.ann.Input2 = pyo.Var(within=pyo.Reals, bounds=(-3, 3))
model.ann.Output = pyo.Var(bounds=(-10000, 10000), within=pyo.Reals)

# Input and output variables are stored in lists to be passes to the AnnModel.
input_vars = [model.ann.Input1, model.ann.Input2]
output_vars = [model.ann.Output]

# A solver instance has to be defined for bound tightening. Make sure that an appropriate MIP solver is installed.
solver = pyo.SolverFactory('glpk')

# Now the AnnModel instance can be created.
ann_model = AnnModel(tf_model=tf_model, modeling_language='PYOMO')

# Input and output variables are connected to the network.
# The block dedicated for the ANN model has to be passed as well.
ann_model.connect_network_input(opt_model=model.ann, input_vars=input_vars)
ann_model.connect_network_output(opt_model=model.ann, output_vars=output_vars)

# This call generates the network formulation inside the block.
# The bound tightening strategy has to be specified, for Pyomo the options are 'MIP' or 'LP' (default).
ann_model.embed_network_formulation(bound_tightening_strategy='LP', solver=solver)

# In this example, no additional model components besides the ANN are considered.
# We choose to minimize the network output and display the solved model.
model.obj = pyo.Objective(expr=model.ann.Output, sense=pyo.minimize)
res = solver.solve(model)
model.display()

# To visualize the computed results, a test data set is generated within the ANN input domain and the tensorflow model
# is evaluated on it. The solution point computed above is extracted and shown on the response surface plot.
sample_input = 6 * np.random.rand(10000, 2) - 3
sample_output = tf_model.predict(sample_input)
sol_point = [input_vars[0].value, input_vars[1].value, output_vars[0].value]
plot_results_2d(sample_input, sample_output, sol_point=sol_point)

# The model parameters computed during bound tightening can be saved for future use of the same model. See the
# 'load_precomputed_parameters_example.py' file on more information on how to load precomputed parameters
ann_model.save_param('data/peaks3x10_param')

