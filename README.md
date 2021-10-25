# reluMIP
> Embed ReLU neural networks into mixed-integer programs.

![](docs/logo.png)

## About
With this package, you can generate mixed-integer linear programming (MIP) models of trained artifical neural networks (ANNs) using the rectified liner unit (ReLU) activation function. At the moment, only `tensorflow` sequential models are supported. Interfaces to either the `Pyomo` or `Gurobi` modelling environments are offered.

ReLU ANNs can be used to approximate complex functions from data. In order to embed these functions into optimization problems, strong formulations of the network are needed. This package employs progressive bound tightening procedures to produce MIP encodings for ReLU networks. This allows the user to embed complex and nonlinear functions into mixed-integer programs. Note that the training of ReLU ANNs is not part of this package and has to be done by the user beforehand. A number of illustrative examples are provided to showcase the functionality of this package.


## Installation
This package is part of PyPI. It can be installed through `pip`:

```sh
pip install reluMIP
```

Alternatively, you can clone the github repository:

```sh
git clone https://git.rwth-aachen.de/avt.svt/public/milp_formulation_for_relu_anns.git
```
Note that either `pyomo` or `gurobipy` (with a Gurobi license) have to be installed to use this package. You can install all requirements from the project root folder by calling:

```sh
pip install -r requirements.txt
```

## Example usages
Two `jupyter` notebooks describing the use of the package are supplied in the `examples` folder. There, an MIP formulation of a ReLU ANN trained on a nonliner, nonconvex function is used to find the global minimum of the network response surface.

## References
Grimstad, B., Andersson, H. (2019). [ReLU networks as surrogate models in mixed-integer linear programs](https://doi.org/10.1016/j.compchemeng.2019.106580). *Computers & Chemical Engineering* (Volume 131, 106580).<br><br>





