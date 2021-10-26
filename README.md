# reluMIP
> Embed ReLU neural networks into mixed-integer programs.

![](docs/reluMIP_methodology.svg)

## About
With this package, you can generate mixed-integer linear programming (MIP) models of trained artifical neural networks (ANNs) using the rectified liner unit (ReLU) activation function. At the moment, only `tensorflow` sequential models are supported. Interfaces to either the `Pyomo` or `Gurobi` modelling environments are offered.

ReLU ANNs can be used to approximate complex functions from data. In order to embed these functions into optimization problems, strong formulations of the network are needed. This package employs progressive bound tightening procedures to produce MIP encodings for ReLU networks. This allows the user to embed complex and nonlinear functions into mixed-integer programs. Note that the training of ReLU ANNs is not part of this package and has to be done by the user beforehand. A number of illustrative examples are provided to showcase the functionality of this package.


## Installation
This package is part of PyPI. It can be installed through `pip`:

```sh
pip install reluMIP
```
After installing, you can use the examples provided in the `examples/` folder to become familiar with the package.

Alternatively, you can clone the github repository:

```sh
git clone https://github.com/ChemEngAI/ReLU_ANN_MILP.git
```
You can install all requirements from the project root folder by calling:

```sh
pip install -r requirements.txt
```
You can add the root folder of the repositors to your `PYTHON_PATH`, so that the package can be accessed from anywhere. 

Note that in order to use the package, a compatible solver has to be installed. This can be [*Gurobi*](https://www.gurobi.com/) (with a valid license) or any MIP solver compatible with *Pyomo* (we recommend [*glpk*](https://www.gnu.org/software/glpk/)). In our experience, the best performance is obtained when using the *Gurobi* interface.

## Example usages
Two `jupyter` notebooks describing the use of the package are supplied in the `examples/` folder in the [`reluMIP` Git repository](https://github.com/ChemEngAI/ReLU_ANN_MILP). There, an MIP formulation of a ReLU ANN trained on a nonliner, nonconvex function is used to find the global minimum of the network response surface. If you installed the package through `pip`, you can simply download the example files that you are interested in.

## Related work on NLP formulations of ANNs
In this tool, ReLU ANNs are formulated as MILPs. Notably, ANNs can also be formulated as nonlinear problems (NLPs) and solved through deterministic gloabl optimization (Schweidtmann and Mitsos (2019)). Please visit the [MeLOn toolbox](https://git.rwth-aachen.de/avt-svt/public/MeLOn) for more information.

## References
Grimstad, B., Andersson, H. (2019). [ReLU networks as surrogate models in mixed-integer linear programs](https://doi.org/10.1016/j.compchemeng.2019.106580). *Computers & Chemical Engineering* (Volume 131, 106580).<br><br>

Schweidtmann, A. M., & Mitsos, A. (2019). [Deterministic global optimization with artificial neural networks embedded](https://doi.org/10.1007/s10957-018-1396-0). *Journal of Optimization Theory and Applications* (Volume 180(3), 925-948).<br><br>



