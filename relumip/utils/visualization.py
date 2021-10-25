import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_results_2d(x, y, sol_point):
    if x.shape[1] == 2 and y.shape[1] == 1:
        y.shape = x[:, 0].shape
        fig = plt.figure(figsize=plt.figaspect(0.4))
        ax = plt.axes(projection='3d')
        #ax = fig.gca(projection='3d')
        p1 = ax.plot_trisurf(x[:, 0], x[:, 1], y, cmap=cm.plasma, label='network response surface')
        p2 = ax.scatter(xs=sol_point[0], ys=sol_point[1], zs=sol_point[2], s=300, c='g', marker='*',
                        label='solution point of MIP minimization')
        # ax.legend()
        ax.set_title('network response surface with computed minimum')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    else:
        raise NotImplementedError('Visualization is only implemented for two-dimensional input and one-dimensional output data.')
