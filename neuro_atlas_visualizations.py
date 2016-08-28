import numpy
import matplotlib
matplotlib.use('AGG')
from matplotlib import pyplot as plt, gridspec
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D  # this is required for projection='3d'


def plot_dict(accuracy_dict, output_path):

    fig = plt.figure()
    grid = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(grid[0, 0])
    independent_axis = sorted(accuracy_dict.iterkeys())
    means = map(lambda x: numpy.mean(accuracy_dict[x]), independent_axis)
    std = map(lambda x: numpy.std(accuracy_dict[x]), independent_axis)
    if any([std != 0]):
        ax.errorbar(independent_axis, means, yerr=std)
    else:
        ax.plot(independent_axis, means)
    fig.savefig(output_path, bbox_inches='tight')
