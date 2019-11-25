# General statistical analysis methods to pre-analyze the data
# such as scatter-plots, correlations...

from classes import Variable
import matplotlib.pyplot as plt


def scatter_xy(data, x: Variable, y: Variable, ax=None, position=None):
    """
    Makes a scatter-plot between x and y
    """
    data.plot(kind='scatter', x=x.col_name, y=y.col_name, s=0.8, ax=ax)
    ax.set_xlabel(x.name.capitalize())
    ax.set_ylabel(y.name.capitalize())

    # <position> is used for generating legible subplots
    if not position:
        plt.title(f"{y.name.capitalize()} against {x.name.capitalize()}")

    else:
        if position == "inner":
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        elif position == "left":
            ax.xaxis.label.set_visible(False)
            ax.set_xticks([])

        elif position == "bottom":
            ax.yaxis.label.set_visible(False)
            ax.set_yticks([])

        elif position == "corner":
            pass


def scatter_grid(data, vars):
    """
    Makes a pairwise correlation grid as subplots
    """
    fig, axes = plt.subplots(nrows=len(vars), ncols=len(vars))
    axis = [0, 0]
    for v1 in vars.values():
        axis[1] = 0
        for v2 in vars.values():
            position = "inner"
            if axis[0] == len(vars) - 1:
                position = "bottom" if axis[1] else "corner"
            elif axis[1] == 0:
                position = "left"
            scatter_xy(data, y=v1, x=v2, ax=axes[axis[0], axis[1]], position=position)
            axis[1] += 1
        axis[0] += 1


def correlation_matrix(data, col_names):
    """
    Returns the correlation matrix between variables in <col_names>
    """
    corr_matrix = data.corr()
    for c in corr_matrix.columns:
        if c not in col_names.keys():
            corr_matrix = corr_matrix.drop(c, axis=0).drop(c, axis=1)

    return corr_matrix
