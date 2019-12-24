import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.palplot(sns.color_palette(colors))


def line_xy(x, y, **kwargs):
    return sns.lineplot(x, y, **kwargs)


def scatter_xy(data, x: str, y: str, axis=None, position=None, **kwargs):
    """
    Makes a scatter-plot between data[x] and data[y]
    :param data: Pandas DataFrame
    :param x: str
    :param y: str
    :param axis: Used by func. scatter_grid
    :param position: Used by func. scatter_grid
    :return plt
    """
    sns.set_palette("husl", 3)
    plot_obj = sns.scatterplot(x=x, y=y, data=data, ax=axis, **kwargs)
    axis.set_xlabel(x) if axis else None
    axis.set_ylabel(y) if axis else None

    # <position> is used for generating nice-looking subplots
    if not position:
        plot_obj.set_title(f"{y} against {x}")

    elif axis:
        if position == "inner":
            axis.xaxis.label.set_visible(False)
            axis.yaxis.label.set_visible(False)
            axis.set_xticks([])
            axis.set_yticks([])

        elif position == "left":
            axis.xaxis.label.set_visible(False)
            axis.set_xticks([])

        elif position == "bottom":
            axis.yaxis.label.set_visible(False)
            axis.set_yticks([])

        elif position == "corner":
            pass

    return plot_obj


def scatter_grid(data, col_names, **kwargs):
    """
    Makes a pairwise correlation grid as subplots
    :param data: DataFrame
    :param col_names: list [str]
    """
    fig, axes = plt.subplots(nrows=len(col_names), ncols=len(col_names))
    axis = [0, 0]
    for v1 in col_names:
        axis[1] = 0
        for v2 in col_names:
            position = "inner"
            if axis[0] == len(col_names) - 1:
                position = "bottom" if axis[1] else "corner"
            elif axis[1] == 0:
                position = "left"
            scatter_xy(data, y=v1, x=v2, axis=axes[axis[0], axis[1]], position=position, **kwargs)
            axis[1] += 1
        axis[0] += 1


def change_labels(plot_obj, labels):
    """
    Change labels of plot
    len(._legend.texts) == len(labels)
    """
    for text, label in zip(plot_obj.legend_.texts, labels):
        text.set_text(label)
