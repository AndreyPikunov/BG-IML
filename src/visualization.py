import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    m,
    labels=None,
    normalize_plot=None,
    include_values=True,
    normalize_values=None,
    values_fontsize="x-small",
    xticks_rotation="horizontal",
    cmap="Greys",
    ax=None,
):

    assert m.shape[0] == m.shape[1]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if normalize_plot == "true":
        m_plot = m / m.sum(axis=1)[:, None]
    elif normalize_plot == "pred":
        m_plot = m / m.sum(axis=0)
    else:
        m_plot = m.copy()

    im = ax.imshow(m_plot, cmap=cmap)
    cmap = im.cmap

    if labels is None:
        labels = list(map(str, range(len(m))))

    assert m.shape[0] == len(labels)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

    if include_values:

        if normalize_values == "true":
            m_values = m / m.sum(axis=1)[:, None]
        elif normalize_values == "pred":
            m_values = m / m.sum(axis=0)
        elif normalize_values == "all":
            m_values = m / m.sum()
        else:
            m_values = m.copy()

        if normalize_values in ["true", "pred", "all"]:
            m_values = (100 * m_values).astype(int)

        for i in range(m.shape[0]):
            for j in range(m.shape[1]):

                c = cmap(255) if m_plot[i, j] / m_plot.max() < 0.5 else cmap(0)

                ax.text(
                    j,
                    i,
                    str(m_values[i, j]),
                    color=c,
                    ha="center",
                    va="center",
                    size=values_fontsize,
                )

    ax.set_xlim(-0.5, len(m) - 0.5)
    ax.set_ylim(len(m) - 0.5, -0.5)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    return fig, ax
