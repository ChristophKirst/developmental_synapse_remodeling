# -*- coding: utf-8 -*-
"""
Plotting
========

Mixins for plotting methods for modeling and data analysis.

Examples
--------
>>> from utils.plotting import lighter_color
>>> lighter_color('green')
array([0.3       , 0.65137255, 0.3       ])
"""
__project__ = 'Developmental Synapse Remodeling'
__author__ = 'Christoph Kirst <christoph.kirst@ucsf.edu>'
__copyright__ = 'Copyright © 2025 by Christoph Kirst'

# imports
import logging
import numpy as np

# logger
logger = logging.getLogger('synapse simulation')

# setup plotting for pycharm
try:
    import IPython
    IPython.get_ipython().run_line_magic("gui", "qt")   # qt for interactive figures in pycharm
except ModuleNotFoundError:
    pass

import matplotlib as mpl
try:
    mpl.use('Qt5Agg')  # qt for interactive figures in pycharm
except ValueError:
    pass
mpl.rcParams['font.size'] = 20  # set global font size

import matplotlib.pyplot as plt
import matplotlib.collections as mplc
import matplotlib.lines

from matplotlib.axes import Axes


# methods

def distribute_plots(n_plots: int) -> (int, int):
    p = int(np.ceil(np.sqrt(n_plots)))
    q = int(np.ceil(n_plots / p))
    return p, q


def set_axes(
        ax: Axes,
        title: str or None = None,
        axes_labels: tuple | None = None,
        legend: bool | str | None = None
):
    ax.set_title(title)
    if axes_labels is not None:
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
    if legend is not None:
        loc = legend if isinstance(legend, str) else None
        ax.legend(loc=loc)


def lighter_color(color, fraction=0.5):
    color = np.array(mpl.colors.to_rgb(color))
    delta = np.ones(3) - color[..., :3]
    color[..., :3] = color[..., :3] + fraction * delta
    return color


def darker_color(color, fraction=0.5):
    color = np.array(mpl.colors.to_rgb(color))
    color[..., :3] = color[..., :3] * fraction
    return color


def default_colors() -> tuple:
    return tuple(mpl.colors.to_rgb(c) for c in mpl.colors.cnames.keys())
