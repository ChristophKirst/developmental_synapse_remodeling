# -*- coding: utf-8 -*-
"""
Example
=======

Analysis script.

Runs the model generates example figures as described in
Nakajo et al. 2025 https://www.biorxiv.org/content/10.1101/2025.02.27.640672v1
"""
__project__ = 'Developmental Synapse Remodeling'
__author__ = 'Christoph Kirst <christoph.kirst@ucsf.edu>'
__license__ = 'GNU GENERAL PUBLIC LICENSE (license.txt)'
__copyright__ = 'Copyright © 2025 by Christoph Kirst'

# imports
import os
import functools as ft
import numpy as np
import tqdm

from scipy.optimize import minimize

from model_2 import Model, ModelResult, ModelParameter, ModelState, ModelSettings, ModelStateHistory, ModelEventHistory
from dataset import ExperimentalData
from event_data import plot_event_sequences, plot_measurements
from utils.plotting import plt, set_axes, distribute_plots

figure_directory = './figures'
figures = dict()


# %% model - example

settings = ModelSettings(
    dt=1,              # time step (h)
    save_states=True,  # save model states
    save_events=True   # save single synapse event sequences
)

parameter = ModelParameter(
    r_b_n=1,      # new birth rate (1 / time)
    g_n_d=0.1,    # new decay rate (1 / (time * #synapses)
    g_n_s=0.01,   # new to stable rate (1 / (time * #synapses)
    g_s_d=0.001   # stable decay rate (1 / (time * #synapses)
)

model = Model(parameter=parameter, settings=settings)

state, result = model.simulate(steps=10000)

figures["model_example"] = plt.figure(1, figsize=(25, 10))
plt.clf()
ax = plt.subplot(1, 2, 1)
result.plot_event_sequences(ax=ax)
ax = plt.subplot(1, 2, 2)
result.plot_states(ax=ax, title='synapse counts', legend="lower right")
plt.tight_layout()


# %% model distributions example

figures["model_example_distributions"] = plt.figure(2, figsize=(30, 10))
plt.clf()

# occurred synapses
first_times = np.linspace(20, result.state_history.max_time - 20, 1000)
occurred = [result.event_history.occurred(ts=tf-10, tf=tf, tl=tf+10, te=tf+20).sum() for tf in first_times]
n = np.arange(np.max(occurred) + 2)
occurred_distribution = result.parameter.measure_distribution(n, measure="occurred_mean", ts=10, tf=20, tl=30, te=40)

ax = plt.subplot(1, 3, 1)
ax.hist(occurred, bins=n-0.5, density=True, label='measured', color='purple', alpha=0.4)
ax.scatter(n, occurred_distribution, label='theory', color='grey')
set_axes(ax, title='occurred synapses', axes_labels=('$O_{f,l}$', '$p(O_{f,l})$'), legend=True)

# appeared synapses
appeared = [result.event_history.appeared(ts=tf-20, tf=tf).sum() for tf in first_times]
n = np.arange(np.max(appeared) + 2)
appeared_distribution = result.parameter.measure_distribution(n, measure="appeared_mean", ts=0, tf=20)

ax = plt.subplot(1, 3, 2)
ax.hist(appeared, bins=n-0.5, density=True, label='measured', color='purple', alpha=0.4)
ax.scatter(n, appeared_distribution, label='theory', color='grey')
set_axes(ax, title='appeared synapses', axes_labels=('$B_{f}$', '$p(B_{f})$'), legend=True)

# lifetimes
result.event_history.update_event_sequences(use_complete_sequences_only=True)
durations = result.event_history.event_durations()
max_duration = np.max(durations) / 10
times = np.linspace(0, max_duration, 100)

ax = plt.subplot(1, 3, 3)
pdf, ref = np.histogram(durations, density=True, bins=times)
valid = pdf != 0
ax.scatter(((ref[:-1] + ref[1:])/2)[valid], np.log(pdf[valid]), label='numerical', color='purple', alpha=0.2)
ax.plot(times, np.log(result.parameter.lifetime_distribution(times)), label='theory', color='grey')
set_axes(ax, title='life time', axes_labels=('life time $\\Delta t$', '$\\log(p(\\Delta t))$'), legend=True)


# %% save figures

for name, figure in figures.items():
    filename = os.path.join(figure_directory, f"{name}.png")
    figure.savefig(filename)
    print(f"saved {filename}")
