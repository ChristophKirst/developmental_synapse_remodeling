# -*- coding: utf-8 -*-
"""
Analysis
========

Analysis script.

Runs the model and analysis and data fitting and generates figures as described in
Nakajo et al. 2025 https://www.biorxiv.org/content/10.1101/2025.02.27.640672v1
"""
__project__ = 'Developmental Synapse Remodeling'
__author__ = 'Christoph Kirst <christoph.kirst@ucsf.edu>'
__license__ = 'GNU GENERAL PUBLIC LICENSE (license.txt)'
__copyright__ = 'Copyright © 2025 by Christoph Kirst'


# imports
import os
import copy
import functools as ft
import warnings
import tqdm
import numpy as np
import scipy.stats as stats

from model_2 import Model, ModelResult, ModelParameter, ModelState, ModelSettings, ModelStateHistory, ModelEventHistory
from dataset import ExperimentalData
from event_data import plot_event_sequences, plot_measurements, plot_survival_data, Measurement
from utils.plotting import plt, set_axes, distribute_plots, Axes, darker_color, lighter_color

figure_directory = './figures'
figures = dict()


# %% experimental data

measurement_times = np.array([0, 6, 12, 24, 48, 72])

data_wt = ExperimentalData(data_sheet='wt').sort(('duration', 'first', 'last'))
data_ko = ExperimentalData(data_sheet='ko').sort(('duration', 'first', 'last'))  # mmp14b deficient condition

# data_wt = ExperimentalData(data_sheet='wt_24').sort(('duration', 'first', 'last'))
# data_ko = ExperimentalData(data_sheet='ko_24').sort(('duration', 'first', 'last'))  # mmp14b deficient condition
#
# coarse_wt = data_wt.coarsen([0, 24])
# coarse_ko = data_ko.coarsen([0, 24])
#
# data_wt = ExperimentalData(data_sheet='wt_24_1').sort(('duration', 'first', 'last'))
# data_ko = ExperimentalData(data_sheet='ko_24_1').sort(('duration', 'first', 'last'))  # mmp14b deficient condition
#
# coarse_wt = coarse_wt.join(data_wt.coarsen([0, 24]))
# coarse_ko = coarse_ko.join(data_ko.coarsen([0, 24]))
#
# data_wt = ExperimentalData(data_sheet='wt_24').sort(('duration', 'first', 'last'))
# data_ko = ExperimentalData(data_sheet='ko_24').sort(('duration', 'first', 'last'))  # mmp14b deficient condition
#
# coarse_wt = data_wt.coarsen([0, 24])
# coarse_ko = data_ko.coarsen([0, 24])

#
# coarse_wt = coarse_wt.join(data_wt.coarsen([0, 24]))
# coarse_ko = coarse_ko.join(data_ko.coarsen([0, 24]))
#
# data_wt = coarse_wt.sort(('duration', 'first', 'last'))
# data_ko = coarse_ko.sort(('duration', 'first', 'last'))

measure_kwargs = dict(
    measure='occurred',  # measure (see event_data.py for definitions)
    measurement_times_borders=(-np.inf, np.inf),  # include -inf, inf to measurement times with borders
)

data_counts_wt = data_wt.measure(**measure_kwargs)
data_counts_ko = data_ko.measure(**measure_kwargs)

data_density_wt = data_wt.measure(**measure_kwargs, normalize=data_wt.total_dendrite_length)
data_density_ko = data_ko.measure(**measure_kwargs, normalize=data_ko.total_dendrite_length)

figures["experimental_data"] = plt.figure(1, figsize=(25, 15))
plt.clf()

ax = plt.subplot(2, 2, 1)
data_wt.plot(ax=ax, title='wt', axes_labels=('time [h]', 'synapse id'), event_sequence_ids=True, legend=False)

ax = plt.subplot(2, 2, 2)
data_ko.plot(ax=ax, title='ko', axes_labels=('time [h]', 'synapse id'), event_sequence_ids=True, legend=False)

ax = plt.subplot(2, 2, 3)
plot_measurements(
    measurements=[data_counts_wt, data_counts_ko],
    labels=['wt', 'ko'],
    colors=['gray', 'darkgreen'],
    ax=ax,
    title='synapse occurrences (absolute count)',
    axes_labels=('occurred $(t_f, t_l)$ [h]', 'synapse counts')
)

ax = plt.subplot(2, 2, 4)
plot_measurements(
    measurements=[data_density_wt, data_density_ko],
    labels=['wt', 'ko'],
    colors=['gray', 'darkgreen'],
    ax=ax,
    title='synapse occurrences (normalized density)',
    axes_labels=('occurred $(t_f, t_l)$ [h]', 'synapse density [1/$\\mu$m]'),
)

plt.tight_layout()


# %% empirical survival curves and life time distributions

measure_continued = dict(
    measure='continued',
    measurement_times_borders=(-np.inf, None),
)
data_continued_wt = data_wt.measure(**measure_continued)
data_continued_ko = data_ko.measure(**measure_continued)

measure_survival = dict(
    measure='survived',
    measurement_times_borders=(-np.inf, None),
)
data_survival_wt = data_wt.measure(**measure_survival)
data_survival_ko = data_ko.measure(**measure_survival)

data_durations_wt_raw = data_wt.event_sequences[:, -1] - data_wt.event_sequences[:, 0]
data_durations_ko_raw = data_ko.event_sequences[:, -1] - data_ko.event_sequences[:, 0]
durations = np.unique(data_durations_wt_raw)

data_durations_wt = {d: (data_durations_wt_raw == d).sum() / len(data_durations_wt_raw) for d in durations}
data_durations_ko = {d: (data_durations_ko_raw == d).sum() / len(data_durations_ko_raw) for d in durations}

data_durations_relative = {d: data_durations_wt[d] - data_durations_ko[d] for d in durations}

figures["experimental_data_survival"] = plt.figure(2, figsize=(25, 15))
plt.clf()

ax = plt.subplot(2, 2, 1)
plot_survival_data(
    measurements=[data_continued_wt, data_continued_ko],
    labels=['wt', 'ko'],
    colors=['gray', 'darkgreen'],
    ax=ax,
    title='empirical synapse survival (absolute count)',
    axes_labels=('time [h]', 'synapse counts'),
)

ax = plt.subplot(2, 2, 2)
plot_survival_data(
    measurements=[data_survival_wt, data_survival_ko],
    labels=['wt', 'ko'],
    colors=['gray', 'darkgreen'],
    ax=ax,
    title='empirical synapse survival (normalized)',
    axes_labels=('time [h]', '% survived'),
)

ax = plt.subplot(2, 2, 3)
plot_measurements(
    measurements=[data_durations_wt, data_durations_ko],
    labels=['wt', 'ko'],
    colors=['gray', 'darkgreen'],
    ax=ax,
    title='empirical synapse life time', axes_labels=('duration [h]', '% counts')
)

ax = plt.subplot(2, 2, 4)
plot_measurements(
    measurements=[data_durations_relative],
    labels=['wt - ko'],
    ax=ax,
    colors=['darkred'],
    title='relative empirical durations', legend='lower left', axes_labels=('duration', '% difference')
)
ax.hlines(0, 0, 9, color='lightgrey')

plt.tight_layout()


# %% model fit - wt

measure_fit_kwargs = measure_kwargs.copy()
measure_fit_kwargs.update(dict(measurement_key_with_borders=True))

data_for_fit_wt = data_wt.measure(**measure_fit_kwargs)
keys_for_fit = list(data_for_fit_wt.keys())

parameter0 = ModelParameter(r_b_n=2.0, g_n_d=0.05, g_n_s=0, g_s_d=1)
parameter = ModelParameter(r_b_n=2.0, g_n_d=0.05, g_n_s=0.01, g_s_d=0.002)

parameter_bounds = dict(g_n_d=(1e-10, 1), g_s_d=(1e-10, 0.1))
fixed_parameter = dict(g_n_s=0, g_s_d=1)

fit0_wt = parameter0.fit(data=data_for_fit_wt, parameter_bounds=parameter_bounds, fixed_parameter=fixed_parameter)
fit0_parameter_wt = fit0_wt.parameter
print(fit0_wt)

fit_wt = parameter.fit(data=data_for_fit_wt, parameter_bounds=parameter_bounds)
fit_parameter_wt = fit_wt.parameter
print(fit_wt)

fit0_counts_wt = fit0_parameter_wt.measure(measurement_times=data_counts_wt.measurement_times())
fit_counts_wt = fit_parameter_wt.measure(measurement_times=data_counts_wt.measurement_times())

aic0_wt = (2 * 2 - 2 * fit0_wt.log_likelihood)
aic_wt = (2 * 4 - 2 * fit_wt.log_likelihood)
delta_aic_wt = aic_wt - aic0_wt
p_1_state_model_wt = np.exp(delta_aic_wt)
fit_summary_wt = {
    "AIC(1-state)": aic0_wt,
    "AIC(2-state)": aic_wt,
    "AIC(2) - AIC(1)": delta_aic_wt,
    "p(1-state model)": p_1_state_model_wt
}
print("Summary: model fit: wt\n", *[f"{k:>20} = {v}\n" for k, v in fit_summary_wt.items()])


# %% model fit - wt: 1- vs 2 state model comparison

def plot_survival(
        ax: Axes,
        fit: ModelParameter,
        data: Measurement,
        times: np.ndarray,
        colors: list = ('lightblue', 'orange'),
        alpha: float = 0.4,
        curves: list | None = None,
        legends: list | None = None,
):
    data_curves = data.to_evolution_curves()
    measurement_times = np.concatenate([[-np.inf], data.measurement_times()])
    curves = range(len(data_curves)) if curves is None else curves
    legends = [] if legends is None else legends
    for i, curve in enumerate(data_curves):
        if i in curves:
            tf = curve[0, 0]
            ts = measurement_times[i]
            fit_times = np.array([t for t in times if t >= tf])
            fit_curve_n = np.array([fit.c_n_mean(ts, tf, t) for t in fit_times])
            fit_curve_s = np.array([fit.c_s_mean(ts, tf, t) for t in fit_times])
            fit_curve = fit_curve_n[0] + fit_curve_s[0]
            ax.stackplot(fit_times, fit_curve_n / fit_curve, fit_curve_s / fit_curve, colors=colors, alpha=alpha,
                         labels=['new', 'stable'])
            ax.scatter(curve[:, 0], curve[:, 1], color='gray', s=50, label='data')
            ax.plot(curve[:, 0], curve[:, 1], color='gray')
            if i in legends:
                ax.legend(loc='upper right')


def plot_one_vs_two(data_counts, data_survival, fit, fit_counts, fit0, fit0_counts, fit_summary, condition='wt'):
    measurement_times = data_counts.measurement_times()
    max_time = np.max(measurement_times)

    keys = list(data_counts.keys())
    keys_large = [k for k in keys if k == (0, max_time)]
    keys_small = [k for k in keys if k not in keys_large]

    import matplotlib.gridspec as grid_spec
    gs = grid_spec.GridSpec(3, 2 * (len(measurement_times) - 1))

    for p, keys_plot in enumerate([keys_small, keys_large]):
        ax = plt.subplot(gs[0, :2*(len(measurement_times)-1)-1] if p == 0 else gs[0, -1])  # noqa

        counts_plot = Measurement({k: data_counts[k] for k in keys_plot})
        counts_fit0_plot = Measurement({k: fit0_counts[k] for k in keys_plot})
        counts_fit_plot = Measurement({k: fit_counts[k] for k in keys_plot})

        title = None if p > 0 else \
            f"1- and 2-state model fit to occurred $O_{{f,l}}$ ($\\Delta$AIC = {fit_summary['AIC(2) - AIC(1)']:.2f})"

        plot_measurements(
            measurements=[counts_plot, counts_fit0_plot, counts_fit_plot],
            labels=[f'data {condition}', '1-state model fit', '2-state model fit'],
            colors=['gray', 'darkgray', 'purple'],
            title=title,
            legend=None if p > 0 else "upper center",
            axes_labels=(None, "synapse count"),
            ax=ax,
            rotate_ticks=20
        )

    # survival curves
    for i in range(len(measurement_times)-1):  # noqa
        times = np.linspace(0, max_time * 1.05, int(100 * max_time * 1.05))

        # ax = plt.subplot(3, 5, i + 6)
        ax = plt.subplot(gs[1, 2*i:2*(i+1)])  # noqa

        plot_survival(ax, fit0.parameter, data_survival, times=times, curves=[i])
        set_axes(ax, title=f"{condition} t={data_survival.measurement_times()[i]}h",
                 axes_labels=("time [h]", "% survival"))

        ax = plt.subplot(gs[2, 2*i:2*(i+1)])  # noqa
        plot_survival(ax, fit.parameter, data_survival, times=times, curves=[i], legends=[i] if i == 2 else [])
        set_axes(ax, title=f"{condition} t={data_survival.measurement_times()[i]}h",
                 axes_labels=("time [h]", "% survival"))

    gs.tight_layout(figure=plt.gcf())


figures["model_fit_one_vs_two_wt"] = plt.figure(3, figsize=(25, 15))
plt.clf()
plot_one_vs_two(
    data_counts_wt, data_survival_wt, fit_wt, fit_counts_wt, fit0_wt, fit0_counts_wt, fit_summary_wt, condition='wt')


# %% model fit - ko

data_for_fit_ko = data_ko.measure(**measure_fit_kwargs)

fit0_ko = parameter0.fit(data=data_for_fit_ko, parameter_bounds=dict(g_n_d=(1e-10, 1)),
                         fixed_parameter=dict(g_n_s=0, g_s_d=1))
fit0_parameter_ko = fit0_ko.parameter
print(fit0_ko)

fit_ko = parameter.fit(data=data_for_fit_ko, parameter_bounds=parameter_bounds)
fit_parameter_ko = fit_ko.parameter
print(fit_ko)  # noqa

fit0_counts_ko = fit0_parameter_ko.measure(measurement_times=data_counts_wt.measurement_times())
fit_counts_ko = fit_parameter_ko.measure(measurement_times=data_counts_wt.measurement_times())
data_counts_ko = data_ko.measure(**measure_kwargs)  # , normalize=data_wt.total_dendritic_length)

aic0_ko = (2 * 2 - 2 * fit0_ko.log_likelihood)
aic_ko = (2 * 4 - 2 * fit_ko.log_likelihood)
delta_aic_ko = aic_ko - aic0_ko
p_1_state_model_ko = np.exp(delta_aic_ko)
fit_summary_ko = {
    "AIC(1-state)": aic0_ko,
    "AIC(2-state)": aic_ko,
    "AIC(2) - AIC(1)": delta_aic_ko,
    "p(1-state model)": p_1_state_model_ko
}
print("Summary: model fit: ko\n", *[f"{k:>20} = {v}\n" for k, v in fit_summary_ko.items()])


# %% model fit - ko: 1- vs 2 state model comparison

figures["model_fit_one_vs_two_ko"] = plt.figure(4, figsize=(25, 15))
plt.clf()
plot_one_vs_two(
    data_counts_ko, data_survival_ko, fit_ko, fit_counts_ko, fit0_ko, fit0_counts_ko, fit_summary_ko, condition='ko')


# %% MLE parameter estimation - boostrap

n_samples = 100  # number of samples

sample_size_wt = int(0.9 * len(data_wt))  # sample size
sample_size_ko = int(0.9 * len(data_ko))
replace = False

scale_wt = len(data_wt) / sample_size_wt
scale_ko = len(data_ko) / sample_size_ko

data_samples_wt = []
data_samples_ko = []

data_counts_samples_wt = np.zeros((n_samples, len(keys_for_fit)), dtype=int)
data_counts_samples_ko = np.zeros((n_samples, len(keys_for_fit)), dtype=int)

fit_parameter_samples_wt = []
fit_parameter_samples_ko = []

for s in tqdm.tqdm(range(n_samples)):
    data_sample_wt = data_wt.resample(replace=False, size=sample_size_wt)
    data_sample_ko = data_ko.resample(replace=False, size=sample_size_ko)

    data_counts_sample_wt = data_sample_wt.measure(**measure_fit_kwargs, normalize=1 / scale_wt)
    data_counts_sample_ko = data_sample_ko.measure(**measure_fit_kwargs, normalize=1 / scale_ko)

    data_counts_samples_wt[s] = [data_counts_sample_wt[key] for key in keys_for_fit]
    data_counts_samples_ko[s] = [data_counts_sample_ko[key] for key in keys_for_fit]

    # with warnings.catch_warnings(record=False):
    #    warnings.simplefilter("ignore")
    parameter_bounds = dict(g_n_d=(1e-9, 0.1), g_n_s=(1e-9, 0.1), g_s_d=(1e-9, 0.01))

    fit_sample_wt = parameter.fit(data_counts_sample_wt, parameter_bounds=parameter_bounds)
    fit_sample_ko = parameter.fit(data_counts_sample_ko, parameter_bounds=parameter_bounds)

    if fit_sample_wt.success:
        fit_parameter_samples_wt.append(fit_sample_wt.parameter)
    if fit_sample_ko.success:
        fit_parameter_samples_ko.append(fit_sample_ko.parameter)

fit_parameter_samples_wt = ModelParameter(
    *[np.array([getattr(p, k) for p in fit_parameter_samples_wt]) for k in parameter._fields]  # noqa
)
fit_parameter_samples_ko = ModelParameter(
    *[np.array([getattr(p, k) for p in fit_parameter_samples_ko]) for k in parameter._fields]  # noqa
)

parameter_fit_mean_wt = ModelParameter(*[np.mean(p) for p in fit_parameter_samples_wt])
parameter_fit_mean_ko = ModelParameter(*[np.mean(p) for p in fit_parameter_samples_ko])

parameter_fit_std_wt = ModelParameter(*[np.std(p, ddof=1) for p in fit_parameter_samples_wt])
parameter_fit_std_ko = ModelParameter(*[np.std(p, ddof=1) for p in fit_parameter_samples_ko])

if False:  # plot and save samples
    plt.figure(10, figsize=(25, 20))  # noqa
    plt.clf()

    ax = plt.subplot(2, 1, 1)
    for k, key in enumerate(keys):
        ax.scatter(np.full(n_samples, fill_value=k), data_counts_samples_wt[:, k], color='lightblue', alpha=0.2)
    ax.set_xticks(np.arange(len(keys)), keys)

    ax = plt.subplot(2, 1, 2)
    for k, key in enumerate(keys):
        ax.scatter(np.full(n_samples, fill_value=k), data_counts_samples_ko[:, k], color='lightblue', alpha=0.2)
    ax.set_xticks(np.arange(len(keys)), keys, rotation=15)

    plt.tight_layout()

    np.savetxt('./data/sample_data_wt_size_norm.txt', data_counts_samples_wt, fmt='%d')
    np.savetxt('./data/sample_data_ko_size_norm.txt', data_counts_samples_ko, fmt='%d')


#%% model fit: MLE parameter

figures["model_fit_parameter"] = plt.figure(5, figsize=(20, 10))
plt.clf()

names = parameter._fields  # noqa
parameter_names = dict(
    r_b_n="$\\rho_{n}$ [1/(h $\\mu$m)]",
    g_n_d="$\\gamma_{n,d}$ [1/h]",
    g_n_s="$\\gamma_{n,s}$ [1/h]",
    g_s_d="$\\gamma_{s,d}$  [1/h]"
)
scales = np.array([[data_wt.total_dendrite_length, data_ko.total_dendrite_length], [1, 1], [1, 1], [1, 1]])
positions = [0, 1]

for i, name in enumerate(names):
    ax = plt.subplot(1, len(names), 1 + i)

    # parameter errors
    for p, samples, std, fit in zip(positions,
                                    [fit_parameter_samples_wt, fit_parameter_samples_ko],
                                    [parameter_fit_std_wt, parameter_fit_std_ko],
                                    [fit_wt, fit_ko]):
        pars = getattr(samples, name) / scales[i, p]
        std = getattr(std, name) / scales[i, p]
        value = getattr(fit.parameter, name) / scales[i, p]

        ax.scatter(np.full(len(pars), fill_value=p), pars, label=name, color=['grey', 'darkgreen'][p], alpha=0.5)
        ax.errorbar([p + 0.1], [value], yerr=std, fmt='o', capsize=5, color=['black', 'darkgreen'][p])

    # tests
    pars_wt = getattr(fit_parameter_samples_wt, name) / scales[i, 0]
    pars_ko = getattr(fit_parameter_samples_ko, name) / scales[i, 1]
    test = stats.mannwhitneyu(pars_wt, pars_ko)

    x1, x2 = positions
    offset = 0.1
    y1 = np.max([pars_wt, pars_ko]) * (1 + offset)
    y2, y3 = np.max(pars_wt) * (1 + 0.8 * offset),  np.max(pars_ko) * (1 + 0.8 * offset)
    plt.plot([x1, x1, x2, x2], [y2, y1, y1, y3], lw=1.5, c='black')
    plt.text((x1 + x2) / 2, y1, f"p={test.pvalue:.2e}", ha='center', va='bottom')

    ax.set_ylim(0, y1 * (1 + offset))
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(parameter_names[name])
    ax.set_xticks(positions, ['wt', 'ko'], rotation=25)

plt.tight_layout()


# %% model fit: simulation

# fit_parameter_wt = ModelParameter(
#     r_b_n=2.641819768988672,
#     g_n_d=0.07524732559632966,
#     g_n_s=0.01762167557656594,
#     g_s_d=0.002253302191273348
# )
#
# fit_parameter_ko = ModelParameter(
#     r_b_n=1.6567208919449408,
#     g_n_d=0.05033187330708671,
#     g_n_s=0.01918883298110662,
#     g_s_d=0.002406874792150074
# )

settings = ModelSettings(
    dt=0.1,  # time step (h)
    save_states=True,  # save model states
    save_events=True  # save single synapse event sequences
)

model_wt = Model(parameter=fit_parameter_wt, settings=settings)
model_ko = Model(parameter=fit_parameter_ko, settings=settings)

steps = int(14 * 24 / settings.dt * 7)

_, model_simulation_wt = model_wt.simulate(steps=steps)
_, model_simulation_ko = model_ko.simulate(steps=steps)

_, model_simulation_long_wt = model_wt.simulate(steps=10 * steps)
_, model_simulation_long_ko = model_ko.simulate(steps=10 * steps)

figures['model_fit'] = plt.figure(6, figsize=(25, 20))
plt.clf()

ax = plt.subplot(2, 2, 1)
model_simulation_wt.plot_event_sequences(ax=ax, title='model fit wt: synapses', axes_labels=("time [h]", "synapse"))

ax = plt.subplot(2, 2, 2)
model_simulation_wt.plot_states(
    ax=ax, title='model fit wt: states', legend='lower right', axes_labels=("time [h]", "synapse counts")
)

ax = plt.subplot(2, 2, 3)
model_simulation_ko.plot_event_sequences(ax=ax, title='model fit ko: synapses', axes_labels=("time [h]", "synapse"))

ax = plt.subplot(2, 2, 4)
model_simulation_ko.plot_states(
    ax=ax, title='model fit ko: states', legend='lower right', axes_labels=("time [h]", "synapse counts")
)

plt.tight_layout()


# %% model fit: simulation measurements vs data

n_measurements = 500
n_measures = len(keys_for_fit)

model_fit_counts_wt = np.zeros((n_measurements, n_measures), dtype=int)
model_fit_counts_ko = np.zeros((n_measurements, n_measures), dtype=int)

model_simulation_long_ko.event_history.update_event_sequences(use_complete_sequences_only=True)
model_simulation_long_ko.event_history.update_event_sequences(use_complete_sequences_only=True)

max_time = int(model_simulation_wt.state_history.max_time) - max(measurement_times)
time_offsets = np.array(np.linspace(1000, max_time, n_measurements), dtype=int)

measure_kwargs_model = measure_fit_kwargs.copy()

for i, t in enumerate(tqdm.tqdm(time_offsets)):
    measure_kwargs_model.update(measurement_times=measurement_times + t)
    model_counts = model_simulation_long_wt.event_history.measure(**measure_kwargs_model)
    model_fit_counts_wt[i] = [model_counts[(ts + t, tf + t, tl + t, te + t)] for ts, tf, tl, te in keys_for_fit]

    model_counts = model_simulation_long_ko.event_history.measure(**measure_kwargs_model)
    model_fit_counts_ko[i] = [model_counts[(ts + t, tf + t, tl + t, te + t)] for ts, tf, tl, te in keys_for_fit]


def plot_stats(data, measurements, parameter, condition, data_color='gray'):
    n_measures = measurements.shape[1]
    m, l = distribute_plots(n_measures)
    m, l = m - 1, l + 1

    fig = plt.gcf()
    ax = None

    for i in range(n_measures):
        ts, tf, tl, te = keys_for_fit[i]
        data_count = data[(ts, tf, tl, te)]
        max_count = max(max(measurements[:, i]), data_count)
        ns = np.arange(int(max_count * 1.15 + 2))

        ax = plt.subplot(m, l, i + 1)
        ax.hist(measurements[:, i], density=True, bins=ns - 0.5, label='model numerical', color='purple', alpha=0.2)

        pdf = parameter.measure_distribution(ns, measure="occurred_mean", ts=ts, tf=tf, tl=tl, te=te)
        ax.plot(ns, pdf, label='model theory', color='purple')

        ax.axvline(data_count, color=data_color, linewidth=5, label='data')

        set_axes(ax, axes_labels=(f'$O_{{({tf}h, {tl}h)}}$', f'$p\\left(O_{{({tf}h, {tl}h)}}\\right)$'), legend=None)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.8, 0.1), fontsize='large')
    fig.suptitle(f'{condition}: occurred synapses - fit (max likelihood)', fontsize='large')
    fig.tight_layout()


figures['model_fit_wt_stats'] = plt.figure(7, figsize=(25, 15))
plt.clf()
plot_stats(data_for_fit_wt, model_fit_counts_wt, fit_parameter_wt, 'wt')

figures['model_fit_ko_stats'] = plt.figure(8, figsize=(25, 15))
plt.clf()
plot_stats(data_for_fit_ko, model_fit_counts_ko, fit_parameter_ko, 'ko', data_color='darkgreen')


# %% model fit: life-time distribution and stable vs synapse pool

times = np.linspace(0, 100, 100)

fit_lifetime_wt = fit_parameter_wt.lifetime_distribution(times)
fit_lifetime_ko = fit_parameter_ko.lifetime_distribution(times)

figures['model_fit_lifetime_distribution'] = plt.figure(9, figsize=(10, 10))
plt.clf()

ax = plt.subplot(1, 1, 1)
ax.plot(times, fit_lifetime_wt, color='gray', lw=1.5, label='wt')
ax.plot(times, fit_lifetime_ko, color='darkgreen', lw=1.5, label='ko')
ax.set_yscale('log')
set_axes(ax, title='synapse lifetime distribution', legend='upper right', axes_labels=('time [h]', '$p(\\Delta t)$'))

plt.tight_layout()


# %% save figures

for name, figure in figures.items():
    filename = os.path.join(figure_directory, f"{name}.png")
    figure.savefig(filename)
    print(f"saved {filename}")

