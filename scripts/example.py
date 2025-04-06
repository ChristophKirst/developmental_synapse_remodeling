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


#% model - example

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

model_wt = Model(parameter=parameter, settings=settings)

state, result_wt = model_wt.simulate(steps=5000)

figures["model_example"] = plt.figure(1, figsize=(25, 10))
plt.clf()
ax = plt.subplot(1, 2, 1)
result_wt.plot_event_sequences(ax=ax)
ax = plt.subplot(1, 2, 2)
result_wt.plot_states(ax=ax, title='synapse counts')
plt.tight_layout()

#% model distributions example

figures["model_distributions_example"] = plt.figure(2, figsize=(25, 10))
plt.clf()

# occurred synapses
first_times = np.linspace(0, result_wt.state_history.max_time - 20, 50)
occurred = np.array([result_wt.event_history.occurred(ts=tf - 10, tf=tf, tl=tf + 10, te=tf + 20).sum() for tf in first_times])
ns = np.arange(np.max(occurred) + 2)
occurred_distribution = result_wt.parameter.measure_distribution(ns, measure="occurred_mean", ts=10, tf=20, tl=30, te=40)

ax = plt.subplot(1, 2, 1)
ax.hist(occurred, bins=ns-0.5, density=True, label='measured', color='purple', alpha=0.4)
ax.scatter(ns, occurred_distribution, label='theory', color='grey')
set_axes(ax, title='occurred synapses', axes_labels=('$O_{f,l}$', '$p(O_{f,l})$'), legend=True)

# lifetimes
result_wt.event_history.update_event_sequences(use_complete_sequences_only=True)
durations = result_wt.event_history.event_durations()
max_duration = np.max(durations) / 10
times = np.linspace(0, max_duration, 100)

ax = plt.subplot(1, 2, 2)
pdf, ref = np.histogram(durations, density=True, bins=times)
log_pdf = np.log(pdf)
ax.scatter((ref[:-1] + ref[1:])/2, np.log(pdf), label='numerical', color='purple', alpha=0.2)
ax.plot(times, np.log(result_wt.parameter.lifetime_distribution(times)), label='theory', color='grey')
set_axes(ax, title='life time', axes_labels=('life time $\\Delta t$', '$\\log(p(\\Delta t))$'), legend=True)


#% experimental data

measurement_times = np.array([0, 6, 12, 24, 48, 72])

data_wt = ExperimentalData(data_sheet='wt').sort(('duration', 'first', 'last'))
data_ko = ExperimentalData(data_sheet='ko').sort(('duration', 'first', 'last'))  # mmp14b deficient condition

measure_kwargs = dict(
    measure='occurred',                               # measure (see event_data.py for definitions)
    measure_with_borders=True,                        # supply neighbouring time points to measure
    measurement_times_add_borders=(-np.inf, np.inf),  # include -inf, inf to measurement times
    measurement_key_include_borders=False             # add the borders to the key in the measurement directoryr
)

counts_wt = data_wt.measure(**measure_kwargs)
counts_ko = data_ko.measure(**measure_kwargs)

density_wt = data_wt.measure(**measure_kwargs, normalize=data_wt.total_dendritic_length)
density_ko = data_ko.measure(**measure_kwargs, normalize=data_ko.total_dendritic_length)

figures["experimental_data"] = plt.figure(3, figsize=(25, 20))
plt.clf()

ax = plt.subplot(2, 2, 1)
data_wt.plot(ax=ax, title='wt', axes_labels=('time [h]', 'synapse id'), event_sequence_ids=True, legend=False)

ax = plt.subplot(2, 2, 2)
data_ko.plot(ax=ax, title='ko', axes_labels=('time [h]', 'synapse id'), event_sequence_ids=True, legend=False)

ax = plt.subplot(2, 2, 3)
plot_measurements(
    measurements=[counts_wt, counts_ko],
    labels=['wt', 'ko'],
    ax=ax,
    title='synapse occurrences (absolute count)',
    axes_labels=('occurred $(t_f, t_l)$ [h]', 'synapse counts')
)

ax = plt.subplot(2, 2, 4)
plot_measurements(
    measurements=[density_wt, density_ko],
    labels=['wt', 'ko'],
    ax=ax,
    title='synapse occurrences (normalized density)',
    axes_labels=('occurred $(t_f, t_l)$ [h]', 'synapse density [1/$\\mu$m]')
)

plt.tight_layout()

#%% wt: model fit

parameter_fit = ModelParameter(  # parameter initial guess
    r_b_n=1,      # new birth rate (1 / time)
    g_n_d=0.1,    # new decay rate (1 / (time * #synapses)
    g_n_s=0.01,   # new to stable rate (1 / (time * #synapses)
    g_s_d=0.001   # stable decay rate (1 / (time * #synapses)
)

data_wt


parameter_fit.log_likelihood()



#%% wt: fitted model

settings = ModelSettings(
    dt=0.1,            # time step (h)
    save_states=True,  # save model states
    save_events=True   # save single synapse event sequences
)


parameter_wt = ModelParameter(  # parameter fitted via maximum likelihood
    r_b_n=2.6418,      # new birth rate (1 / time)
    g_n_d=0.0752432,   # new decay rate (1 / (time * #synapses)
    g_n_s=0.0176194,   # new to stable rate (1 / (time * #synapses)
    g_s_d=0.00225309   # stable decay rate (1 / (time * #synapses)
)

model_wt = Model(parameter=parameter_wt, settings=settings)

steps = int(14 * 24 / settings.dt * 100)
_, result_wt = model_wt.simulate(steps=steps)

figures['model_wt'] = plt.figure(4, figsize=(20, 20))
plt.clf()
ax = plt.subplot(1, 2, 1)
result_wt.plot_event_sequences(ax=ax)
ax = plt.subplot(1, 2, 2)
result_wt.plot_states(ax=ax)


#%% wt: numerical model measurements

measure_kwargs = dict(
    measure="occurred",
    measurement_times=measurement_times,
    measure_with_borders=True,
    measurement_times_add_borders=True
)

model_wt_measurement = result_wt.event_history.measure(**measure_kwargs, measurement_key_with_borders=True)
time_pairs_full = list(model_wt_measurement.keys())
time_pairs = [pair[1:3] for pair in time_pairs_full]

n_measures = len(model_wt_measurement)
n_samples = 500
model_wt_measurements = np.zeros((n_samples, n_measures), dtype=int)

max_time = int(result_wt.state_history.max_time)
time_offsets = np.array(np.linspace(0, max_time - 72, n_samples), dtype=int)

for i, t in enumerate(tqdm.tqdm(time_offsets)):
    measure_kwargs.update(measurement_times=measurement_times + t)
    model_wt_measurement = result_wt.event_history.measure(**measure_kwargs)
    model_wt_measurements[i] = np.array([model_wt_measurement[(tf + t, tl + t)] for tf, tl in time_pairs])

#%%  wt: model performance: statistics - theory - data

fig = figures['model_wt_stats'] = plt.figure(5, figsize=(25, 20))
plt.clf()
m, l = distribute_plots(n_measures)
l += 1
m -= 1

for i in range(n_measures):
    max_count = max(model_wt_measurements[:, i])
    ts, tf, tl, te = time_pairs_full[i]
    ns = np.arange(max_count + 1)

    ax = plt.subplot(m, l, i + 1)
    ax.hist(model_wt_measurements[:, i], density=True, bins=ns-0.5, label='model numerical', color='purple', alpha=0.2)

    pdf = parameter_wt.measure_distribution(ns, measure="occurred_mean", ts=ts, tf=tf, tl=tl, te=te)
    ax.plot(ns, pdf, label='model theory', color='grey')

    ax.axvline(counts_wt[(tf, tl)], color='purple', linewidth=5, label='data')

    set_axes(ax, axes_labels=(f'$O_{{({tf}h, {tl}h)}}$', '$p\\left(O_{{({tf}h, {tl}h)}}\\right)$'), legend=None)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.8, 0.1), fontsize='large')
fig.suptitle('wt: occurred synapses - fit (max likelihood)', fontsize='large')
fig.tight_layout()











#%% plot statistics

figures['model_wt_stats'] = plt.figure(5, figsize=(20, 20))
plt.clf()

for i in range(n_measures):
    plt.scatter(np.full(n_samples, fill_value=i), model_wt_measurements[:, i], color='lightblue', alpha=0.05, s=100)

count_theory = parameter_wt.measure(measurement_times=measurement_times)
ids, counts = np.array([(key_to_id[k], v) for k, v in count_theory.items()]).T
plt.scatter(ids, counts, color='orange')

ids, counts = np.array([(key_to_id[k], v) for k, v in count_wt.items()]).T
plt.scatter(ids, counts, color='purple')

#%%  statistics theory

from utils.plotting import distribute_plots
m, l = distribute_plots(n_measures)

plt.figure(10, figsize=(20, 20))
plt.clf()

for i in range(n_measures):
    key = list(key_to_id.keys())[i]
    id = key_to_id[key]
    max_count = int(max(model_wt_measurements[:, id]))

    ax = plt.subplot(m, l, i + 1)
    ax.hist(model_wt_measurements[:, id], density=True, bins=np.arange(max_count + 1) - 0.5)

    mean = count_theory[key]
    n = 20 * max_count
    p = mean / n
    from scipy.stats import binom
    x = np.arange(0, max_count + 10)
    pmf = binom.pmf(x, n, p)
    ax.plot(x, pmf, color='orange')

    ax.axvline(count_wt[key], color='purple')

    ax.set_title(f"{key}")


#%% max likelihood apprrox

#%%
measurement_times = np.array([0, 6, 12, 24, 48, 72])

measure_kwargs = dict(
    measure="occurred",
    measurement_times=measurement_times,
    measure_with_borders=True,
    measurement_times_add_borders=True
)

count_wt = data_wt.measure(**measure_kwargs, normalize=1)
count_ko = data_ko.measure(**measure_kwargs, normalize=data_ko.total_dendritic_length / data_wt.total_dendritic_length)

time_offset = int(result_wt.state_history.max_time - 72)
# time_offset = int(24 * 14 * 10)
measure_kwargs.update(measurement_times=measurement_times + time_offset)
model_wt_measurement = result_wt.event_history.measure(**measure_kwargs)
count_model = {(k[0] - time_offset, k[1] - time_offset): v for k, v in model_wt_measurement.items()}

figure = plt.figure(3, figsize=(20, 20))
plt.clf()
ax = plt.subplot(1, 1, 1)
plot_measurements(
    measurements=[count_wt, count_ko, count_model],
    labels=['wt', 'ko', 'model']
)

#%% ko model

# fit wt
parameter_ko = ModelParameter(
    r_b_n=2.4013,     # new birth rate (1 / time)
    g_n_d=0.14,      # new decay rate (1 / (time * #synapses)
    g_n_s=0.021132,    # new to stable rate (1 / (time * #synapses)
    g_s_d=0.002054   # stable decay rate (1 / (time * #synapses)
)

state = ModelState()

model_wt = Model(parameter=parameter, settings=settings)

state, result_wt = model_wt.simulate(steps=int(14 * 24 * 1 / settings.dt * 50), state=state)

model_figure = plt.figure(2, figsize=(20, 20))
plt.clf()
ax = plt.subplot(1, 2, 1)
result_wt.plot_event_sequences(ax=ax)
ax = plt.subplot(1, 2, 2)
result_wt.plot_states(ax=ax)

#%
measurement_times = np.array([0, 6, 12, 24, 48, 72])

measure_kwargs = dict(
    measure="occurred",
    measurement_times=measurement_times,
    measure_with_borders=True,
    measurement_times_add_borders=True
)

count_wt = data_wt.measure(**measure_kwargs, normalize=1)
count_ko = data_ko.measure(**measure_kwargs, normalize=data_ko.total_dendritic_length / data_wt.total_dendritic_length)

time_offset = int(result_wt.state_history.max_time - 72)
# time_offset = int(24 * 14 * 10)
measure_kwargs.update(measurement_times=measurement_times + time_offset)
model_wt_measurement = result_wt.event_history.measure(**measure_kwargs)
count_model = {(k[0] - time_offset, k[1] - time_offset): v for k, v in model_wt_measurement.items()}

figure = plt.figure(3, figsize=(20, 20))
plt.clf()
ax = plt.subplot(1, 1, 1)
plot_measurements(
    measurements=[count_wt, count_ko, count_model],
    labels=['wt', 'ko', 'model']
)




#%%

eh = result_wt.event_history
eh.update_event_sequences(max_event_time=400000)

sh = result_wt.state_history
a = sh.n[:] = sh.s[:]

# es = eh.simulated_event_sequences[:]
es = eh.event_sequences
start = es[:, 0]
end = es[:, -1]
mt = measure_kwargs['measurement_times']

np.all(start == eh[(0,-1)][:,0])
np.all(end == eh[(0,-1)][:,1])

occurred = np.logical_and(start <= mt[0], mt[-1] <= end)
print('manual:', occurred.sum())

dict(eh.measure("occurred", measurement_times=[mt[0], mt[-1]], measure_with_borders=True, measurement_times_add_borders=True, measurement_key_with_borders=True))













#%% fit data

def to_mathematica(measurement):
    str = "{"
    for k, v in measurement.items():
        str += f"{{{k[0]},{k[1]},{v}}},"
    str = str[:-1] + "}"
    return str

to_mathematica(count_ko)

#%%

# wt data numerical fit
parameter = ModelParameter(
    r_b_n=0.00175 * 1000,   # new birth rate (1 / time / um) * 100 um
    g_n_d=0.05279,          # new decay rate (1 / (time * #synapses)
    g_n_s=0.02002,          # new to stable rate (1 / (time * #synapses)
    g_s_d=0.00183           # stable decay rate (1 / (time * #synapses)
)

#%% plot model results


plt.figure(1); plt.clf()
ax = plt.subplot(1, 1, 1)
parameter.plot_measure_occurred_a_mean(ax=ax, measurement_times=measurement_times, indices=True,
                                       other_measurements=[model_wt_measurement, data_wt_measurement], other_labels=['model', 'data'])


#%% measure

hours = (0, 6, 12, 24, 48, 72)
scale = 5
measurement_times = tuple(result_wt.state_history.max_time - scale * hours[-1] + h * scale for h in hours)

model_wt_measurement = result_wt.event_history.measure(
    "occurred", measurement_times=measurement_times, measurement_times_add_borders=True,
    measure_with_borders=True, indices=True)

data_wt_measurement = data_wt.measure('occurred', measurement_times_borders=True, measure_with_borders=True, indices=True)

data_ko_measurement = data_ko.measure('occurred', measurement_times_borders=True, measure_with_borders=True, indices=True)

plt.figure(1); plt.clf()
ax = plt.subplot(1, 1, 1)
parameter.plot_measure_occurred_a_mean(ax=ax, measurement_times=measurement_times, indices=True,
                                       other_measurements=[model_wt_measurement, data_wt_measurement], other_labels=['model', 'data'])

#%%
from event_data import EventData

plt.figure(1)
plt.clf()
ax = plt.subplot(2, 2, 1)

et = data_wt.event_sequences
et = et[np.logical_not(np.logical_and(et[:,0]== 0, et[:, -1] ==72))]
dat = EventData(event_sequences=et)
dat.plot_event_times(ax=ax, sort_order=('duration', 'first', 'last'))


# data.plot_events(ax=ax, sort_order=('duration', 'first', 'last'))

dl = np.sum(list(data_wt.dendrite_lengths.values()))

ax = plt.subplot(2, 2, 2)

et = data_ko.event_sequences
et = et[np.logical_not(np.logical_and(et[:,0]== 0, et[:, -1] ==72))]
dat_ko = EventData(event_sequences=et)
dat_ko.plot_event_times(ax=ax, sort_order=('duration', 'first', 'last'))


# data_ko.plot_events(ax=ax, sort_order=('duration', 'first', 'last'))

dl_ko = np.sum(list(data_ko.dendrite_lengths.values()))

ax = plt.subplot(2, 2, 3)
measurement = data_wt.measure(measurement_times_borders=True, measurement_times=data_wt.unique_event_times, measure='occurred', measure_with_borders=True)
ko_measurement = data_ko.measure(measurement_times_borders=True, measurement_times=data_wt.unique_event_times, measure='occurred', measure_with_borders=True)





#%%

def subsample_data(data: ExperimentalData, percent=0.95, normalize = 1.0):
    event_times = data.event_sequences
    n = len(event_times)
    m = int(percent * n)
    sub_sample=np.random.permutation(event_times)[:m]
    scale = m / n
    sub_data = EventData(event_sequences=sub_sample)
    measurement = sub_data.measure(measurement_times_borders=True, measurement_times=data.unique_event_times, indices=True,
                                   measure='occurred', measure_with_borders=True, normalize=normalize * scale
                                   )
    return measurement


def to_mathematica(measurement):
    str = "{"
    for k, v in measurement.items():
        str += f"{{{k[0]},{k[1]},{v}}},"
    str = str[:-1] + "}"
    return str

#%% generate subsampled data

samples = [subsample_data(data_wt, 0.95, normalize=dl) for i in range(20)]

mathematica = "{" + ",".join([to_mathematica(m) for m in samples]) + "}"
mathematica


#%%

samples_ko = [subsample_data(data_ko, 0.95, normalize=dl_ko) for i in range(20)]

mathematica_ko = "{" + ",".join([to_mathematica(m) for m in samples_ko]) + "}"
mathematica_ko





#%%

def reduce(m):
    return {k: v for k, v in m.items() if k != (0, 72)}  # (k[0] != 1 and k[1] != 6)}

measurement, ko_measurement = [reduce(m) for m in [measurement, ko_measurement]]

from event_data import EventDataUtils
EventDataUtils._plot_measurements(ax=ax, measurements=[measurement, ko_measurement], labels=['wt', 'ko'])

ax = plt.subplot(2, 2, 4)
measurement = data_wt.measure(measurement_times_borders=True, measurement_times=data_wt.unique_event_times, measure='occurred', measure_with_borders=True, normalize=dl)
ko_measurement = data_ko.measure(measurement_times_borders=True, measurement_times=data_wt.unique_event_times, measure='occurred', measure_with_borders=True, normalize=dl_ko)

measurement, ko_measurement = [reduce(m) for m in [measurement, ko_measurement]]

EventDataUtils._plot_measurements(ax=ax, measurements=[measurement, ko_measurement], labels=['wt', 'ko'], title='normalized')

#%%

measurement


#%%
measurement = data_wt.measure(measurement_times_borders=True, measurement_times=data_wt.unique_event_times, measure='occurred', measure_with_borders=True, normalize=dl, indices=True)
measurement_ko = data_ko.measure(measurement_times_borders=True, measurement_times=data_wt.unique_event_times, measure='occurred', measure_with_borders=True, normalize=dl_ko, indices=True)

str = ""
for k, v in measurement_ko.items():
    str += f"{{{{{k[0]}, {k[1]}}}, {v}}}, "
str

#%% lets fit the data to this model and model with g_n_s = 0

from scipy.optimize import curve_fit

data_wt_measurement = data_wt.measure('occurred', measurement_times_borders=True, measure_with_borders=True, indices=True)

# functions in put tuple of time window output the observed data (using model parameter)
hours = (0, 6, 12, 24, 48, 72)  # noqa
hours_inf = np.array((-np.inf,) + hours + (np.inf,))


def model_curve(x, r_b_n, g_n_d, g_n_s, g_s_d, t_offset):
    t_scale = 1
    parameter = model_wt.ModelParameter(r_b_n, g_n_d, g_n_s, g_s_d)
    tf, tl = np.asarray(x, dtype=int)
    ts, te = tf - 1, tl + 1
    ts, tf, tf, te = tuple(t_scale * hours_inf[t] + t_offset for t in (ts, tf, tf, te))
    return parameter.occurred_a_mean(ts, tf, tf, te)


def model_1_curve(x, r_b_n, g_n_d, t_offset):
    t_scale = 1
    parameter = model_wt.ModelParameter(r_b_n, g_n_d, 0, 1)
    tf, tl = np.asarray(x, dtype=int)
    ts, te = tf - 1, tl + 1
    ts, tf, tf, te = tuple(t_scale * hours_inf[t] + t_offset for t in (ts, tf, tf, te))
    return parameter.occurred_a_mean(ts, tf, tf, te)


# filter
def reduce(m):
    return {k: v for k, v in m.items() if k != (1, 6)}  # (k[0] != 1 and k[1] != 6)}


x_data = list(data_wt_measurement.keys())
y_data = list(data_wt_measurement[k] for k in x_data)
x_data_np = np.array(x_data, dtype=int).T


p0 = tuple(parameter) + (1000000,)
p0 = (2, 0.01, 0.01, 0.005, 100000)
parameter_fit, _ = curve_fit(model_curve, x_data_np, y_data, p0=p0, maxfev=15000)  # noqa
error = np.array([model_curve(x, *parameter_fit) - y for x, y in zip(x_data_np.T, y_data)])

model_parameter_fit = model_wt.ModelParameter(*parameter_fit[:4])
t_scale, t_off = 1, parameter_fit[4]
measurement_times = tuple(t_off + t_scale * h for h in hours)
fit_measurement = model_parameter_fit.measure_occurred_a_mean(measurement_times=measurement_times, indices=True)

p0 = (2, 0.0065, 1000000,)
parameter_1_fit, _ = curve_fit(model_1_curve, x_data_np, y_data, p0=p0, maxfev=15000)  # noqa
error_1 = np.array([model_1_curve(x, *parameter_1_fit) - y for x, y in zip(x_data_np.T, y_data)])

model_1_parameter_fit = model_wt.ModelParameter(*parameter_1_fit[:2])
t_1_scale, t_1_off = 1, parameter_1_fit[2]
measurement_1_times = tuple(t_1_off + t_1_scale * h for h in hours)
fit_1_measurement = model_1_parameter_fit.measure_occurred_a_mean(measurement_times=measurement_1_times, indices=True)

hand = model_wt.ModelParameter(r_b_n=0.3, g_n_d=0.0014, g_n_s=0.00, g_s_d=1.0)
hand_measurement = hand.measure_occurred_a_mean(measurement_times=measurement_times, indices=True)
error_hand = np.array([hand_measurement[x] - y for x, y in zip(x_data, y_data)])

measurements = [data_wt_measurement, fit_measurement, fit_1_measurement, hand_measurement]
labels = ['data', f'fit ({np.sum(error**2)})', f'fit1 ({np.sum(error_1**2)})', f'hand ({np.sum(error_hand**2)})']

from event_data import EventDataUtils

plt.figure(1); plt.clf()
ax = plt.subplot(1, 2, 1)
EventDataUtils._plot_measurements(ax=ax, measurements=measurements, labels=labels)
ax = plt.subplot(1, 2, 2)
EventDataUtils._plot_measurements(ax=ax, measurements=[reduce(m) for m in measurements], labels=labels)


#%%

t_off = 1000
t_scale = 10

hours = (0, 6, 12, 24, 48, 72)  # noqa
measurement_times = tuple(t_off + t_scale * h for h in hours)

plt.figure(1); plt.clf()
ax = plt.subplot(1, 1, 1)
parameter.plot_measure_occurred_a_mean(ax=ax, measurement_times=measurement_times, indices=True,
                                       other_measurements=[data_wt_measurement], other_labels=['experiment'])




#%%

parameter.occurred_a_mean(0, 1, 2, 3)





#%%

plt.figure(1); plt.clf(); ax = plt.subplot(1, 3, 1)
result_wt.plot_states(ax=ax)
ax = plt.subplot(1, 3, 2)
data_wt = result_wt.event_history

result_wt.event_history.plot_event_times()
ax = plt.subplot(1, 3, 3)

#%%
hours = (0, 6, 12, 24, 48, 72)
scale = 10
measurement_times = tuple(result_wt.state_history.max_time - scale * hours[-1] + h * scale for h in hours)

result_wt.event_history.measure("occurred", measurement_times=measurement_times)


#%%

results.plot(
    sort_order=("decay", "stable", "new"),
    observe=(-200, -190),
    observation_times=observations,
    include_active=True
)

 #%%

parameter = model_wt.ModelParameter(
    r_b_n=1,    # new birth rate (1 / time)
    g_n_d=0.2,   # new decay rate (1 / (time * #synapses)
    g_n_s=0.2,    # new to stable rate (1 / (time * #synapses)
    g_s_d=0.01   # stable decay rate (1 / (time * #synapses)
)


settings = model_wt.ModelSettings(
    dt=0.1,  # time / step
    save= True,
    save_synapses=True
)

results = []
measurements = []

n0 = 50
s0 = 50
t0 = 1000

# t0s = np.linspace(1000, 4000, 10)
t0s = [0]
t1s = np.linspace(1, 25, 10)
t_max = np.max(t0s) + np.max(t1s)
steps = int(t_max/settings.dt+1)

for i in tqdm.tqdm(range(300)):
    state = model_wt.ModelState(n=n0, s=s0, t=0)
    result_wt = model_wt.ModelResult(parameter=parameter)
    result_wt.save_state(-1, state.n, state.s)
    result_wt.new.extend(np.full((state.n, 1), fill_value=-1))
    result_wt.stable.extend(np.full((state.s, 2), fill_value=-1))
    _, result_wt = model_wt.simulate(steps=steps, state=state, results=result_wt, parameter=parameter, settings=settings)
    results.append(result_wt)
    for t0 in t0s:
        measurement = [result_wt.measure_changes(t0=t0, t1=t0 + t1) for t1 in t1s]
        measurement = {k: [m[k] for m in measurement] for k in measurement[0].keys()}
        measurements.append(measurement)


#%%
def plot_changes(measurements, parameter, figure=2):
    figure = plt.figure(figure) if figure is not None else plt.figure()
    plt.clf()

    colors = list(mpl.colors.get_named_colors_mapping().values())

    procs = ['n0_n1', 'n0_nd1', 'n0_r1', 'n0_s1', 'n0_sd1', 's0_s1', 's0_sd1',
             'g0', 'g0_n1', 'g0_s1', 'g0_nd1', 'g0_sd1']
    n_plots = len(procs)
    q, p = model_wt.ModelResult._distribute_plots(n_plots)  # noqa

    n0 = measurements[0]['n0'][0]
    s0 = measurements[0]['s0'][0]

    ns = np.arange(n0+1)
    for i, proc  in enumerate(procs):
        theory = f"p_{proc}"
        times1 = measurements[0]["t1"]
        pdf = None
        if theory is not None and hasattr(parameter, theory):
            x0 = (n0,) if 'n0' in theory else (s0,) if 's0' in theory else ()
            pdf = np.array([getattr(parameter, theory)(ns, t1, *x0, t0=0) for t1 in t1s])  # (T, N)
        data = np.array([measurement[proc] for measurement in measurements]).T  # (T, S)

        ax = plt.subplot(p, q, 1 + i)
        for j, t1 in enumerate(times1):
            ax.hist(data[j], bins=np.arange(n0+2)-0.5, density=True,
                    label=f"{proc} {t1=:.0f}",
                    color=mpl.colors.to_rgba(colors[j], 0.2) )
            if pdf is not None:
                ax.plot(ns, pdf[j], color=colors[j])  # label=f"{theory} {t1=}",
        ax.legend()
        ax.set_title(proc)
        ax.set_ylim(0, 0.5)


plot_changes(measurements, parameter)

#%%



#%%

result_wt = results[100]

plt.figure(6)
plt.clf()
result_wt.plot_synapses()

synapses = result_wt.synapse_event_times(include_active=True)

t0=0
n0 = np.logical_and(synapses[:, 0] <= t0, t0 < synapses[:, 1])
pts = np.zeros(np.sum(n0))
plt.scatter(pts, np.where(n0)[0], color='r')

for t1 in times1:
        # n -> n
        n1 = np.logical_and(n0, t1 < synapses[:, 1])
        plt.scatter(np.zeros(np.sum(n1)) + t1 + 0.1, np.where(n1)[0], c='b', label='n1')

        # n -> d
        nd1 = np.logical_and(synapses[:, 1] == synapses[:, 2], synapses[:, 2] <= t1)
        nd1 = np.logical_and(n0, nd1)
        plt.scatter(np.zeros(np.sum(nd1)) + t1 + 0.2, np.where(nd1)[0], c='black', label='nd1')

        # n -> s
        s1 = np.logical_and(synapses[:, 1] <= t1, t1 < synapses[:, 2])
        s1 = np.logical_and(n0, s1)
        plt.scatter(np.zeros(np.sum(s1)) + t1 + 0.3, np.where(s1)[0], c='orange', label='s1')

        # n -> s -> d
        sd1 = np.logical_and(synapses[:, 1] < synapses[:, 2], synapses[:, 2] <= t1)
        sd1 = np.logical_and(n0, sd1)
        plt.scatter(np.zeros(np.sum(sd1)) + t1 + 0.4, np.where(sd1)[0], c='brown', label='sd1')

        print(np.sum(n1) + np.sum(nd1) + np.sum(s1) + np.sum(sd1))

plt.legend()

#%%

switch_times = []
for result_wt in results:
    synapses = result_wt.synapse_event_times()
    switch = synapses[:, 1] < synapses[:, 2]
    switch_times.append(synapses[switch, 1])
switch_times = np.concatenate(switch_times)

plt.figure(3)
plt.clf()
plt.hist(switch_times, density=True, bins=np.linspace(0, 25, 50))

times = np.linspace(0.1, np.max(switch_times), 50)
plt.plot(times, parameter.p_n0_tns(times, t0=0, t1=50))

plt.figure(4)
plt.clf()
result_wt.plot_synapses(sort_order=('stable', 'new', 'decay'))





#%% switch times between two time points

t0 = 0
t1 = 3

switch_times = []
for result_wt in results:
    synapses = result_wt.synapse_event_times()
    switch = synapses[:, 1] < synapses[:, 2]
    switch = np.logical_and(switch, synapses[:, 1] <= t1)
    switch_times.append(synapses[switch, 1])
switch_times = np.concatenate(switch_times)

plt.figure(3)
plt.clf()
plt.hist(switch_times, density=True, bins=np.linspace(0, 5, 50))

times = np.linspace(0, np.max(switch_times), 50)

pdf = parameter.p_n0_tns(times, t0=0, t1=t1)
# pdf = pdf / np.sum(pdf * (times[1]-times[0]))
plt.plot(times, pdf)

plt.figure(4)
plt.clf()
result_wt.plot_synapses(sort_order=('stable', 'new', 'decay'))

#%% generation times between two time points


t0 = 0
t1 = 25

generation_times = []
for result_wt in results:
    synapses = result_wt.synapse_event_times()
    generation = np.logical_and(t0 < synapses[:, 0], synapses[:, 0] <= t1)
    generation_times.append(synapses[generation, 0])
generation_times = np.concatenate(generation_times)

plt.figure(7)
plt.clf()
plt.hist(generation_times, density=True, bins=np.linspace(t0, t1, 10))

times = np.linspace(0, np.max(generation_times), 50)
pdf = [1/(t1-t0)] * len(times)
plt.plot(times, pdf)

plt.figure(8)
plt.clf()
result_wt.plot_synapses(sort_order=('stable', 'new', 'decay'))




#%%

synapses = results.synapses()

observations = results.observations((-1500, -1000, -500, -10e-20))
life = observations['life']

ts = results.observation_times((-1500, -1000, -500, -100, -10e-20), bounds=True)

i = 1
j = 0

np.sum(
    np.logical_and(
        np.logical_and(ts[i + 0] < synapses[:,  0], synapses[:,  0] <= ts[i + 1]),
        np.logical_and(ts[j + 1] <= synapses[:, -1], synapses[:, -1] < ts[j + 2])
    )
)

ts = results.observation_times((-1500, -1000,-500, -10e-20))
survived = observations["survived"]


#%%

np.sum(synapses[:, 0] == synapses[:, -1])

#%%

plt.figure(7)

x = np.zeros((10, 4))
x[0, 1] = 1
x[0, 2] = 2
x[9, 0] = 10
plt.imshow(x.T, origin="lower")


#%% save figures

for name, figure in figures.items():
    filename = os.path.join(figure_directory, f"{name}.png")
    figure.savefig(filename)
    print(f"saved {filename}")
