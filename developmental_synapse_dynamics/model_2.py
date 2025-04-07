# -*- coding: utf-8 -*-
"""
Two state model
===============

Synapse types
-------------
    * *n*: new synapses
    * *s*: stable synapses
    * *a*: all synapses (= n + s)

Examples
--------
>>> from model_2 import Model
>>> model = Model()
>>> model
 Model(...)

>>> state, result = model.simulate(10000)
>>> state
 ModelState(t=10000.0, n=..., s=...)

>>> result
 ModelResult{ModelStateHistory(10000)}{ModelEventHistory('new', 'stable', 'decay')[...]}

>>> from utils.plotting import plt
>>> plt.figure(1); plt.clf()
>>> ax = plt.subplot(1, 2, 1)
>>> result.plot_states(ax=ax)
>>> ax = plt.subplot(1, 2, 2)
>>> result.event_history.plot()

References
----------
Haruna Nakajo, et al., Extracellular matrix proteolysis maintains synapse plasticity during brain development,
bioRxiv (2025), https://www.biorxiv.org/content/10.1101/2025.02.27.640672v1

Christoph Kirst, et al. Developmental Synapse Remodeling
github (2025), https://www.github.com/ChristophKirst/developmental_synapse_dynamics
"""
__project__ = 'Developmental Synapse Remodeling'
__author__ = 'Christoph Kirst <christoph.kirst@ucsf.edu>'
__copyright__ = 'Copyright © 2025 by Christoph Kirst'

# typing
from typing import NamedTuple, Callable, Iterator, Self
from logging import Logger

from utils.plotting import Axes

# imports
import copy
import logging
import numpy as np

from functools import lru_cache
from tqdm import tqdm as progress_bar
from scipy import optimize
from scipy.stats import binom, poisson

from event_data import MeasurableEventData, Measurement, MeasurementIterator
from utils.buffered_array import BufferedArray
from utils.plotting import plt, default_colors, lighter_color, darker_color, set_axes

logger = logging.getLogger('synapse remodeling')


class ModelSettings(NamedTuple):
    dt: float = 1.0             # time step
    save_states: bool = True    # save history of model states
    save_events: bool = True    # save history of individual synapse events


class ModelState(NamedTuple):
    t: float = 0.0   # time
    n: int = 0       # number of new synapses
    s: int = 0       # number of stable synapses

    # derived quantities

    @property
    def a(self):  # all synapses
        return n + s

    @property
    def state_colors(self):
        return 'blue', 'orange', 'purple'  # new, stable, active


class ModelParameter(NamedTuple):
    r_b_n: float = 1.0    # new synapse birth rate
    g_n_d: float = 0.1    # new synapse decay rate
    g_n_s: float = 0.01   # new to stable synapse transition rate
    g_s_d: float = 0.001  # stable synapse decay rate

    # derived quantities

    @property
    def g_n_l(self) -> float:
        """Total new synapse loss rate"""
        return self.g_n_d + self.g_n_s

    @property
    def p_n_s(self) -> float:
        """Probability for a new synapse to transition to stable state"""
        return self.g_n_s / self.g_n_l

    @property
    def n_inf(self) -> float:
        """New synapses steady state"""
        return self.r_b_n / self.g_n_l

    @property
    def s_inf(self):
        """Stable synapses steady state"""
        return self.n_inf * self.g_n_s / self.g_s_d

    @property
    def a_inf(self):
        """All synapses steady state"""
        return self.n_inf + self.s_inf

    # single synapse transition probabilities

    def w_n_n(self, t, t0=0):
        """Transition probability for a single synapse new -> new"""
        g_n_l = self.g_n_l
        return np.exp(-g_n_l * (t - t0))

    def w_n_d(self, t, t0=0):
        """Transition probability for a single synapse new -> decayed"""
        g_n_d = self.g_n_d
        g_n_l = self.g_n_l
        return g_n_d / g_n_l * (1 - np.exp(-g_n_l * (t - t0)))

    def w_n_s(self, t, t0=0):
        """Transition probability for a single synapse new -> stable"""
        g_n_l = self.g_n_l
        g_n_s = self.g_n_s
        g_s_d = self.g_s_d
        return g_n_s / (g_n_l - g_s_d) * (np.exp(-g_s_d * (t - t0)) - np.exp(-g_n_l * (t - t0)))

    def w_n_s_d(self, t, t0=0):
        """Transition probability for a single synapse new -> stable -> decayed"""
        g_n_s = self.g_n_s
        g_n_l = self.g_n_l
        g_s_d = self.g_s_d
        return g_n_s / g_n_l \
            - g_n_s / (g_n_l - g_s_d) * np.exp(-g_s_d * (t - t0)) \
            + g_n_s * g_s_d / (g_n_l * (g_n_l - g_s_d)) * np.exp(-g_n_l * (t - t0))

    def w_s_s(self, t, t0=0):
        """Transition probability for a single synapse stable -> stable"""
        g_s_d = self.g_s_d
        return np.exp(-g_s_d * (t - t0))

    def w_s_d(self, t, t0=0):
        """Transition probability for a single synapse stable -> decayed"""
        g_s_d = self.g_s_d
        return 1 - np.exp(-g_s_d * (t - t0))

    # means

    def n_mean(self, t, n0=None, t0=None, state0: ModelState | None = None):
        """Mean number of new synapses starting with n0 new synapses at t0"""
        n_inf = self.n_inf
        g_n_l = self.g_n_l
        n0 = n0 if n0 is not None else state0.n if state0 is not None else 0
        t0 = t0 if t0 is not None else state0.t if state0 is not None else 0.0
        return n_inf + (n0 - n_inf) * np.exp(-g_n_l * (t - t0))

    def s_mean(self, t, n0=None, s0=None, t0=None, state0: ModelState | None = None):
        """Mean number of new synapses starting with n0 new synapses and s0 stable synapses at t0"""
        n0 = n0 if n0 is not None else state0.n if state0 is not None else 0
        s0 = s0 if s0 is not None else state0.s if state0 is not None else 0
        t0 = t0 if t0 is not None else state0.t if state0 is not None else 0.0

        n_inf = self.n_inf
        s_inf = self.s_inf
        g_n_l = self.g_n_l
        g_n_s = self.g_n_s
        g_s_d = self.g_s_d
        t = t - t0

        return s_inf + (s0 - s_inf) * np.exp(-g_s_d * t) \
            + (n0 - n_inf) * g_n_s / (g_n_l - g_s_d) * (np.exp(-g_s_d * t) - np.exp(-g_n_l * t))

    def a_mean(self, t, n0=None, s0=None, t0=None, state0: ModelState | None = None):
        """Mean number of active synapses starting with n0 new synapses and s0 stable synapses at t0"""
        return self.n_mean(t, n0=n0, t0=t0, state0=state0) + self.s_mean(t, n0=n0, s0=s0, t0=t0, state0=state0)

    def b_n_mean(self, ts, tf):
        """Mean number of new synapses becoming visible."""
        n_inf = self.n_inf
        g_n_l = self.g_n_l
        t = tf - ts
        return n_inf * (1 - np.exp(-g_n_l * t))

    def b_s_mean(self, ts, tf):
        """Mean number of stable synapses becoming visible."""
        n_inf = self.n_inf
        s_inf = self.s_inf
        g_n_l = self.g_n_l
        g_n_s = self.g_n_s
        g_s_d = self.g_s_d
        t = tf - ts

        return s_inf * (1 - np.exp(-g_s_d * t)) \
            - n_inf * g_n_s / (g_n_l - g_s_d) * (np.exp(-g_s_d * t) - np.exp(-g_n_l * t))

    def b_mean(self, ts, tf):
        """Mean number of synapses becoming visible."""
        return self.b_n_mean(ts, tf) + self.b_s_mean(ts, tf)

    def c_n_mean(self, ts, tf, tl):
        """Mean number of new synapses continuing to be visible."""
        g_n_l = self.g_n_l
        t2 = tl - tf

        return self.b_n_mean(ts, tf) * np.exp(-g_n_l*t2)

    def c_s_mean(self, ts, tf, tl):
        """Mean number of stable synapses continuing to be visible."""
        g_n_l = self.g_n_l
        g_n_s = self.g_n_s
        g_s_d = self.g_s_d
        t2 = tl - tf

        return self.b_s_mean(ts, tf) * np.exp(-g_s_d * t2) \
            + self.b_n_mean(ts, tf) * g_n_s / (g_n_l - g_s_d) * (np.exp(-g_s_d * t2) - np.exp(-g_n_l * t2))

    def c_mean(self, ts, tf, tl):
        """Mean number of synapses continuing to be visible."""
        return self.c_n_mean(ts, tf, tl) + self.c_s_mean(ts, tf, tl)

    def o_mean(self, ts, tf, tl, te):
        """Mean number of synapses occurred."""
        return self.c_mean(ts, tf, tl) - self.c_mean(ts, tf, te)

    def appeared_mean(self, ts, tf):
        """Mean number of synapses appeared."""
        return self.b_mean(ts, tf)

    def continued_mean(self, ts, tf, tl):
        """Mean number of synapses continued to be observed."""
        return self.c_mean(ts, tf, tl)

    def occurred_mean(self, ts, tf, tl, te):
        """Mean number of synapses occurred."""
        return self.o_mean(ts, tf, tl, te)

    def survival_mean(self, ts, tf, tl):
        return self.continued_mean(ts, tf, tl) / self.continued_mean(ts, tf, tf)

    # distributions

    def measure_distribution(
            self,
            n: float | list | np.ndarray | None = None,
            measure: str | Callable = 'occurred_mean',
            ts=None, tf=None, tl=None, te=None
    ):
        """Probability for the given measure for counts n"""
        measure = getattr(self, measure) if isinstance(measure, str) else measure
        kwargs = {k: v for k, v in dict(ts=ts, tf=tf, tl=tl, te=te).items() if v is not None}
        mean = measure(**kwargs)
        if n is None:
            n = np.arange(int(3 * mean))
        return self.p_poisson(n, mean)

    def lifetime_distribution(self, t):
        """Probability for synapse lifetimes of t"""
        g_n_d = self.g_n_d
        g_n_s = self.g_n_s
        g_n_l = self.g_n_l
        g_s_d = self.g_s_d

        return (g_n_d - g_n_s * g_s_d / (g_n_l - g_s_d)) * np.exp(-g_n_l*t) \
            + g_n_s * g_s_d / (g_n_l - g_s_d) * np.exp(-g_s_d * t)

    # likelihood

    def log_likelihood(
        self,
        data: dict | Measurement,
        measure: str | Callable = "occurred_mean",
        epsilon: float = 10e-15,
    ):
        measure = getattr(self, measure) if isinstance(measure, str) else measure

        ll = 0
        for key, value in data.items():
            mean = measure(*key)
            if mean <= 0:
                mean = epsilon
            ll += value * np.log(mean) - mean

        return ll

    def fit(
            self,
            data,
            measure: str = "occurred_mean",
            fixed_parameter: dict | None = None,
            parameter_bounds: dict | None = None,
            method: str | None = "SLSQP",  # noqa
            epsilon: float = 1e-9,
            verbose: bool = False
    ):
        # fixed parameter
        fixed_parameter = dict() if fixed_parameter is None else fixed_parameter

        # fit parameter
        fit_parameter = {name: getattr(self, name) for name in self._fields if name not in fixed_parameter}

        # likelihood
        def negative_log_likelihood(x):
            parameter = {k: v for k, v in zip(fit_parameter.keys(), x)}
            parameter.update(fixed_parameter)
            return -ModelParameter(**parameter).log_likelihood(data=data, measure=measure)

        parameter_bounds = dict() if parameter_bounds is None else parameter_bounds
        bounds = [parameter_bounds.get(name, (epsilon, None)) for name in fit_parameter.keys()]
        initial = np.array(list(fit_parameter.values()))

        mle = optimize.minimize(negative_log_likelihood, initial, method=method, bounds=bounds)

        parameter = self.to_dict()
        parameter.update({k: v for k, v in zip(fit_parameter.keys(), mle.x)})
        mle.parameter = ModelParameter(**parameter)

        mle.log_likelihood = -mle.fun

        if verbose:
            if mle.success:
                print("Fitting success:")
                print(mle.parameter)
                print("log-likelihood:", mle.log_likelihood)
            else:
                print("Fitting failed:", mle.message)

        return mle

    # random

    @classmethod
    def p_binomial(cls, n, n0, p):
        return binom._pmf(n, n0, p)  # noqa

    @classmethod
    def p_poisson(cls, n, r):
        return poisson._pmf(n, r)  # noqa

    @classmethod
    def binomial(cls, n, p, size=None):
        return np.random.binomial(n, p, size=size)

    @classmethod
    def poisson(cls, r=1.0, size=None):
        return np.random.poisson(lam=r, size=size)

    # measurement

    def measure(
            self,
            measurement_times: list | np.ndarray,
            measure: str = 'occurred_mean',
            measurement_times_borders: bool | tuple | None = True,
            measurement_iterator: MeasurementIterator | None = None,
            measurement_key_as_indices: bool = False,
            measurement_key_with_borders: bool = False,
            max_event_time: float | None = None,
            measurement: Measurement | None = None,
            apply: Callable | None = None,
            normalize: bool | int | float = False,
    ):
        name = measure if isinstance(measure, str) else measure.__name__
        measure = getattr(self, measure) if isinstance(measure, str) else measure
        measurement = MeasurableEventData(name=name).measure(
            measure=measure,
            measurement_times=measurement_times,
            measurement_times_borders=measurement_times_borders,
            measurement_iterator=measurement_iterator,
            measurement_key_as_indices=measurement_key_as_indices,
            measurement_key_with_borders=measurement_key_with_borders,
            max_event_time=max_event_time,
            measurement=measurement,
            apply=apply,
            normalize=normalize
        )
        return measurement

    def single_synapse_transition_probabilities(
            self,
            t: list | np.ndarray
    ):
        t = np.asarray(t)
        result = dict(w_n_n=None, w_n_s=None, w_n_d=None, w_n_s_d=None, w_s_d=None)
        for k in result.keys():
            theory = getattr(self, k)
            result[k] = theory(t)
        result.update(dict(t=t))

        return result

    # conversion

    def to_dict(self):
        return {k: v for k, v in zip(self._fields, self)}

    # plotting

    def plot_states(
            self,
            t: np.ndarray,
            n0: float = 0,
            s0: float = 0,
            plot_state_dynamics: bool = True,
            plot_steady_state: bool = True,
            ax: Axes | None = None,
            axes_labels: tuple = ("time", "count"),
            title: str | None = 'states',
            state_colors: tuple | None = ('blue', 'orange', 'purple'),
            legend: str | None = "upper left"
    ):
        ax = ax if ax is not None else plt.gca()
        state_colors = state_colors if state_colors is not None else ModelState().state_colors

        if plot_state_dynamics:
            n_mean = self.n_mean(t=t, n0=n0, t0=t[0])
            ax.plot(t, n_mean, color=lighter_color(state_colors[0]), linestyle='--', label="$\\bar{n}$")

            s_mean = self.s_mean(t=t, n0=n0, s0=s0, t0=t[0])
            ax.plot(t, s_mean, color=lighter_color(state_colors[1]), linestyle='--', label="$\\bar{s}$")

            a_mean = n_mean + s_mean
            ax.plot(t, a_mean, color=lighter_color(state_colors[2]), linestyle='--', label="$\\bar{a}$")

        if plot_steady_state:
            ts = (t[0], t[-1])
            n_inf = self.n_inf
            ax.plot(ts, (n_inf, n_inf), color=lighter_color(state_colors[0]), linestyle=':', label="$n_\infty$")  # noqa

            s_inf = self.s_inf
            ax.plot(ts, (s_inf, s_inf), color=lighter_color(state_colors[1]), linestyle=':', label="$s_\infty$")  # noqa

            a_inf = n_inf + s_inf
            ax.plot(ts, (a_inf, a_inf), color=lighter_color(state_colors[2]),  linestyle=':', label="$a_\infty$")  # noqa

        set_axes(ax, title, axes_labels, legend=legend)

        return ax

    def plot_single_synapse_transition_probabilities(
            self,
            t: list | np.ndarray | None = None,
            ax: Axes = None,
            title: str | None = "transition probabilities",
            axes_labels: tuple | None = ("time", "probability"),
            colors: dict | None = None,
            legend: str | None = 'lower right'
    ) -> Axes:
        transition_probabilities = self.single_synapse_transition_probabilities(t)
        return self._plot_single_synapse_transition_probabilities(
            transition_probabilities=transition_probabilities,
            ax=ax,
            title=title,
            axes_labels=axes_labels,
            colors=colors,
            legend=legend
        )

    @classmethod
    def _plot_single_synapse_transition_probabilities(
            cls,
            transition_probabilities: dict,
            ax: Axes = None,
            title: str | None = "transition probabilities",
            axes_labels: tuple | None = ("time", "probability"),
            colors: dict | None = None,
            legend: str | None = 'lower right'
    ) -> Axes:
        """Helper to plot transition probabilities."""
        colors_ = dict(
            p_n_n="blue", p_n_s="darkblue", p_n_d="red", p_n_s_d="lightblue", p_s_s="orange", p_s_d="brown"
        )
        colors_.update(colors if colors is not None else dict())
        colors = colors_

        ax = ax if ax is not None else plt.gca()

        t = transition_probabilities['t']
        for k in transition_probabilities.keys():
            if k != 't':
                ax.plot(t, transition_probabilities[k], color=colors[k], label=f"theory {k}")

        set_axes(ax=ax, title=title, axes_labels=axes_labels, legend=legend)

        return ax


class ModelEventHistory(MeasurableEventData):
    """Model history of the synaptic event sequences.

    Notes
    -----
    The raw simulation data is held in simulated_event_sequences and filtered to the event_sequences property
    on calling update_event_sequences.
    The event_sequences property is used for all measures and analysis.
    """

    # settings
    event_types = ('new', 'stable', 'decay')
    event_colors: tuple = ('darkblue', 'orange', 'black')

    # event sequence filter
    sort_order: tuple[str, ...] | None = ('new', 'decay', 'duration')  # default sort order
    use_complete_sequences_only: bool = False  # if True only use synapses already decayed.
    undefined_event_time: float = -np.inf

    def __init__(self):
        MeasurableEventData.__init__(self, event_sequences=None)
        self.simulated_event_sequences_dict: dict = dict()
        self.reset()

    def reset(self):
        self.simulated_event_sequences_dict: dict = {
            k: BufferedArray(shape=(100, self.n_event_types), fill_value=self.undefined_event_time)
            for k in self.event_types
        }
        self.update_event_sequences()

    def update_event_sequences(
            self,
            max_event_time: float | None = None,
            sort_order: tuple[str, ...] | None = None,
            use_complete_sequences_only: bool | None = None
    ):
        max_event_time = max_event_time if max_event_time is not None else self.max_event_time
        sort_order = sort_order if sort_order is not None else self.sort_order
        use_complete_sequences_only = use_complete_sequences_only if use_complete_sequences_only is not None \
            else self.use_complete_sequences_only

        if use_complete_sequences_only:
            self.event_sequences = self.simulated_event_sequences(-1)[:].copy()
        else:
            self.event_sequences = \
                np.concatenate([self.simulated_event_sequences(e)[:].copy() for e in self.event_types], axis=0)
        self.event_sequences[self.event_sequences == self.undefined_event_time] = max_event_time
        self.sort(sort_order=sort_order)

    def simulated_event_sequences(self, event_type: str | int) -> BufferedArray:
        key = self.event_index_to_type(event_type)
        return self.simulated_event_sequences_dict[key]

    def single_synapse_transition_probabilities(
            self,
            t: list | np.ndarray | None = None
    ):
        t = t if t is not None else np.linspace(0, self.max_event_time, 1000)
        t = np.asarray(t)
        n_times = len(t)

        starts, transitions, ends = self.event_sequences
        n_synapses = len(synapses)

        t_b = starts
        t_t = transitions - t_b
        t_d = ends - t_b
        t_s = ends - transitions

        # synapses: birth time, transition time, decay time  and  transition time = decay time if decay from new
        decay_from_n = transitions == ends
        decay_from_s = transitions != ends

        p_n_n = np.zeros(n_times)
        p_n_d = np.zeros(n_times)
        p_n_s = np.zeros(n_times)
        p_n_s_d = np.zeros(n_times)
        p_s_s = np.zeros(n_times)
        p_s_d = np.zeros(n_times)

        for i, t in enumerate(t):
            p_n_n[i] = np.sum(t <= t_t) / n_synapses
            p_n_s[i] = np.sum(np.logical_and(t_t < t, t <= t_d)) / n_synapses
            p_n_d[i] = np.sum(t_d[decay_from_n] < t) / n_synapses
            p_n_s_d[i] = np.sum(t_d[decay_from_s] < t) / n_synapses

            p_s_d[i] = np.sum(t_s[decay_from_s] < t) / np.sum(decay_from_s)
            p_s_s[i] = np.sum(np.logical_and(t_t[decay_from_s] < t, t <= t_d[decay_from_s])) / np.sum(decay_from_s)

        return dict(p_n_n=p_n_n, p_n_s=p_n_s, p_n_d=p_n_d, p_n_s_d=p_n_s_d, p_s_d=p_s_d, t=t)

    # plotting

    def plot_single_synapse_transitions_probabilities(
            self,
            t: list | np.ndarray | None = None,
            parameter: ModelParameter | None = None,
            ax: Axes = None,
            title: str | None = "transition probabilities",
            axes_labels: tuple | None = ("time", "probability"),
            colors: dict | None = None,
            legend: str | None = 'lower right'
    ) -> Axes:
        transition_probabilities = self.single_synapse_transition_probabilities(t)

        ax = ModelParameter._plot_single_synapse_transition_probabilities(  # noqa
            transition_probabilities=transition_probabilities,
            ax=ax,
            title=title,
            axes_labels=axes_labels,
            colors=colors,
            legend=legend
        )

        if parameter is not None:
            ax = parameter.plot_single_synapse_transitions(
                t=t,
                ax=ax,
                title=title,
                axes_labels=axes_labels,
                colors=colors,
                legend=legend
            )

        return ax


class ModelStateHistory:
    """Model state history

    Stores the history of the states in the model, i.e. the counts of synapses in each state at each time.
    """
    def __init__(self):
        self.t: BufferedArray = BufferedArray(shape=100, dtype=float)
        self.n: BufferedArray = BufferedArray(shape=100, dtype=int)
        self.s: BufferedArray = BufferedArray(shape=100, dtype=int)

    def __len__(self):
        return len(self.t)

    def __enter__(self, other: tuple):
        if isinstance(other, tuple) and len(other) == 3 and all(isinstance(o, BufferedArray) for o in other):
            self.t, self.n, self.s = other
        else:
            raise ValueError

    def __iter__(self):
        return iter((self.t, self.n, self.s))

    @property
    def times(self):
        return self.t[:]

    @property
    def max_time(self):
        return np.max(self.times)

    def append_t_n_s(self, t, n, s):
        self.t.append(t)
        self.n.append(n)
        self.s.append(s)

    def append_state(self, state: ModelState):
        self.append_t_n_s(*state)

    def append_history(self, history: Self):
        self.t.extend(history.t[:])
        self.n.extend(history.n[:])
        self.s.extend(history.s[:])

    def __add__(self, other: ModelState | Self):
        if isinstance(other, ModelState):
            self.append_state(other)
        elif isinstance(other, ModelStateHistory):
            self.append_history(other)
        else:
            raise ValueError

    def plot(
            self,
            parameter: ModelParameter | None = None,
            plot_state_dynamics: bool = True,
            plot_steady_state: bool = True,
            ax: Axes | None = None,
            axes_labels: tuple | None = ("time", "count"),
            title: str | None = 'states',
            state_colors: tuple | None = ('darkblue', 'orange', 'purple'),
            legend: str | None = "upper left",
    ):
        t = self.t[:]
        n = self.n[:]
        s = self.s[:]
        a = n + s

        ax = ax if ax is not None else plt.gca()
        state_colors = state_colors if state_colors is not None else ModelEventHistory.event_colors

        ax.plot(t, n, color=state_colors[0], label="new")
        ax.plot(t, s, color=state_colors[1], label="stable")
        ax.plot(t, a, color=state_colors[2], label="total")

        if parameter is not None:
            parameter.plot_states(
                t=t, n0=n[0], s0=s[0], ax=ax,
                plot_state_dynamics=plot_state_dynamics, plot_steady_state=plot_steady_state, state_colors=state_colors
            )

        set_axes(ax, title, axes_labels, legend=legend)  # noqa

        return ax

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"


class ModelResult:
    """Model results

    Stores the state history as well as the synaptic event sequence history of a model simulation.
    """

    def __init__(
            self,
            state_history: ModelStateHistory | None = None,
            event_history: ModelEventHistory | None = None,
            parameter: ModelParameter | None = None
    ):
        self.state_history: ModelStateHistory = state_history if state_history is not None else ModelStateHistory()
        self.event_history: ModelEventHistory = event_history if event_history is not None else ModelEventHistory()
        self.parameter: ModelParameter = parameter

    def save_state(self, t_n_s: tuple | ModelState):
        self.state_history.append_state(t_n_s)

    def save_events(self, event_times: np.ndarray):
        self.event_history.simulated_event_sequences.extend(event_times)

    # plotting

    def plot_states(
            self,
            plot_state_dynamics: bool = True,
            plot_steady_state: bool = True,
            ax: Axes | None = None,
            axes_labels: tuple | None = ("time", "count"),
            title: str | None = 'states',
            state_colors: tuple | None = ('darkblue', 'orange', 'purple'),
            legend: str | None = "upper left",
    ):
        return self.state_history.plot(
            parameter=self.parameter,
            plot_state_dynamics=plot_state_dynamics,
            plot_steady_state=plot_steady_state,
            ax=ax,
            axes_labels=axes_labels,
            title=title,
            state_colors=state_colors,
            legend=legend
        )

    def plot_event_sequences(
            self,
            event_sequence_positions: np.ndarray | tuple | list | None = None,
            event_sequence_colors: np.ndarray | tuple | list | None = None,
            event_sequence_labels: np.ndarray | tuple | list | None = None,
            ax: object | None = None,
            axes_labels: tuple = ('time', 'synapse'),
            title: str | None = 'synapse events',
            legend: str | bool = True,
            line_width: float | None = None,
            line_padding: str | float | tuple = 'auto',
            axes_padding: tuple | None = None
    ) -> Axes:
        return self.event_history.plot(
            event_sequence_positions=event_sequence_positions,
            event_sequence_colors=event_sequence_colors,
            event_sequence_labels=event_sequence_labels,
            ax=ax,
            axes_labels=axes_labels,
            title=title,
            legend=legend,
            line_width=line_width,
            line_padding=line_padding,
            axes_padding=axes_padding
        )

    def plot_single_synapse_transition_probabilities(
            self,
            t: list | np.ndarray | None = None,
            ax: Axes = None,
            title: str | None = "transition probabilities",
            axes_labels: tuple | None = ("time", "probability"),
            colors: dict | None = None,
            legend: str | None = 'lower right'
    ):
        return self.event_history.plot_single_synapse_transitions_probabilities(
            t=t,
            parameter=self.parameter,
            ax=ax,
            title=title,
            axes_labels=axes_labels,
            colors=colors,
            legend=legend
        )

    def __repr__(self):
        return f"{self.__class__.__name__}{{{self.state_history}}}{{{self.event_history}}}"


class Model(NamedTuple):
    parameter: ModelParameter = ModelParameter()
    sate: ModelState = ModelState()
    settings: ModelSettings = ModelSettings()

    def simulate(
            self,
            steps: int,
            state: ModelState | None = None,
            result: ModelResult | None = None,
            parameter: ModelParameter | None = None,
            settings: ModelSettings | None = None,
            logger: Logger | None = None
    ) -> tuple[ModelState, ModelResult]:
        parameter = parameter if parameter is not None else self.parameter
        state = state if state is not None else self.sate
        settings = settings if settings is not None else self.settings
        result = result if result is not None else ModelResult(parameter=parameter)

        r_n_b, g_n_d, g_n_s, g_s_d = parameter
        t, n, s = state
        dt, save_states, save_events = settings

        g_n_l, p_n_s = parameter.g_n_l, parameter.p_n_s
        lambda_n_b, gamma_n_l, gamma_s_d = r_n_b * dt, g_n_l * dt, g_s_d * dt

        p_n_l = np.exp(-gamma_n_l)
        p_s_d = np.exp(-gamma_s_d)

        binomial, poisson = parameter.binomial, parameter.poisson

        for _ in progress_bar(range(steps)):
            # time update
            t += dt

            # loss of new synapses
            n_survive = binomial(n, p_n_l)  # see radio-active decay paper
            dn_l = n - n_survive

            # decay vs transition
            dn_s = binomial(dn_l, p_n_s)
            dn_d = dn_l - dn_s

            # decay stable synapses
            s_survive = binomial(s, p_s_d)  # see radio-active decay paper
            ds_d = s - s_survive

            # generation of new synapses
            dn_b = poisson(lambda_n_b)

            # update state
            n = n_survive + dn_b
            s = s_survive + dn_s

            # save
            if save_states:
                result.save_state((t, n, s))
            if save_events:
                self.simulate_events(result.event_history, t, dn_b, dn_d, dn_s, ds_d)
                # check consistency
                # assert(n == len(result.event_history.simulated_event_sequences('new')))
                # assert(s == len(result.event_history.simulated_event_sequences('stable')))

            if logger:
                logger.info(ModelState(t, n, s))

        state = ModelState(t, n, s)
        result.event_history.update_event_sequences(max_event_time=t)
        return state, result

    @classmethod
    def simulate_events(
            cls,
            event_history: ModelEventHistory,
            t: float,   # time
            dn_b: int,  # new synapses born
            dn_d: int,  # new synapses decayed
            dn_s: int,  # new synapses transitioned to stable
            ds_d: int   # stable synapses decayed
    ):
        """Simulate the individual event sequences of synapses."""

        new = event_history.simulated_event_sequences('new')
        stable = event_history.simulated_event_sequences('stable')
        decay = event_history.simulated_event_sequences('decay')

        index_new = event_history.event_type_to_index('new')
        index_stable = event_history.event_type_to_index('stable')
        index_decay = event_history.event_type_to_index('decay')
        undefined = event_history.undefined_event_time

        # stable synapse decay
        if ds_d > 0:
            s_d = np.random.choice(len(stable), ds_d, replace=False)

            stable[s_d, index_decay] = t
            decay.extend(stable[s_d])
            stable.delete(s_d)

        # new synapse transition and decay
        dn_l = dn_d + dn_s
        if dn_l > 0:
            n_l = np.random.choice(len(new), dn_l, replace=False)
            n_d = n_l[:dn_d]
            n_t = n_l[dn_d:]

            new[n_d, index_stable:] = t
            decay.extend(new[n_d])

            new[n_t, index_stable] = t
            stable.extend(new[n_t])

            new.delete(n_l)

        # new synapse generation
        if dn_b > 0:
            new_sequences = np.full((dn_b, 3), fill_value=undefined)
            new_sequences[:, index_new] = t
            new.extend(new_sequences)
