# -*- coding: utf-8 -*-
"""
Event data
==========

This class provides routines to save and measure event based time series where
events of different types happen in a predefined order, e.g. creation of an object and its subsequent decay.

We use the following conventions:

Event type
----------
These are different types of events that will occur in the given sequence.

Event sequence
--------------
An event sequence is a list of times at which the events occured.

Even data
---------
This is an array of event sequences.
Given d event types and n event sequences the data will be an (n,d) array of event times.

States
------
Each event sequence indicates that associated objects are in defined states between two subsequent event times.
The event type of the first of these two subsequent event times typically correspond to the state
into which the object transitioned.

Measurement times
-----------------
These are discrete times at which we can measure events or changes in state due to events.

We use these conventions for certain measurement times:

    * *ts*: reference start time before first measurement time (can be -inf).

    * *tf*: first measurement time. The event happened between ts and tf, and is observed here for the first time.

    * *tl*: last measurement time. An event happened between tl and te that changed the observation.

    * *te*: reference end time after the last observation time (can be inf).

    * *t1*, *t2*, ...: event times

Measurement of states
---------------------
We use the following terms to denote certain measurements where t1 and t2 are event times of selected event types:

    * *seen* at t: t1 <= t <= t2

    * *generated* in (ts, tf): ts < t1 <= tf

    * *appeared* or *become visible* in (ts, tf): generated in (ts, tf) and seen at tf

    * *disappeared* or *decayed* in (tl, te): tl <= t1 < te

    * *continued* in (ts, tf, tl): events appeared in (ts, tf) and continued until tl: ts < t1 <= tf && tl <= t2

    * *occurred* in (ts, tf, tl, te): events appeared in (ts, tf) and disappeared in (tl, te).

    * *duration*: the duration between two events, i.e. t2 - t1.


Examples
--------
>>> import numpy as np
>>> event_sequences = np.array([np.random.rand(10), np.random.rand(10) + 2]).T

>>> from event_data import EventData
>>> data = EventData(event_sequences=event_sequences)
>>> data
 EventData('first', 'last'){10}

>>> from event_data import MeasurableEventData
>>> data = MeasurableEventData(event_sequences=event_sequences)
>>> data
 MeasurableEventData('first', 'last'){10}

>>> data.sort(sort_order=('last', 'first'))
 MeasurableEventData('first', 'last')[10]

>>> measurement = data.measure('occurred', measurement_times=[0, 0.5], measurement_times_borders=True)
>>> measurement
 Measurement(occurred){...}

>>> from utils.plotting import plt
>>> plt.figure(1)
>>> ax = plt.subplot(1, 2, 1)
>>> _ = data.plot(ax=ax)
>>> ax = plt.subplot(1, 2, 2)
>>> _ = measurement.plot(ax=ax)
"""
__project__ = 'Developmental Synapse Remodeling'
__author__ = 'Christoph Kirst <christoph.kirst@ucsf.edu>'
__copyright__ = 'Copyright © 2025 by Christoph Kirst'


# typing
from typing import NamedTuple, Callable, Iterator, Sequence, Self
from logging import Logger
from matplotlib.axes import Axes

from utils.buffered_array import BufferedArray

# imports
import copy
import logging
import numpy as np

from inspect import signature
from itertools import combinations_with_replacement, product
from functools import lru_cache, partial
from tqdm import tqdm as progress_bar

from utils.plotting import mpl, mplc, plt, set_axes, default_colors

logger = logging.getLogger('synapse simulation')


def identity(x):
    return x


class EventData(object):
    """Base class for event data."""

    event_types: tuple[str, ...] = ('first', 'last')
    event_type_colors: tuple = ('darkblue', 'orange')

    def __init__(self, event_sequences: np.ndarray | BufferedArray | Self | None = None):
        if isinstance(event_sequences, EventData):
            event_sequences = event_sequences.event_sequences
        if event_sequences is None:
            event_sequences = np.zeros((0, self.n_event_types))
        if event_sequences.ndim != 2 or event_sequences.shape[1] != self.n_event_types:
            raise ValueError
        self.event_sequences: np.ndarray | BufferedArray = event_sequences

    @property
    def n_event_types(self):
        return len(self.event_types)

    @property
    def n_event_sequences(self):
        return len(self.event_sequences)

    def __len__(self):
        return self.n_event_sequences

    @property
    def max_event_time(self) -> float:
        if self.n_event_sequences == 0:
            return 0
        return np.max(self.event_sequences)

    @property
    def unique_event_times(self):
        return np.unique(self.event_sequences)

    def sorted_event_sequences(
            self,
            event_sequences: np.ndarray | BufferedArray | None = None,
            sort_order: tuple[int | str, ...] | None = None,
            sort_data: np.ndarray | None = None,
            event_type_to_index_map: dict | None = None,
            return_order: bool = False
    ):
        sort_data = sort_data if sort_data is not None else event_sequences if event_sequences is not None \
            else self.event_sequences_with_duration
        event_sequences = event_sequences if event_sequences is not None else self.event_sequences
        event_type_to_index_map = event_type_to_index_map if event_type_to_index_map is not None \
            else self.event_type_to_index_map(with_duration=True, with_index=True)

        if sort_order is None:
            return event_sequences, slice(None)

        sort_order = tuple(event_type_to_index_map.get(s, None) for s in sort_order)
        if None in sort_order:
            raise ValueError(f"cannot sort using {sort_order=}")

        order = np.lexsort([sort_data[:, i] for i in sort_order[::-1]])
        event_sequences = event_sequences[order]

        if return_order:
            return event_sequences, order
        else:
            return event_sequences

    def sort(
            self,
            sort_order: tuple[int | str, ...] | None = None,
            return_order: bool = False
    ):
        event_sequences, order = self.sorted_event_sequences(sort_order=sort_order, return_order=True)
        self.event_sequences = event_sequences
        if return_order:
            return order
        else:
            return self

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def join(self, other: Self):
        if not isinstance(other, EventData):
            raise ValueError
        if not self.event_types == other.event_types:
            raise ValueError
        joined = self.copy()
        joined.event_sequences = np.concatenate([joined.event_sequences, other.event_sequences], axis=0)
        return joined

    def choose(self, indices):
        selected = self.copy()
        selected.event_sequences = selected.event_sequences[indices]
        return selected

    def resample(self, size: int | None = None, replace: bool = True):
        size = size if size is not None else len(self)
        sample_ids = np.random.choice(range(len(self)), size=size, replace=replace)
        return self.choose(sample_ids)

    def coarsen(self, measurement_times: list | np.ndarray) -> Self:
        coarse_times = list(measurement_times)
        n_coarse_times = len(coarse_times)
        coarse_times_inf = np.array([-np.inf] + coarse_times + [np.inf])

        event_sequences = self.event_sequences
        t1, t2 = event_sequences[:, (0, -1)].T

        # remove event sequences non-overlapping coarse times
        remove = np.zeros(len(self), dtype=bool)
        for i in range(n_coarse_times + 1):
            remove = np.logical_or(remove, np.logical_and(coarse_times_inf[i] < t1, t2 < coarse_times_inf[i+1]))
        coarsened = self.choose(np.logical_not(remove))

        # coarsen sequences
        coarse_times_inf = coarse_times_inf[1:]
        event_sequences = coarsened.event_sequences
        for i, sequence in enumerate(event_sequences):
            coarse_sequence = []

            # first event types orient to next future coarse time
            for time in sequence[:-1]:
                index = np.where(coarse_times_inf >= time)[0][0]
                coarse_time = coarse_times[index] if index < n_coarse_times else coarse_times[-1]
                coarse_sequence.append(coarse_time)

            # last event type orients to last previous coarse time
            index = np.where(coarse_times_inf > sequence[-1])[0][0] - 1
            coarse_sequence.append(coarse_times[index])

            event_sequences[i] = coarse_sequence

        return coarsened

    def event_type_to_index_map(
            self,
            with_duration: bool = False,
            with_index: bool = True
    ) -> dict:
        type_to_index = {t: i for i, t in enumerate(self.event_types)}
        if with_index:
            type_to_index.update({i: i for i in range(self.n_event_types)})
            type_to_index.update({-i: self.n_event_types - i for i in range(1, self.n_event_types + 1)})
        if with_duration:
            type_to_index.update(duration=self.n_event_types)
        return type_to_index

    def event_index_to_type_map(
            self,
            with_duration: bool = False,
            with_type: bool = True
    ) -> dict:
        type_to_index = {i: t for i, t in enumerate(self.event_types)}
        type_to_index.update({i - self.n_event_types: t for i, t in enumerate(self.event_types)})
        if with_type:
            type_to_index.update({t: t for t in self.event_types})
        if with_duration:
            type_to_index.update(duration=self.n_event_types)
        return type_to_index

    def event_type_to_index(
            self,
            types: str | int | tuple[str | int, ...] | None | slice = None,
            with_duration: bool = False
    ) -> int | tuple[int, ...]:
        n = self.n_event_types + with_duration
        if types is None:
            return tuple(range(n))
        if isinstance(types, slice):
            return tuple(range(n)[types])
        type_to_index_map = self.event_type_to_index_map(with_duration=with_duration)
        if isinstance(types, str | int):
            return type_to_index_map[types]
        else:
            return tuple(type_to_index_map[t] for t in types)

    def event_index_to_type(
            self,
            index: int | tuple[int, ...] | str | slice | None = None,
            with_duration: bool = False
    ) -> str | tuple[str, ...]:
        event_types = self.event_types + (('duration',) if with_duration else ())
        n = self.n_event_types + with_duration
        if index is None:
            return event_types
        if isinstance(index, str):
            if index in self.event_types:
                return index
            else:
                raise KeyError(f"{index} not in {self.event_types}")
        if isinstance(index, slice):
            return tuple(event_types[i] for i in range(n)[index])
        index_to_type_map = self.event_index_to_type_map(with_duration=with_duration)
        if isinstance(index, tuple):
            return tuple(index_to_type_map[i] for i in index)
        else:
            return index_to_type_map[index]

    def event_durations(
            self,
            start_type: str | int | None = None,
            end_type: str | int | None = None
    ):
        start_type = start_type if start_type is not None else 0
        end_type = end_type if end_type is not None else -1
        start_index, end_index = self.event_type_to_index((start_type, end_type))
        return self.event_sequences[..., end_index] - self.event_sequences[..., start_index]

    @property
    def event_sequences_with_duration(self):
        return np.concatenate([self.event_sequences, self.event_durations()[:, None]], axis=-1)

    def event_sequences_for_types(self, types: tuple[str, ...] | None = None) -> np.ndarray:
        return self.event_sequences_with_duration[..., self.event_type_to_index(types)]

    def __getitem__(self, item):
        return self.event_sequences_for_types(item)

    def plot(
            self,
            event_types: np.ndarray | tuple | list | None = None,
            event_type_colors: np.ndarray | tuple | list | None = None,
            event_sequence_positions: np.ndarray | tuple | list | None = None,
            event_sequence_colors: np.ndarray | tuple | list | None = None,
            event_sequence_labels: np.ndarray | tuple | list | None = None,
            event_sequence_ids: bool | np.ndarray | tuple | None = None,
            event_sequence_id_colors: np.ndarray | tuple | None = None,
            event_sequence_id_labels: np.ndarray | tuple | None = None,
            ax: object | None = None,
            axes_labels: tuple = ('time', 'synapse'),
            title: str | None = 'events',
            legend: str | bool = True,
            line_width: float | None = None,
            line_padding: str | float | tuple = 'auto',
            axes_padding: tuple | None = None
    ):
        return plot_event_sequences(
            event_sequences=self,
            event_types=event_types,
            event_type_colors=event_type_colors,
            event_sequence_positions=event_sequence_positions,
            event_sequence_colors=event_sequence_colors,
            event_sequence_labels=event_sequence_labels,
            event_sequence_ids=event_sequence_ids,
            event_sequence_id_colors=event_sequence_id_colors,
            event_sequence_id_labels=event_sequence_id_labels,
            ax=ax,
            axes_labels=axes_labels,
            title=title,
            legend=legend,
            line_width=line_width,
            line_padding=line_padding,
            axes_padding=axes_padding,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}{self.event_types}[{len(self)}]"


class Measurement(dict):

    name: str | None = None
    data: EventData | None = None

    def __init__(self, *args, name: str | None = None, data: EventData | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.data = data

    def measurement_times(self):
        return np.unique([key for key in self.keys()])

    def normalize(
            self,
            normalize: float | None = True
    ) -> Self:
        if normalize is True:
            normalize = np.sum(self.values())
        if isinstance(normalize, float):
            measurement = {k: v / normalize for k, v in self.items()}
            return Measurement(measurement, name=self.name, data=self.data)
        else:
            return Measurement(self.copy(), name=self.name, data=self.data)

    def combine(
            self,
            combine_key: Callable,
            combine_measure: Callable = np.sum,
            initial_measure: object = 0,
    ) -> Self:
        measurement = self.copy()
        combined_measurement = {}
        for i, j in measurement.keys():
            k = combine_key(i, j)
            if k is not None:
                if k not in combined_measurement:
                    combined_measurement[k] = initial_measure
                combined_measurement[k] = combine_measure(combined_measurement[k], measurement[(i, j)])
        return Measurement(combined_measurement, name=self.name, data=self.data)

    def to_matrix(self, keys: tuple | None = None, axes_indices: tuple | None = None) -> np.ndarray:
        measurement = self
        if axes_indices is None:
            keys = keys if keys is not None else np.array(measurement.keys())
            n_axes = len(keys[0])
            axes_indices = [np.unique(keys[:, i]) for i in range(n_axes)]
        shape = tuple(len(indices) for indices in axes_indices)
        matrix = np.zeros(shape, dtype=type(measurement[keys[0]]))
        for index in product(*axes_indices):
            matrix[index] = measurement.get(index, 0)
        return matrix

    def to_event_sequences(
            self,
            remove_unseen: bool = False,
            return_seen: bool = False,
            fill_value: float = 0
    ):
        """Aims to convert event sequence wise measurements to event times."""
        measurement = self
        key = list(measurement.keys())[0]
        if not isinstance(measurement[key], np.ndarray | list | tuple):  # cannot convert summary data
            raise ValueError
        n = len(measurement[key])
        if n == 0:
            raise ValueError
        d = len(key)
        if d == 0:
            raise ValueError
        event_sequences = np.full((n, d), dtype=dtype(measurement[key][0]), fill_value=fill_value)

        seen = np.zeros(n, dtype=bool)
        for k, v in measurement.items():
            seen[v] = True
            event_sequences[v] = k

        if remove_unseen:
            event_sequences = event_sequences[seen]

        if return_seen:
            return event_sequences, seen
        else:
            return event_sequences

    def to_event_data(self, remove_unseen: bool = False):
        return MeasurableEventData(event_sequences=self.to_event_sequences(remove_unseen=remove_unseen))

    def to_evolution_curves(
            self,
            initial_measurement_times: list | np.ndarray | None = None,
            normalize: bool = False
    ):
        times = initial_measurement_times if initial_measurement_times is not None else self.measurement_times()
        curves = [np.array([(t, self[(times[i], t)]) for t in times[i:]]) for i in range(len(times))]
        if normalize:
            curves = [curve / curve[0] for curve in curves]
        return curves

    def plot(
            self,
            other_measurements: list | None = None,
            other_labels: list | None = None,
            ax: Axes | None = None,
            axes_labels: tuple = ('measurement', 'count'),
            title: str | None = 'measurement',
            label: str | None = None,
            legend: bool | str | None = "upper left"
    ):
        measurements = [self]
        labels = [label if label is not None else self.name]

        if other_measurements is not None:
            measurements += other_measurements
            labels += other_labels

        return plot_measurements(
            measurements, labels, ax=ax, axes_labels=axes_labels, title=title, legend=legend
        )

    def plot_as_matrix(
            self,
            keys: list | None = None,
            axes_indices: list | None = None,
            ax: Axes | None = None,
            axes_labels: tuple = ("first time $t_i$", "last time $t_j$"),
            title: str | None = 'measurement',
            legend: str | None = 'upper left'
    ):
        image = self.to_matrix(keys=keys, axes_indices=axes_indices)
        if image.ndim > 2:
            image = image[:, :, (0,) * (image.ndim - 2)]
        ax = ax if ax is not None else plt.gca()
        iax = ax.imshow(image.T, origin="lower")
        ax.figure.colorbar(iax, ax=ax)
        ax.set_xticks(first_indices)
        ax.set_yticks(last_indices)
        set_axes(ax=ax, title=title, axes_labels=axes_labels, legend=legend)
        return ax

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name if self.name is not None else ''}){dict(self)}"


class MeasurementIterator:

    depth: int = 2
    borders: tuple | None = None
    key_as_indices: bool = False
    key_with_borders: bool = False
    default_borders: tuple = (-np.inf, np.inf)

    def __init__(
            self,
            measurement_times: list | np.ndarray,
            depth: int | Callable | None = None,
            borders: tuple | bool | None = None,
            indices: list | None = None,
            key_as_indices: bool | None = None,
            key_with_borders: bool | None = None,
    ):
        self.measurement_times: list | np.ndarray = measurement_times
        self.key_as_indices: bool = key_as_indices if key_as_indices is not None else self.key_as_indices
        self.key_with_borders: bool = key_with_borders if key_with_borders is not None else self.key_with_borders
        self.indices: list | None = indices
        self.iterator: Iterator | None = None

        # borders
        if borders is True:
            borders = self.default_borders
        if borders is False:
            borders = None
        borders = borders if borders is not None else self.borders
        self.borders = (None, None) if borders is None else borders

        # initialize depth
        if isinstance(depth, Callable):
            s = signature(depth)
            depth = len([p for p in s.parameters.values() if p.default == s.empty])
            depth -= sum(self.add_borders)
        self.depth: int = depth if depth is not None else self.depth

    @property
    def add_borders(self) -> tuple:
        return tuple(b is not None for b in self.borders)

    @property
    def has_borders(self) -> bool:
        return any(self.add_borders)

    @property
    def n_measurement_times(self) -> int:
        return len(self.measurement_times)

    @property
    def n_iterations(self) -> int:
        return len(list(iter(self)))

    def index_apply_borders(self, index: tuple) -> tuple:
        add_left, add_right = self.add_borders
        if add_left:
            index = (index[0] - 1,) + index
        if add_right:
            index = index + (index[-1] + 1,)
        return index

    def key_apply_borders(self, key: tuple) -> tuple:
        if self.has_borders and not self.key_with_borders:
            add_left, add_right = self.add_borders
            key = key[1 if add_left else None: -1 if add_right else None]
        return key

    def measurement_times_from_index(self, index: tuple) -> tuple:
        left_border, right_border = self.borders
        measurement_times = self.measurement_times
        n = self.n_measurement_times
        return tuple(measurement_times[i] if 0 <= i < n else left_border if i == -1 else right_border for i in index)

    def __iter__(self) -> Iterator | Self:
        if self.indices is not None:
            self.iterator = iter(self.indices)
        else:
            self.iterator = iter(combinations_with_replacement(range(self.n_measurement_times), self.depth))
        return self

    def __next__(self):
        index = self.index_apply_borders(next(self.iterator))
        times = self.measurement_times_from_index(index)
        key = self.key_apply_borders(index if self.key_as_indices else times)
        return key, times


class MeasurableEventData(EventData):

    name: str | None = None

    def __init__(
            self,
            event_sequences: np.ndarray | BufferedArray | None = None,
            measurement_times: np.ndarray | None = None,
            name: str | None = None
    ):
        EventData.__init__(self, event_sequences=event_sequences)
        self.measurement_times: np.ndarray | BufferedArray | None = measurement_times
        self.name: str | None = name

    # measure definitions

    @classmethod
    def _seen(cls, t1, t2, t):
        """Events seen at time t."""
        return np.logical_and(t1 <= t, t <= t2)

    @classmethod
    def _generated(cls, t1, ts, tf):
        """Events generated between ts and tf."""
        return np.logical_and(ts < t1, t1 <= tf)

    @classmethod
    def _appeared(cls, t1, t2, ts, tf):
        """Events generated between ts and tf."""
        return cls._continued(t1, t2, ts, tf, tf)

    @classmethod
    def _continued(cls, t1, t2, ts, tf, tl):
        """Events appeared with start times between ts and tf and remained visible until tl."""
        generated = cls._generated(t1, ts, tf)
        remained = tl <= t2
        return np.logical_and(generated, remained)

    @classmethod
    def _disappeared(cls, t1, tl, te):
        """Events disappeared between tl and te"""
        return np.logical_and(tl <= t1, t1 < te)

    @classmethod
    def _occurred(cls, t1, t2, ts, tf, tl, te):
        """Events occurred with start times between ts and tf and end times between tl and te."""
        appeared = cls._generated(t1, ts, tf)
        disappeared = cls._disappeared(t2, tl, te)
        return np.logical_and(appeared, disappeared)

    # measures

    def seen(self, t, event_types: tuple = (0, -1)):
        return self._seen(*self[event_types].T, t)

    def generated(self, ts, tf, event_types: tuple = (0,)):
        return self._generated(*self[event_types].T, ts, tf)

    def appeared(self, ts, tf, event_types: tuple = (0, -1)):
        return self._appeared(*self[event_types].T, ts, tf)

    def continued(self, ts, tf, tl, event_types: tuple = (0, 1)):
        return self._continued(*self[event_types].T, ts, tf, tl)

    def disappeared(self, tl, te, event_types: tuple = (-1,)):
        return self._disappeared(*self[event_types].T, tl, te)

    def occurred(self, ts, tf, tl, te, event_types: tuple = (0, -1)):
        return self._occurred(*self[event_types].T, ts, tf, tl, te)

    def survived(self, ts, tf, tl, event_types: tuple = (0, -1)):
        norm = max(1, self._continued(*self[event_types].T, ts, tf, tf).sum())
        return self._continued(*self[event_types].T, ts, tf, tl).sum() / norm

    # measurements

    def measurement_durations(
            self,
            measurement_times: np.ndarray | None
    ) -> tuple:
        measurement_times = self.measurement_times if measurement_times is None else measurement_times
        durations = []
        for tf in measurement_times:
            for tl in measurement_times:
                durations.append(tl - tf)
        return tuple(np.unique(durations))

    def standardize_measurement_times(
            self,
            measurement_times: float | tuple[float, ...] | np.ndarray | None = None,
            max_event_time: float | None = None,
    ) -> list:
        max_event_time = max_event_time if max_event_time is not None else self.max_event_time

        measurement_times = self.measurement_times if measurement_times is None else measurement_times
        if measurement_times is None:
            raise ValueError(f"no measurement times.")
        if isinstance(measurement_times, float):
            measurement_times = (measurement_times,)

        if any(t < 0 for t in measurement_times):
            if max_event_time is None:
                raise ValueError('negative observation times without t_max')
            measurement_times = tuple(t if t >= 0 else max_event_time + t for t in measurement_times)
            if any(-np.inf < t < 0 for t in measurement_times):
                raise ValueError('negative observation times')

        return list(sorted(measurement_times))

    def measurement_iterator(
            self,
            measurement_times: list | np.ndarray | None = None,
            depth: int | Callable | None = None,
            borders: bool | tuple | None = None,
            key_as_indices: bool = False,
            key_with_borders: bool = False,
            max_event_time: float | None = None,
    ):
        return MeasurementIterator(
            measurement_times=self.standardize_measurement_times(measurement_times, max_event_time),
            depth=depth,
            borders=borders,
            key_as_indices=key_as_indices,
            key_with_borders=key_with_borders
        )

    def measure(
            self,
            measure: str | Callable,
            measurement_times: list | np.ndarray | None = None,
            measurement_iterator: Iterator | None = None,
            measurement_times_indices: list | None = None,
            measurement_times_borders: bool | tuple | None = None,
            measurement_key_as_indices: bool = False,
            measurement_key_with_borders: bool = False,
            max_event_time: float | None = None,
            measure_event_types: tuple[str, ...] | None = None,
            measurement: Measurement | None = None,
            apply: Callable | None = np.sum,
            normalize: bool | float = False
    ) -> Measurement:
        name = None
        if isinstance(measure, str):
            name = measure
            measure = getattr(self, measure)
        if apply is None:
            apply = identity
        if measure_event_types is not None:
            measure = partial(measure, event_types=measure_event_types)

        measurement_times = self.standardize_measurement_times(
            measurement_times=measurement_times,
            max_event_time=max_event_time
        )

        if measurement_iterator is None:
            measurement_iterator = MeasurementIterator(
                measurement_times=measurement_times,
                depth=measure,
                borders=measurement_times_borders,
                indices=measurement_times_indices,
                key_as_indices=measurement_key_as_indices,
                key_with_borders=measurement_key_with_borders,
            )

        measurement = measurment if measurement is not None else Measurement()
        measurement.name = name

        for key, times in iter(measurement_iterator):
            measurement[key] = apply(measure(*times))

        return measurement.normalize(normalize=normalize)


def _plot_event_sequences_base(
        event_sequences: np.ndarray,
        event_sequence_positions: int | float | np.ndarray | None = None,
        event_sequence_colors: tuple | list | np.ndarray | None = None,
        event_sequence_labels: tuple | list | np.ndarray | None = None,
        plotting: dict | None = None
) -> dict:
    """Event plotting helper."""
    event_sequences = np.asarray(event_sequences)
    if event_sequences.ndim != 2 or event_sequences.shape[1] != 2:
        raise ValueError
    n_lines = len(event_sequences)

    plotting = plotting if plotting is not None else dict()
    ax = plotting.get('ax', None)
    ax = ax if ax is not None else plt.gca()
    legend = plotting.get('legend', dict())
    limits = plotting.get('limits', dict())
    line_width = plotting.get('line_width', 2)
    line_padding = plotting.get('line_padding', 'auto')
    axes_padding = plotting.get('axes_padding', None)
    axes_padding = axes_padding if axes_padding is not None else (0.5, 0.5)

    if line_padding == 'auto':
        times = np.sort(np.unique(event_sequences))
        if len(times) <= 1:
            line_padding = 0.25
        else:
            line_padding = np.min(times[1:] - times[:-1]) / 4
    if line_padding is not None:
        line_padding = (-line_padding, line_padding) if not isinstance(line_padding, tuple) else line_padding
        event_sequences = event_sequences + np.array(line_padding)[None, :]

    if event_sequence_positions is None:
        event_sequence_positions = np.arange(n_lines)
    elif isinstance(event_sequence_positions, float | int):
        event_sequence_positions = np.arange(n_lines) + event_sequence_positions

    line_kwargs = dict(colors=event_sequence_colors, linewidths=line_width)

    start_times, end_times = event_sequences.T
    lines = np.array([np.array([start_times, event_sequence_positions]).T,
                      np.array([end_times, event_sequence_positions]).T]).transpose([1, 0, 2])
    lines = mplc.LineCollection(lines, **line_kwargs)

    ax.add_collection(lines)
    plotting['ax'] = ax

    x_lim = limits.get('x_lim', (np.inf, -np.inf))
    limits['x_lim'] = (
               min(x_lim[0], np.min(event_sequences) - axes_padding[0]),
               max(x_lim[1], np.max(event_sequences) + axes_padding[1])
    )
    y_lim = limits.get('y_lim', (np.inf, -np.inf))
    limits['y_lim'] = (
               min(y_lim[0], np.min(event_sequence_positions) - axes_padding[0]),
               max(y_lim[1], np.max(event_sequence_positions) + axes_padding[1])
    )
    plotting['limits'] = limits

    if event_sequence_labels is not None:
        for label, color in zip(event_sequence_labels, event_sequence_colors):
            if label is not None and label not in legend:
                legend[label] = mpl.lines.Line2D([0, 1], [0, 1], color=color, linewidth=line_width)
    plotting['legend'] = legend

    return plotting


def _plot_event_sequences_using_event_types(
        event_sequences: np.ndarray,
        event_types: tuple[str | int, ...] | None = None,
        event_sequence_positions: int | float | np.ndarray | None = None,
        event_type_colors: np.ndarray | tuple | list | None = None,
        plotting: dict | None = None
) -> dict:
    """Helper plotting events coloring different event types."""
    n_event_sequences = event_sequences.shape[0]
    n_event_types = event_sequences.shape[1] - 1
    if n_event_types < 1:
        raise ValueError

    event_type_colors = event_type_colors if event_type_colors is not None else default_colors()[:n_event_types]
    event_types = event_types if event_types is not None else (None,) * n_event_types

    for i in range(n_event_types):
        label, color = event_types[i], event_type_colors[i]
        event_sequence_labels = (label,) * n_event_sequences
        event_sequence_colors_ = (color,) * n_event_sequences
        plotting = _plot_event_sequences_base(
            event_sequences=event_sequences[:, [i, i + 1]],
            event_sequence_positions=event_sequence_positions,
            event_sequence_labels=event_sequence_labels,
            event_sequence_colors=event_sequence_colors_,
            plotting=plotting
        )

    return plotting


def _plot_event_sequences_using_sequence_ids(
            event_sequences: np.ndarray,
            event_sequence_positions: int | float | np.ndarray | None = None,
            event_sequence_ids: np.ndarray | None = None,
            event_sequence_id_colors: dict | None = None,
            event_sequence_id_labels: dict | None = None,
            plotting: dict | None = None
) -> dict:
    """Helper plotting events coloring different objects according to their ids."""

    n_event_sequences = event_sequences.shape[0]
    n_event_types = event_sequences.shape[1] - 1
    if n_event_types < 1:
        raise ValueError
    event_sequences = event_sequences[:, [0, -1]]

    event_sequence_ids = event_sequence_ids if event_sequence_ids is not None else np.arange(n_event_sequences)

    event_sequence_id_colors = event_sequence_id_colors if event_sequence_id_colors is not None \
        else {i: default_colors()[i] for i in range(len(event_sequence_ids))}
    event_sequence_colors = [event_sequence_id_colors[i] for i in event_sequence_ids]
    event_sequence_labels = [event_sequence_id_labels[i] for i in event_sequence_ids] \
        if event_sequence_id_labels is not None else None

    return _plot_event_sequences_base(
        event_sequences=event_sequences,
        event_sequence_positions=event_sequence_positions,
        event_sequence_colors=event_sequence_colors,
        event_sequence_labels=event_sequence_labels,
        plotting=plotting
    )


def plot_event_sequences(
        event_sequences: np.ndarray | EventData,
        event_types: np.ndarray | tuple | list | None = None,
        event_type_colors: np.ndarray | tuple | list | None = None,
        event_sequence_positions: np.ndarray | tuple | list | None = None,
        event_sequence_colors: np.ndarray | tuple | list | None = None,
        event_sequence_labels: np.ndarray | tuple | list | None = None,
        event_sequence_ids: bool | np.ndarray | tuple | None = None,
        event_sequence_id_colors: np.ndarray | tuple | None = None,
        event_sequence_id_labels: np.ndarray | tuple | None = None,
        ax: object | None = None,
        axes_labels: tuple = ('time', 'synapse'),
        title: str | None = None,
        legend: str | bool = True,
        line_width: float | None = None,
        line_padding: str | float | tuple = 'auto',
        axes_padding: tuple | None = None
) -> Axes:
    if event_sequence_ids is True:
        event_sequence_ids = getattr(event_sequences, 'event_sequence_ids')
        event_sequence_id_colors = getattr(event_sequences, 'event_sequence_id_colors') \
            if event_sequence_id_colors is None else event_sequence_id_colors
        event_sequence_id_labels = getattr(event_sequences, 'event_sequence_id_labels') \
            if event_sequence_id_labels is None else event_sequence_id_labels

    event_types = event_types if event_types is not None else event_sequences.event_types
    event_type_colors = event_type_colors if event_type_colors is not None else event_sequences.event_type_colors
    event_sequences = event_sequences.event_sequences if isinstance(event_sequences, EventData) else event_sequences

    plotting = dict(
        ax=ax,
        axes_labels=axes_labels,
        line_padding=line_padding,
        line_width=line_width,
        axes_padding=axes_padding
    )

    plot_function = _plot_event_sequences_base
    plot_kwargs = dict(
        event_sequences=event_sequences,
        event_sequence_positions=event_sequence_positions,
    )

    if event_sequence_colors is not None:
        plot_kwargs.update(
            event_sequence_colors=event_sequence_colors,
            event_sequence_labels=event_sequence_labels
        )
    elif event_sequence_ids is not None:
        plot_kwargs.update(
            event_sequence_ids=event_sequence_ids,
            event_sequence_id_colors=event_sequence_id_colors,
            event_sequence_id_labels=event_sequence_id_labels
        )
        plot_function = _plot_event_sequences_using_sequence_ids
    elif event_types is not None:
        plot_kwargs.update(
            event_types=event_types,
            event_type_colors=event_type_colors
        )
        plot_function = _plot_event_sequences_using_event_types

    plotting = plot_function(**plot_kwargs, plotting=plotting)

    ax = plotting['ax']
    limits = plotting['limits']
    ax.set_xlim(limits['x_lim'])
    ax.set_ylim(limits['y_lim'])
    set_axes(ax=ax, title=title, axes_labels=axes_labels, legend=None)
    if legend is not None and legend is not False:
        loc = legend if isinstance(legend, str) else "upper left"
        legend = plotting['legend']
        ax.legend(list(legend.values()), list(legend.keys()), loc=loc)

    return ax


def plot_measurements(
        measurements: list | dict | Measurement,
        labels: list | str | None = None,
        colors: list | None = None,
        ax=None,
        axes_labels=('measurement', 'count'),
        title='measurement',
        legend: bool | str | None = 'upper left',
        rotate_ticks=75
):
    if not isinstance(measurements, list | tuple):
        measurements = [measurements]
    if labels is None or isinstance(labels, str):
        labels = [labels] * len(measurements)
    colors = colors if colors is not None else default_colors()

    types = []
    for measurement in measurements:
        for key in measurement.keys():
            if key not in types:
                types.append(key)

    pos = np.arange(len(types))
    type_width = 1.0
    bar_width = type_width / (len(measurements) + 1)

    ax = ax if ax is not None else plt.gca()
    for s, measurement in enumerate(measurements):
        data = [measurement.get(t, 0) for t in types]
        ax.bar(pos + s * bar_width, data, width=bar_width, label=labels[s], color=colors[s])
    ax.set_xticks(pos, types, rotation=rotate_ticks)
    legend = 'upper left' if legend is True else legend
    set_axes(ax=ax, title=title, axes_labels=axes_labels, legend=legend)

    return ax


def plot_survival_data(
        measurements: list | dict | Measurement,
        labels: list | str | None = None,
        colors: list | None = None,
        ax=None,
        axes_labels=('measurement', 'count'),
        title='measurement',
        legend: bool | str | None = 'upper right',
):
    if not isinstance(measurements, list | tuple):
        measurements = [measurements]
    if labels is None or isinstance(labels, str):
        labels = [labels] * len(measurements)
    colors = colors if colors is not None else default_colors()

    curves = [measurement.to_evolution_curves() for measurement in measurements]

    ax = ax if ax is not None else plt.gca()

    times = []
    for curve, label, color in zip(curves, labels, colors):
        for i, survival_curve in enumerate(curve):
            ax.plot(survival_curve[:, 0], survival_curve[:, 1], color=color, label=label if i == 0 else None)
            times.append(survival_curve[:, 0])
    times = np.unique(np.concatenate(times))
    if all(t % 1 == 0 for t in times):
        times = np.array(times, dtype=int)

    ax.set_xticks(times, times)
    legend = 'upper right' if legend is True else legend
    set_axes(ax=ax, title=title, axes_labels=axes_labels, legend=legend)

    return ax
