# -*- coding: utf-8 -*-
"""
Dataset
=======

EventData for experimentally observed data.

Examples
--------
>>> from dataset import ExperimentalData
>>> data = ExperimentalData()
>>> data
Dataset('first', 'last'){363}

>>> data.unique_event_times
array([ 0,  6, 12, 24, 48, 72])

>>> data.measurement_times
array([ 0,  6, 12, 24, 48, 72])

>>> measurement = data.measure('occurred', measurement_times_borders=True, measure_with_borders=True)
>>> measurement
Measurement[occurred]{21}

>>> data.dendrite_lengths
{0: ... }

>>> data.sort(sort_order=('duration', 'first', 'last'))

>>> from utils.plotting import plt
>>> plt.figure(1); plt.clf()
>>> ax = plt.subplot(1, 2, 1)
>>> _ = data.plot(ax=ax, legend=True, event_sequence_ids=True)
>>> ax = plt.subplot(1, 2, 2)
>>> _ = measurement.plot(ax=ax)

>>> data_ko = ExperimentalData(data_sheet='ko')
>>> data_ko
ExperimentalData('first', 'last'){296}
"""
__project__ = 'Developmental Synapse Remodeling'
__author__ = 'Christoph Kirst <christoph.kirst@ucsf.edu>'
__copyright__ = 'Copyright © 2025 by Christoph Kirst'


# typing
from typing import Self
from pathlib import Path

# imports
import copy
import numpy as np
import pandas as pd

from event_data import MeasurableEventData, Measurement
from utils.plotting import plt, default_colors


class ExperimentalData(MeasurableEventData):
    data_file: Path = Path.cwd() / 'data' / 'synapse_data.xlsx'
    data_sheet: int | str = 'wt'

    data_sheet_map: dict = {
        'wt': 8,
        'ko': 10,
        'wt_24': 7,
        'ko_24': 9,
        'wt_24_1': 6,
        'ko_24_1': 6,
    }

    data_type_map: dict = {
        'wt': 'synapses',
        'ko': 'synapses',
        'wt_24': 'synapses',
        'ko_24': 'synapses',
        'wt_24_1': 'counts',
        'ko_24_1': 'counts',
    }

    condition_map: dict = {
        'wt': 'mmp14b+/+',
        'ko': 'mmp14b-/-',
    }

    def __init__(self, data_file: str | None = None, data_sheet: int | str | None = None):
        MeasurableEventData.__init__(self, event_sequences=None)

        self.fish_ids: np.array | None = None
        self.cell_ids: np.array | None = None
        self.dendrite_lengths: dict | None = None

        self.data_file: str = data_file or self.data_file
        self.data_sheet: int | str = data_sheet or self.data_sheet
        if self.data_file is not None:
            self.load()

    @classmethod
    def fish_cell_ids_to_ids(cls, fish_ids, cell_ids):
        assert(np.max(cell_ids) < 1000)
        return 1000 * fish_ids + cell_ids

    @property
    def ids(self):
        return self.fish_cell_ids_to_ids(self.fish_ids, self.cell_ids)

    @classmethod
    def data_frame(cls, data_file: str | Path, data_sheet: int, cols: str = "A:L"):
        return pd.read_excel(data_file, sheet_name=data_sheet, header=3, usecols=cols)  # noqa

    @classmethod
    def parse_data_frame_labels(cls, df):
        # identity
        df_fish_label = df['animal']
        fish_label = [l for l in df_fish_label.unique() if l is not np.nan]
        fish_id_start = np.sort(
            np.array([np.where(df_fish_label == l)[0][0] for l in fish_label] + [len(df_fish_label)])
        )
        fish_id_length = [fish_id_start[i + 1] - fish_id_start[i] for i in range(len(fish_id_start) - 1)]
        fish_ids = np.concatenate([np.full(fish_id_length[i], fill_value=i) for i in range(len(fish_id_length))])

        df_cell_label = df['cell']
        cell_label = [l for l in df_cell_label.unique() if l is not np.nan]
        cell_id_start = np.sort(
            np.concatenate([np.where(df_cell_label == l)[0] for l in cell_label] + [[len(df_cell_label)]])
        )
        cell_id_length = [cell_id_start[i + 1] - cell_id_start[i] for i in range(len(cell_id_start) - 1)]
        cell_ids = np.concatenate([np.full(cell_id_length[i], fill_value=i) for i in range(len(cell_id_length))])

        ids = cls.fish_cell_ids_to_ids(fish_ids, cell_ids)
        starts = np.concatenate([[0], np.where(np.diff(ids) > 0)[0] + 1])
        # ends = np.concatenate([starts[1:], [len(ids)]])

        # dendritic lengths
        dendrite_lengths = np.array(df["dendrite length (µm)"])
        dendrite_lengths = {(fish_ids[i], cell_ids[i]): l for i, l in zip(starts, dendrite_lengths[starts])}

        return fish_ids, cell_ids, dendrite_lengths

    def parse_data_frame_synapses(self, df: pd.DataFrame, condition: str):  # noqa
        # events
        event_sequences = np.array([eval(o) for o in df['life']])
        self.event_sequences = event_sequences

        # identity
        self.fish_ids, self.cell_ids, self.dendrite_lengths = self.parse_data_frame_labels(df)

    def parse_data_frame_counts(self, df: pd.DataFrame, condition: str):
        # read counts
        condition_ = self.condition_map[condition]
        df = df[df['condition'] == condition_]

        count_headers = {
            (0, 0): '# of Lost synapses',
            (0, 24): '# of Stable synapses',
            (24, 24): '# of New synapses',
        }
        counts = {k: np.array(df[h], dtype=int) for k, h in count_headers.items()}

        fish_ids_cell, cell_ids_cell, dendrite_lengths = self.parse_data_frame_labels(df)

        # generate event sequences
        n_events = np.sum([v for v in counts.values()])
        event_sequences = np.zeros((n_events, 2))
        fish_ids = np.zeros(n_events, dtype=int)
        cell_ids = np.zeros(n_events, dtype=int)

        n_cells = len(fish_ids_cell)
        i = 0
        for c in range(n_cells):
            for event, count in counts.items():
                n_events_cell = count[c]
                event_sequences[i: i + n_events_cell] = event
                fish_ids[i: i + n_events_cell] = fish_ids_cell[c]
                cell_ids[i: i + n_events_cell] = cell_ids_cell[c]
                i += n_events_cell

        # save
        self.event_sequences = event_sequences
        self.fish_ids, self.cell_ids = fish_ids, cell_ids
        self.dendrite_lengths = dendrite_lengths

    def load(self, data_file: str | Path | None = None, data_sheet: int | str | None = None):
        """Load data set."""
        data_file = data_file if data_file is not None else self.data_file
        self.data_file = data_file

        # synapse data
        data_sheet = data_sheet if data_sheet is not None else self.data_sheet
        self.data_sheet = data_sheet

        if isinstance(data_sheet, str):
            data_sheet_name = data_sheet
            data_sheet = self.data_sheet_map[data_sheet]
        else:
            data_sheet_name = [k for k, v in self.data_sheet_map.items() if data_sheet == v][0]

        data_type = self.data_type_map[data_sheet_name]

        df = self.data_frame(data_file=data_file, data_sheet=data_sheet)
        parse = getattr(self, f'parse_data_frame_{data_type}')
        return parse(df, condition=data_sheet_name[:2])

    @property
    def measurement_times(self):
        return self.unique_event_times

    @measurement_times.setter
    def measurement_times(self, value):
        pass

    def sort(
            self,
            sort_order: tuple[int | str, ...] | None = None,
            return_order: bool = False
    ):
        event_times, order = self.sorted_event_sequences(sort_order=sort_order, return_order=True)
        self.event_sequences = event_times
        self.fish_ids = self.fish_ids[order]
        self.cell_ids = self.cell_ids[order]
        if return_order:
            return order
        else:
            return self

    @property
    def event_sequence_ids(self):
        return self.ids

    @property
    def event_sequence_id_colors(self) -> dict:
        color_list = default_colors()
        return {i: color_list[k] for k, i in enumerate(np.unique(self.ids))}

    @property
    def event_sequence_id_labels(self) -> dict:
        return {i: f"fish {fid} cell {cid}" for i, fid, cid in zip(self.ids, self.fish_ids, self.cell_ids)}

    @property
    def total_dendrite_length(self):
        return np.sum(list(self.dendrite_lengths.values()))

    @property
    def cell_ids_to_dendrite_lengths(self):
        return {k[1]: v for k, v in self.dendrite_lengths.items()}

    @property
    def fish_ids_to_dendrite_lengths(self):
        dendrite_lengths = {k[0]: 0 for k in self.dendrite_lengths.keys()}
        for k, v in self.dendrite_lengths.items():
            dendrite_lengths[k[0]] = dendrite_lengths[k[0]] + v
        return dendrite_lengths

    def choose(self, indices):
        selected = MeasurableEventData.choose(self, indices)
        selected.cell_ids = selected.cell_ids[indices]
        selected.fish_ids = selected.fish_ids[indices]
        return selected

    def join(self, other) -> Self:
        if not isinstance(other, ExperimentalData):
            raise ValueError
        fish_id_offset = np.max(self.fish_ids)
        cell_id_offset = np.max(self.cell_ids)
        joined = MeasurableEventData.join(self, other)
        joined.fish_ids = np.concatenate([self.fish_ids, other.fish_ids + fish_id_offset], axis=0)
        joined.cell_ids = np.concatenate([self.cell_ids, other.cell_ids + cell_id_offset], axis=0)
        joined.dendrite_lengths.update(
            {(k[0] + fish_id_offset, k[1] + cell_id_offset): v for k, v in other.dendrite_lengths.items()}
        )

        return joined
