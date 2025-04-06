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

import copy
import numpy as np
import pandas as pd

from pathlib import Path

from event_data import MeasurableEventData
from utils.plotting import plt, default_colors


class ExperimentalData(MeasurableEventData):
    data_file: Path = Path.cwd() / 'data' / 'synapse_data.xlsx'
    data_sheet: int | str = 'wt'

    data_sheet_map: dict = {
        'wt': 8,
        'ko': 10
    }

    def __init__(self, data_file: str | None = None, data_sheet: int | str | None = None):
        MeasurableEventData.__init__(self, event_sequences=None)

        self.fish_ids: np.array | None = None
        self.cell_ids: np.array | None = None
        self.ids: np.array | None = None
        self.dendrite_lengths: dict | None = None

        self.data_file: str = data_file or self.data_file
        self.data_sheet: int | str = data_sheet or self.data_sheet
        if self.data_file is not None:
            self.load()

    def load(self, data_file: str | Path | None = None, data_sheet: int | str | None = None):
        """Load data set."""
        data_file = data_file if data_file is not None else self.data_file
        self.data_file = data_file

        # synapse data
        data_sheet = data_sheet if data_sheet is not None else self.data_sheet
        self.data_sheet = data_sheet

        data_sheet = self.data_sheet_map[data_sheet] if isinstance(data_sheet, str) else data_sheet

        df = pd.read_excel(data_file, sheet_name=data_sheet, header=3, usecols="B:L")  # noqa
        event_sequences = np.array([eval(o) for o in df['life']])
        self.event_sequences = event_sequences

        # identity
        df_fish_label = df['animal']
        fish_label = [l for l in df_fish_label.unique() if l is not np.nan]
        fish_id_start = np.sort(
            np.array([np.where(df_fish_label == l)[0][0] for l in fish_label] + [len(df_fish_label)])
        )
        fish_id_length = [fish_id_start[i+1] - fish_id_start[i] for i in range(len(fish_id_start)-1)]
        fish_ids = np.concatenate([np.full(fish_id_length[i], fill_value=i) for i in range(len(fish_id_length))])
        self.fish_ids = fish_ids

        df_cell_label = df['cell']
        cell_label = [l for l in df_cell_label.unique() if l is not np.nan]
        cell_id_start = np.sort(
            np.concatenate([np.where(df_cell_label == l)[0] for l in cell_label] + [[len(df_cell_label)]])
        )
        cell_id_length = [cell_id_start[i+1] - cell_id_start[i] for i in range(len(cell_id_start)-1)]
        cell_ids = np.concatenate([np.full(cell_id_length[i], fill_value=i) for i in range(len(cell_id_length))])
        self.cell_ids = cell_ids

        assert (np.all(cell_ids < 1000))
        ids = fish_ids * 1000 + cell_ids
        self.ids = ids

        starts = np.concatenate([[0], np.where(np.diff(ids) > 0)[0] + 1])
        # ends = np.concatenate([starts[1:], [len(ids)]])
        # self.id_to_synapses = {i: range(s, e) for i, (s, e) in enumerate(zip(starts, ends))}

        # dendritic lengths
        dendrite_lengths = np.array(df["dendrite length (µm)"])
        self.dendrite_lengths = {i: l for i, l in zip(ids[starts], dendrite_lengths[starts])}

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
        self.ids = self.ids[order]
        self.cell_ids = self.cell_ids[order]
        self.fish_ids = self.fish_ids[order]
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
    def total_dendritic_length(self):
        return np.sum(list(self.dendrite_lengths.values()))

    def choose(self, indices):
        selected = copy.deepcopy(self)
        selected.event_sequences = selected.event_sequences[indices]
        selected.ids = selected.ids[indices]
        selected.cell_ids = selected.cell_ids[indices]
        selected.fish_ids = selected.fish_ids[indices]
        return selected

    def resample(self, size: int | None = None, replace: bool = True):
        size = size if size is not None else len(self)
        sample_ids = np.random.choice(range(len(self)), size=size, replace=replace)
        return self.choose(sample_ids)
