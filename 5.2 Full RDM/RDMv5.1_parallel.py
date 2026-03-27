# -*- coding: utf-8 -*-
"""
RDM v5.1 — Parallel stress-test runner for all optimisation portfolios.

Reads Pywr model JSONs from output_json/json_RDM_{RUN_ID}/po_*/,
runs simulations, and saves results as CSV.

Updated from RDMv4.0_parallel.py with v5.1 custom classes
(DelayedActivationParameter, PercentileParameterRecorder).

@author: guill
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import os
import gc
import json
import pandas as pd
import numpy as np
import time
import shutil

from pywr.core import Model
from pywr.parameters import Parameter, load_parameter
from pywr.recorders import (
    Recorder,
    BaseConstantParameterRecorder,
    recorder_registry,
)

import warnings

# Suppress common runtime warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=".*Document requires version.*",
    category=RuntimeWarning
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "output_json")
EXECUTION_RUN_ID = "20260306-184644" # Replace with the actual run ID used in optimization
JSON_OUTPUT_FOLDER_NAME = f"json_RDM_{EXECUTION_RUN_ID}"
JSON_OUTPUT_FINAL_DIR = os.path.join(JSON_OUTPUT_BASE_DIR, JSON_OUTPUT_FOLDER_NAME)
RDM_DIR_PATH = os.path.join(SCRIPT_DIR, "RDM_results")
RESULTS_DIR_PATH = os.path.join(RDM_DIR_PATH, f"RDM_results_{EXECUTION_RUN_ID}")
DATA_INPUTS_PATH = os.path.join(SCRIPT_DIR, "2. Data Inputs")

csv_files = ['supply_corrected.csv', 'demand_corrected.csv', 'anc_corrected.csv', 'projects.csv']

# Definir los directorios de trabajo divididos en bloques
directorios_trabajo = [
    d for d in os.listdir(JSON_OUTPUT_FINAL_DIR)
    if d.startswith("po_") and os.path.isdir(os.path.join(JSON_OUTPUT_FINAL_DIR, d))
]

os.makedirs(RDM_DIR_PATH, exist_ok=True)

# Crear el directorio de resultados si no existe
if not os.path.exists(RESULTS_DIR_PATH):
    os.makedirs(RESULTS_DIR_PATH)


# =============================================================================
# CUSTOM PARAMETERS
# =============================================================================

class TimeWindowBinaryParameter(Parameter):
    """A parameter that represents binary decisions (0 or 1) over specific time windows."""

    all_parameters = {}

    def __init__(self, model, windows, dependencies=None, strict_dependencies=None,
                 mutual_exclusion_group=None, values=None, **kwargs):
        super(TimeWindowBinaryParameter, self).__init__(model, **kwargs)

        self.windows = windows
        self.dependencies = dependencies or {}
        self.strict_dependencies = strict_dependencies or {}
        self.mutual_exclusion_group = mutual_exclusion_group or {}

        if values is not None:
            assert len(values) == len(windows), (
                f"Values length {len(values)} must match number of decision windows {len(windows)}."
            )
            self.values = np.array(values, dtype=np.float64)
        else:
            self.values = np.zeros(len(windows), dtype=np.float64)

        self._double_size = len(self.windows)
        self._lower_bounds = np.zeros(len(self.windows), dtype=np.float64)
        self._upper_bounds = np.ones(len(self.windows), dtype=np.float64)

        self.enforce_activation()

    @classmethod
    def set_all_parameters(cls, parameters):
        cls.all_parameters = parameters

    def register_dependencies(self):
        if not self.dependencies:
            return

        self._dependent_params = {}
        for dependent, prerequisite in self.dependencies.items():
            if prerequisite in TimeWindowBinaryParameter.all_parameters:
                self._dependent_params[dependent] = TimeWindowBinaryParameter.all_parameters[prerequisite]
            else:
                raise ValueError(f"Dependency mapping failed for {dependent} or {prerequisite}.")

    def enforce_mutual_exclusion(self):
        if not self.mutual_exclusion_group:
            return

        group_members = self.mutual_exclusion_group.get('members', [])
        max_simultaneous = self.mutual_exclusion_group.get('max_simultaneous', 1)

        if self.name not in group_members:
            return

        group_params = []
        for member_name in group_members:
            if member_name in TimeWindowBinaryParameter.all_parameters:
                group_params.append(TimeWindowBinaryParameter.all_parameters[member_name])

        if len(group_params) <= max_simultaneous:
            return

        for window_idx in range(len(self.windows)):
            activations_in_window = []

            for param in group_params:
                if window_idx == 0:
                    prev_value = 0.0
                else:
                    prev_value = param.values[window_idx - 1]

                current_value = param.values[window_idx]

                if prev_value == 0.0 and current_value == 1.0:
                    activations_in_window.append(param)

            if len(activations_in_window) > max_simultaneous:
                for i, param in enumerate(activations_in_window):
                    if i >= max_simultaneous:
                        param.values[window_idx] = 0.0
                        for future_idx in range(window_idx + 1, len(self.windows)):
                            param.values[future_idx] = 0.0

    def enforce_sequential_activation(self):
        for dependent_name, prereq_name in self.dependencies.items():
            if dependent_name != self.name:
                continue

            prereq = TimeWindowBinaryParameter.all_parameters.get(prereq_name)
            if prereq is None:
                raise ValueError(f"Dependency mapping failed for {dependent_name} or {prereq_name}.")

            for i in range(len(self.windows)):
                if prereq.values[i] == 0.0:
                    self.values[i] = 0.0

        for dependent_name, prereq_name in self.strict_dependencies.items():
            if dependent_name != self.name:
                continue

            prereq = TimeWindowBinaryParameter.all_parameters.get(prereq_name)
            if prereq is None:
                raise ValueError(f"Strict dependency mapping failed for {dependent_name} or {prereq_name}.")

            prereq_vals = np.array(prereq.values)
            activated_idxs = np.where(prereq_vals == 1.0)[0]

            if activated_idxs.size == 0:
                self.values[:] = 0.0
            else:
                first_idx = activated_idxs[0]
                self.values[:first_idx+1] = 0.0

        self.enforce_mutual_exclusion()

    def enforce_activation(self):
        for i in range(1, len(self.values)):
            if self.values[i - 1] == 1.0:
                self.values[i] = 1.0

    def value(self, timestep, scenario_index):
        try:
            value = 0.0
            for i, window in enumerate(self.windows):
                if timestep.index >= window:
                    value = float(self.values[i])
                else:
                    break
            return value
        except Exception as e:
            print(f"Error in value() for {self.name}: {e}")
            return 0.0

    def set_double_variables(self, values):
        assert len(values) == len(self.windows), (
            f"Decision values length {len(values)} must match the number of decision windows {len(self.windows)}."
        )
        self.values = values.astype(np.float64)
        self.enforce_sequential_activation()
        self.enforce_activation()

    def get_double_variables(self):
        return np.array(self.values, dtype=np.float64)

    def get_double_lower_bounds(self):
        return self._lower_bounds

    def get_double_upper_bounds(self):
        return self._upper_bounds

    @property
    def double_size(self):
        return self._double_size

    @double_size.setter
    def double_size(self, value):
        self._double_size = value

    @classmethod
    def load(cls, model, data):
        windows = data.pop('windows')
        values = data.pop('values', None)
        dependencies = data.pop('dependencies', None)
        strict_dependencies = data.pop('strict_dependencies', None)
        mutual_exclusion_group = data.pop('mutual_exclusion_group', None)

        return cls(model, windows=windows, dependencies=dependencies,
                   strict_dependencies=strict_dependencies,
                   mutual_exclusion_group=mutual_exclusion_group,
                   values=values, **data)

TimeWindowBinaryParameter.register()


class DelayedActivationParameter(Parameter):
    """
    Parámetro que retorna el estado de activación solo después del delay de construcción.
    """

    def __init__(self, model, activation_parameter, delay, **kwargs):
        super().__init__(model, **kwargs)

        self._activation_parameter = None
        self.activation_parameter = activation_parameter
        self.delay = int(delay)
        self._first_activation_ts = None

    @property
    def activation_parameter(self):
        return self._activation_parameter

    @activation_parameter.setter
    def activation_parameter(self, value):
        if self._activation_parameter is not None:
            self.children.remove(self._activation_parameter)
        self._activation_parameter = value
        if value is not None:
            self.children.add(value)

    def setup(self):
        super().setup()
        n_scenarios = len(self.model.scenarios.combinations)
        self._first_activation_ts = np.full(n_scenarios, -1, dtype=np.int32)

    def reset(self):
        if self._first_activation_ts is not None:
            self._first_activation_ts[:] = -1

    def value(self, timestep, scenario_index):
        try:
            current_activation = self._activation_parameter.get_value(scenario_index)
            si = scenario_index.global_id

            if current_activation >= 0.5:
                if self._first_activation_ts[si] < 0:
                    self._first_activation_ts[si] = timestep.index

                time_since_activation = timestep.index - self._first_activation_ts[si]
                if time_since_activation >= self.delay:
                    return 1.0

            return 0.0

        except Exception as e:
            print(f"Error in DelayedActivationParameter.value() for {self.name}: {e}")
            return 0.0

    @classmethod
    def load(cls, model, data):
        activation_parameter = load_parameter(model, data.pop('activation_parameter'))
        delay = int(data.pop('delay'))
        return cls(model, activation_parameter, delay, **data)


DelayedActivationParameter.register()


# =============================================================================
# CUSTOM RECORDERS
# =============================================================================

class MinParameterRecorder(BaseConstantParameterRecorder):
    """Record the minimum value of a Parameter during a simulation."""

    def __init__(self, *args, **kwargs):
        self.factor = kwargs.pop('factor', 1.0)
        super(MinParameterRecorder, self).__init__(*args, **kwargs)

    def reset(self):
        self._values = np.full(len(self.model.scenarios.combinations), np.inf)

    def after(self):
        factor = self.factor
        values = self._param.get_all_values()

        for scenario_index in self.model.scenarios.combinations:
            i = scenario_index.global_id
            scaled_value = values[i] * factor
            if scaled_value < self._values[i]:
                self._values[i] = scaled_value

    def aggregated_value(self):
        return self._values.min()

MinParameterRecorder.register()


class MaxParameterRecorder(BaseConstantParameterRecorder):
    """Record the maximum value of a Parameter during a simulation."""

    def __init__(self, *args, **kwargs):
        self.factor = kwargs.pop('factor', 1.0)
        super(MaxParameterRecorder, self).__init__(*args, **kwargs)

    def reset(self):
        self._values = np.full(len(self.model.scenarios.combinations), -np.inf)

    def after(self):
        factor = self.factor
        values = self._param.get_all_values()

        for scenario_index in self.model.scenarios.combinations:
            i = scenario_index.global_id
            scaled_value = values[i] * factor
            if scaled_value > self._values[i]:
                self._values[i] = scaled_value

    def aggregated_value(self):
        return self._values.max()

MaxParameterRecorder.register()


class PercentileParameterRecorder(Recorder):
    """
    Calcula el percentil especificado de un parámetro a lo largo de la simulación.
    """

    def __init__(self, model, parameter, percentile=5, **kwargs):
        name = kwargs.pop('name', None)
        if name is None:
            name = f"percentileparameterrecorder.{parameter.name}"

        super().__init__(model, name=name, **kwargs)

        self._param = parameter
        self.percentile = float(percentile)

        if not 0 <= self.percentile <= 100:
            raise ValueError(f"percentile debe estar entre 0 y 100, se recibió {self.percentile}")

        parameter.parents.add(self)

    @property
    def parameter(self):
        return self._param

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        self._data = np.zeros((nts, ncomb), dtype=np.float64)

    def reset(self):
        self._data[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        self._data[ts.index, :] = self._param.get_all_values()
        return 0

    def values(self):
        return np.percentile(self._data, self.percentile, axis=0)

    def aggregated_value(self):
        values = self.values()
        return self._scenario_aggregator.aggregate_1d(values, ignore_nan=self.ignore_nan)

    @property
    def data(self):
        return np.array(self._data)

    def to_dataframe(self):
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        return pd.DataFrame(data=self._data, index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter
        parameter = load_parameter(model, data.pop('parameter'))
        percentile = data.pop('percentile')
        return cls(model, parameter, percentile=percentile, **data)


PercentileParameterRecorder.register()
recorder_registry['percentileparameterrecorder'] = PercentileParameterRecorder


# =============================================================================
# WORKER FUNCTION
# =============================================================================

def process_directory(directorio):
    dir_path = os.path.join(JSON_OUTPUT_FINAL_DIR, directorio)

    if not os.path.exists(dir_path):
        return f"{directorio} skipped (missing)"

    # ---- Ensure input data exists
    data_subdir = os.path.join(dir_path, "2. Data Inputs")
    os.makedirs(data_subdir, exist_ok=True)

    for csv_file in csv_files:
        src = os.path.join(DATA_INPUTS_PATH, csv_file)
        dst = os.path.join(data_subdir, csv_file)

        if os.path.exists(dst):
            continue

        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    # ---- Run models
    json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]

    for json_file in json_files:
        try:
            model = Model.load(os.path.join(dir_path, json_file))
            model.run()

            df = model.to_dataframe()
            out_csv = os.path.join(
                RESULTS_DIR_PATH,
                json_file.replace('.json', '.csv')
            )

            df.to_csv(out_csv, index=False)

        except Exception as e:
            print(f"✗ {directorio}/{json_file} failed: {e}")
            continue

        finally:
            # Clean up memory after each model
            try:
                del model
            except NameError:
                pass
            TimeWindowBinaryParameter.all_parameters = {}
            gc.collect()

    return f"{directorio} done"


if __name__ == "__main__":

    max_workers = min(os.cpu_count(), 48)  # keep conservative for Pywr

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(process_directory, d): d
            for d in directorios_trabajo
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Running RDM directories",
            unit="dir"
        ):
            result = future.result()
            tqdm.write(result)

    print("\n✅ All RDM directories processed.")
