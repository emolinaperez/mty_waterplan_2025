# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:31:23 2024

@author: guill
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import os
from pywr.core import Model
from pywr.recorders import Recorder
from pywr.recorders._recorders import NodeRecorder
from pywr.optimisation.platypus import PlatypusWrapper
from pywr.parameters import Parameter
from pywr.parameters._activation_functions import BinaryStepParameter
from pywr.recorders import BaseConstantParameterRecorder, recorder_registry
import json
import pandas as pd
import numpy as np
import platypus
from platypus.core import _EvaluateJob, Algorithm, PlatypusConfig, Solution, Variator
import platypus.evaluator
from platypus.evaluator import Job, ProcessPoolEvaluator, MultiprocessingEvaluator, MapEvaluator
from platypus.types import Binary
import time
from matplotlib import pyplot as plt
import logging
import copy
import random
import pickle
import shutil

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



json_base_path = r"D:\guillermo_sim_3\5.2 Full RDM\json_RDM"
results_path = r"D:\guillermo_sim_3\6.2 Full RDM Results\RDM_results"
data_inputs_path = r"D:\guillermo_sim_3\5.2 Full RDM\2. Data Inputs"

csv_files = ['supply_corrected.csv', 'demand_corrected.csv', 'anc_corrected.csv', 'projects.csv']

# Definir los directorios de trabajo divididos en bloques
directorios_trabajo = [
    d for d in os.listdir(json_base_path)
    if d.startswith("po_") and os.path.isdir(os.path.join(json_base_path, d))
]


# Crear el directorio de resultados si no existe
if not os.path.exists(results_path):
    os.makedirs(results_path)

# os.chdir('C:\\Users\\guill\\OneDrive\\Documents\\Freelance\\FAMM - Plan Hídrico NL\\Working Files')

# # Cargar archivos CSV de entrada
# supply_ts = pd.read_csv("supply.csv", delimiter=";")
# demand_ts = pd.read_csv("demand.csv", delimiter=";")

# supply_ts['Date'] = pd.to_datetime(supply_ts['Date'], format='%d/%m/%Y', dayfirst=True)
# demand_ts['Date'] = pd.to_datetime(demand_ts['Date'], format='%d/%m/%Y', dayfirst=True)

# supply_ts.to_csv("supply_corrected.csv", index=False)
# demand_ts.to_csv("demand_corrected.csv", index=False)

class TimeWindowBinaryParameter(Parameter):
    """A parameter that represents binary decisions (0 or 1) over specific time windows."""

    all_parameters = {}

    def __init__(self, model, windows, dependencies=None, strict_dependencies=None, 
                 mutual_exclusion_group=None, values=None, **kwargs):
        """
        Parameters:
        - model: Pywr model instance.
        - windows: List of integers representing decision time windows.
        - dependencies: Dictionary mapping dependent projects to prerequisite projects (simultaneous).
        - strict_dependencies: Dictionary mapping dependent projects to prerequisite projects (lagged).
        - mutual_exclusion_group: Dictionary with 'group_id', 'max_simultaneous', and 'members'.
        - values: Initial binary values for each window.
        """
        # IMPORTANTE: Inicializar el padre PRIMERO antes de asignar atributos
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

        # Inicializar double_size y bounds ANTES de enforce_activation
        self._double_size = len(self.windows)
        self._lower_bounds = np.zeros(len(self.windows), dtype=np.float64)
        self._upper_bounds = np.ones(len(self.windows), dtype=np.float64)
        
        self.enforce_activation()

    @classmethod
    def set_all_parameters(cls, parameters):
        """Set the class-level dictionary of all parameters."""
        cls.all_parameters = parameters

    def register_dependencies(self):
        """Register dependencies for the current parameter."""
        if not self.dependencies:
            return

        self._dependent_params = {}
        for dependent, prerequisite in self.dependencies.items():
            if prerequisite in TimeWindowBinaryParameter.all_parameters:
                self._dependent_params[dependent] = TimeWindowBinaryParameter.all_parameters[prerequisite]
            else:
                raise ValueError(f"Dependency mapping failed for {dependent} or {prerequisite}.")

    def enforce_mutual_exclusion(self):
        """Enforce mutual exclusion constraints: max N projects can activate in same window."""
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
        """Enforce dependencies between projects."""
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
        """Once a project is activated (1), all subsequent time steps must also be 1."""
        for i in range(1, len(self.values)):
            if self.values[i - 1] == 1.0:
                self.values[i] = 1.0

    def value(self, timestep, scenario_index):
        """Return the binary decision for the current timestep.
        
        CRÍTICO: Este método DEBE retornar un float, nunca None.
        """
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
            return 0.0  # Retornar 0.0 en caso de error, nunca None

    def set_double_variables(self, values):
        """Set binary decision values for each window."""
        assert len(values) == len(self.windows), (
            f"Decision values length {len(values)} must match the number of decision windows {len(self.windows)}."
        )
        self.values = values.astype(np.float64)
        self.enforce_sequential_activation()
        self.enforce_activation()

    def get_double_variables(self):
        """Return current variable values."""
        return np.array(self.values, dtype=np.float64)

    def get_double_lower_bounds(self):
        """Return lower bounds for optimization."""
        return self._lower_bounds

    def get_double_upper_bounds(self):
        """Return upper bounds for optimization."""
        return self._upper_bounds
    
    @property
    def double_size(self):
        """Number of decision variables."""
        return self._double_size
    
    @double_size.setter
    def double_size(self, value):
        """Set number of decision variables."""
        self._double_size = value

    @classmethod
    def load(cls, model, data):
        """Custom load method to handle all parameter types."""
        windows = data.pop('windows')
        values = data.pop('values', None)
        dependencies = data.pop('dependencies', None)
        strict_dependencies = data.pop('strict_dependencies', None)
        mutual_exclusion_group = data.pop('mutual_exclusion_group', None)
        
        return cls(model, windows=windows, dependencies=dependencies, 
                   strict_dependencies=strict_dependencies,
                   mutual_exclusion_group=mutual_exclusion_group,
                   values=values, **data)

TimeWindowBinaryParameter.register()  # Register the class so that it can be loaded from JSON.

class MinParameterRecorder(BaseConstantParameterRecorder):
    """Record the minimum value of a `Parameter` during a simulation.
    
    This recorder tracks the minimum value returned by a `Parameter`
    over the course of a model's simulation. A factor can be provided to
    apply a linear scaling to the values before recording.
    
    Parameters
    ----------
    model : `pywr.core.Model`
        The model instance.
    param : `pywr.parameters.Parameter`
        The parameter whose minimum value will be recorded.
    name : str, optional
        The name of the recorder.
    factor : float, default=1.0
        A scaling factor for the values of `param`.
    """

    def __init__(self, *args, **kwargs):
        self.factor = kwargs.pop('factor', 1.0)
        super(MinParameterRecorder, self).__init__(*args, **kwargs)

    def reset(self):
        """Initialize _values to positive infinity before starting the simulation.
        
        This ensures that the minimum value can be accurately recorded
        by starting from a high initial value for comparison.
        """
        self._values = np.full(len(self.model.scenarios.combinations), np.inf)  # Initialize with positive infinity

    def after(self):
        """Update the recorded minimum value for each scenario combination.
        
        This method is called at each timestep to record the minimum value
        observed across all timesteps.
        """
        factor = self.factor
        values = self._param.get_all_values()

        for scenario_index in self.model.scenarios.combinations:
            i = scenario_index.global_id
            # Apply scaling factor and update the minimum value if needed
            scaled_value = values[i] * factor
            if scaled_value < self._values[i]:
                self._values[i] = scaled_value
            # Debugging information to check the values
            # print(f"Scenario {i}: Current scaled value: {scaled_value}, Recorded min value: {self._values[i]}")

    def aggregated_value(self):
        """Return the minimum value recorded across all scenarios.
        
        This is the value that will be returned to the model as the final
        aggregated minimum value.
        """
        min_value = self._values.min()
        # print(f"Aggregated minimum value: {min_value}")
        return min_value

MinParameterRecorder.register()

class MaxParameterRecorder(BaseConstantParameterRecorder):
    """Record the maximum value of a `Parameter` during a simulation.
    
    This recorder tracks the maximum value returned by a `Parameter`
    over the course of a model's simulation. A factor can be provided to
    apply a linear scaling to the values before recording.
    
    Parameters
    ----------
    model : `pywr.core.Model`
        The model instance.
    param : `pywr.parameters.Parameter`
        The parameter whose maximum value will be recorded.
    name : str, optional
        The name of the recorder.
    factor : float, default=1.0
        A scaling factor for the values of `param`.
    """

    def __init__(self, *args, **kwargs):
        self.factor = kwargs.pop('factor', 1.0)
        super(MaxParameterRecorder, self).__init__(*args, **kwargs)

    def reset(self):
        """Initialize _values to negative infinity before starting the simulation.
        
        This ensures that the maximum value can be accurately recorded
        by starting from a low initial value for comparison.
        """
        self._values = np.full(len(self.model.scenarios.combinations), -np.inf)  # Initialize with negative infinity

    def after(self):
        """Update the recorded maximum value for each scenario combination.
        
        This method is called at each timestep to record the maximum value
        observed across all timesteps.
        """
        factor = self.factor
        values = self._param.get_all_values()

        for scenario_index in self.model.scenarios.combinations:
            i = scenario_index.global_id
            # Apply scaling factor and update the maximum value if needed
            scaled_value = values[i] * factor
            if scaled_value > self._values[i]:
                self._values[i] = scaled_value
            # Debugging information to check the values
            # print(f"Scenario {i}: Current scaled value: {scaled_value}, Recorded max value: {self._values[i]}")

    def aggregated_value(self):
        """Return the maximum value recorded across all scenarios.
        
        This is the value that will be returned to the model as the final
        aggregated maximum value.
        """
        max_value = self._values.max()
        # print(f"Aggregated maximum value: {max_value}")
        return max_value

MaxParameterRecorder.register()

# Función para extraer datos de los recorders y unirlos con demand_ts
def extract_and_display_recorders(model, demand_ts):
    recorder_df = model.to_dataframe()
    if isinstance(recorder_df.index, pd.PeriodIndex):
        recorder_df.index = recorder_df.index.to_timestamp()
    demand_ts['Date'] = demand_ts['Date'].dt.to_period('M').dt.to_timestamp()
    demand_ts.set_index('Date', inplace=True)
    demand_ts_dates = pd.DataFrame(index=demand_ts.index)
    recorder_df_with_date = pd.concat([demand_ts_dates, recorder_df], axis=1, join='inner')
    return recorder_df_with_date

def process_directory(directorio):
    dir_path = os.path.join(json_base_path, directorio)

    if not os.path.exists(dir_path):
        return f"{directorio} skipped (missing)"

    # ---- Ensure input data exists
    data_subdir = os.path.join(dir_path, "2. Data Inputs")
    os.makedirs(data_subdir, exist_ok=True)

    for csv_file in csv_files:
        src = os.path.join(data_inputs_path, csv_file)
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
                results_path,
                f"{directorio}_{json_file.replace('.json', '.csv')}"
            )

            df.to_csv(out_csv, index=False)

        except Exception as e:
            print(f"✗ {directorio}/{json_file} failed: {e}")
            continue


    return f"{directorio} done"


if __name__ == "__main__":

    max_workers = min(os.cpu_count(), 32)  # keep conservative for Pywr

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
