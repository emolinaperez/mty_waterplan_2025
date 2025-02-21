# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:31:23 2024

@author: guill
"""

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

json_base_path = 'C:\\Users\\guill\\OneDrive\\Documents\\Freelance\\FAMM - Plan Hídrico NL\\Working Files\\json_RDM'
results_path = 'C:\\Users\\guill\\OneDrive\\Documents\\Freelance\\FAMM - Plan Hídrico NL\\Working Files\\RDM_results'

# Definir los directorios de trabajo divididos en bloques
directorios_trabajo = [f"po{i}" for i in range(1, 973)]

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

    all_parameters = {}  # Class-level attribute to store all parameters

    def __init__(self, model, windows, dependencies=None, values=None, **kwargs):
        """
        Parameters:
        - model: Pywr model instance.
        - windows: List of integers representing decision time windows.
        - dependencies: Dictionary mapping dependent projects to prerequisite projects.
        - values: Initial binary values for each window. If None, initialized to 0.
        """
        super(TimeWindowBinaryParameter, self).__init__(model, **kwargs)
        self.windows = windows
        self.dependencies = dependencies or {}  # Dependencies between projects

        if values is not None:
            assert len(values) == len(windows), (
                f"Values length {len(values)} must match number of decision windows {len(windows)}."
            )
            self.values = np.array(values, dtype=np.float64)
        else:
            self.values = np.zeros(len(windows), dtype=np.float64)

        self.double_size = len(self.windows)  # Set the double size to the number of decision windows
        self.enforce_activation()  # Ensure that once activated, the project stays activated.

    @classmethod
    def set_all_parameters(cls, parameters):
        """Set the class-level dictionary of all parameters."""
        cls.all_parameters = parameters

    def register_dependencies(self):
        """Register dependencies for the current parameter."""
        if not self.dependencies:
            # print(f"[DEBUG] No dependencies to register for parameter {self.name}")
            return

        self._dependent_params = {}
        for dependent, prerequisite in self.dependencies.items():
            if prerequisite in TimeWindowBinaryParameter.all_parameters:
                self._dependent_params[dependent] = TimeWindowBinaryParameter.all_parameters[prerequisite]
                # print(f"[DEBUG] Registered dependency for {dependent}: prerequisite is {prerequisite}")
            else:
                print(f"[ERROR] Dependency mapping failed for {dependent} or {prerequisite}.")
                raise ValueError(f"Dependency mapping failed for {dependent} or {prerequisite}.")

    def enforce_sequential_activation(self):
        """Enforce sequential activation logic based on dependencies."""
        for dependent, prerequisite in self.dependencies.items():
            dependent_param = TimeWindowBinaryParameter.all_parameters.get(dependent)
            prerequisite_param = TimeWindowBinaryParameter.all_parameters.get(prerequisite)
    
            if dependent_param and prerequisite_param:
                # print(f"[DEBUG] Enforcing activation: {dependent} depends on {prerequisite}")
                for i in range(len(self.windows)):
                    if prerequisite_param.values[i] == 0.0:
                        self.values[i] = 0.0
    
                # Debugging output
                # print(f"[DEBUG] Sequential activation enforced for {dependent}")
                # print(f"Dependent values: {self.values}")
                # print(f"Prerequisite values: {prerequisite_param.values}")
            else:
                print(f"[ERROR] Dependency mapping failed for {dependent} or {prerequisite}.")
                raise ValueError(f"Dependency mapping failed for {dependent} or {prerequisite}.")

    def enforce_activation(self):
        """Once a project is activated (1), all subsequent time steps must also be 1."""
        for i in range(1, len(self.values)):
            if self.values[i - 1] == 1.0:
                self.values[i] = 1.0

    def value(self, timestep, scenario_index):
        """Return the binary decision for the current timestep."""
        value = 0.0
        for i, window in enumerate(self.windows):
            if timestep.index >= window:  # If the timestep is within or after the decision window
                value = self.values[i]
            else:
                break
        return value  # Return the binary decision for the current timestep

    def set_double_variables(self, values):
        """Set binary decision values for each window."""
        assert len(values) == len(self.windows), (
            f"Decision values length {len(values)} must match the number of decision windows {len(self.windows)}."
        )
        self.values = values.astype(np.float64)
        self.enforce_sequential_activation()  # Reapply sequential activation logic
        self.enforce_activation()  # Reapply activation enforcement logic

    def get_double_variables(self):
        return self.values

    def get_double_lower_bounds(self):
        return np.zeros(len(self.values))  # Lower bounds for the binary variable (0).

    def get_double_upper_bounds(self):
        return np.ones(len(self.values))  # Upper bounds for the binary variable (1).

    @classmethod
    def load(cls, model, data):
        """Custom load method to handle 'windows', 'values', and 'dependencies'."""
        windows = data.pop('windows')
        values = data.pop('values', None)
        dependencies = data.pop('dependencies', None)
        return cls(model, windows=windows, dependencies=dependencies, values=values, **data)

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

# Iterar sobre cada directorio en directorios_trabajo
for directorio in directorios_trabajo:
    dir_path = os.path.join(json_base_path, directorio)

    # Verificar si el directorio existe
    if not os.path.exists(dir_path):
        print(f"Directorio {dir_path} no existe. Saltando...")
        continue

    # Listar archivos JSON en el directorio
    json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]

    # Procesar cada archivo JSON
    for json_file in json_files:
        json_file_path = os.path.join(dir_path, json_file)

        # Cargar el modelo desde el archivo JSON
        model = Model.load(json_file_path)

        # Ejecutar el modelo
        stats = model.run()
        print(f"Modelo ejecutado para {json_file} con estadísticas: {stats}")

        # Convertir los resultados a DataFrame
        results_df = model.to_dataframe()

        # Crear el nombre del archivo CSV de salida en results_path
        csv_filename = f"{json_file.replace('.json', '.csv')}"
        csv_output_path = os.path.join(results_path, csv_filename)

        # Guardar los resultados en el archivo CSV
        results_df.to_csv(csv_output_path, index=False)
        print(f"Resultados guardados en {csv_output_path}")