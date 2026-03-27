# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 19:53:52 2026

@author: guill

VERSIÓN 5.1 FINAL: Pre-filtro usando portafolio_max.json como única fuente de verdad
================================================================================
ESTRATEGIA:
1. Para cada futuro, simular con portafolio_max.json (respeta restricciones)
2. Si P_percentil(margen) >= threshold → ese futuro es "optimizable" → optimizar
3. Si P_percentil(margen) < threshold → ese futuro es "infactible" → usar portafolio_max

CONFIGURACIÓN:
- percentile y threshold se leen del recorder 'par_margen_de_reserva_rest' 
  en portafolio_max.json. El JSON es la ÚNICA fuente de verdad.
================================================================================
"""


import os
from pywr.core import Model
from pywr.recorders import recorder_registry, Recorder
from pywr.recorders._recorders import NodeRecorder
from pywr.optimisation.platypus import PlatypusWrapper
from pywr.parameters import Parameter, load_parameter
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
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed

try:

    from tqdm import tqdm

except Exception:

    tqdm = None

# Set the seed for reproducibility
SEED = 12345

# === RUTAS BASE ===
BASE_DIR = os.getcwd()
JSON_DIR = os.path.join(BASE_DIR, "1. OptModel_JSONs")
DATA_DIR = os.path.join(BASE_DIR, "2. Data Inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
# Use a single run id across parallel workers to avoid multiple timestamped dirs.
RUN_ID = os.environ.get("OPT_MODEL_RUN_ID")
if RUN_ID is None:
    RUN_ID = time.strftime("%Y%m%d-%H%M%S")
    os.environ["OPT_MODEL_RUN_ID"] = RUN_ID
RESULT_DIR = os.path.join(OUTPUT_DIR, f"opt_model_results_{RUN_ID}")

# === RUTA AL PORTAFOLIO MÁXIMO ===
PORTAFOLIO_MAX_PATH = os.path.join(JSON_DIR, "portafolio_max.json")

# === CONFIGURACIÓN DEL PRE-FILTRO ===
ENABLE_PREFILTRO = True


def cargar_config_prefiltro(portafolio_max_path):
    """
    Lee percentile y threshold del recorder en portafolio_max.json.
    
    El JSON es la ÚNICA fuente de verdad. Si faltan campos, lanza KeyError.
    """
    with open(portafolio_max_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    recorder = data['recorders']['par_margen_de_reserva_rest']
    percentile = recorder['percentile']
    threshold = recorder['constraint_lower_bounds']
    
    return percentile, threshold


# Cargar configuración del prefiltro desde el JSON
PREFILTRO_PERCENTILE, PREFILTRO_THRESHOLD = cargar_config_prefiltro(PORTAFOLIO_MAX_PATH)

# Create Result Dir if it does not exists
os.makedirs(name=OUTPUT_DIR, exist_ok=True)
os.makedirs(name=RESULT_DIR, exist_ok=True)

# === CARGA DE INPUTS CORREGIDOS ===
supply_ts  = pd.read_csv(os.path.join(DATA_DIR, "supply_corrected.csv"),  delimiter=",", parse_dates=["Date"])
demand_ts  = pd.read_csv(os.path.join(DATA_DIR, "demand_corrected.csv"),  delimiter=",", parse_dates=["Date"])
anc_ts     = pd.read_csv(os.path.join(DATA_DIR, "anc_corrected.csv"),     delimiter=",", parse_dates=["Date"])
projects   = pd.read_csv(os.path.join(DATA_DIR, "projects.csv"),          delimiter=",")


class custom_HUX(Variator):

    def __init__(self, probability=1.0):
        super().__init__(2)
        self.probability = probability

    def evolve(self, parents):
        result1 = copy.deepcopy(parents[0])
        result2 = copy.deepcopy(parents[1])
        problem = result1.problem

        if random.uniform(0.0, 1.0) <= self.probability:
            for i in range(len(problem.types)):
                if isinstance(problem.types[i], Binary):
                    for j in range(problem.types[i].nbits):
                        if result1.variables[i][j] != result2.variables[i][j]:
                            if bool(random.getrandbits(1)):
                                result1.variables[i][j] = not result1.variables[i][j]
                                result2.variables[i][j] = not result2.variables[i][j]
                                result1.evaluated = False
                                result2.evaluated = False

        return [result1, result2]


def custom_run_job(job):
    job.run()
    return job


platypus.evaluator.run_job = custom_run_job


def custom_call(self, solution):
    problem = solution.problem
    variables = solution.variables

    all_variables = []

    for i in range(len(variables)):
        if isinstance(problem.types[i], platypus.Binary):
            binary_decoded = list(problem.types[i].decode(variables[i]))
            binary_decoded_numeric = [1 if val else 0 for val in binary_decoded]
            all_variables.extend(binary_decoded_numeric)
        else:
            all_variables.append(variables[i])

    assert len(all_variables) == problem.nvars, f"Expected {problem.nvars} variables but got {len(all_variables)}"

    self.evaluate(solution)
    
    for i in range(min(len(variables), self.nvars)):
        try:
            encoded_var = self.types[i].encode(solution.variables[i])
            solution.variables[i] = encoded_var
        except IndexError as e:
            print(f"Error encoding variable {i}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during encoding for variable {i}: {e}")
            raise

    solution.constraint_violation = sum([abs(f(x)) for (f, x) in zip(solution.problem.constraints, solution.constraints)])
    solution.feasible = solution.constraint_violation == 0.0
    solution.evaluated = True


platypus.Problem.__call__ = custom_call

objective_values = []


def custom_evaluate_all(self, solutions):
    if self.evaluator is None:
        self.evaluator = PlatypusConfig.default_evaluator

    unevaluated = [s for s in solutions if not s.evaluated]

    if not unevaluated:
        print("No hay soluciones no evaluadas.")
        return

    jobs = [_EvaluateJob(s) for s in unevaluated]
    
    try:
        results = self.evaluator.evaluate_all(jobs, log_frequency=50)
    except Exception as e:
        print(f"Error during evaluate_all execution: {str(e)}")
        return

    for i, result in enumerate(results):
        try:
            unevaluated[i].variables[:] = [[int(val) if isinstance(val, bool) else val for val in sublist] for sublist in result.solution.variables]
            unevaluated[i].objectives[:] = result.solution.objectives[:]
            unevaluated[i].constraints[:] = result.solution.constraints[:]

            if hasattr(result.solution, 'feasible'):
                unevaluated[i].feasible = result.solution.feasible
            else:
                print(f"Warning: Solution {i} has no 'feasible' attribute.")

            unevaluated[i].evaluated = result.solution.evaluated

            for solution in solutions:
                if any(obj is None for obj in solution.objectives):
                    print(f"Asignando valor predeterminado a los objetivos para la solución con None: {solution}")
                    solution.objectives = [float('inf')] * len(solution.objectives)

            objective_values.append(unevaluated[i].objectives[0])

            for sublist in unevaluated[i].variables:
                for var in sublist:
                    if isinstance(var, bool):
                        print(f"Warning: Variable {var} is still boolean after conversion in Solution {i}")
                    if var not in [0, 1]:
                        print(f"Error: Variable {var} is not binary (0 or 1) in Solution {i}")

        except Exception as e:
            print(f"An error occurred while processing solution {i}: {str(e)}")
            
        for obj in unevaluated[i].objectives:
            if obj is None:
                print(f"Error: Found None value in objectives for solution {i}.")
                raise ValueError(f"None objective found in solution {i}: {unevaluated[i].objectives}")

    self.nfe += len(solutions)


Algorithm.evaluate_all = custom_evaluate_all


def custom_evaluate(self):
    self.problem(self)


Solution.evaluate = custom_evaluate


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


# =============================================================================
# PRE-FILTRO: FUNCIONES
# =============================================================================

def crear_modelo_para_prefiltro(json_futuro_path, portafolio_max_path):
    """
    Crea un modelo combinando los datos de oferta/demanda del futuro 
    con las activaciones del portafolio máximo.
    
    Parameters
    ----------
    json_futuro_path : str
        Path al JSON del futuro (contiene datos de oferta/demanda específicos)
    portafolio_max_path : str
        Path al JSON del portafolio máximo (tiene las activaciones correctas)
    
    Returns
    -------
    pywr.core.Model
        Modelo listo para simular
    """
    with open(portafolio_max_path, 'r', encoding='utf-8') as f:
        portafolio_max_data = json.load(f)
    
    with open(json_futuro_path, 'r', encoding='utf-8') as f:
        futuro_data = json.load(f)
    
    parametros_a_actualizar = ['demanda_req', 'anc_req', 'oferta_sostenible_base']
    
    for param_name in parametros_a_actualizar:
        if param_name in futuro_data.get('parameters', {}):
            futuro_param = futuro_data['parameters'][param_name]
            if param_name in portafolio_max_data.get('parameters', {}):
                if 'column' in futuro_param:
                    portafolio_max_data['parameters'][param_name]['column'] = futuro_param['column']
                    print(f"  [prefiltro] {param_name}: usando columna '{futuro_param['column']}'")
    
    model = Model.load(portafolio_max_data)
    
    return model


def ejecutar_prefiltro(json_path, portafolio_max_path, percentile, threshold,
                       margen_recorder_name='par_margen_de_reserva'):
    """
    Ejecuta el pre-filtro para determinar si un futuro es optimizable.
    
    Parameters
    ----------
    json_path : str
        Path al JSON del futuro a evaluar
    portafolio_max_path : str
        Path al JSON del portafolio máximo pre-construido
    percentile : float
        Percentil a evaluar (leído del JSON)
    threshold : float
        Umbral de la restricción (leído del JSON)
    margen_recorder_name : str
        Nombre del recorder que contiene los datos del margen de reserva
    
    Returns
    -------
    dict
        Resultados del pre-filtro
    """
    print(f"\n{'─'*60}")
    print(f"PRE-FILTRO: Evaluando con portafolio máximo")
    print(f"  Archivo portafolio_max: {os.path.basename(portafolio_max_path)}")
    print(f"  Configuración: P{percentile} >= {threshold}")
    print(f"{'─'*60}")
    
    start_time = time.time()
    
    try:
        model = crear_modelo_para_prefiltro(json_path, portafolio_max_path)
        run_stats = model.run()
        tiempo_simulacion = time.time() - start_time
        
        margen_data = None
        recorder_encontrado = None
        
        posibles_nombres = [
            margen_recorder_name,
            'par_margen_de_reserva',
            'margen_de_reserva',
            'reserve_margin'
        ]
        
        for nombre in posibles_nombres:
            if nombre in model.recorders:
                recorder = model.recorders[nombre]
                if hasattr(recorder, 'data'):
                    margen_data = recorder.data[:, 0]
                    recorder_encontrado = nombre
                    break
                elif hasattr(recorder, '_data'):
                    margen_data = recorder._data[:, 0]
                    recorder_encontrado = nombre
                    break
        
        if margen_data is None:
            print(f"  [Error] No se encontró recorder del margen. Recorders disponibles:")
            for rec_name in model.recorders.keys():
                print(f"    - {rec_name}")
            
            return {
                'es_optimizable': True,
                'percentil_valor': None,
                'timesteps_negativos': None,
                'pct_negativos': None,
                'margen_minimo': None,
                'margen_maximo': None,
                'margen_medio': None,
                'tiempo_simulacion': tiempo_simulacion,
                'error': 'No se encontró recorder del margen de reserva'
            }
        
        print(f"  Recorder encontrado: {recorder_encontrado}")
        
        percentil_valor = np.percentile(margen_data, percentile)
        timesteps_negativos = (margen_data < threshold).sum()
        total_timesteps = len(margen_data)
        pct_negativos = 100.0 * timesteps_negativos / total_timesteps
        
        es_optimizable = percentil_valor >= threshold
        
        resultado = {
            'es_optimizable': es_optimizable,
            'percentil_valor': float(percentil_valor),
            'timesteps_negativos': int(timesteps_negativos),
            'pct_negativos': float(pct_negativos),
            'margen_minimo': float(margen_data.min()),
            'margen_maximo': float(margen_data.max()),
            'margen_medio': float(margen_data.mean()),
            'tiempo_simulacion': tiempo_simulacion,
            'total_timesteps': total_timesteps
        }
        
        print(f"\n  Resultados del pre-filtro:")
        print(f"    P{percentile} del margen:        {percentil_valor:,.2f} m³/s")
        print(f"    Margen mínimo:            {resultado['margen_minimo']:,.2f} m³/s")
        print(f"    Margen máximo:            {resultado['margen_maximo']:,.2f} m³/s")
        print(f"    Margen medio:             {resultado['margen_medio']:,.2f} m³/s")
        print(f"    Timesteps con margen < 0: {timesteps_negativos} de {total_timesteps} ({pct_negativos:.1f}%)")
        print(f"    Tiempo de simulación:     {tiempo_simulacion:.2f} segundos")
        print(f"\n  ¿Es optimizable? {'✓ SÍ' if es_optimizable else '✗ NO'}")
        
        return resultado
        
    except Exception as e:
        print(f"  [Error] Error en pre-filtro: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'es_optimizable': True,
            'percentil_valor': None,
            'timesteps_negativos': None,
            'pct_negativos': None,
            'margen_minimo': None,
            'margen_maximo': None,
            'margen_medio': None,
            'tiempo_simulacion': time.time() - start_time,
            'error': str(e)
        }


def guardar_resultado_portafolio_max(json_path, portafolio_max_path, result_csv_path, 
                                      result_plot_path, prefiltro_resultado, demand_ts,
                                      percentile, threshold):
    """
    Guarda los resultados cuando se usa el portafolio máximo (futuro infactible).
    """
    print(f"\n{'─'*60}")
    print(f"GUARDANDO RESULTADO: Portafolio Máximo (futuro infactible)")
    print(f"{'─'*60}")
    
    model = crear_modelo_para_prefiltro(json_path, portafolio_max_path)
    model.run()
    
    recorder_df = model.to_dataframe()
    
    if isinstance(recorder_df.index, pd.PeriodIndex):
        recorder_df.index = recorder_df.index.to_timestamp()
    
    recorder_df['tipo_solucion'] = 'portafolio_max'
    recorder_df['es_factible'] = False
    recorder_df['prefiltro_percentil'] = prefiltro_resultado.get('percentil_valor', np.nan)
    
    recorder_df.to_csv(result_csv_path)
    print(f"  CSV guardado en: {result_csv_path}")
    
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, 
             f"FUTURO INFACTIBLE\n\n"
             f"P{percentile} del margen: {prefiltro_resultado.get('percentil_valor', 'N/A'):.2f} m³/s\n"
             f"(Umbral requerido: >= {threshold})\n\n"
             f"Solución: Portafolio Máximo\n"
             f"(26 proyectos activados, P5 deshabilitado)",
             ha='center', va='center', fontsize=12,
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    plt.axis('off')
    plt.title(f"Resultado: {os.path.basename(json_path)}")
    plt.savefig(result_plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Plot guardado en: {result_plot_path}")
    
    return recorder_df


# =============================================================================
# CUSTOM PLATYPUS WRAPPER
# =============================================================================

class CustomPlatypusWrapper(PlatypusWrapper):
    def evaluate(self, solution):
        for ivar, var in enumerate(self.model_variables):
            project_variables = np.array(solution[ivar])
            binary_decoded = [1 if val else 0 for val in project_variables]
            assert len(binary_decoded) == var.double_size, f"Expected {var.double_size} variables, got {len(binary_decoded)}"
            var.set_double_variables(np.array(binary_decoded, dtype=np.float64))

            if isinstance(var, TimeWindowBinaryParameter):
                if not np.all(np.isin(var.values, [0, 1])):
                    print(f"Warning: Binary parameter {var} has invalid values: {var.values}")

        self.run_stats = self.model.run()
        # print(f"Run stats: {self.run_stats}")

        objectives = []
        for r in self.model_objectives:
            sign = 1.0 if r.is_objective == "minimise" else -1.0
            value = r.aggregated_value()

            if value is None:
                print(f"Error: Objective {r.name} has None value.")
                raise ValueError(f"Objective {r.name} is None for solution {solution}")

            objectives.append(sign * value)

        constraints = []
        for c in self.model_constraints:
            x = c.aggregated_value()
            if c.is_double_bounded_constraint:
                constraints.extend([x, x])
            else:
                constraints.append(x)

        if any(constraint is None for constraint in constraints):
            print(f"Error: Found a None value in constraints: {constraints}")
            raise ValueError(f"Invalid constraint values for solution {solution}")

        # print(f"Setting objectives: {objectives}")

        if len(constraints) > 0:
            return objectives, constraints
        else:
            return objectives


# =============================================================================
# FUNCIONES DE EJECUCIÓN
# =============================================================================

def extract_and_display_recorders(model, demand_ts, algorithm_name):
    """Extract recorder data and store as a variable."""
    recorder_df = model.to_dataframe()

    if isinstance(recorder_df.index, pd.PeriodIndex):
        recorder_df.index = recorder_df.index.to_timestamp()

    if 'Date' not in demand_ts.columns:
        raise KeyError("'Date' column not found in demand_ts DataFrame.")
    
    demand_ts['Date'] = demand_ts['Date'].dt.to_period('M').dt.to_timestamp()
    demand_ts.set_index('Date', inplace=True)
    
    demand_ts_dates = pd.DataFrame(index=demand_ts.index)
    recorder_df_with_date = pd.concat([demand_ts_dates, recorder_df], axis=1, join='inner')

    globals()[f'recorder_df_{algorithm_name}'] = recorder_df_with_date

    # print(f"Recorder Data for {algorithm_name}:")
    return recorder_df_with_date


def apply_best_solution_and_evaluate(model, solution):
    """Applies the best solution and evaluates the model."""
    for ivar, var in enumerate(model.variables):
        decision_value = np.array(solution.variables[ivar]).flatten()

        if isinstance(var, TimeWindowBinaryParameter):
            if not np.all(np.isin(decision_value, [0, 1])):
                # print(f"Error: Invalid binary values in variable {ivar}: {decision_value}")
                raise ValueError(f"Variable {ivar} has invalid binary values.")
                
            var.set_double_variables(np.array(decision_value))
        else:
            # print(f"Warning: Variable {ivar} is not a TimeWindowBinaryParameter.")
            pass
    
    model.run()


def run_optimisation(algorithm, wrapper, algorithm_name, demand_ts, max_evaluations=6000):
    """Ejecuta la optimización con el algoritmo seleccionado y mide el tiempo."""
    if demand_ts.index.name == 'Date':
        demand_ts_reset = demand_ts.reset_index()
    else:
        demand_ts_reset = demand_ts.copy()
    
    start_time = time.time()
    algorithm.run(max_evaluations)
    end_time = time.time()

    # print(f"{algorithm_name} completado en {end_time - start_time:.2f} segundos.")
    
    best_solution = algorithm.result[0]
    apply_best_solution_and_evaluate(wrapper.model, best_solution)
    recorder_df = extract_and_display_recorders(wrapper.model, demand_ts_reset, algorithm_name)
    
    # print(f"Mejor solución con {algorithm_name}: {best_solution}")
    
    return best_solution, end_time - start_time, recorder_df


def run_model_with_json(json_path, result_csv_path, result_plot_path, 
                        portafolio_max_path, percentile, threshold,
                        enable_prefiltro=True):
    """
    Run the model using a specified JSON configuration and save results.
    """
    global objective_values
    objective_values = []
    
    nombre_futuro = os.path.basename(json_path)
    
    print(f"\n{'='*70}")
    print(f"PROCESANDO: {nombre_futuro}")
    print(f"{'='*70}")
    
    # =========================================================================
    # PASO 1: PRE-FILTRO
    # =========================================================================
    prefiltro_resultado = None
    
    if enable_prefiltro:
        prefiltro_resultado = ejecutar_prefiltro(
            json_path,
            portafolio_max_path,
            percentile=percentile, 
            threshold=threshold
        )
        
        if not prefiltro_resultado['es_optimizable']:
            print(f"\n{'─'*60}")
            print(f"DECISIÓN: Futuro INFACTIBLE → Usando portafolio máximo")
            print(f"{'─'*60}")
            
            if demand_ts.index.name == 'Date':
                demand_ts_reset = demand_ts.reset_index()
            else:
                demand_ts_reset = demand_ts.copy()
            
            recorder_df = guardar_resultado_portafolio_max(
                json_path, portafolio_max_path, result_csv_path, result_plot_path,
                prefiltro_resultado, demand_ts_reset, percentile, threshold
            )
            
            return {
                'tipo_solucion': 'portafolio_max',
                'tiempo_total': prefiltro_resultado.get('tiempo_simulacion', 0),
                'prefiltro': prefiltro_resultado,
                'recorder_df': recorder_df
            }
        
        print(f"\n{'─'*60}")
        print(f"DECISIÓN: Futuro OPTIMIZABLE → Ejecutando optimización")
        print(f"{'─'*60}")
    
    # =========================================================================
    # PASO 2: OPTIMIZACIÓN
    # =========================================================================
    
    with open(json_path) as f:
        model_data = json.load(f)

    wrapper = CustomPlatypusWrapper(model_data)

    all_parameters = {param.name: param for param in wrapper.model.parameters}
    TimeWindowBinaryParameter.set_all_parameters(all_parameters)

    for param in all_parameters.values():
        if isinstance(param, TimeWindowBinaryParameter):
            param.register_dependencies()

    for param in all_parameters.values():
        if isinstance(param, TimeWindowBinaryParameter):
            param.enforce_sequential_activation()

    wrapper.problem.types = [
        platypus.Binary(var.double_size) if isinstance(var, TimeWindowBinaryParameter) else platypus.Binary(1)
        for var in wrapper.model_variables
    ]

    ga = platypus.GeneticAlgorithm(
        wrapper.problem,
        population_size=100,
        offspring_size=100,
        selector=platypus.TournamentSelector(2),
        comparator=platypus.ParetoDominance(),
        variator=custom_HUX(probability=1.0),
        evaluator=evaluator
    )

    best_solution_ga, time_ga, recorder_df_ga = run_optimisation(ga, wrapper, "Genetic Algorithm", demand_ts)

    if enable_prefiltro and prefiltro_resultado:
        recorder_df_ga['tipo_solucion'] = 'optimizada'
        recorder_df_ga['es_factible'] = best_solution_ga.feasible
        recorder_df_ga['prefiltro_percentil'] = prefiltro_resultado.get('percentil_valor', np.nan)

    recorder_df_ga.to_csv(result_csv_path, index=False)

    plt.figure()
    plt.plot(objective_values)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Objective Function Evolution Over Iterations")
    plt.savefig(result_plot_path)
    plt.close()

    # print(f"Recorder Data for Genetic Algorithm stored in {result_csv_path}")
    # print(f"Execution time for GA: {time_ga} seconds")
    # print(f"Objectives for GA: {best_solution_ga.objectives}")
    # print(f"Constraints for GA: {best_solution_ga.constraints}")
    # print(f"Feasibility of GA Solution: {best_solution_ga.feasible}")
    # print(f"Objective plot saved in {result_plot_path}")

    return {
        'tipo_solucion': 'optimizada',
        'best_solution': best_solution_ga,
        'tiempo_optimizacion': time_ga,
        'prefiltro': prefiltro_resultado if enable_prefiltro else None,
        'recorder_df': recorder_df_ga
    }


# =============================================================================
# MAIN
# =============================================================================

evaluator = MapEvaluator()

# Parallel workers for multiple directorios (set to 1 to run sequentially)
PARALLEL_WORKERS = 48


def run_dir_trabajo(numero_dir_trabajo):
    """Process all JSON files in a single copias_generadas_X directory."""
    global SEED

    # === DIRECTORIOS DE TRABAJO (JSONs) ===
    directorio = os.path.join(JSON_DIR, f"copias_generadas_{numero_dir_trabajo}")

    if not os.path.isdir(directorio):
        return {'total': 0, 'optimizados': 0, 'portafolio_max': 0, 'errores': 0}

    # Obtener lista de archivos JSON en el directorio
    json_files = sorted([f for f in os.listdir(directorio) if f.endswith(".json")])

    # Define custom results path (subdirectorio por dir_trabajo)
    custom_results_path = os.path.join(RESULT_DIR, f"copias_generadas_{numero_dir_trabajo}")
    os.makedirs(custom_results_path, exist_ok=True)

    total_files = len(json_files)
    iterator = json_files
    if tqdm is not None:
        iterator = tqdm(json_files, desc=f"dir_trabajo {numero_dir_trabajo}", total=total_files)

    resumen = {'total': 0, 'optimizados': 0, 'portafolio_max': 0, 'errores': 0}

    # Procesar cada archivo JSON en el directorio
    for idx, json_file in enumerate(iterator, start=1):
        json_path = os.path.join(directorio, json_file)
        base_filename = os.path.splitext(json_file)[0]
        result_csv_path = os.path.join(custom_results_path, f"{base_filename}.csv")
        result_plot_path = os.path.join(custom_results_path, f"{base_filename}.png")

        if tqdm is None:
            print(f"[{idx}/{total_files}] {json_file}")

        # Set seed for reproducibility
        print(f"Estableciendo semilla = {SEED}")
        random.seed(SEED)
        np.random.seed(SEED)

        try:
            resultado = run_model_with_json(
                json_path,
                result_csv_path,
                result_plot_path,
                portafolio_max_path=PORTAFOLIO_MAX_PATH,
                percentile=PREFILTRO_PERCENTILE,
                threshold=PREFILTRO_THRESHOLD,
                enable_prefiltro=ENABLE_PREFILTRO
            )

            resumen['total'] += 1
            if resultado['tipo_solucion'] == 'optimizada':
                resumen['optimizados'] += 1
            else:
                resumen['portafolio_max'] += 1

        except Exception as e:
            print(f"Error procesando {json_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            resumen['errores'] += 1

        # Setting new seed value
        SEED = SEED + 1

    return resumen


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║     OPTIMIZACIÓN RDM v5.1 - PRE-FILTRO POR PERCENTIL                 ║
    ║          Configuración leída desde portafolio_max.json               ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  Estrategia:                                                          ║
    ║  1. Simular con portafolio_max.json (respeta restricciones)          ║
    ║  2. Si P_percentil(margen) >= threshold → Optimizar                  ║
    ║  3. Si P_percentil(margen) <  threshold → Usar portafolio máximo     ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    print(f"Configuración del pre-filtro (leída de {os.path.basename(PORTAFOLIO_MAX_PATH)}):")
    print(f"  - Habilitado:       {ENABLE_PREFILTRO}")
    print(f"  - Percentil:        P{PREFILTRO_PERCENTILE} (cumplir en {100-PREFILTRO_PERCENTILE}% del tiempo)")
    print(f"  - Umbral:           >= {PREFILTRO_THRESHOLD} m³/s")
    print(f"  - Workers:          {PARALLEL_WORKERS}")
    print()

    # Auto-descubrir directorios de trabajo
    dir_trabajo_list = []
    for entry in os.listdir(JSON_DIR):
        full_path = os.path.join(JSON_DIR, entry)
        if os.path.isdir(full_path) and entry.startswith("copias_generadas_"):
            suffix = entry.replace("copias_generadas_", "", 1)
            if suffix:
                dir_trabajo_list.append(suffix)

    dir_trabajo_list = sorted(dir_trabajo_list)
    if not dir_trabajo_list:
        print(f"No dir_trabajo folders found in {JSON_DIR}")
        sys.exit(1)

    print(f"Directorios encontrados: {len(dir_trabajo_list)}")
    print()

    resultados_resumen = {
        'total': 0,
        'optimizados': 0,
        'portafolio_max': 0,
        'errores': 0
    }

    if PARALLEL_WORKERS > 1 and len(dir_trabajo_list) > 1:
        with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(run_dir_trabajo, d): d for d in dir_trabajo_list}
            for future in as_completed(futures):
                d = futures[future]
                try:
                    resumen = future.result()
                    if resumen:
                        resultados_resumen['total'] += resumen.get('total', 0)
                        resultados_resumen['optimizados'] += resumen.get('optimizados', 0)
                        resultados_resumen['portafolio_max'] += resumen.get('portafolio_max', 0)
                        resultados_resumen['errores'] += resumen.get('errores', 0)
                except Exception as exc:
                    print(f"Error en proceso paralelo (dir {d}): {exc}")
                    resultados_resumen['errores'] += 1
    else:
        for d in dir_trabajo_list:
            resumen = run_dir_trabajo(d)
            if resumen:
                resultados_resumen['total'] += resumen.get('total', 0)
                resultados_resumen['optimizados'] += resumen.get('optimizados', 0)
                resultados_resumen['portafolio_max'] += resumen.get('portafolio_max', 0)
                resultados_resumen['errores'] += resumen.get('errores', 0)

    print(f"\n{'='*70}")
    print(f"RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"  Total de futuros procesados: {resultados_resumen['total']}")
    print(f"  - Optimizados:               {resultados_resumen['optimizados']}")
    print(f"  - Portafolio máximo:         {resultados_resumen['portafolio_max']}")
    print(f"  - Errores:                   {resultados_resumen['errores']}")
    print("RUN_ID: ", RUN_ID)

    evaluator.close()


if __name__ == "__main__":
    main()