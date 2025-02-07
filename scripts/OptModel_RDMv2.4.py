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

# Set the seed for reproducibility
SEED = 12345

# Definir los directorios de trabajo divididos en bloques
directorios_trabajo = [
    "./copias_generadas_1",
    "./copias_generadas_2",
    "./copias_generadas_3",
    "./copias_generadas_4",
    "./copias_generadas_5",
    "./copias_generadas_6",
    "./copias_generadas_7",
    "./copias_generadas_8",
    "./copias_generadas_9",
    "./copias_generadas_10",
    "./copias_generadas_11",
    "./copias_generadas_12",
    "./copias_generadas_13",
    "./copias_generadas_14",
    "./copias_generadas_15",
    "./copias_generadas_16",
    "./copias_generadas_17",
    "./copias_generadas_18",
    "./copias_generadas_19",
    "./copias_generadas_20",
    "./copias_generadas_21",
    "./copias_generadas_22",
    "./copias_generadas_23",
    "./copias_generadas_24",
    "./copias_generadas_25",
    "./copias_generadas_26",
    "./copias_generadas_27"#,
    # "./copias_generadas_pd",
    # "pruebas_2.0",
    # "pruebas_2.0_v2",
    # "pruebas_2.0_v3",
    # "pruebas_2.0_v4",
    # "pruebas_2.2_v1",
    # "prueba_2.3",
    # "prueba_2.4"
]

# logger = logging.getLogger(__name__)

# # Configurar el logger para imprimir en consola y escribir en un archivo .txt
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(levelname)s - %(message)s', 
#                     handlers=[
#                         # logging.FileHandler("output_terminal_v18.txt"),
#                         logging.StreamHandler()
#                     ])

# # Redirige los prints normales a logging
# print = logging.info

# Ruta del directorio de resultados
result_directory = "./model_results_pruebas_2.4_v3-4"

# Crear el directorio de resultados si no existe
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

# Cargar el archivo CSV, especificando que el delimitador es ';'
supply_ts = pd.read_csv("supply.csv", delimiter=";")
demand_ts = pd.read_csv("demand.csv", delimiter=";")
anc_ts = pd.read_csv("anc.csv", delimiter=";")
projects = pd.read_csv("projects_final_v4.csv", delimiter=";")

# Revisar cómo se interpretan las fechas antes de modificarlas
# print(supply_ts.head())  # Asegúrate de que la columna 'Date' esté en el formato correcto.
# print(demand_ts.head())  # Asegúrate de que la columna 'Date' esté en el formato correcto.

# Asegurarse de que la columna 'Date' esté en formato de fecha, detectando si el día está primero
supply_ts['Date'] = pd.to_datetime(supply_ts['Date'], format='%d/%m/%Y', dayfirst=True)
demand_ts['Date'] = pd.to_datetime(demand_ts['Date'], format='%d/%m/%Y', dayfirst=True)
anc_ts['Date'] = pd.to_datetime(anc_ts['Date'], format='%d/%m/%Y', dayfirst=True)

# Guardar de nuevo el archivo CSV con el formato correcto
supply_ts.to_csv("supply_corrected.csv", index=False)
demand_ts.to_csv("demand_corrected.csv", index=False)
anc_ts.to_csv("anc_corrected.csv", index=False)

class custom_HUX(Variator):

    def __init__(self, probability=1.0):
        super().__init__(2)  # Necesitas dos padres para esta operación
        self.probability = probability

    def evolve(self, parents):
        result1 = copy.deepcopy(parents[0])
        result2 = copy.deepcopy(parents[1])
        problem = result1.problem

        # Depuración para ver el número de variables y tipos
        # print(f"Total nvars: {problem.nvars}, Length of types: {len(problem.types)}")

        if random.uniform(0.0, 1.0) <= self.probability:
            for i in range(len(problem.types)):  # Limitar el bucle con len(problem.types)
                # print(f"Variable index i: {i}, Total variables: {len(problem.types)}")

                if isinstance(problem.types[i], Binary):
                    for j in range(problem.types[i].nbits):
                        # Depuración para imprimir el valor de las variables
                        # print(f"Comparing bit {j} of variable {i}: Parent1={result1.variables[i][j]}, Parent2={result2.variables[i][j]}")
                        
                        if result1.variables[i][j] != result2.variables[i][j]:
                            if bool(random.getrandbits(1)):
                                result1.variables[i][j] = not result1.variables[i][j]
                                result2.variables[i][j] = not result2.variables[i][j]
                                # print(f"Swapped bit {j} in variable {i}: Result1={result1.variables[i][j]}, Result2={result2.variables[i][j]}")
                                result1.evaluated = False
                                result2.evaluated = False

        return [result1, result2]

# Sobreescribir la función run_job
def custom_run_job(job):
    # Imprimir las variables y objetivos antes de ejecutar el trabajo
    # print(f"Running job for solution with variables: {job.solution.variables}")
    # print(f"Objectives before job run: {job.solution.objectives}")
    
    # Ejecutar el trabajo (evaluación de la solución)
    job.run()
    
    # Imprimir las variables y objetivos después de ejecutar el trabajo
    # print(f"Completed job with objectives: {job.solution.objectives}")
    
    return job

# Sobreescribir la función run_job en Platypus con la versión personalizada
platypus.evaluator.run_job = custom_run_job

# Sobrescribimos el método __call__ dentro de la clase Problem
def custom_call(self, solution):
    # Acceder al problema asociado a la solución
    problem = solution.problem

    # Asegúrate de que estamos accediendo a las variables de la solución
    variables = solution.variables
    # print(f"solution.variables before decoding: {variables}, length: {len(variables)}, type: {type(variables)}")
    # print(f"Expected number of variables (problem.nvars): {problem.nvars}")

    all_variables = []  # Nueva lista para almacenar todas las variables expandidas

    # # Imprimir el tamaño de las variables y types
    # print(f"Number of problem types: {len(problem.types)}")
    # print(f"Number of solution variables: {len(variables)}")

    # Iteramos sobre las variables del problema
    for i in range(len(variables)):
        if isinstance(problem.types[i], platypus.Binary):
            # Decodificar variables binarias
            binary_decoded = list(problem.types[i].decode(variables[i]))  # Asegúrate de que sea una lista
            # print(f"Decoded Binary variable {i}: {binary_decoded}, length: {len(binary_decoded)}, type: {type(binary_decoded)}")

            # Convertir los valores True/False a 1/0
            binary_decoded_numeric = [1 if val else 0 for val in binary_decoded]
            # print(f"Binary variable {i} after conversion to numeric: {binary_decoded_numeric}")

            # Añadir las variables binarias decodificadas y convertidas a la lista
            all_variables.extend(binary_decoded_numeric)
        else:
            # Manejo para variables no binarias
            # print(f"Handling non-Binary variable {i}")
            all_variables.append(variables[i])

    # Verificamos que el número total de variables coincida con lo esperado
    assert len(all_variables) == problem.nvars, f"Expected {problem.nvars} variables but got {len(all_variables)}"
    # print(f"All expanded variables: {all_variables}")

    # **Evaluar la solución utilizando las variables decodificadas**
    # print("Calling self.evaluate(solution) to evaluate objectives...")
    self.evaluate(solution)  # Utiliza la evaluación completa de la solución

    # Verificar si la evaluación estableció los objetivos
    # print(f"After function call - Solution Objectives: {solution.objectives}")
    
    # Verificar los tipos y las variables antes del encoding
    for i in range(min(len(variables), self.nvars)):  # Asegurarse de no salir del rango
        # print(f"Encoding variable {i}: {variables[i]}, Type: {self.types[i]}")
        try:
            # Intentar codificar la variable específica
            encoded_var = self.types[i].encode(solution.variables[i])
            # print(f"Encoded variable {i}: {encoded_var}")
        
        # Almacenar el resultado en solution.variables
            solution.variables[i] = encoded_var

        except IndexError as e:
            print(f"Error encoding variable {i}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during encoding for variable {i}: {e}")
            raise
            
    # Asegurarse de que todo se ha codificado correctamente
    # print(f"solution.variables after encoding: {solution.variables}")

    # Evaluar la violación de restricciones
    solution.constraint_violation = sum([abs(f(x)) for (f, x) in zip(solution.problem.constraints, solution.constraints)])
    solution.feasible = solution.constraint_violation == 0.0
    solution.evaluated = True

    # Imprimir los objetivos después de la evaluación
    # print(f"Final Solution Objectives: {solution.objectives}")

# Sobreescribimos el método __call__ dentro de la clase Problem
platypus.Problem.__call__ = custom_call

# Lista para almacenar los valores de la función objetivo en cada iteración
objective_values = []

# Sobrescribir el método evaluate_all de la clase Algorithm
def custom_evaluate_all(self, solutions):
    # print("Entrando en custom_evaluate_all...")  # Verificación de que la función se está ejecutando

    # Verifica si el evaluator está asignado correctamente
    if self.evaluator is None:
        # print("No hay un evaluator asignado, usando el default evaluator.")
        self.evaluator = PlatypusConfig.default_evaluator

    # Imprimir el tipo de evaluator asignado
    # print(f"Using evaluator: {type(self.evaluator)}")

    unevaluated = [s for s in solutions if not s.evaluated]

    # **Nuevo print**: Imprimir el estado de las soluciones antes de la evaluación
    # for i, solution in enumerate(unevaluated):
    #     print(f"Before evaluation - Solution {i}: Variables: {solution.variables}, Objectives: {solution.objectives}")

    if not unevaluated:
        print("No hay soluciones no evaluadas.")
        return

    jobs = [_EvaluateJob(s) for s in unevaluated]
    
    try:
        # Ejecutar el evaluador y capturar excepciones potenciales
        results = self.evaluator.evaluate_all(jobs, log_frequency=50)
    except Exception as e:
        print(f"Error during evaluate_all execution: {str(e)}")
        return  # Terminar la ejecución si ocurre un error en evaluate_all

    # **Nuevo print**: Verificar que los resultados contengan las soluciones y objetivos correctos
    # for i, result in enumerate(results):
    #     print(f"Evaluated Result {i}: Objectives: {result.solution.objectives}, Constraints: {result.solution.constraints}")
    
    # Imprimir el estado antes de actualizar
    # for i, solution in enumerate(unevaluated):
    #     print(f"Before updating - Solution {i}: Variables: {solution.variables}, Objectives: {solution.objectives}")

    for i, result in enumerate(results):
        try:
            # Debug: Imprimir las variables originales antes de la conversión
            # print(f"Solution {i} before conversion: {result.solution.variables}")

            # Convertir True/False dentro de cada lista a 1/0
            unevaluated[i].variables[:] = [[int(val) if isinstance(val, bool) else val for val in sublist] for sublist in result.solution.variables]

            # Debug: Imprimir las variables después de la conversión
            # print(f"Solution {i} after conversion to binary: {unevaluated[i].variables}")
            
            # print(f"Objectives in result.solution for solution {i}: {result.solution.objectives}")

            # Actualizar los objetivos con los resultados obtenidos
            unevaluated[i].objectives[:] = result.solution.objectives[:]
            
            # print(f"Updated objectives for solution {i}: {unevaluated[i].objectives}")

            unevaluated[i].constraints[:] = result.solution.constraints[:]

            # Manejar el atributo 'feasible' solo si existe
            if hasattr(result.solution, 'feasible'):
                unevaluated[i].feasible = result.solution.feasible
            else:
                print(f"Warning: Solution {i} has no 'feasible' attribute.")

            unevaluated[i].evaluated = result.solution.evaluated

            # Revisar y reemplazar objetivos None con un valor predeterminado
            for solution in solutions:
                if any(obj is None for obj in solution.objectives):
                    print(f"Asignando valor predeterminado a los objetivos para la solución con None: {solution}")
                    solution.objectives = [float('inf')] * len(solution.objectives)  # Usa un valor grande si es minimización

            # Guardar el valor de la función objetivo actual
            objective_values.append(unevaluated[i].objectives[0])  # Asume que tienes un solo objetivo

            # Verificación adicional
            # Verificar si las variables binarias fueron correctamente convertidas a 1/0
            for sublist in unevaluated[i].variables:
                for var in sublist:
                    if isinstance(var, bool):
                        print(f"Warning: Variable {var} is still boolean after conversion in Solution {i}")
                    if var not in [0, 1]:
                        print(f"Error: Variable {var} is not binary (0 or 1) in Solution {i}")

        except Exception as e:
            print(f"An error occurred while processing solution {i}: {str(e)}")
            
        # **Nuevo código**: Verificación de valores objetivo
        for obj in unevaluated[i].objectives:
            if obj is None:
                print(f"Error: Found None value in objectives for solution {i}.")
                raise ValueError(f"None objective found in solution {i}: {unevaluated[i].objectives}")

    # # Imprimir el estado después de actualizar
    # for i, solution in enumerate(unevaluated):
    #     print(f"After evaluation - Solution {i}: Variables: {solution.variables}, Objectives: {solution.objectives}, Constraints: {solution.constraints}")

    self.nfe += len(solutions)

# Sobrescribir el método dentro de la clase Algorithm
# print(f"Sobrescribiendo el método evaluate_all de Algorithm con custom_evaluate_all.")  # Verificación

Algorithm.evaluate_all = custom_evaluate_all

# Definir el nuevo método evaluate
def custom_evaluate(self):
    """Sobrescribir el método evaluate para agregar prints."""
    # print(f"Evaluating solution: Variables: {self.variables}, Objectives before: {self.objectives}")
    
    # Llamar al problema para realizar la evaluación
    self.problem(self)

    # Verificar si los objetivos se han asignado correctamente
    # print(f"After evaluation - Objectives: {self.objectives}")

# Sobrescribir el método evaluate en la clase Solution
Solution.evaluate = custom_evaluate

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

# Define the enforce_all_sequential_activation function
def enforce_all_sequential_activation(all_parameters):
    """Ensure sequential activation logic is applied to all TimeWindowBinaryParameter instances."""
    for param in all_parameters.values():
        if isinstance(param, TimeWindowBinaryParameter):
            param.enforce_sequential_activation(all_parameters)

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

# Custom wrapper subclass to handle the array transformation
class CustomPlatypusWrapper(PlatypusWrapper):
    def evaluate(self, solution):
        # logger.info("Evaluating solution ...")

        # Iterar sobre los proyectos (filas)
        for ivar, var in enumerate(self.model_variables):
            # Extraer la fila correspondiente al proyecto actual
            project_variables = np.array(solution[ivar])  # Seleccionar la fila 'ivar'

            # Verificar si estamos extrayendo la fila correcta
            # print(f"Project {ivar+1}: Extracted variables (single row): {project_variables}")

            # Convertir booleanos a binarios (1/0)
            binary_decoded = [1 if val else 0 for val in project_variables]

            # Verificar que el tamaño sea correcto
            assert len(binary_decoded) == var.double_size, f"Expected {var.double_size} variables, got {len(binary_decoded)}"
            var.set_double_variables(np.array(binary_decoded, dtype=np.float64))

            # Verificar si los valores son binarios (0 o 1)
            if isinstance(var, TimeWindowBinaryParameter):
                if not np.all(np.isin(var.values, [0, 1])):
                    print(f"Warning: Binary parameter {var} has invalid values: {var.values}")

        # Ejecutar el modelo Pywr y guardar estadísticas de la ejecución
        self.run_stats = self.model.run()
        print(f"Run stats: {self.run_stats}")

        # # Verificar la salida final de las variables
        # print("Evaluando con los siguientes parámetros:")
        # for ivar, var in enumerate(self.model_variables):
        #     print(f"Project {ivar+1}: {var.get_double_variables()}")

        # Evaluar los objetivos y restricciones
        objectives = []
        for r in self.model_objectives:
            # print(f"Evaluating objective {r.name} ...")
            sign = 1.0 if r.is_objective == "minimise" else -1.0
            value = r.aggregated_value()
            # print(f"Aggregated value for {r.name}: {value}")

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

        # Verificar restricciones inválidas
        if any(constraint is None for constraint in constraints):
            print(f"Error: Found a None value in constraints: {constraints}")
            raise ValueError(f"Invalid constraint values for solution {solution}")

        # Guardar los resultados
        # logger.info(f"Evaluation completed in {self.run_stats.time_taken:.2f} seconds "
                    # f"({self.run_stats.speed:.2f} ts/s).")

        print(f"Setting objectives: {objectives}")

        # Devolver los objetivos y restricciones
        if len(constraints) > 0:
            return objectives, constraints
        else:
            return objectives

# Function to extract recorder data, display, and store as variable
def extract_and_display_recorders(model, demand_ts, algorithm_name):
    """Extract recorder data and store as a variable in Spyder."""

    # Convert the recorder data to a DataFrame
    recorder_df = model.to_dataframe()

    # Convert PeriodIndex to DatetimeIndex
    if isinstance(recorder_df.index, pd.PeriodIndex):
        recorder_df.index = recorder_df.index.to_timestamp()  # Convert PeriodIndex to DatetimeIndex

    # Ensure that 'Date' is set as the index for demand_ts
    if 'Date' not in demand_ts.columns:
        raise KeyError("'Date' column not found in demand_ts DataFrame.")
    
    # Format the 'Date' in demand_ts to match Pywr (year-month only)
    demand_ts['Date'] = demand_ts['Date'].dt.to_period('M').dt.to_timestamp()
    demand_ts.set_index('Date', inplace=True)  # Set 'Date' as index if it's not already
    
    # Extract only the 'Date' index from demand_ts for joining purposes
    demand_ts_dates = pd.DataFrame(index=demand_ts.index)  # Create a DataFrame with just the dates as index

    # Merge the recorder data with the demand_ts DataFrame based on the index
    recorder_df_with_date = pd.concat([demand_ts_dates, recorder_df], axis=1, join='inner')  # Use 'inner' join to avoid NaNs

    # Store the DataFrame in a variable corresponding to the algorithm name
    globals()[f'recorder_df_{algorithm_name}'] = recorder_df_with_date

    # # Plot the recorder data
    # recorder_df_with_date.plot(subplots=True)
    # plt.title(f"Recorder Data for {algorithm_name}")
    # plt.show()

    print(f"Recorder Data for {algorithm_name}:")
    # print(recorder_df_with_date.head())  # Display first few rows of the DataFrame

    return recorder_df_with_date

# Función para evaluar el modelo con la mejor solución obtenida del algoritmo
def apply_best_solution_and_evaluate(model, solution):
    """Applies the best solution and evaluates the model."""
    # print(f"Applying solution: {solution.variables}")
    for ivar, var in enumerate(model.variables):
        decision_value = np.array(solution.variables[ivar]).flatten()  # Ensure it's treated as an array
        # print(f"Setting variable for var {ivar}: {decision_value} (Length of var: {len(var.get_double_variables())})")

        if isinstance(var, TimeWindowBinaryParameter):
            if not np.all(np.isin(decision_value, [0, 1])):
                print(f"Error: Invalid binary values in variable {ivar}: {decision_value}")
                raise ValueError(f"Variable {ivar} has invalid binary values.")
                
            var.set_double_variables(np.array(decision_value))  # Pass the full array of decision values
        else:
            print(f"Warning: Variable {ivar} is not a TimeWindowBinaryParameter.")
    
    model.run()

# Función para correr el modelo con un algoritmo específico
def run_optimisation(algorithm, wrapper, algorithm_name, demand_ts, max_evaluations=6000):
    """ Ejecuta la optimización con el algoritmo seleccionado y mide el tiempo """
    # print(f"Iniciando optimización con {algorithm_name}...")

    # Reset the demand_ts index before using it, to ensure the 'Date' column exists
    if demand_ts.index.name == 'Date':
        demand_ts_reset = demand_ts.reset_index()
    else:
        demand_ts_reset = demand_ts.copy()  # Ensure we don't alter the original DataFrame
    
    # Medir el tiempo de ejecución
    start_time = time.time()
    algorithm.run(max_evaluations)
    end_time = time.time()

    # Mostrar tiempo de ejecución
    print(f"{algorithm_name} completado en {end_time - start_time:.2f} segundos.")
    
    # Obtén la mejor solución del algoritmo
    best_solution = algorithm.result[0]
    
    # Aplica la mejor solución al modelo y re-evalúa
    apply_best_solution_and_evaluate(wrapper.model, best_solution)
    
    # Extraer y mostrar los datos del recorder después de aplicar la solución
    recorder_df = extract_and_display_recorders(wrapper.model, demand_ts_reset, algorithm_name)
    
    # Mostrar la mejor solución obtenida
    print(f"Mejor solución con {algorithm_name}: {best_solution}")
    
    return best_solution, end_time - start_time, recorder_df

# Función para ejecutar el modelo y guardar los resultados
def run_model_with_json(json_path, result_csv_path, result_plot_path):
    """Run the model using a specified JSON configuration and save results to CSV and PNG."""

    global objective_values
    objective_values = []  # Reset objective value list for each execution

    # Load model data from the JSON file
    with open(json_path) as f:
        model_data = json.load(f)

    # Create the PlatypusWrapper with the model data
    wrapper = CustomPlatypusWrapper(model_data)

    # Gather all parameters into a dictionary
    all_parameters = {param.name: param for param in wrapper.model.parameters}
    TimeWindowBinaryParameter.set_all_parameters(all_parameters)

    # Verify all_parameters population
    # print("All parameters loaded into TimeWindowBinaryParameter:")
    # for name, param in TimeWindowBinaryParameter.all_parameters.items():
        # print(f"Parameter name: {name}, Type: {type(param)}, Values: {getattr(param, 'values', None)}")

    # Register dependencies for TimeWindowBinaryParameter instances
    for param in all_parameters.values():
        if isinstance(param, TimeWindowBinaryParameter):
            param.register_dependencies()

    # Enforce sequential activation logic after registering dependencies
    for param in all_parameters.values():
        if isinstance(param, TimeWindowBinaryParameter):
            param.enforce_sequential_activation()

    # Adjust the number of binary decision variables used in optimization
    wrapper.problem.types = [
        platypus.Binary(var.double_size) if isinstance(var, TimeWindowBinaryParameter) else platypus.Binary(1)
        for var in wrapper.model_variables
    ]

    # Configure the genetic algorithm (GA)
    ga = platypus.GeneticAlgorithm(
        wrapper.problem,  # Problem to optimize
        population_size=100,  # Population size
        offspring_size=100,  # Number of offspring generated per iteration
        selector=platypus.TournamentSelector(2),  # Tournament selection
        comparator=platypus.ParetoDominance(),  # Pareto dominance criterion
        variator=custom_HUX(probability=1.0),
        evaluator=evaluator
    )

    # Run optimization with GA
    best_solution_ga, time_ga, recorder_df_ga = run_optimisation(ga, wrapper, "Genetic Algorithm", demand_ts)

    # Save results to a CSV file
    recorder_df_ga.to_csv(result_csv_path, index=False)

    # Plot and save the objective function evolution
    plt.figure()
    plt.plot(objective_values)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Objective Function Evolution Over Iterations")
    plt.savefig(result_plot_path)
    plt.close()

    # Include original comments and prints
    print(f"Recorder Data for Genetic Algorithm stored in {result_csv_path}")
    print(f"Execution time for GA: {time_ga} seconds")
    print(f"Objectives for GA: {best_solution_ga.objectives}")
    print(f"Constraints for GA: {best_solution_ga.constraints}")
    print(f"Feasibility of GA Solution: {best_solution_ga.feasible}")
    print(f"Objective plot saved in {result_plot_path}")

    return best_solution_ga, time_ga, recorder_df_ga

    evaluator.close()  # Cerrar el evaluator después de la ejecución

# Configura el evaluador con 4 procesos
evaluator = MapEvaluator()

# # Intenta serializar el wrapper de Platypus con pickle
# try:
#     with open("test_pickle_wrapper.pkl", "wb") as f:
#         pickle.dump(CustomPlatypusWrapper, f)
#     print("Serialización del wrapper exitosa.")
# except Exception as e:
#     print(f"Error en la serialización del wrapper: {e}")

# Ejecutar el modelo con cada archivo JSON en el directorio
# Ejecutar el modelo en cada directorio, con pausas entre bloques

for directorio in directorios_trabajo:
    print(f"Procesando archivos en {directorio}")
    
    # Obtener lista de archivos JSON en el directorio
    json_files = [f for f in os.listdir(directorio) if f.endswith(".json")]
    
    # Procesar cada archivo JSON en el directorio
    for json_file in json_files:
        json_path = os.path.join(directorio, json_file)
        base_filename = os.path.splitext(json_file)[0]
        result_csv_path = os.path.join(result_directory, f"{base_filename}.csv")
        result_plot_path = os.path.join(result_directory, f"{base_filename}.png")
        
        # Set seed for reproducibility
        print(f"Estableciendo semilla = {SEED}")
        random.seed(SEED)           # Python's random module
        np.random.seed(SEED)        # NumPy's random module
        
        # Ejecutar el modelo y guardar resultados
        run_model_with_json(json_path, result_csv_path, result_plot_path)
        
        # Setting new seed value
        SEED = SEED + 1
    
    # # Descanso entre bloques de directorios
    # print(f"Descansando después de procesar {directorio}...")
    # time.sleep(300)  # Descanso de 5 minutos (300 segundos) entre bloques