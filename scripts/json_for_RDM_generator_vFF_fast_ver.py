# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:22:24 2024

@author: guill
"""

import os
import json
import pandas as pd
import re
import copy
from utils.utils import Utils

# --- Precompile Regular Expressions ---
col_clean_re = re.compile(r"^\('|',? 0\)$|'$")
activation_pattern = re.compile(r'^par_activacion_p[0-9]+$', re.IGNORECASE)
project_pattern = re.compile(r'P(\d+)')

# --- Define Placeholders ---
reemplazos_demanda = [
    "ANC02_CLIMIN_ECOACE", "ANC02_CLIMIN_ECODIN", "ANC02_CLIMIN_ECOHIS",
    "ANC02_CLIMRCP45_ECOACE", "ANC02_CLIMRCP45_ECODIN", "ANC02_CLIMRCP45_ECOHIS",
    "ANC02_CLIMRCP85_ECOACE", "ANC02_CLIMRCP85_ECODIN", "ANC02_CLIMRCP85_ECOHIS",
    "ANC22_CLIMIN_ECOACE", "ANC22_CLIMIN_ECODIN", "ANC22_CLIMIN_ECOHIS",
    "ANC22_CLIMRCP45_ECOACE", "ANC22_CLIMRCP45_ECODIN", "ANC22_CLIMRCP45_ECOHIS",
    "ANC22_CLIMRCP85_ECOACE", "ANC22_CLIMRCP85_ECODIN", "ANC22_CLIMRCP85_ECOHIS",
    "ANCTEND_CLIMIN_ECOACE", "ANCTEND_CLIMIN_ECODIN", "ANCTEND_CLIMIN_ECOHIS",
    "ANCTEND_CLIMRCP45_ECOACE", "ANCTEND_CLIMRCP45_ECODIN", "ANCTEND_CLIMRCP45_ECOHIS",
    "ANCTEND_CLIMRCP85_ECOACE", "ANCTEND_CLIMRCP85_ECODIN", "ANCTEND_CLIMRCP85_ECOHIS"
]

reemplazos_oferta = [
    "1A, 0.90", "1A, 0.95", "1A, 0.97", "2A, 0.90", "2A, 0.95", "2A, 0.97",
    "3A, 0.90", "3A, 0.95", "3A, 0.97", "3B, 0.90", "3B, 0.95", "3B, 0.97",
    "4A, 0.90", "4A, 0.95", "4A, 0.97", "4B, 0.90", "4B, 0.95", "4B, 0.97",
    "5A, 0.90", "5A, 0.95", "5A, 0.97", "5B, 0.90", "5B, 0.95", "5B, 0.97",
    "6A, 0.90", "6A, 0.95", "6A, 0.97", "6B, 0.90", "6B, 0.95", "6B, 0.97",
    "6C, 0.90", "6C, 0.95", "6C, 0.97", "7A, 0.90", "7A, 0.95", "7A, 0.97"
]

# --- Define Paths ---
SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(SCRIPTS_DIR)
CONFIG_DIR = os.path.join(PARENT_DIR, 'config')
JSON_DIR = os.path.join(CONFIG_DIR, 'json')
YAML_DIR = os.path.join(CONFIG_DIR, 'yaml')
MODEL_RESULTS_DIR = os.path.join(PARENT_DIR, 'model_results')
JSON_RDM_DIR = os.path.join(PARENT_DIR, 'json_RDM')
os.makedirs(JSON_RDM_DIR, exist_ok=True)

# --- Load Config ---
utils = Utils()
archivo_base = utils.read_yaml_file(os.path.join(YAML_DIR, 'copias_generadas_script_config.yaml'))['archivo_base']

# --- Recursive Replacement Function ---
def recursive_replace(obj, replacements):
    """
    Recursively traverse the JSON-like object and replace occurrences of keys in the
    'replacements' dictionary with their corresponding values.
    """
    if isinstance(obj, dict):
        return {k: recursive_replace(v, replacements) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_replace(item, replacements) for item in obj]
    elif isinstance(obj, str):
        for old, new in replacements.items():
            obj = obj.replace(old, new)
        return obj
    else:
        return obj

# --- Initialize Output Folder Counter ---
output_folder_counter = 1

# --- Process Each Folder and CSV File ---
for folder_num in range(1, 28):
    csv_folder_path = os.path.join(MODEL_RESULTS_DIR, f"copias_generadas_{folder_num}")
    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder_path, csv_file)
        csv_data = pd.read_csv(csv_path)

        # Clean column names using the precompiled regex
        csv_data.columns = [col_clean_re.sub("", col) for col in csv_data.columns]

        # Get activation columns using the precompiled pattern
        activacion_columns = [col for col in csv_data.columns if activation_pattern.match(col)]

        # Load the base JSON once for this CSV file
        base_json_path = os.path.join(JSON_DIR, archivo_base)
        with open(base_json_path, "r") as f:
            base_json = json.load(f)

        # Precompute activation values for each activation column
        activation_values_dict = {}
        for col in activacion_columns:
            match = project_pattern.search(col)
            if match:
                project_index = match.group(1)
                param_name = f"activacion_P{project_index}"
                if param_name in base_json['parameters']:
                    windows = base_json['parameters'][param_name]['windows']
                    adjusted_windows = windows[:-1] + [windows[-1] - 1]
                    activation_values = csv_data[col].iloc[adjusted_windows].tolist()
                    activation_values_dict[param_name] = activation_values

        # Create output folder for this CSV file
        output_folder_name = f"po{output_folder_counter}"
        output_folder_path = os.path.join(JSON_RDM_DIR, output_folder_name)
        os.makedirs(output_folder_path, exist_ok=True)
        output_folder_counter += 1

        # Generate modified JSON files for each combination of demanda and oferta
        for demanda in reemplazos_demanda:
            for oferta in reemplazos_oferta:
                # Clone the base JSON using deepcopy
                modified_json = copy.deepcopy(base_json)

                # Update activation parameters with precomputed values
                for param_name, act_values in activation_values_dict.items():
                    if param_name in modified_json['parameters']:
                        modified_json['parameters'][param_name]['values'] = act_values

                # Replace placeholder texts using the recursive replacement function
                replacements = {
                    "ANC02_CLIMIN_ECOACE": demanda,
                    "1A, 0.90": oferta
                }
                modified_json = recursive_replace(modified_json, replacements)

                # Construct the output filename and save the modified JSON
                oferta_modificada = oferta.replace(", ", "_")
                output_filename = f"{output_folder_name}_{demanda}_{oferta_modificada}.json"
                output_path = os.path.join(output_folder_path, output_filename)

                with open(output_path, "w") as f:
                    json.dump(modified_json, f, indent=4)

                print(f"Archivo generado: {output_path}")
