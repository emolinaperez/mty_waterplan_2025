# -*- coding: utf-8 -*-
"""
Parallel JSON generation with progress bar
"""

import os
import json
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ----------------------------
# Static configuration
# ----------------------------
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
OPTIMIZATION_OUTPUT_DIR = os.path.join(PARENT_DIR, "3. OptModel for Production", "output")
JSON_OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "output_json")
BASE_JSON_FILE_PATH = os.path.join(SCRIPT_DIR, "basefile.json")
EXECUTION_RUN_ID = "20260128-122156" # Replace with the actual run ID used in optimization
OPTIMIZATION_RESULTS_FOLDER_NAME = f"opt_model_results_{EXECUTION_RUN_ID}"
OPTIMIZATION_RESULTS_DIR = os.path.join(OPTIMIZATION_OUTPUT_DIR, OPTIMIZATION_RESULTS_FOLDER_NAME)
JSON_OUTPUT_FOLDER_NAME = f"json_RDM_{EXECUTION_RUN_ID}"
JSON_OUTPUT_FINAL_DIR = os.path.join(JSON_OUTPUT_BASE_DIR, JSON_OUTPUT_FOLDER_NAME)



# ----------------------------
# Worker function
# ----------------------------
def process_folder(folder_num: int):

    csv_folder = os.path.join(OPTIMIZATION_RESULTS_DIR, f"copias_generadas_{folder_num}")
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    with open(BASE_JSON_FILE_PATH, "r") as f:
        base_json = json.load(f)

    for i, csv_file in enumerate(csv_files, start=1):
        csv_path = os.path.join(csv_folder, csv_file)
        csv_data = pd.read_csv(csv_path)

        csv_data.columns = [
            re.sub(r"^\('|',? 0\)$|'$", "", col) for col in csv_data.columns
        ]

        activacion_columns = [
            c for c in csv_data.columns
            if re.match(r"^par_activacion_p[0-9]+$", c, re.IGNORECASE)
        ]

        output_folder_name = f"po_{folder_num}_{i}"
        output_folder = os.path.join(JSON_OUTPUT_FINAL_DIR, output_folder_name)
        os.makedirs(output_folder, exist_ok=True)

        for demanda in reemplazos_demanda:
            for oferta in reemplazos_oferta:

                modified_json = json.loads(json.dumps(base_json))

                for column in activacion_columns:
                    project_index = re.search(r'P(\d+)', column).group(1)
                    param_name = f"activacion_P{project_index}"

                    if param_name in modified_json["parameters"]:
                        windows = modified_json["parameters"][param_name]["windows"]
                        adjusted_windows = windows[:-1] + [windows[-1] - 1]
                        values = csv_data[column].iloc[adjusted_windows].tolist()
                        modified_json["parameters"][param_name]["values"] = values

                modified_json_str = (
                    json.dumps(modified_json)
                    .replace("ANC02_CLIMIN_ECOACE", demanda)
                    .replace("1A, 0.90", oferta)
                )
                modified_json = json.loads(modified_json_str)

                oferta_mod = oferta.replace(", ", "_")
                filename = f"{output_folder_name}_{demanda}_{oferta_mod}.json"
                out_path = os.path.join(output_folder, filename)

                with open(out_path, "w") as f:
                    json.dump(modified_json, f, indent=4)

    return folder_num


# ----------------------------
# Main (with progress bar)
# ----------------------------
if __name__ == "__main__":

    max_workers = min(27, os.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(process_folder, folder_num)
            for folder_num in range(1, 28)
        ]

        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing folders",
            unit="folder"
        ):
            pass

    print("✅ All folders processed.")
