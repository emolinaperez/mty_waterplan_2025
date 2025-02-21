# MTY WATERPLAN 2025

## Get Started

Create a conda env with python 3.11 (You can use any name)

```sh
conda create -n wp_env python=3.11
```

Activate the env

```sh
conda activate wp_env
```

Install additional libraries

```sh
pip install -r requirements.txt
```

## How to Run Simulation 1

1. Run `json_file_generator_allscenarios.py` to generate the copias_generadas_all directories containing configuration json files.

2. Then you can run `OptModel_RDM_server_ver.py`or `OptModel_RDMv2.4.py` which creates model_results.

## How to Run Simulation 2