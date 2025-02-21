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

2. Then you can run `OptModel_RDM_server_ver.py` with the working dir as an argument.

Example:
```sh
python3 OptModel_RDM_server_ver.py copias_generadas_1
```
## How to Run Simulation 2

1. Run `json_for_RDM_generator_vFF_fast_ver.py` to generate the `json_RDM`.
2. Run `RDMv2.0_server_ver.py` with the batch yaml file name as an argument.

Example:
```sh
python3 RDMv2.0_server_ver.py batch_1.yaml
```