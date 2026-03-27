RDM Optimization + Stress-Test Pipeline
======================================

This repo runs a two-stage pipeline:
1) Optimize portfolios with Pywr/Platypus.
2) Stress-test the optimized portfolios with RDM simulations.

The run is keyed by a single `RUN_ID`. Keep it consistent across steps so
the JSON generator and RDM runner read the correct optimization outputs.

Folder Map (what matters)
-------------------------
- `3. OptModel for Production/`
  - `OptModel_RDMv5.1_prod_v2.py`: main optimization script.
  - `1. OptModel_JSONs/`: input JSONs for optimization (per scenario).
  - `2. Data Inputs/`: input CSVs (supply/demand/anc/projects).
  - `output/opt_model_results_<RUN_ID>/`: optimization outputs.
- `5.2 Full RDM/`
  - `json_file_generator_RDM_parallel_v.py`: builds RDM JSONs from opt results.
  - `RDMv5.1_parallel.py`: runs the RDM simulations in parallel.
  - `config.yaml`: holds the `run_id` used by the JSON generator.
  - `basefile.json`: template JSON for the RDM runs.
  - `2. Data Inputs/`: input CSVs used by the RDM runs.
  - `output_json/json_RDM_<RUN_ID>/`: generated RDM JSONs.
  - `RDM_results/RDM_results_<RUN_ID>/`: RDM output CSVs.

Quick Start (new users)
-----------------------
1) Create the environment (Conda).
   - `conda env create -f "3. OptModel for Production/environment.yml"`
   - `conda activate pywr_opt`

2) Update inputs (both folders).
   - `3. OptModel for Production/2. Data Inputs/`
   - `5.2 Full RDM/2. Data Inputs/`
   - Use the `*_corrected.csv` files; run `date_adjustment.py` if needed.

3) (Optional) Regenerate optimization JSONs.
   - If you change `basefile.json` or scenarios:
     - `cd "3. OptModel for Production/1. OptModel_JSONs"`
     - `python json_file_generator_opt.py`

4) Run the optimization.
   - Choose a run id (example): `20260306-184644`
   - Set it so outputs are repeatable:
     - `setx OPT_MODEL_RUN_ID 20260306-184644` (new shell)
     - or `powershell`: `$env:OPT_MODEL_RUN_ID="20260306-184644"`
   - Run:
     - `cd "3. OptModel for Production"`
     - `python OptModel_RDMv5.1_prod_v2.py`
   - Output folder:
     - `3. OptModel for Production/output/opt_model_results_<RUN_ID>/`

5) Generate RDM JSONs from the optimization outputs.
   - Set the same run id in `5.2 Full RDM/config.yaml`:
     - `run_id: 20260306-184644`
   - Run:
     - `cd "5.2 Full RDM"`
     - `python json_file_generator_RDM_parallel_v.py`
   - Output folder:
     - `5.2 Full RDM/output_json/json_RDM_<RUN_ID>/po_*/`

6) Run the RDM simulations.
   - Set the same run id in `5.2 Full RDM/RDMv5.1_parallel.py`:
     - `EXECUTION_RUN_ID = "20260306-184644"`
   - Run:
     - `cd "5.2 Full RDM"`
     - `python RDMv5.1_parallel.py`
   - Output folder:
     - `5.2 Full RDM/RDM_results/RDM_results_<RUN_ID>/`

Notes and Tips
--------------
- Parallelism:
  - `OptModel_RDMv5.1_prod_v2.py`: set `PARALLEL_WORKERS`.
  - `json_file_generator_RDM_parallel_v.py`: uses up to 27 processes.
  - `RDMv5.1_parallel.py`: uses up to 48 processes (adjust as needed).
- If you see missing JSONs in RDM, check that:
  - `config.yaml` run_id matches the optimization `RUN_ID`.
  - `RDMv5.1_parallel.py` `EXECUTION_RUN_ID` matches the same run.
- Large runs are CPU- and RAM-heavy. Start small if you are new.
