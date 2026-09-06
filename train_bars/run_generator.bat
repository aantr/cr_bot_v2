@echo off
set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%..\portable_generator\generator.py" --dataset-root "%SCRIPT_DIR%..\dataset" --output "%SCRIPT_DIR%..\generation" --count 200 --units-min 1 --units-max 500 --classes-yaml "%SCRIPT_DIR%bar-dataset.yaml" --generation-filter bars
