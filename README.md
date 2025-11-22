# AI_RES
AI for Renewable Energy project repository. This project builds the model to predict the carbon dioxide content and level in the atmosphere on an hourly basis.  


Repo Structure
configs/
xgboost.yaml
src/
__init__.py
data/
(placeholders)
evaluation/
time_cv.py
models/
__init__.py
train_xgboost.py
train_lightgbm.py
nn/
lstm.py
notebooks/
03_Modeling_baselines.ipynb (skeleton + runnable cells)
04_Modeling_advanced.ipynb (skeleton + experiments)
results/
metrics/
baseline_vs_models.csv
models/
(serialized models saved here)
logs/
experiments.csv
