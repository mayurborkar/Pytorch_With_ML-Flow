# Pytorch With ML-Flow

## Step1: Run The Bash File
```
bash init_setup.sh
```

## Step2: Create The Utils Dir inside src for creating some python repeated file
```
mkdir utils
```
```
touch utils/common.py
```

## Step3: To Use src as package we have run the setup.py file
```
pip install -e .
```

## Step4: Create The src/stage_01_get_data.py file 
```
touch src/stage_01_get_data.py
```

## Step5: Create The src/stage_01_get_data.py file 
```
touch src/stage_02_data_loader.py
```

## Step6: Create The src/stage_03_model_creation.py file 
```
touch src/stage_03_model_creation.py
```

## Step7: Create The src/stage_04_training_model.py file 
```
touch src/stage_04_training_model.py
```

## Step8: Create The src/stage_05_prediction.py file 
```
touch src/stage_05_prediction.py
```

## Step9: Create The main.py For Starting Point 
```
touch src/main.py
```

### Step10: Create the MLproject File To Use Mlops & Run It by Below Command
```
mlflow run . --no-conda
```
### Step11: To run any specific entry point in MLproject file
```
mlflow run . -e get_data --no-conda
```

### Step12: To run any specific entry point in MLproject file
```
mlflow run . -e get_data -P config=configs/your_config.yaml --no-conda
```