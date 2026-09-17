# Lab 2 — Model Training and Experiment Tracking with MLflow

## Objective

The objective of this lab was to train a Food-11 image classifier using a pretrained ResNet-18 model, track its parameters and metrics with MLflow, compare several hyperparameter configurations, and identify the best run.

## Environment setup

The work was completed in the repository created during Lab 1:

```powershell
cd C:\Users\husse\Desktop\3eme\mlops\mlops-lab-1
```

Because the experiments were run on CPU, the following PyTorch CPU index configuration was added to `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

The required dependencies were installed with:

```powershell
uv add mlflow torch torchvision scikit-learn
```

### Question 1: What changed in `pyproject.toml` and `uv.lock`?

`pyproject.toml` was updated to declare MLflow, PyTorch, torchvision, and scikit-learn as project dependencies. It also contains the CPU-only PyTorch package index configuration.

`uv.lock` was regenerated to contain the exact resolved versions of these packages and all their transitive dependencies. While `pyproject.toml` describes the project's direct dependency requirements, `uv.lock` makes the environment reproducible by locking exact package versions.

## Local MLflow tracking server

The local tracking server was started from the repository root using:

```powershell
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

The MLflow interface was then available at:

```text
http://127.0.0.1:5000
```

### Question 2: What are the backend store and artifact root used for?

`--backend-store-uri sqlite:///mlflow.db` tells MLflow to store structured experiment and run metadata in the local SQLite database named `mlflow.db`. This metadata includes experiment names, run IDs, parameters, metric values, timestamps, statuses, and tags.

`--default-artifact-root ./mlruns` tells MLflow where to store files produced by runs. These artifacts can include trained models, model metadata, dependency specifications, plots, and other generated files.

The difference is that metadata consists mainly of structured, searchable information about experiments, while artifacts are the actual files generated or saved by a run.

## Ignoring local MLflow outputs

The local MLflow outputs were added to `.gitignore`:

```powershell
Add-Content .gitignore @("mlflow.db", "mlruns/")
git add .gitignore
git commit -m "Ignore local mlflow tracking files"
git push
```

### Question 3: Why should `mlflow.db` and `mlruns/` not be tracked by Git or DVC?

These paths contain generated experiment outputs rather than source code. They can be large and change every time a training run is performed. Tracking them in Git would make the repository unnecessarily large and create frequent irrelevant changes.

They also should not be tracked by DVC because MLflow already provides the appropriate system for tracking experiment parameters, metrics, run metadata, and model artifacts. DVC is being used for versioned datasets and data pipelines, while MLflow is being used for experiments.

## MLflow experiment configuration

The training script connects to the server and selects the experiment with:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("food11")
```

### Question 4: What happens when `set_experiment` is called with a new name?

When `mlflow.set_experiment("food11")` was called for the first time, MLflow automatically created an experiment named `food11` and made it the active experiment. On this machine, it was created with experiment ID `1` and artifact location:

```text
file:///C:/Users/husse/Desktop/3eme/mlops/mlops-lab-1/mlruns/1
```

## Training implementation

The training code was created in:

```text
src/food11/train.py
```

The script performs the following operations:

1. Reads command-line arguments for the dataset, number of epochs, learning rate, and batch size.
2. Loads the Food-11 training, validation, and evaluation splits using `ImageFolder` and `DataLoader`.
3. Loads a pretrained ResNet-18 and replaces its final layer with a layer that produces 11 class scores.
4. Trains the model using cross-entropy loss and the Adam optimizer.
5. Logs the fixed hyperparameters at the start of each MLflow run.
6. Logs training loss, validation loss, and validation accuracy after every epoch.
7. Evaluates the final model on the evaluation split and logs its test accuracy.
8. Saves the trained model as an MLflow logged model.

The first training run was launched with:

```powershell
uv run python .\src\food11\train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
```

### Question 5: What is the difference between a parameter and a metric?

An MLflow parameter is a value selected before training that remains fixed throughout a run. Examples include the learning rate, batch size, number of epochs, dataset, architecture, and optimizer.

An MLflow metric is a measured result produced during or after training. Examples include training loss, validation loss, validation accuracy, and test accuracy.

Metrics accept a `step` because their values can change over time. In this lab, the epoch number was used as the step, which allows MLflow to draw a metric curve across epochs. Parameters do not use a step because they remain constant for the entire run.

### Question 6: What was visible in the run, and where was the model stored?

The MLflow run contained the following information:

- Parameters such as `dataset`, `epochs`, `lr`, `batch_size`, `architecture`, `optimizer`, and `device`.
- Metric histories for `train_loss`, `val_loss`, and `val_accuracy`.
- A final `test_accuracy` metric.
- A logged ResNet-18 model named `model`.

The model artifacts were physically stored inside the repository's `mlruns` directory. Because MLflow 3 treats logged models as separate entities, the model uses a structure similar to:

```text
mlruns/<experiment-id>/models/<model-id>/artifacts
```

For the initial run, the exact artifact directory was:

```text
mlruns/1/models/m-61b9eab30b7e4a2ab584cc5d66633084/artifacts
```

In MLflow 3.16.1, the model can be inspected through the **Models** or **Logged Models** view rather than only through the run's normal artifact list.

## Hyperparameter experiments

The following commands were used to compare learning rates and batch sizes:

```powershell
uv run python .\src\food11\train.py --dataset mini --epochs 5 --lr 0.01 --batch-size 32
uv run python .\src\food11\train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
uv run python .\src\food11\train.py --dataset mini --epochs 5 --lr 0.0001 --batch-size 32
uv run python .\src\food11\train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 64
```

Five runs appear in MLflow because the original development run was followed by the four comparison runs.

## Experiment results

The following values were retrieved from the local MLflow tracking database:

| Run name | Run ID | Learning rate | Batch size | Final train loss | Final validation loss | Final validation accuracy | Test accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| inquisitive-duck-214 | `6a1af6af65db4b6393b142fa1bef1ce0` | 0.0001 | 32 | 0.032772 | 0.622341 | **0.806569** | **0.822080** |
| incongruous-cub-445 | `6604e1b40edf4f69ae0ac35fa507c449` | 0.001 | 64 | 0.195546 | 1.539644 | 0.555657 | 0.588504 |
| abrasive-skunk-54 | `a2c87363b93c4e84a0f7ba84ffb2a4fa` | 0.001 | 32 | 0.435094 | 2.571871 | 0.492701 | 0.468978 |
| masked-zebra-515 | `cf5572488e974c38b82b448c416f0561` | 0.001 | 32 | 0.435094 | 2.571871 | 0.492701 | 0.468978 |
| chill-colt-517 | `69816f96d7bc40c38fca7d385be8e29a` | 0.01 | 32 | 2.329180 | 2.770823 | 0.145985 | 0.135036 |

The two runs with `lr=0.001` and `batch_size=32` produced identical results because they used the same configuration and deterministic random seed.

### Question 7: Which learning rate gave the best validation accuracy? Is higher always better?

The best learning rate was `0.0001`, which produced a validation accuracy of approximately `0.80657` with a batch size of 32.

A higher learning rate was not better. Increasing the learning rate to `0.01` reduced the validation accuracy to approximately `0.14599`. This suggests that the larger learning rate caused updates that were too aggressive for this pretrained network and dataset.

### Question 8: What pattern appeared in the parallel-coordinates plot?

The parallel-coordinates plot showed that learning rate had a strong effect on performance:

- `lr=0.0001` and `batch_size=32` produced the highest validation accuracy, approximately `0.80657`.
- `lr=0.001` and `batch_size=32` produced a validation accuracy of approximately `0.49270`.
- Keeping `lr=0.001` and increasing the batch size from 32 to 64 improved validation accuracy in this experiment from approximately `0.49270` to `0.55566`.
- `lr=0.01` and `batch_size=32` produced the lowest validation accuracy, approximately `0.14599`.

Therefore, the smaller learning rate performed best. Batch size 64 performed slightly better than batch size 32 when comparing the two runs that used `lr=0.001`, but learning rate had the clearest effect. These results describe this set of runs and should not be treated as a universal rule for every dataset or model.

### Question 9: Which run was the best?

After sorting the runs by `val_accuracy` in descending order, the best run was:

```text
Run name: inquisitive-duck-214
Run ID: 6a1af6af65db4b6393b142fa1bef1ce0
Learning rate: 0.0001
Batch size: 32
Validation accuracy: 0.8065693430656934
Test accuracy: 0.8220802919708029
```

The best run's model was stored at:

```text
mlruns/1/models/m-2a21dbaa05c94725ae96eefe85c0fd17/artifacts
```

## Committing the training code

The versioned source and dependency files can be committed with:

```powershell
git add src/food11/train.py pyproject.toml uv.lock lab2-report.md
git commit -m "Add training script with mlflow tracking"
git push
```

The source code and dependency definitions are stored in Git, while the local MLflow database and artifacts remain ignored.

