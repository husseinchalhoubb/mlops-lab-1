# Lab 3 - Containerizing the Model with Docker

## Environment / Setup Summary

- Development host: Windows, using Docker Desktop with Linux containers.
- Python environment: Python 3.11 managed with `uv`.
- Model: an 11-class Food-11 ResNet18 classifier. Training logs the model artifact to MLflow under the artifact name `model`.
- MLflow Registry: registered model `food11`; version `1` comes from run `6a1af6af65db4b6393b142fa1bef1ce0` and is assigned the alias `champion`.
- Serving application: FastAPI in `src/food11/serve.py`. It exposes `GET /health` and `POST /predict`.
- Container image measured locally: `food11-api:latest` has a Docker **content size of about 429 MB**. Docker also reports 2.01 GB disk usage; that is Docker's local storage accounting and should not be confused with the image content size.

## Question 1

### Answer

`mlflow.pytorch.log_model(..., name="model")` logs a model **artifact for one particular MLflow run**. It records the serialized model and its MLmodel metadata under that run's artifacts, so it is tied to the experiment/run that produced it.

The Model Registry is a separate, named lifecycle layer on top of those artifacts. Registering the artifact creates a versioned model such as `food11` version `1`, with its source run recorded. The registry makes it possible to attach aliases, descriptions, and later versions without changing the original run.

For this lab, the best run was registered as `food11` version `1`, sourced from run `6a1af6af65db4b6393b142fa1bef1ce0`, then assigned the alias `champion`.

## Question 2

### Answer

An alias is a mutable, human-meaningful pointer to a specific registered model version. For example, `champion` can point to the version approved for production, while `challenger` can point to a candidate version being evaluated.

Aliases are preferable to hard-coding a version in serving code because deployment code stays stable. Moving `champion` from version 1 to version 2 changes the model that is resolved at startup, without editing `serve.py` or rebuilding the API image.

## Question 3

### Answer

The serving code loads:

```python
model = mlflow.pyfunc.load_model("models:/food11@champion")
```

This is a Registry URI. It asks MLflow to resolve the alias `champion` for the registered model `food11`, then obtain the corresponding model artifact. In contrast, a local `.pth` path such as `models/food11_best.pth` names one file on the local filesystem and requires the application to know where that file exists.

To deploy a new champion, I would register the new trained artifact as a new version and move the `champion` alias to it in MLflow. No code change is required. The MLflow tracking/artifact service must remain reachable by the container and be able to serve the artifact; the API loads the selected model when its process starts.

## Question 4

### Answer

The Dockerfile uses two stages:

```text
builder stage
  python:3.11-slim
  install uv
  copy pyproject.toml + uv.lock
  uv sync --frozen --no-dev --no-install-project
        |
        v
runtime stage
  python:3.11-slim
  copy only /app/.venv from builder
  copy src/
  run uvicorn src.food11.serve:app
```

The builder creates the dependency virtual environment; the runtime image receives that environment and the application source, but not the builder's package-install tooling or unrelated build context.

Copying `pyproject.toml` and `uv.lock` before `src/` makes dependency installation a cacheable layer. Source-code edits normally invalidate only the later `COPY src ./src` layer, so Docker can reuse the dependency layer. `--frozen` ensures the installed dependency set matches `uv.lock`, while `--no-dev` excludes development dependencies. `--no-install-project` is necessary here because dependencies are installed before the local `src/` package is copied into the builder; attempting to install the project itself at that point would fail because its source is not present.

## Question 5

### Answer

Only the multi-stage image was built and measured in this repository, so an exact numerical comparison with a naive single-stage image is not available and should not be invented. The verified `food11-api:latest` content size is about **429 MB**.

From `docker history`, the largest current image layer is the copied virtual environment (`COPY /app/.venv /app/.venv`), shown as about **1.44 GB** in the layer history. Docker layer accounting can differ from the displayed content size because of compression/shared data, but it clearly shows that Python/ML dependencies dominate this image.

A naive image that copied the full repository and kept build tooling would usually be larger, but the actual difference must be measured by building that alternative image before quoting a number.

## Question 6

### Answer

`.dockerignore` controls which files Docker sends as the **build context**. It reduces transfer time and prevents ignored files from being available to any `COPY` instruction. It does not by itself decide what ends up in the final image: the Dockerfile's `COPY` instructions do that.

This project ignores `.venv/`, `data/`, `mlruns/`, `mlflow.db`, `.git/`, `.dvc/`, Python caches, test caches, and editor folders. Important reasons include:

- `.venv/` is a host Windows virtual environment; copying it into a Linux image can introduce incompatible executables and duplicate the container's Linux environment.
- `data/` can make the build context very large and is not needed by the prediction API.
- `mlruns/` and `mlflow.db` are local tracking state, not application code; baking them into an image is brittle and may expose experiment metadata.
- `.git/`, `.dvc/`, caches, and editor configuration add unrelated files and can leak history or machine-specific state.

With this particular Dockerfile, only `pyproject.toml`, `uv.lock`, the built virtual environment, and `src/` are copied into image stages. Therefore most of the ignored folders would not be in the final image even if present, but excluding them still keeps the context clean and protects against later broad copies such as `COPY . .`.

## Question 7

### Answer

Inside a Docker container, `localhost` means the container itself, not the Windows host. The MLflow server runs on the host, so the container uses Docker Desktop's special hostname:

```text
FastAPI container -- http://host.docker.internal:5000 --> MLflow server on Windows host
```

The runtime command supplies this through `MLFLOW_TRACKING_URI`:

```powershell
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
```

This lets `models:/food11@champion` resolve against the host MLflow server rather than trying to find MLflow inside the API container.

## Question 8

### Answer

The final image successfully ran after the MLflow host/artifact-serving configuration was corrected; that configuration correction did not require rebuilding the image. The image contains the Linux Python runtime, the locked production dependencies, the FastAPI source, and the Uvicorn launch command. It does **not** bake in the selected MLflow model artifact.

At container startup, `serve.py` reads `MLFLOW_TRACKING_URI`, connects to MLflow, resolves `food11@champion`, and loads that artifact once at module import. Requests then reuse the in-memory model. This separation means a later alias change can select a newer model for a newly started container without changing the application image.

## Question 9

### Answer

To deploy this image outside the local machine, push it to a container registry such as Docker Hub, GitHub Container Registry (GHCR), or a cloud registry such as Amazon ECR. A deployment platform can then pull the image using an immutable version tag or, preferably, an image digest.

For a production release I would tag both a readable release identifier (for example, `food11-api:1.0.0`) and record the immutable `sha256` digest used by the deployment. I would also keep the MLflow tracking/artifact endpoint reachable from the deployment environment, because the container resolves the champion model from MLflow at startup.

## Commands Used

```powershell
# Check Docker installation and run a first test container
docker info
docker run hello-world

# Add serving dependencies
uv add fastapi uvicorn python-multipart
uv add pillow

# Run the API locally
uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000

# Build and inspect the container image
docker build -t food11-api:latest .
docker images food11-api:latest
docker history food11-api:latest

# Run it while pointing it to MLflow on the Windows host
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest

# Test the health endpoint
curl.exe http://127.0.0.1:8000/health

# Test prediction (replace the final path with an actual image in the dataset)
curl.exe -X POST -F "file=@data/food11_processed_mini/validation/Bread/<image>.jpg" http://127.0.0.1:8000/predict
```

## Key Concepts Learned

- An MLflow run artifact is not the same thing as a versioned, aliased Registry model.
- Registry aliases decouple deployment code from model-version selection.
- Docker multi-stage builds and dependency-first copying improve reproducibility and cache reuse.
- `.dockerignore` reduces the build context; Dockerfile `COPY` instructions define image contents.
- Containers use their own network namespace, so host services need an explicit reachable hostname.
- A container image can package the serving application while MLflow supplies the currently selected model artifact at runtime.
