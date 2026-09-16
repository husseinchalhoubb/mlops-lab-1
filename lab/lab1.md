# Lab 1 — Git, DVC, and data preparation

## Implementation summary

- Git repository: GitHub `husseinchalhoubb/mlops-lab-1`
- DVC data remote: DagsHub `husseinchalhoubb/mlops-lab-1`
- Data upload approach: the complete dataset was uploaded to DagsHub. The local-remote and reduced-dataset alternatives were not needed.
- Raw images: 16,643
- Processed images: 16,643, resized to 128×128 and arranged by class
- Mini images: 3,292, limited to at most 100 images per class in each split

The preparation script can be run from the repository root with:

```bash
uv run python ./src/food11/data.py
```

## Question 1

`uv init` created the basic Python project files:

- `.python-version` selects the Python version used by the project.
- `pyproject.toml` contains project metadata, the supported Python version, dependencies, command-line entry points, and build-system configuration.
- `README.md` is the project documentation file.
- `src/mlops_lab_1/__init__.py` contains the starter package and generated command-line function.

After Pillow was added, `uv.lock` was also created. It records exact resolved package versions so that the environment can be reproduced.

## Question 2

`dvc init` created:

- `.dvc/config`, which contains repository-level, non-secret DVC configuration.
- `.dvc/.gitignore`, which prevents DVC's cache, temporary files, and local configuration from being committed.
- `.dvcignore`, which tells DVC which files to ignore while scanning the workspace.

The shared configuration and ignore files should be committed to Git. DVC cache files, temporary state, and `.dvc/config.local` must remain local.

## Question 3

With `--global`, DVC stores configuration in the user's global DVC configuration outside the Git repository. Other configuration scopes are:

- the repository configuration, `.dvc/config`, when no scope option is supplied;
- the project-local configuration, `.dvc/config.local`, with `--local`;
- the machine-wide configuration with `--system`.

Credentials must never be pushed to GitHub. In this project, the authentication values are in `.dvc/config.local`, which `.dvc/.gitignore` excludes from Git. Only the non-secret default-remote selection is committed in `.dvc/config`.

## Question 4

Running `dvc add data` added `/data` to `.gitignore`. The image files are therefore not stored by Git. Git tracks `data.dvc`, while DVC stores and transfers the data contents.

## Question 5

The generated `data.dvc` file is a small data pointer. It records:

- a content hash identifying the tracked version;
- the total data size;
- the number of tracked files;
- the tracked path, `data`.

Changing the contents of `data` and running `dvc add data` produces a new pointer version for Git to track.

## Question 6

GitHub contains the source code, configuration files, `.gitignore`, and `data.dvc`; it does not contain the image files under `data`. `data.dvc` is the pointer that identifies the data version. The actual data is stored in the configured DagsHub DVC remote. `dvc status --cloud` confirmed that the local DVC cache and the DagsHub remote are in sync.

## Question 7

A fresh Git clone does not initially contain the `data` directory because Git ignores it. After DVC is installed and credentials are configured, the tracked data can be downloaded with:

```bash
dvc pull
```

## Question 8

After checking out commit `16a4b13` and running `dvc checkout`, the `food11_processed` and `food11_processed_mini` directories are absent because that commit's `data.dvc` points to the raw-only data version. Checking out `main` and running `dvc checkout` restores the current data version, including both processed directories.

## Prepared data structure

The script maps the numeric filename labels to the class names required by Food-11 and produces an ImageFolder-compatible structure:

```text
data/
├── food11_raw/
├── food11_processed/
│   ├── training/<class name>/*.jpg
│   ├── evaluation/<class name>/*.jpg
│   └── validation/<class name>/*.jpg
└── food11_processed_mini/
    ├── training/<class name>/*.jpg
    ├── evaluation/<class name>/*.jpg
    └── validation/<class name>/*.jpg
```

All 19,935 generated images were validated as RGB images with dimensions of exactly 128×128 pixels. The full processed split counts match the raw split counts. Every mini class contains at most 100 images.
