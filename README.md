# Symbolic Branch Networks (SBN)

This repository contains the implementation and experimental code accompanying the following research article:

**Nöth-Rodríguez, D. (2026).** *Symbolic Branch Networks: Tree-Inherited Neural Models for Interpretable Multiclass Classification.* Neurocomputing. https://doi.org/10.1016/j.neucom.2026.135532

The repository provides the code to reproduce the experiments reported in the paper, including benchmark evaluations, statistical comparisons, and local and global interpretability analyses.

## 1. Environment setup

The experiments reported in the paper were run on **Ubuntu 24.04.5 LTS**.

First, create a Python virtual environment named `SBN`:

```bash
python3 -m venv SBN
```

Activate the environment:

```bash
source SBN/bin/activate
```

Install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The dependencies and their versions are specified in `requirements.txt`.

### Reproducibility note

The experiments were conducted on Ubuntu 24.04.5 LTS. Although random seeds are set throughout the implementation, exact numerical reproducibility across different operating systems or hardware configurations is not guaranteed. Differences in numerical libraries, hardware, and CPU/GPU computation may affect results. Reproducing the experiments in the same software and hardware environment is therefore recommended.

## 2. Download the datasets

Run:

```bash
python download_datasets.py
```

This script downloads datasets from the OpenML-CC18 benchmark suite and creates the following directories:

- `data/csv/`: downloaded datasets in CSV format.
- `data/config/`: YAML configuration files used by the training scripts.

**By default, only the 10 smallest datasets are downloaded.**

To include additional datasets, edit **line 41** in `download_datasets.py`:

```python
tabular_multiclass = smallest_datasets #+ middle_datasets + largest_datasets
```

For example, to download all datasets listed in the script, change it to:

```python
tabular_multiclass = smallest_datasets + middle_datasets + largest_datasets
```

The three lists contain the smallest, middle-sized, and largest datasets, respectively.

## 3. Run the benchmark experiments

The next step is to execute:

```bash
python benchmarks_metrics.py
```

This script runs the benchmark experiments on all datasets found in `data/config/`. For each dataset, it evaluates the models over 10 random seeds (10000 to 10009).

The default models are:

- `TeacherSBN`
- `SBN_1_0`
- `SBN_0_0`
- `XGBoost`
- `TabPFN`

The teacher model is trained before the SBN student models.

### TabPFN authentication

Running TabPFN requires an API access token from the **Prior Labs User Interface**.

Declare your token as an environment variable before running the benchmark:

```bash
export TABPFN_TOKEN="YOUR_API_ACCESS_TOKEN"
```

The token is then available to the script through the `TABPFN_TOKEN` environment variable.

If you do not intend to run TabPFN, make the following two changes in `benchmarks_metrics.py`:

1. Comment out line 17:

   ```python
   # os.environ["TABPFN_TOKEN"] = os.getenv("TABPFN_TOKEN")
   ```

2. Remove `"TabPFN"` from the `models` list:

   ```python
   models = ["TeacherSBN", "SBN_1_0", "SBN_0_0", "XGBoost"]
   ```

The benchmark outputs are saved by default in:

```text
output/benchmarks/
├── models/
└── <dataset_name>.txt
```

The `models/` directory contains the trained models and associated files. Each dataset has a text file containing its metadata and the metrics obtained for each model and seed.

The script can also be configured with alternative output and dataset-configuration directories through its command-line arguments:

```bash
python benchmarks_metrics.py --save_path output/benchmarks --config_path data/config
```

## 4. Summarise benchmark results

After running the benchmark experiments, execute:

```bash
python metrics_summary.py
```

This script reads the results in `output/benchmarks/` and provides tables summarising model performance and dataset characteristics.

The current examples in the script's `__main__` section include:

**Algorithm comparison**

```python
means, tests = get_algorithm_comparison(
    "TeacherSBN",
    ["SBN_1_0"],
    "auto",
    "greater",
    data_dir="output/benchmarks"
)
```

This returns:

- A table containing the mean metric values for each dataset and algorithm.
- A table containing the Wilcoxon test statistic and p-value for each comparison.

With `"auto"`, the script uses AUC for binary classification datasets and accuracy for multiclass datasets. The `"greater"` hypothesis tests whether the comparison algorithm performs better than the base algorithm.

**Dataset characteristics and model properties**

```python
values = get_algorithm_values(
    "SBN_1_0",
    ["Labels", "Features", "hidden_neurons",
     "Train", "Test", "Val", "fitting_time"],
    data_dir="output/benchmarks"
)
```

This returns a table with dataset characteristics and the selected model metrics, averaged across runs where applicable.

You can adapt these function calls to obtain other comparisons or metrics. For example, to compare `SBN_1_0` against XGBoost:

```python
means, tests = get_algorithm_comparison(
    "SBN_1_0",
    ["XGBoost"],
    "auto",
    "greater",
    data_dir="output/benchmarks"
)
```

To compare using a specific metric, replace `"auto"` with a metric name available in the benchmark results, such as `"accuracy"` or `"AUC"`.

## 5. Local interpretability experiments

The local interpretability experiments use the Vehicle dataset and are performed in two steps.

### 5.1. Train the models

First, run:

```bash
python vehicle_models.py
```

This script trains and saves **10 teacher/student model pairs**.

An important distinction is that these models are not the same models generated by `benchmarks_metrics.py`.

In this experiment, all 10 runs use the same train/validation/test split and the same student-side seed. The teacher model is trained with a different seed in each run (10000 to 10009). Each corresponding SBN student is then trained using its respective teacher.

The default output directory is:

```text
output/benchmarks/local/
```

### 5.2. Generate local interpretability results

Once the models have been trained, run:

```bash
python vehicle.py
```

This script loads the models generated by `vehicle_models.py` and produces the local interpretability results.

It includes a detailed local explanation for a selected test sample, as well as an analysis of local-explanation stability across the 10 teacher seeds.

The output files are saved under:

```text
output/benchmarks/local/local_path_stability/vehicle/
```

## 6. Global interpretability experiments

The global interpretability experiment is implemented in:

```bash
python mnist.py
```

Before running this script, the MNIST dataset must be downloaded and its configuration file must be available at:

```text
data/config/mnist_784.yaml
```

You must also train at least one SBN model on MNIST. In particular, the current implementation of `mnist.py` expects the following seed-10000 model files:

```text
output/benchmarks/models/
├── mnist_784_TeacherSBN_10000.joblib
└── mnist_784_SBN_1_0_10000.pt
```

These are generated by running `benchmarks_metrics.py` with the MNIST dataset included in the downloaded configurations.

The script compares the global feature importance of the SBN student with the teacher model. It calculates their Spearman rank correlation and generates a visual comparison of the MNIST centroid and the two feature-importance maps.

The figures are saved in:

```text
output/benchmarks/global/
├── mnist_global_importance.pdf
└── mnist_global_importance.png
```

## 7. Recommended execution order

For convenience, the complete workflow is:

```bash
# 1. Create and activate the environment
python3 -m venv SBN
source SBN/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download datasets
python download_datasets.py

# 4. Run benchmark experiments
python benchmarks_metrics.py

# 5. Summarise benchmark results
python metrics_summary.py

# 6. Train models for local interpretability
python vehicle_models.py

# 7. Generate local interpretability results
python vehicle.py

# 8. Generate global interpretability results
python mnist.py
```

**Note:** For the global MNIST interpretability experiment, make sure that `mnist_784` is included in the downloaded datasets and that the required teacher and SBN student checkpoints have been generated before running `mnist.py`.
