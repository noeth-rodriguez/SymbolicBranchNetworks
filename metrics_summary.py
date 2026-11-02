
from pathlib import Path
import ast
import re
import pandas as pd
from scipy.stats import wilcoxon


def _read_dataset_file(file_path):
    """Read dataset metadata and run-level metric dictionaries."""
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()

    if not lines:
        raise ValueError(f"Empty file: {file_path}")

    metadata = {}
    for key, value in re.findall(
        r"(Labels|Features|Train|Test|Val):\s*(\d+)", lines[0]
    ):
        metadata[key] = int(value)

    records = []
    for line in lines[1:]:
        try:
            record = ast.literal_eval(line.strip())
            if isinstance(record, dict):
                records.append(record)
        except (ValueError, SyntaxError):
            continue

    return metadata, records


def _normalise_metric(metric):
    aliases = {
        "acuracy": "accuracy",
        "acc": "accuracy",
    }
    return aliases.get(metric.lower(), metric)


def _get_metric_name(metric, metadata):
    if metric == "auto":
        return "AUC" if metadata.get("Labels") == 2 else "accuracy"
    return _normalise_metric(metric)


def get_algorithm_comparison(
    base_algorithm,
    list_of_algorithms_to_compare_against,
    metric_to_compare="auto",
    hypothesis="greater",
    data_dir="."
):
    """
    Compare algorithms using dataset-level mean metrics.

    Returns:
        means_table: datasets x algorithms DataFrame.
        tests_table: Wilcoxon statistic and p-value for each comparison.

    hypothesis:
        'greater': comparison algorithm > base algorithm.
        'equal': two-sided test for a difference between algorithms.
    """
    if hypothesis not in ("greater", "equal"):
        raise ValueError("hypothesis must be 'greater' or 'equal'")

    algorithms = [
        base_algorithm,
        *list_of_algorithms_to_compare_against
    ]

    means = {}

    for file_path in sorted(Path(data_dir).glob("*.txt")):
        dataset = file_path.stem
        metadata, records = _read_dataset_file(file_path)

        metric = _get_metric_name(metric_to_compare, metadata)
        row = {}

        for algorithm in algorithms:
            values = [
                record[metric]
                for record in records
                if record.get("model_name") == algorithm
                and metric in record
                and isinstance(record[metric], (int, float))
            ]

            row[algorithm] = (
                sum(values) / len(values) if values else float("nan")
            )

        means[dataset] = row

    means_table = pd.DataFrame.from_dict(means, orient="index")
    means_table.index.name = "Dataset"

    test_results = []

    for algorithm in list_of_algorithms_to_compare_against:
        paired = means_table[
            [base_algorithm, algorithm]
        ].dropna()

        base_values = paired[base_algorithm].to_numpy()
        comparison_values = paired[algorithm].to_numpy()

        if len(paired) == 0:
            statistic, pvalue = float("nan"), float("nan")

        elif (comparison_values == base_values).all():
            statistic, pvalue = 0.0, 1.0

        else:
            result = wilcoxon(
                comparison_values,
                base_values,
                alternative=(
                    "greater" if hypothesis == "greater"
                    else "two-sided"
                )
            )
            statistic = result.statistic
            pvalue = result.pvalue

        test_results.append({
            "Base algorithm": base_algorithm,
            "Compared algorithm": algorithm,
            "Metric": metric_to_compare,
            "Hypothesis": hypothesis,
            "Number of datasets": len(paired),
            "Wilcoxon statistic": statistic,
            "p-value": pvalue
        })

    tests_table = pd.DataFrame(test_results)

    return means_table, tests_table


def get_algorithm_values(
    algorithm,
    list_of_metrics,
    data_dir="."
):
    """
    Return a datasets x metrics DataFrame for a specific algorithm.

    Dataset metadata is extracted from the first line.
    Model-specific metrics are averaged across runs.
    """
    metadata_names = {
        "Labels", "Features", "Train", "Test", "Val"
    }

    results = {}

    for file_path in sorted(Path(data_dir).glob("*.txt")):
        dataset = file_path.stem
        metadata, records = _read_dataset_file(file_path)

        algorithm_records = [
            record for record in records
            if record.get("model_name") == algorithm
        ]

        row = {}

        for metric in list_of_metrics:

            if metric in metadata_names:
                row[metric] = metadata.get(metric, float("nan"))

            else:
                values = [
                    record[metric]
                    for record in algorithm_records
                    if metric in record
                    and isinstance(record[metric], (int, float))
                ]

                row[metric] = (
                    sum(values) / len(values)
                    if values else float("nan")
                )

        results[dataset] = row

    table = pd.DataFrame.from_dict(results, orient="index")
    table.index.name = "Dataset"

    return table

if __name__ == "__main__":

    # 1. Algorithm comparison
    means, tests = get_algorithm_comparison(
        "TeacherSBN",
        ["SBN_1_0"],
        "auto",
        "greater",
        data_dir="output/benchmarks"
    )

    print(means)
    print(tests)


    # 2. Dataset characteristics and model properties
    values = get_algorithm_values(
        "SBN_1_0",
        ["Labels", "Features", "hidden_neurons",
         "Train", "Test", "Val", "fitting_time"],
        data_dir="output/benchmarks"
    )

    print(values)