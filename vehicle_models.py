# Adapted from https://github.com/SquareResearchCenter-AI/BEExAI
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

import argparse
import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import torch

from preprocessing import Dataset
from beexai.dataset.load_data import load_data
from beexai.utils.time_seed import set_seed
from train import Trainer


os.environ["TABPFN_TOKEN"] = os.getenv("TABPFN_TOKEN", "")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_path",
    type=str,
    default="output/benchmarks/local",
    help="Directory in which to save models and metrics",
)
parser.add_argument(
    "--config_path",
    type=str,
    default="data/config/vehicle.yaml",
    help="Path to vehicle.yaml",
)
args = parser.parse_args()

SAVE_PATH = args.save_path
CONFIG_PATH = args.config_path
DATA_NAME = "vehicle"

MODELS_DIR = os.path.join(SAVE_PATH, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# All data preparation and student-side randomness use this seed.
FIXED_SEED = 10000

# Each teacher gets a different seed, while the corresponding student
# is trained using FIXED_SEED and the same dataset split.
TEACHER_SEEDS = range(10000, 10010)

MODEL_NAME = "SBN_1_0"


def train_and_save(model_name, model_params, model_path, X_train, y_train,
                   X_val, y_val, X_test, y_test, seed):
    """Train, evaluate, and save a model and its complete metrics."""
    metrics_path = model_path + ".pkl"

    model_extension = ".pt" if "SBN_" in model_name else ".joblib"
    model_file = model_path + model_extension

    if os.path.exists(metrics_path) and os.path.exists(model_file):
        with open(metrics_path, "rb") as metrics_file:
            metrics = pickle.load(metrics_file)
        print(f"Model and metrics already exist, skipping: {model_path}")
        return metrics

    trainer = Trainer(
        model_name,
        model_params=model_params,
        device=DEVICE,
    )

    loss_file = model_path + ".png"

    trainer.train(
        X_train,
        y_train,
        X_val,
        y_val,
        seed,
        loss_file=loss_file,
    )

    if "SBN_" in model_name:
        trainer.model.eval()

    trainer.infer(
        X_test,
        np.asarray(y_test).reshape(-1),
    )

    # Save after infer() so the metrics include both training and
    # inference results. Trainer.save_model() in the supplied train.py
    # opens the pickle in text mode, so write the metrics pickle here
    # explicitly in binary mode as well.
    trainer.save_model(model_path)

    metrics = trainer.model.metrics
    with open(metrics_path, "wb") as metrics_file:
        pickle.dump(metrics, metrics_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(metrics)

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main():
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(
            f"Could not find vehicle configuration: {CONFIG_PATH}"
        )

    metrics_output = os.path.join(SAVE_PATH, f"{DATA_NAME}.txt")

    with open(metrics_output, "w", encoding="utf-8") as output_file:
        for run_index, teacher_seed in enumerate(TEACHER_SEEDS, start=1):
            print(
                f"\n=== Dataset: {DATA_NAME} | "
                f"Run {run_index}/10 | Teacher seed: {teacher_seed} ==="
            )

            # Reset before data loading/splitting so all ten runs use
            # exactly the same train/validation/test partition.
            set_seed(FIXED_SEED)

            dataf, target_col, task, _ = load_data(
                from_cleaned=True,
                config_path=CONFIG_PATH,
                keep_corr_features=True,
            )
            data = Dataset(dataf, target_col)

            X_train, X_test, y_train, y_test = data.get_train_test(
                test_size=0.2,
                scaler_params={
                    "x_cat_encoder_name": "onehot",
                    "y_scaler_name": "labelencoder",
                },
            )

            X_train, X_val, y_train, y_val = data.get_train_val(
                X_train,
                y_train,
                val_size=0.125,
            )

            num_labels = np.unique(np.asarray(y_train)).shape[0]
            n_samples, n_features = X_train.shape

            if run_index == 1:
                header = (
                    f"Labels: {num_labels} Features: {n_features} "
                    f"Train: {n_samples} Test: {len(y_test)} "
                    f"Val: {len(y_val)}"
                )
                print(header, file=output_file)
                print(header)

            # 1. Train/evaluate the teacher with its run-specific seed.
            teacher_model_path = os.path.join(
                MODELS_DIR,
                f"{DATA_NAME}_TeacherSBN_{teacher_seed}",
            )

            teacher_metrics = train_and_save(
                model_name="TeacherSBN",
                model_params={},
                model_path=teacher_model_path,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                seed=teacher_seed,
            )

            # Load the teacher corresponding to this run.
            teacher_file = teacher_model_path + ".joblib"
            if not os.path.isfile(teacher_file):
                raise FileNotFoundError(
                    "Teacher model was not saved where expected: "
                    f"{teacher_file}. Check Trainer.save_model()."
                )
            teacher_model = joblib.load(teacher_file)

            # 2. Train one SBN_1_0 student using the corresponding teacher.
            student_model_path = os.path.join(
                MODELS_DIR,
                f"{DATA_NAME}_{MODEL_NAME}_{teacher_seed}",
            )

            student_metrics = train_and_save(
                model_name=MODEL_NAME,
                model_params={"teacher_model": teacher_model},
                model_path=student_model_path,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                seed=FIXED_SEED,
            )

            # Keep the output file in the same per-run dictionary format
            # used by the original benchmarking script.
            print(student_metrics, file=output_file)
            print(
                f"Completed run {run_index}/10 "
                f"(teacher seed {teacher_seed})."
            )

    print(f"\nFinished. Metrics saved to: {metrics_output}")


if __name__ == "__main__":
    main()
