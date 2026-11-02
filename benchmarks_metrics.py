# Adapted from https://github.com/SquareResearchCenter-AI/BEExAI
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
import argparse
import glob
import os
import numpy as np
import torch
import joblib
import pickle
from preprocessing import Dataset
from beexai.dataset.load_data import load_data
from beexai.utils.time_seed import set_seed
from train import Trainer

os.environ["TABPFN_TOKEN"] = os.getenv("TABPFN_TOKEN")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_path",
    type=str,
    default="output/benchmarks",
    help="Path to folder to save results",
)
parser.add_argument(
    "--config_path",
    type=str,
    default="data/config",
    help="Path to folder to save results",
)
args = parser.parse_args()
SAVE_PATH = args.save_path
models_dir = SAVE_PATH+"/models/"
def is_sbn(name):
    return "SBN_" in name

if not os.path.exists(models_dir):
            os.makedirs(models_dir)

CONFIG_PATH = args.config_path

device = "cuda" if torch.cuda.is_available() else "cpu"

all_config_names = glob.glob(f"{CONFIG_PATH}/*.yaml")

models = ["TeacherSBN","SBN_1_0","SBN_0_0","XGBoost","TabPFN"]

if any(model.startswith("SBN") for model in models):
    if "TeacherSBN" in models:
        models.remove("TeacherSBN")
    models.insert(0, "TeacherSBN")  # TeacherSBN must be trained before an SBN

for path in all_config_names: #for each dataset
    skip_dataset = False
    DATA_NAME = path.split("/")[-1].split(".")[0].replace("\\", "_")
    print("=== Dataset: ", DATA_NAME," ===")
    with (open(f"{SAVE_PATH}/{DATA_NAME}.txt", 'w') as f): #save in this file the metrics for each model
        for i,SEED in enumerate(range(10000,10010)): #for each seed
            set_seed(SEED)
            teacher_metrics = False
            dataf, target_col, task, _ = load_data(
                from_cleaned=True, config_path=path, keep_corr_features=True
            )
            data = Dataset(dataf, target_col)

            X_train, X_test, y_train, y_test = data.get_train_test(
                test_size=0.2, scaler_params={"x_cat_encoder_name": "onehot","y_scaler_name": "labelencoder"}
            )

            X_train, X_val, y_train, y_val = data.get_train_val(X_train, y_train, val_size=0.125)

            num_labels = np.unique(np.asarray(y_val)).shape[0]
            n_samples, n_features = X_train.shape

            if i ==0: #first seed, print dataset header
                print("Labels:",num_labels, "Features:",n_features,"Train:", n_samples,"Test:", y_test.shape[0],"Val:", y_val.shape[0], file=f)
                print("Labels:",num_labels, "Features:",n_features,"Train:", n_samples,"Test:", y_test.shape[0],"Val:", y_val.shape[0])

            for MODEL_NAME in models:
                if is_sbn(MODEL_NAME):
                    teacher_path = models_dir + f"{DATA_NAME}_TeacherSBN_{SEED}.joblib" #The teacher was trained before the SBN
                    teacher_model = joblib.load(teacher_path)
                    model_params = {"teacher_model":teacher_model}
                    ext = ".pt"
                else:
                    model_params = {}
                    ext = ".joblib"
                model_file_name = models_dir + f"{DATA_NAME}_{MODEL_NAME}_{SEED}"

                if os.path.exists(model_file_name + '.pkl'): #metrics already computed
                    with open(model_file_name + ".pkl", "rb") as metrics_file:
                        metrics_dict = pickle.load(metrics_file)
                else:  # fit and infer to compute metrics
                    trainer = Trainer(
                        MODEL_NAME,
                        model_params=model_params,
                        device=device
                    )

                    # Train
                    loss_file = model_file_name + ".png"

                    trainer.train(
                        X_train, y_train,
                        X_val, y_val,
                        SEED,
                        loss_file=str(loss_file)
                    )

                    # Infer
                    trainer.infer(
                        X_test,
                        np.asarray(y_test).reshape(-1)
                    )

                    # Save model and complete metrics dictionary
                    trainer.save_model(model_file_name)

                    metrics_dict = trainer.model.metrics

                    torch.cuda.empty_cache()
                print(metrics_dict, file=f)
                print(metrics_dict)
        f.close()