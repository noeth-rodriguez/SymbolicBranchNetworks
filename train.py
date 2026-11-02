# Adapted from https://github.com/SquareResearchCenter-AI/BEExAI/src/beexai/training/
"""Fitting and evaluating models on benchmark datasets."""

from typing import Callable, Optional
import time
import joblib
import numpy as np
import pandas as pd
import torch
import pickle
import os
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,roc_curve)
from SBNFramwork import SBNModel
from tabpfn import TabPFNClassifier

class Trainer:
    """Trainer class
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict] = None,
        device: str = None,
    ):
        self.models = {
            "XGBoost": XGBClassifier,
            "TabPFN": TabPFNClassifier,
            "SBN_0_0": SBNModel,
            "SBN_0_1": SBNModel,
            "SBN_1_0": SBNModel,
            "SBN_1_1": SBNModel,
            "TeacherSBN": ExtraTreesClassifier
        }
        self.model_name = model_name
        self.model_params = model_params if model_params is not None else {}
        self.device = device
        assert (
            self.model_name in self.models
        ), f"Model name must be one of {self.models.keys()}"
        if "SBN_" in model_name:
            w1 = model_name.split("_")[-2]
            w2 = model_name.split("_")[-1]
            self.model = SBNModel(teacher_model=model_params["teacher_model"], training_w1=int(w1)==True, training_w2=int(w2)==True, device=device)
        elif model_name == "TabPFN":
            self.model = TabPFNClassifier(**self.model_params, device=device)
        else:
            self.model = self.models[model_name](**self.model_params)


    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        seed: int,
        loss_file: Optional[str] = None,
    ) -> Callable:
        """Perform training on the whole training set.
        Returns:
            callable: trained model
        """
        metrics = {}
        metrics["model_name"] = self.model_name
        def sync_cuda():
            if torch.cuda.is_available():
                torch.cuda.synchronize()


        if "SBN_" in self.model_name:

            sync_cuda()
            t0_build = time.perf_counter()
            self.model.build_model_from_teacher()
            sync_cuda()
            t1_build = time.perf_counter()

            metrics["building_time"] = t1_build - t0_build
            m1 = self.model.m1.to('cpu').detach().float().numpy()
            m2 = self.model.m2.to('cpu').detach().float().numpy()
            metrics["hidden_neurons"] = m1.shape[0]
            metrics["sparsity_w1"] = m1.sum() / m1.size
            metrics["sparsity_w2"] = m2.sum() / m2.size
            del m1, m2

            sync_cuda()
            t0_fit = time.perf_counter()
            self.model = self.model.fit(
                np.asarray(x_train),
                np.asarray(y_train).reshape(-1),
                np.asarray(x_val),
                np.asarray(y_val).reshape(-1),
                loss_file=loss_file,
            )
            sync_cuda()
            t1_fit = time.perf_counter()

        elif self.model_name == "TeacherSBN":
            _, n_features = x_train.shape
            n_labels = len(list(set(list(np.asarray(y_train).ravel()))))
            current_estimators = 0
            max_estimators = n_labels + round(np.log2(n_features))
            depth_min = 2
            depth_max = round(np.log2(n_features)) + 4
            current_depth = depth_min
            steps = 2
            t0_fit = time.perf_counter()
            tree_ensemble = ExtraTreesClassifier(random_state=seed, warm_start=True, n_estimators=0)
            while current_estimators <= max_estimators:
                if current_depth > depth_max:
                    current_depth = depth_min
                current_estimators += 1
                current_depth += steps
                tree_ensemble.n_estimators = current_estimators
                tree_ensemble.max_leaf_nodes = 2 ** current_depth
                tree_ensemble.fit(x_train, y_train)
            t1_fit = time.perf_counter()

            self.model = tree_ensemble


        elif self.model_name == "XGBClassifier":


            param_grid = {
                "n_estimators": [10, 50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.03, 0.1, 0.3],
            }

            y_train_1d = np.asarray(y_train).ravel()
            y_val_1d = np.asarray(y_val).ravel()

            n_classes = len(np.unique(y_train_1d))
            eval_metric = "mlogloss" if n_classes > 2 else "logloss"

            best_score = -np.inf
            best_model = None
            best_params = None

            t0_fit = time.perf_counter()
            for n_estimators in param_grid["n_estimators"]:
                for max_depth in param_grid["max_depth"]:
                    for learning_rate in param_grid["learning_rate"]:

                        params = {
                            **self.model_params,
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "early_stopping_rounds": 10,
                            "eval_metric": eval_metric,
                            "random_state": seed,
                        }

                        model = XGBClassifier(**params)

                        model.fit(
                            x_train,
                            y_train_1d,
                            eval_set=[(x_val, y_val_1d)],
                            verbose=False,
                        )

                        y_val_pred = model.predict(x_val)
                        score = accuracy_score(y_val_1d, y_val_pred)

                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_params = {
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "learning_rate": learning_rate,
                            }
            t1_fit = time.perf_counter()

            self.model = best_model
            print(
                f"Best XGBoost validation accuracy: {best_score:.4f}; "
                f"parameters: {best_params}"
            )
        else:
            t0_fit = time.perf_counter()
            self.model.fit(x_train, y_train)
            t1_fit = time.perf_counter()

        metrics["fitting_time"] = t1_fit - t0_fit

        self.model.metrics = metrics
        return self.model


    def infer(self, x_test, y_test):

        def sync_cuda():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        if "SBN_" in self.model_name:
            self.model.eval()
        metrics = {}
        # y_true must be 1D
        y_true = np.array(y_test).ravel()


        if len(np.unique(y_true)) == 2:
            sync_cuda()
            t0_inference = time.perf_counter()
            proba = self.model.predict_proba(x_test)  # shape [N, C]
            sync_cuda()
            t1_inference = time.perf_counter()

            y_score = proba[:, 1]

            # convert to numpy
            if torch.is_tensor(y_score):
                y_score = y_score.detach().cpu().numpy()
            if torch.is_tensor(y_true):
                y_true = y_true.detach().cpu().numpy()

            # AUC
            metrics["AUC"] = round(roc_auc_score(y_true, y_score), 3)

            # balanced threshold (TPR ≈ TNR)
            fpr, tpr, thresh = roc_curve(y_true, y_score)
            balance_idx = np.abs(tpr - (1 - fpr)).argmin()
            best_thresh = float(thresh[balance_idx])

            # balanced prediction
            y_pred_balanced = (y_score >= best_thresh).astype(int)

            metrics["threshold"] = round(best_thresh, 4)
            metrics["accuracy"] = round(accuracy_score(y_true, y_pred_balanced), 6)
            metrics["f1_score"] = round(f1_score(y_true, y_pred_balanced), 6)
        else:
            sync_cuda()
            t0_inference = time.perf_counter()
            y_pred = self.model.predict(x_test)
            sync_cuda()
            t1_inference = time.perf_counter()
            metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 6)
            metrics["f1 score"] = round(f1_score(y_true, y_pred, average="weighted"), 6)

        metrics["inference_time"] = t1_inference - t0_inference
        self.model.metrics = self.model.metrics | metrics

    def save_model(self, path: str):
        """Save the model and its metrics."""

        is_nn = "SBN_" in self.model_name

        if is_nn:
            torch.save(self.model.state_dict(), path + '.pt')
        else:
            joblib.dump(self.model, path + '.joblib')

        # Save metrics for every model type
        metrics_path = path + '.pkl'

        with open(metrics_path, 'wb') as metrics_file:
            pickle.dump(
                self.model.metrics,
                metrics_file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def load_model(self, path: str):
        """Load the model"""
        if "SBN_" in self.model_name:
            self.model.build_model_from_teacher()
            fn_dict = torch.load(path,weights_only=True)
            self.model.load_state_dict(fn_dict)
            self.model.to(self.device)
        else:
            self.model = joblib.load(path)

        if os.path.exists(path + '.pkl'): #metrics already computed
            with open(path+ ".pkl", "rb") as metrics_file:
                self.model.metrics = pickle.load(metrics_file)
        else:
            self.model.metrics = {}


