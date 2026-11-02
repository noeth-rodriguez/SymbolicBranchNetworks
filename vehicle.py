"""Unified local explanation and local-explanation stability analysis for SBN.

Both the illustrative local explanation (seed 10000) and cross-seed
stability analysis use the same method:
  1. Select up to three branches with the largest positive contributions to
     the predicted-class score. These branches support the predicted class.
  2. Preserve signed feature contributions for directional interpretation.
  3. Aggregate nonnegative weighted path-feature scores:
         score_i += branch contribution * |effective W1[j, i] * BN0(x)[i]|
     for features on selected branches' tree paths.
  4. Rank only features with a positive score; zero-score features are not
     treated as explanatory features or used to fill the top-k sets.

This script computes a detailed local explanation for SAMPLE_IDX using
EXPLANATION_SEED and repeats the same scoring method across SEEDS.
"""

import warnings

#from benchmarks_metrics import teacher_model

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib

from beexai.dataset.load_data import load_data
from beexai.utils.time_seed import set_seed
from preprocessing import Dataset
from train import Trainer

# ---------------------------- Configuration ----------------------------
SAVE_PATH = Path("output/benchmarks/local")
MODEL_NAME = "SBN_1_0"
CONFIG_PATH = "data/config/vehicle.yaml"

SEEDS = list(range(10000, 10010))
EXPLANATION_SEED = 10000
TEST_SIZE = 0.2
SPLIT_SEED = 10000
SAMPLE_IDX = 39
TOP_NEURONS = 3
TOP_K_FEATURES = (3, 5)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_NAME = Path(CONFIG_PATH).stem
MODELS_DIR = SAVE_PATH / "models"
OUT_DIR = SAVE_PATH / "local_path_stability" / DATA_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_fixed_data():
    """Load a reproducible train/test split and the selected test example."""
    set_seed(SPLIT_SEED)
    torch.manual_seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)

    dataf, target_col, task, _ = load_data(
        from_cleaned=True,
        config_path=CONFIG_PATH,
        keep_corr_features=True,
    )
    data = Dataset(dataf, target_col)
    X_train, X_test, y_train, y_test = data.get_train_test(
        test_size=TEST_SIZE,
        scaler_params={
            "x_cat_encoder_name": "onehot",
            "y_scaler_name": "labelencoder",
        },
    )

    if SAMPLE_IDX < 0 or SAMPLE_IDX >= len(X_test):
        raise IndexError(
            f"SAMPLE_IDX={SAMPLE_IDX}, but test set has {len(X_test)} rows."
        )

    original_feature_names = dataf.drop(columns=[target_col]).columns.tolist()
    processed_feature_names = list(X_test.columns)
    original_sample = dataf.drop(columns=[target_col]).loc[
        X_test.index[SAMPLE_IDX]
    ].to_numpy(dtype=np.float32)
    x_np = X_test.iloc[SAMPLE_IDX].to_numpy(dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
    actual_class = int(np.asarray(y_test)[SAMPLE_IDX].reshape(-1)[0])
    class_names = sorted(dataf[target_col].unique())

    return (
        dataf, target_col, X_test, y_test, x_tensor, original_sample,
        actual_class, original_feature_names, processed_feature_names,
        class_names,
    )


def explain_weighted_paths(model, x, processed_feature_names, top_neurons=3):
    """Explain using positive class-supporting branches and magnitude scores."""
    model.eval()
    with torch.no_grad():
        x = x.to(model.device)
        z = model.bn0(x)
        w1_eff = model.r1 * model.w1
        h_pre = F.linear(z, w1_eff)
        h_bn = model.bn1(h_pre)
        h = F.silu(h_bn)
        h_tilde = model.bn2(h)
        w2_eff = model.r2 * model.w2
        logits = F.linear(h_tilde, w2_eff)
        predicted_class = int(torch.argmax(logits, dim=1).item())

        # Select only branches that positively support the predicted class.
        class_contrib = h_tilde[0] * w2_eff[predicted_class]
        positive_indices = torch.where(class_contrib > 0)[0]
        k = min(top_neurons, positive_indices.numel())
        if k > 0:
            positive_values = class_contrib[positive_indices]
            _, order = torch.topk(positive_values, k=k)
            selected_indices = positive_indices[order]
        else:
            selected_indices = positive_indices

        feature_scores = np.zeros(len(processed_feature_names), dtype=np.float64)
        selected_details = []

        for neuron_idx in selected_indices:
            j = int(neuron_idx.item())
            signed_neuron_contrib = float(class_contrib[j].item())
            branch = model.branch_info[j]
            feature_contrib = w1_eff[j] * z[0]

            path_indices = sorted({int(i) for i in branch["path_features"]})
            per_feature = []
            for i in path_indices:
                if i < 0 or i >= len(feature_scores):
                    continue
                raw_contrib = float(feature_contrib[i].item())
                # Selected branch contributions are positive. The absolute
                # feature term measures influence magnitude, not direction.
                weighted_abs_contrib = max(signed_neuron_contrib, 0.0) * abs(raw_contrib)
                feature_scores[i] += weighted_abs_contrib
                per_feature.append({
                    "feature_index": i,
                    "feature": processed_feature_names[i],
                    "signed_feature_contribution": raw_contrib,
                    "signed_class_contribution": signed_neuron_contrib,
                    "weighted_abs_contribution": weighted_abs_contrib,
                })

            class_contributions = (
                h_tilde[0, j] * w2_eff[:, j]
            )
            selected_details.append({
                "neuron_index": j,
                "tree_index": branch.get("tree_index"),
                "parent_node": branch.get("parent_node"),
                "signed_class_contribution": signed_neuron_contrib,
                "branch_activation": float(h_tilde[0, j].item()),
                "path_features": "; ".join(
                    processed_feature_names[i] for i in path_indices
                    if 0 <= i < len(processed_feature_names)
                ),
                "conditions": branch.get("conditions", []),
                "parent_split": branch.get("parent_split"),
                "feature_details": per_feature,
                "class_contributions": {
                    c: float(v.item())
                    for c, v in enumerate(class_contributions)
                },
            })

    total = feature_scores.sum()
    normalized_scores = feature_scores / total if total > 0 else feature_scores.copy()

    # Only strictly positive scores are active explanatory features. Leave
    # inactive feature ranks as NaN rather than arbitrarily ranking tied zeros.
    active_indices = np.flatnonzero(feature_scores > 1e-12)
    ranks = np.full(len(feature_scores), np.nan, dtype=float)
    if active_indices.size:
        active_order = active_indices[np.argsort(-normalized_scores[active_indices], kind="stable")]
        ranks[active_order] = np.arange(1, len(active_order) + 1)

    return {
        "predicted_class": predicted_class,
        "logits": logits[0].detach().cpu().numpy(),
        "scores": normalized_scores,
        "raw_scores": feature_scores,
        "ranks": ranks,
        "selected_details": selected_details,
    }


def load_model(seed):
    model_path = MODELS_DIR / f"{DATA_NAME}_{MODEL_NAME}_{seed}.pt"
    teacher_path = MODELS_DIR / f"{DATA_NAME}_TeacherSBN_{seed}.joblib"
    if not model_path.exists() or not teacher_path.exists():
        return None, model_path
    teacher_model = joblib.load(teacher_path)
    trainer = Trainer(MODEL_NAME, model_params={"teacher_model":teacher_model,"w1":1,"w2":0}, device=DEVICE)
    trainer.load_model(str(model_path))
    model = trainer.model

    # Preserve the model reconstruction sequence from the supplied scripts.
    if hasattr(model, "build_model_from_ensemble") and hasattr(model, "tree_ensemble"):
        model.build_model_from_ensemble(model.tree_ensemble)
        trainer.load_model(str(model_path))
        model = trainer.model

    return trainer, model_path


def print_detailed_explanation(result, model, x, original_sample,
                               original_feature_names, class_names, actual_class):
    """Print the selected branches and their signed/magnitude interpretations."""
    print("\n" + "#" * 78)
    print("DETAILED LOCAL EXPLANATION")
    print("#" * 78)
    print("Actual class:", class_names[actual_class])
    print("Predicted class:", class_names[result["predicted_class"]])
    print("Logits:", np.array2string(result["logits"], precision=5))

    # Recompute transformed input and effective weights for readable tables.
    model.eval()
    with torch.no_grad():
        x = x.to(model.device)
        z = model.bn0(x)
        w1_eff = model.r1 * model.w1
        w2_eff = model.r2 * model.w2

    for rank, branch in enumerate(result["selected_details"], start=1):
        j = branch["neuron_index"]
        print("\n" + "=" * 72)
        print(f"Branch rank {rank}: neuron {j}")
        print(f"Tree: {branch['tree_index']} | parent node: {branch['parent_node']}")
        print(
            "Predicted-class contribution: "
            f"{branch['signed_class_contribution']:+.6f} "
        )
        print("Interpretation:", "supports" if branch["signed_class_contribution"] >= 0
              else "opposes", "the predicted-class score")

        print("\nPath feature contributions (signed values shown):")
        print(f"{'Feature':<28} {'Original X':>12} {'Effective W1':>14} {'W1*z':>14}")
        print("-" * 72)
        path_indices = sorted({
            int(i) for i in model.branch_info[j]["path_features"]
        })
        for i in path_indices:
            if i < 0 or i >= len(z[0]):
                continue
            name = original_feature_names[i] if i < len(original_feature_names) else f"feature_{i}"
            x_value = original_sample[i] if i < len(original_sample) else float("nan")
            print(
                f"{name:<28} {x_value:>12.4f} "
                f"{float(w1_eff[j, i].item()):>+14.6f} "
                f"{float((w1_eff[j, i] * z[0, i]).item()):>+14.6f}"
            )

        print("\nContribution of this branch to each class:")
        print(f"{'Class':<20} {'Effective W2':>16} {'Contribution':>16}")
        print("-" * 56)
        for c in range(w2_eff.shape[0]):
            w = float(w2_eff[c, j].item())
            contrib = float((result_activation := branch["branch_activation"]) * w)
            cname = str(class_names[c]) if c < len(class_names) else str(c)
            print(f"{cname:<20} {w:>+16.6f} {contrib:>+16.6f}")

        conditions = branch["conditions"]
        if conditions:
            print("\nBranch condition verification:")
            for cond in conditions:
                i = int(cond["feature"])
                name = original_feature_names[i] if i < len(original_feature_names) else f"feature_{i}"
                value = float(original_sample[i]) if i < len(original_sample) else float("nan")
                threshold = float(cond["threshold"])
                direction = cond["direction"]
                passed = value > threshold if direction == "right" else value <= threshold
                op = ">" if direction == "right" else "<="
                mark = "PASS" if passed else "FAIL"
                print(f"  {name} {op} {threshold:.6f} (value={value:.6f}): {mark}")

        parent_split = branch["parent_split"]
        if parent_split:
            i = int(parent_split["feature"])
            name = original_feature_names[i] if i < len(original_feature_names) else f"feature_{i}"
            value = float(original_sample[i]) if i < len(original_sample) else float("nan")
            threshold = float(parent_split["threshold"])
            print("\nParent split:")
            print(f"  {name} <= {threshold:.6f} (value={value:.6f})")
            for side in ("left", "right"):
                leaf_key = f"{side}_leaf"
                majority_key = f"{side}_majority_class"
                if parent_split.get(leaf_key) is not None and majority_key in parent_split:
                    cls = int(parent_split[majority_key])
                    label = class_names[cls] if cls < len(class_names) else cls
                    print(f"  {side} child majority class: {label}")

    print("\nTop-ranked path features by magnitude-based weighted score:")
    ranked = np.flatnonzero(result["scores"] > 1e-12)
    ranked = ranked[np.argsort(-result["scores"][ranked], kind="stable")]
    if not len(ranked):
        print("  No positive feature scores on the selected branch paths.")
    for i in ranked[:min(10, len(ranked))]:
        print(f"  rank {int(result['ranks'][i]):>2}: {i} "
              f"{result_feature_name(i, model, original_feature_names)} "
              f"score={result['scores'][i]:.6f}")


def result_feature_name(i, model, original_feature_names):
    # The score vector uses processed feature indices. For this dataset they
    # normally align with the original columns; fallback names are explicit.
    return original_feature_names[i] if i < len(original_feature_names) else f"feature_{i}"

def summarize_feature_usage(
    w1_eff,
    feature_scores,
    feature_names,
    seed,
    zero_tol=1e-12,
    top_k=(3, 5),
):
    """
    Summarize structural and local feature usage for one seed.

    w1_eff: effective input-to-hidden weight matrix,
            shape (neurons, features).
    feature_scores: nonnegative local path-feature scores.
    feature_names: names in the same order as W1 columns.
    """
    W = w1_eff.detach().cpu().numpy() if hasattr(
        w1_eff, "detach"
    ) else np.asarray(w1_eff)

    scores = np.asarray(feature_scores, dtype=float).reshape(-1)
    names = list(feature_names)

    if W.ndim != 2:
        raise ValueError("w1_eff must be a 2D matrix.")
    if W.shape[1] != len(names) or len(scores) != len(names):
        raise ValueError(
            "W1 columns, feature names, and feature scores "
            "must have the same number of features."
        )

    # A column is structurally unused if every effective weight
    # for that feature is approximately zero.
    column_max_abs = np.max(np.abs(W), axis=0)
    structurally_unused = column_max_abs <= zero_tol

    # Local scores should be nonnegative for the magnitude-based method.
    if np.any(scores < -zero_tol):
        raise ValueError(
            "Negative local scores found. Check the scoring formula."
        )

    locally_inactive = scores <= zero_tol
    locally_low_ranked = scores > zero_tol

    # Rank only features with positive local scores.
    active_indices = np.flatnonzero(locally_low_ranked)
    active_indices = sorted(
        active_indices,
        key=lambda i: scores[i],
        reverse=True,
    )

    ranks = {i: rank + 1 for rank, i in enumerate(active_indices)}

    result = pd.DataFrame({
        "seed": seed,
        "feature": names,
        "max_abs_effective_W1": column_max_abs,
        "structurally_unused": structurally_unused,
        "local_score": scores,
        "locally_inactive": locally_inactive,
        "local_rank": [ranks.get(i, np.nan) for i in range(len(names))],
    })

    for k in top_k:
        result[f"top{k}"] = result["local_rank"].le(k).fillna(False)

    # An explicit mutually exclusive classification for this run.
    result["status"] = np.select(
        [
            result["structurally_unused"],
            result["locally_inactive"],
            result["local_rank"].notna(),
        ],
        [
            "structurally unused",
            "locally inactive (nonzero weights)",
            "locally active",
        ],
        default="unclassified",
    )

    return result


def main():
    print(f"Device: {DEVICE}")
    print(f"Dataset config: {CONFIG_PATH}")
    print(f"Model directory: {MODELS_DIR}")
    print(f"Output directory: {OUT_DIR}")

    (
        dataf, target_col, X_test, y_test, x_tensor, original_sample,
        actual_class, original_feature_names, processed_feature_names,
        class_names,
    ) = load_fixed_data()

    # -------- Detailed local explanation for the manuscript example --------
    trainer, model_path = load_model(EXPLANATION_SEED)
    if trainer is None:
        raise FileNotFoundError(f"Missing model for local explanation: {model_path}")
    model = trainer.model
    result = explain_weighted_paths(
        model, x_tensor.to(model.device), processed_feature_names, TOP_NEURONS
    )
    print_detailed_explanation(
        result, model, x_tensor, original_sample, original_feature_names,
        class_names, actual_class
    )

    # Save detailed local explanation records.
    detail_rows = []
    for branch_rank, branch in enumerate(result["selected_details"], start=1):
        for feat in branch["feature_details"]:
            detail_rows.append({
                "branch_rank_by_positive_contribution": branch_rank,
                "neuron_index": branch["neuron_index"],
                "tree_index": branch["tree_index"],
                "parent_node": branch["parent_node"],
                "predicted_class": result["predicted_class"],
                "signed_class_contribution": branch["signed_class_contribution"],
                **feat,
            })
    pd.DataFrame(detail_rows).to_csv(
        OUT_DIR / "single_example_local_explanation.csv", index=False
    )
    pd.DataFrame([{
        "actual_class": actual_class,
        "predicted_class": result["predicted_class"],
        **{f"logit_{i}": float(v) for i, v in enumerate(result["logits"])}
    }]).to_csv(OUT_DIR / "single_example_prediction.csv", index=False)

    # ---------------- Stability analysis over saved model seeds ----------------
    score_rows = []
    neuron_rows = []
    usage_rows = []
    run_results = {}
    for seed in SEEDS:
        run_trainer, path = load_model(seed)
        if run_trainer is None:
            print(f"WARNING: missing model, skipping seed {seed}: {path}")
            continue

        print(f"Processing stability run for seed {seed}")
        run_model = run_trainer.model
        run_result = explain_weighted_paths(
            run_model, x_tensor.to(run_model.device),
            processed_feature_names, TOP_NEURONS
        )
        run_results[seed] = run_result

        # Summarize whether each input feature is structurally unused by the
        # effective first-layer weights and whether it contributes locally to
        # this example's selected positive branches.
        w1_eff = run_model.r1 * run_model.w1
        usage = summarize_feature_usage(
            w1_eff=w1_eff,
            feature_scores=run_result["raw_scores"],
            feature_names=processed_feature_names,
            seed=seed,
        )
        usage_rows.append(usage)

        for i, feature_name in enumerate(processed_feature_names):
            score_rows.append({
                "seed": seed,
                "feature": feature_name,
                "weighted_score": float(run_result["scores"][i]),
                "rank": (int(run_result["ranks"][i]) if np.isfinite(run_result["ranks"][i]) else np.nan),
                "active_feature": bool(run_result["scores"][i] > 1e-12),
                "in_top3": bool(np.isfinite(run_result["ranks"][i]) and run_result["ranks"][i] <= 3),
                "in_top5": bool(np.isfinite(run_result["ranks"][i]) and run_result["ranks"][i] <= 5),
            })
        for branch_rank, branch in enumerate(run_result["selected_details"], start=1):
            neuron_rows.append({
                "seed": seed,
                "branch_rank_by_positive_contribution": branch_rank,
                "neuron_index": branch["neuron_index"],
                "tree_index": branch["tree_index"],
                "parent_node": branch["parent_node"],
                "predicted_class": run_result["predicted_class"],
                "signed_class_contribution": branch["signed_class_contribution"],
                "path_features": branch["path_features"],
            })

    if not run_results:
        raise FileNotFoundError(
            f"No models found in {MODELS_DIR} matching "
            f"{DATA_NAME}_{MODEL_NAME}_<seed>.pt"
        )

    scores_df = pd.DataFrame(score_rows)
    neurons_df = pd.DataFrame(neuron_rows)
    usage_df = pd.concat(usage_rows, ignore_index=True)
    usage_df.to_csv(OUT_DIR / "feature_usage_by_seed.csv", index=False)

    # Aggregate run-level usage indicators. "Structurally unused" is assessed
    # from the full effective W1 column; local inactivity is sample-specific.
    usage_summary = usage_df.groupby("feature").agg(
        structurally_unused_runs=("structurally_unused", "sum"),
        locally_inactive_runs=("locally_inactive", "sum"),
        top3_runs=("top3", "sum"),
        top5_runs=("top5", "sum"),
        mean_local_score=("local_score", "mean"),
        median_local_rank=("local_rank", "median"),
    ).reset_index()
    usage_summary.to_csv(OUT_DIR / "feature_usage_summary.csv", index=False)

    scores_df.to_csv(OUT_DIR / "weighted_path_feature_scores_long.csv", index=False)
    neurons_df.to_csv(OUT_DIR / "selected_top_neurons.csv", index=False)

    rank_table = scores_df.pivot(index="feature", columns="seed", values="rank")
    rank_table.columns = [f"Seed {int(c)}" for c in rank_table.columns]
    rank_table.to_csv(OUT_DIR / "weighted_path_feature_ranks.csv")

    score_table = scores_df.pivot(index="feature", columns="seed", values="weighted_score")
    score_table.columns = [f"Seed {int(c)}" for c in score_table.columns]
    score_table.to_csv(OUT_DIR / "weighted_path_feature_scores.csv")

    frequency = scores_df.groupby("feature").agg(
        top3_count=("in_top3", "sum"),
        top5_count=("in_top5", "sum"),
        median_rank=("rank", "median"),
        mean_rank=("rank", "mean"),
    ).sort_values(["top3_count", "median_rank"], ascending=[False, True])
    frequency.to_csv(OUT_DIR / "weighted_path_feature_frequency.csv")

    predicted_classes = {
        seed: int(res["predicted_class"]) for seed, res in run_results.items()
    }
    print("\nActual encoded class for fixed sample:", actual_class)
    print("Predicted encoded classes by seed:", predicted_classes)
    print("All runs predict the same class:",
          len(set(predicted_classes.values())) == 1)
    print("\nFeature frequency summary:")
    print(frequency.to_string())
    print(f"\nSaved all outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
