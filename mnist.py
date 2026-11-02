import warnings
from sklearn.exceptions import DataConversionWarning

#warnings.filterwarnings("ignore", category=DataConversionWarning)
import os
import numpy as np
from preprocessing import Dataset
from beexai.dataset.load_data import load_data
from beexai.utils.time_seed import set_seed
from train import Trainer
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import torch
import joblib

SAVE_PATH = "output/benchmarks/global/"
models_path = "output/benchmarks/models/"

if not os.path.exists(f"{SAVE_PATH}"):
            os.makedirs(f"{SAVE_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "SBN_1_0"
SEED = 10000
path = "data/config/mnist_784.yaml"
DATA_NAME = "mnist_784"
set_seed(SEED)

dataf, target_col, task, _ = load_data(
    from_cleaned=True, config_path=path, keep_corr_features=True
)
data = Dataset(dataf, target_col)

X_train, _, _, _ = data.get_train_test(
    test_size=0.2, scaler_params={"x_cat_encoder_name": "onehot","y_scaler_name": "labelencoder"}
)

original_columns = dataf.drop(columns=[target_col]).columns.tolist()
processed_columns = X_train.columns.tolist()

original_to_processed = {
    col: i for i, col in enumerate(processed_columns)
}

processed_indices = [
    original_to_processed[col]
    for col in original_columns
]

X_raw = dataf.drop(columns=[target_col]).to_numpy(dtype=np.float32)
mnist_centroid = X_raw.mean(axis=0).reshape(28, 28)

teacher_path = models_path + f"{DATA_NAME}_TeacherSBN_{SEED}.joblib"
student_path = models_path + f"{DATA_NAME}_SBN_1_0_{SEED}.pt"

if os.path.exists(student_path) and os.path.exists(teacher_path):
    print("Loading models...")
    teacher_model = joblib.load(teacher_path)
    trainer = Trainer("SBN_1_0", model_params={"teacher_model":teacher_model,"w1":True,"w2":False}, device=device)
    trainer.model.build_model_from_teacher()
    trainer.load_model(student_path)
    print("Models loaded.")
    #SBN structural importance
    m1 = trainer.model.m1.to('cpu').detach().float().numpy()
    gi = m1.sum(axis=0)
    gi = gi / gi.sum()

    # Teacher ensemble importance
    fi = np.asarray(teacher_model.feature_importances_)

    fi = fi[processed_indices]
    gi = gi[processed_indices]

    rho, p_value = spearmanr(gi,fi)

    print("Spearman rho:", rho)
    print("p-value:", p_value)


    # --------------------------------------------------
    # Combined MNIST global importance figure
    # --------------------------------------------------

    fig = plt.figure(figsize=(12, 4.5))

    # Main layout:
    # left  = correlation plot
    # right = three visual maps
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.65],
        wspace=0.25
    )

    # ==================================================
    # (a) Correlation plot
    # ==================================================

    ax = fig.add_subplot(gs[0, 0])

    ax.scatter(
        fi,
        gi,
        s=18,
        alpha=0.7
    )

    # y = x reference
    lower = min(fi.min(), gi.min())
    upper = max(fi.max(), gi.max())

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel("SBN teacher")
    ax.set_ylabel("SBN")
    ax.text(
        0.05,
        0.95,
        rf"Spearman $\rho={rho:.3f}$",
        transform=ax.transAxes,
        verticalalignment="top"
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    # Panel label
    ax.text(
        -0.12, 1.04,
        "(a)",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top"
    )

    # ==================================================
    # (b) Visual comparison
    # ==================================================

    gs_right = gs[0, 1].subgridspec(
        1, 3,
        wspace=0.08
    )

    # Same scale for SBN teacher and SBN
    fi_img = fi.reshape(28, 28)
    gi_img = gi.reshape(28, 28)

    vmax = max(
        fi_img.max(),
        gi_img.max()
    )

    # ---- MNIST centroid ----
    ax0 = fig.add_subplot(gs_right[0, 0])

    ax0.imshow(
        mnist_centroid,
        cmap="gray"
    )

    ax0.set_title("MNIST centroid")
    ax0.axis("off")

    # ---- SBN teacher ----
    ax1 = fig.add_subplot(gs_right[0, 1])

    ax1.imshow(
        fi_img,
        vmin=0,
        vmax=vmax
    )

    ax1.set_title("SBN teacher")
    ax1.axis("off")

    # ---- SBN structural importance ----
    ax2 = fig.add_subplot(gs_right[0, 2])

    im = ax2.imshow(
        gi_img,
        vmin=0,
        vmax=vmax
    )

    ax2.set_title("SBN")
    ax2.axis("off")

    # Colorbar for the two importance maps
    cbar = fig.colorbar(
        im,
        ax=[ax1, ax2],
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label("Normalized importance")

    # Panel label
    ax0.text(
        -0.2, 1.04,
        "(b)",
        transform=ax0.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top"
    )

    # ==================================================
    # Overall title
    # ==================================================

    fig.suptitle(
        "MNIST global importance",
        fontsize=14,
        y=1.0
    )


    # Save for paper
    fig.savefig(
        SAVE_PATH+"mnist_global_importance.pdf",
        bbox_inches="tight"
    )

    # Optional PNG
    fig.savefig(
        SAVE_PATH+"mnist_global_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    torch.cuda.empty_cache()

    print("Done!")