import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


# --- Configuration ---

N_BOOT = 5000
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
SHOW_PLOTS = True

files_to_analyze = {
    'Molphase': 'Sensitivity_Test/Positive_0.3_Test/MolPhase_0.3_For_AUROC.csv',
    'DeePhase': 'Sensitivity_Test/Positive_0.3_Test/DeePhase_0.3_For_AUROC.csv',
    'Fuzdrop': 'Sensitivity_Test/Positive_0.3_Test/FuzDrop_0.3_For_AUROC.csv',
    'PSPHunter': 'Sensitivity_Test/Positive_0.3_Test/PSPHunter_0.3_For_AUROC.csv',
    'PICNIC': 'Sensitivity_Test/Positive_0.3_Test/PICNIC_0.3_For_AUROC.csv',
    'PSPire': 'Sensitivity_Test/Positive_0.3_Test/PSPire_0.3_For_AUROC.csv',
    'Phaseek': 'Sensitivity_Test/Positive_0.3_Test/Phaseek_0.3_For_AUROC.csv',
    'PSAP': 'Sensitivity_Test/Positive_0.3_Test/PSAP_0.3_For_AUROC.csv',
    'LLPhyScore': 'Sensitivity_Test/Positive_0.3_Test/LLPhyScore_0.3_For_AUROC.csv',
}

output_dir = "Sensitivity_Test/Positive_0.3_Test/Graphs"
os.makedirs(output_dir, exist_ok=True)


# --- Stratified Bootstrap ---

def stratified_bootstrap_ci_2d(y, structure, pred, metric_fn,
                               n_boot=2000, rng=None):

    y = np.asarray(y)
    structure = np.asarray(structure)
    pred = np.asarray(pred)

    # Identify strata (label, structure)
    strata = {}
    for i in range(len(y)):
        key = (y[i], structure[i])
        strata.setdefault(key, []).append(i)

    # Convert lists → arrays
    for k in strata:
        strata[k] = np.array(strata[k])

    # Must have both classes present
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan

    boot_vals = []

    for _ in range(n_boot):
        sampled_indices = []

        # sample within each stratum
        for key, idxs in strata.items():
            if len(idxs) == 0:
                continue
            sampled = rng.choice(idxs, size=len(idxs), replace=True)
            sampled_indices.append(sampled)

        if len(sampled_indices) == 0:
            return np.nan, np.nan, np.nan

        sampled_indices = np.concatenate(sampled_indices)

        # skip invalid sample (e.g., all 0s or all 1s)
        if len(np.unique(y[sampled_indices])) < 2:
            continue

        try:
            m = metric_fn(y[sampled_indices], pred[sampled_indices])
            boot_vals.append(m)
        except:
            pass

    if len(boot_vals) == 0:
        return np.nan, np.nan, np.nan

    point = metric_fn(y, pred)
    low, high = np.percentile(boot_vals, [2.5, 97.5])
    return point, low, high



# --- Process each algorithm ---

results = []

for algo, file_path in files_to_analyze.items():

    df = pd.read_csv(file_path)

    y = df["true_label"].values
    s = df["prediction_score"].values
    struct = df["protein_type"].values

    # --- MIXED ---
    mix_auroc_pt, mix_auroc_lo, mix_auroc_hi = stratified_bootstrap_ci_2d(
        y, struct, s, roc_auc_score, N_BOOT, rng
    )
    mix_auprc_pt, mix_auprc_lo, mix_auprc_hi = stratified_bootstrap_ci_2d(
        y, struct, s, average_precision_score, N_BOOT, rng
    )

    # --- FOLDED ---
    df_fold = df[df["protein_type"] == "Folded"]
    y_fold = df_fold["true_label"].values
    s_fold = df_fold["prediction_score"].values
    struct_fold = df_fold["protein_type"].values

    fold_auroc_pt, fold_auroc_lo, fold_auroc_hi = stratified_bootstrap_ci_2d(
        y_fold, struct_fold, s_fold, roc_auc_score, N_BOOT, rng
    )
    fold_auprc_pt, fold_auprc_lo, fold_auprc_hi = stratified_bootstrap_ci_2d(
        y_fold, struct_fold, s_fold, average_precision_score, N_BOOT, rng
    )

    # --- DISORDERED ---
    df_dis = df[df["protein_type"] == "Disordered"]
    y_dis = df_dis["true_label"].values
    s_dis = df_dis["prediction_score"].values
    struct_dis = df_dis["protein_type"].values

    dis_auroc_pt, dis_auroc_lo, dis_auroc_hi = stratified_bootstrap_ci_2d(
        y_dis, struct_dis, s_dis, roc_auc_score, N_BOOT, rng
    )
    dis_auprc_pt, dis_auprc_lo, dis_auprc_hi = stratified_bootstrap_ci_2d(
        y_dis, struct_dis, s_dis, average_precision_score, N_BOOT, rng
    )

    results.append({
        "algorithm": algo,

        "mix_AUROC": mix_auroc_pt,
        "mix_AUROC_low": mix_auroc_lo,
        "mix_AUROC_high": mix_auroc_hi,
        "mix_AUPRC": mix_auprc_pt,
        "mix_AUPRC_low": mix_auprc_lo,
        "mix_AUPRC_high": mix_auprc_hi,

        "fold_AUROC": fold_auroc_pt,
        "fold_AUROC_low": fold_auroc_lo,
        "fold_AUROC_high": fold_auroc_hi,
        "fold_AUPRC": fold_auprc_pt,
        "fold_AUPRC_low": fold_auprc_lo,
        "fold_AUPRC_high": fold_auprc_hi,

        "dis_AUROC": dis_auroc_pt,
        "dis_AUROC_low": dis_auroc_lo,
        "dis_AUROC_high": dis_auroc_hi,
        "dis_AUPRC": dis_auprc_pt,
        "dis_AUPRC_low": dis_auprc_lo,
        "dis_AUPRC_high": dis_auprc_hi,
    })


results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "LLPS_three_panel_CIs.csv"), index=False)



# --- Determine shared Y axis limit ---

y_min = min(
    results_df["mix_AUROC_low"].min(),
    results_df["fold_AUROC_low"].min(),
    results_df["dis_AUROC_low"].min(),
    results_df["mix_AUPRC_low"].min(),
    results_df["fold_AUPRC_low"].min(),
    results_df["dis_AUPRC_low"].min(),
)

y_max = max(
    results_df["mix_AUROC_high"].max(),
    results_df["fold_AUROC_high"].max(),
    results_df["dis_AUROC_high"].max(),
    results_df["mix_AUPRC_high"].max(),
    results_df["fold_AUPRC_high"].max(),
    results_df["dis_AUPRC_high"].max(),
)



# --- Plotting ---

blue = "#0072B2"
orange = "#E69F00"

def make_panel(df, prefix, title, out_path):
    
    # Sort independently by panel AUROC
    df = df.sort_values(f"{prefix}_AUROC", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(18, 8))

    x = np.arange(len(df))
    width = 0.38

    ax.bar(x - width/2, df[f"{prefix}_AUROC"], width,
           color=blue, label="AUROC",
           yerr=[df[f"{prefix}_AUROC"] - df[f"{prefix}_AUROC_low"],
                 df[f"{prefix}_AUROC_high"] - df[f"{prefix}_AUROC"]],
           capsize=4, ecolor="black")

    ax.bar(x + width/2, df[f"{prefix}_AUPRC"], width,
           color=orange, label="AUPRC",
           yerr=[df[f"{prefix}_AUPRC"] - df[f"{prefix}_AUPRC_low"],
                 df[f"{prefix}_AUPRC_high"] - df[f"{prefix}_AUPRC"]],
           capsize=4, ecolor="black")

    ax.set_ylim(0, 1.1)   # shared y-axis limits

    ax.set_xticks(x)
    ax.set_xticklabels(df["algorithm"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(False)

    for sp in ["bottom", "top", "left", "right"]:
        ax.spines[sp].set_color("black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)



# --- Generate panels ---

make_panel(
    results_df, "mix",
    "Mixed Proteins (Folded + Disordered): AUROC & AUPRC with 95% CIs",
    os.path.join(output_dir, "panel_mixed.png")
)

make_panel(
    results_df, "fold",
    "Folded Proteins Only: AUROC & AUPRC with 95% CIs",
    os.path.join(output_dir, "panel_folded.png")
)

make_panel(
    results_df, "dis",
    "Disordered Proteins Only: AUROC & AUPRC with 95% CIs",
    os.path.join(output_dir, "panel_disordered.png")
)

print("\nAll panels created successfully!")