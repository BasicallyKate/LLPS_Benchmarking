import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


# --- Configuration ---

SHOW_PLOTS = True
N_BOOT = 5000
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

files_to_analyze = {
    'Molphase': 'Mutations/AUROC/Molphase_Mutations.csv',
    'DeePhase': 'Mutations/AUROC/DeePhase_Mutations.csv',
    'Fuzdrop': 'Mutations/AUROC/Fuzdrop_Mutations.csv',
    'PSPHunter': 'Mutations/AUROC/PSPHunter_Mutations.csv',
    'LLPhyScore': 'Mutations/AUROC/LLPhyScore_Mutations.csv',
    'PSAP': 'Mutations/AUROC/PSAP_Mutations.csv',
    'PSPire': 'Mutations/AUROC/PSPire_Mutations.csv',
    'Phaseek': 'Mutations/AUROC/Phaseek_Mutations.csv',
    'PICNIC': 'Mutations/AUROC/PICNIC_Mutations.csv',
}

output_dir = "Mutations/AUROC/bootstrap_plots"
os.makedirs(output_dir, exist_ok=True)


# --- Stratified Bootstrap ---

def stratified_bootstrap_ci(y, pred, metric_fn, n_boot=2000, rng=None):

    y = np.asarray(y)
    pred = np.asarray(pred)
    
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos, n_neg = len(pos), len(neg)

    boot_scores = []

    for _ in range(n_boot):
        bpos = rng.choice(pos, size=n_pos, replace=True)
        bneg = rng.choice(neg, size=n_neg, replace=True)
        idx = np.concatenate([bpos, bneg])
        rng.shuffle(idx)

        if len(np.unique(y[idx])) < 2:
            continue

        try:
            score = metric_fn(y[idx], pred[idx])
            boot_scores.append(score)
        except:
            pass

    boot_scores = np.array(boot_scores)
    point = metric_fn(y, pred)

    if len(boot_scores) == 0:
        return point, np.nan, np.nan

    lo, hi = np.percentile(boot_scores, [2.5, 97.5])
    return point, lo, hi


# --- Collect Results ---


rows = []

for algo, file_path in files_to_analyze.items():
    df = pd.read_csv(file_path)
    y = df["true_label"].values
    s = df["prediction_score"].values

    # AUROC CI
    auroc_pt, auroc_lo, auroc_hi = stratified_bootstrap_ci(
        y, s, roc_auc_score, N_BOOT, rng
    )

    # AUPRC CI
    auprc_pt, auprc_lo, auprc_hi = stratified_bootstrap_ci(
        y, s, average_precision_score, N_BOOT, rng
    )

    rows.append({
        "algorithm": algo,
        "AUROC": auroc_pt,
        "AUROC_low": auroc_lo,
        "AUROC_high": auroc_hi,
        "AUPRC": auprc_pt,
        "AUPRC_low": auprc_lo,
        "AUPRC_high": auprc_hi,
    })

results_df = pd.DataFrame(rows)


# --- Sort algorithms by AUROC ---


results_df = results_df.sort_values("AUROC", ascending=False).reset_index(drop=True)


# --- Save confidence intervals to CSV ---


ci_csv_path = os.path.join(output_dir, "LLPS_algorithm_CIs.csv")
results_df.to_csv(ci_csv_path, index=False)
print(f"Saved confidence intervals to: {ci_csv_path}")


# --- Plot combined graph ---

plt.style.use("seaborn-v0_8-whitegrid")

# Colour-blind friendly palette (Okabe–Ito)
color_auroc = "#0072B2"   # blue
color_auprc = "#E69F00"   # orange

fig, ax = plt.subplots(figsize=(18, 8))

x = np.arange(len(results_df))
width = 0.38

# AUROC bars
ax.bar(
    x - width/2,
    results_df["AUROC"],
    width,
    color=color_auroc,
    label="AUROC",
    yerr=[
        results_df["AUROC"] - results_df["AUROC_low"],
        results_df["AUROC_high"] - results_df["AUROC"]
    ],
    capsize=4,
    ecolor="black"
)

# AUPRC bars
ax.bar(
    x + width/2,
    results_df["AUPRC"],
    width,
    color=color_auprc,
    label="AUPRC",
    yerr=[
        results_df["AUPRC"] - results_df["AUPRC_low"],
        results_df["AUPRC_high"] - results_df["AUPRC"]
    ],
    capsize=4,
    ecolor="black"
)

# Labels
ax.set_xticks(x)
ax.set_xticklabels(results_df["algorithm"], rotation=45, ha="right")
ax.set_ylabel("Score", fontsize=12)
ax.set_title("LLPS Predictor Performance with 95% Bootstrap Confidence Intervals", fontsize=15)

# No gridlines
ax.grid(visible=False)

# Black axes
for spine in ["bottom", "top", "right", "left"]:
    ax.spines[spine].set_color("black")

ax.legend()

plt.tight_layout()

# Save figure
plot_path = os.path.join(output_dir, "combined_bar_chart_CIs_sorted.png")
fig.savefig(plot_path, dpi=300)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

print(f"Saved combined bar chart to: {plot_path}")
print("Done!")