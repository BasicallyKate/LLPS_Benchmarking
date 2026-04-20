import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os

# --- Configuration ---
files_to_analyze = {
    'Molphase': 'Disordered_Folded/Molphase_Dis_Fold.csv',
    'DeePhase': 'Disordered_Folded/DeePhase_Dis_Fold.csv',
    'Fuzdrop': 'Disordered_Folded/Fuzdrop_Dis_Fold.csv',
    'PSPHunter': 'Disordered_Folded/PSPHunter_Dis_Fold.csv',
    'LLPhyScore': 'Disordered_Folded/LLPhyScore_Dis_Fold.csv',
    'PSAP': 'Disordered_Folded/PSAP_Dis_Fold.csv',
    'PICNIC': 'Disordered_Folded/PICNIC_Dis_Fold.csv',
    'PSPire': 'Disordered_Folded/PSPire_Dis_Fold.csv',
    'Phaseek': 'Disordered_Folded/Phaseek_Dis_Fold.csv',

}

output_dir = "Disordered_Folded/New_plots"
os.makedirs(output_dir, exist_ok=True)

# --- Colors and Styling ---
plt.style.use("seaborn-v0_8-whitegrid")
COLOR_CYCLE = list(plt.cm.tab10.colors)
algo_to_color = {}

def get_color_for(algo_name):
    if algo_name not in algo_to_color:
        algo_to_color[algo_name] = COLOR_CYCLE[len(algo_to_color) % len(COLOR_CYCLE)]
    return algo_to_color[algo_name]

def style_axes(ax):
    for side in ("bottom", "top", "right", "left"):
        ax.spines[side].set_color("black")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.grid(visible=False)

# --- Individual Algorithm Plot ---
def save_individual_plot(subset_name, algorithm_name, color, fpr, tpr, precision, recall, roc_auc, pr_auc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    # ROC
    ax1.plot(fpr, tpr, lw=2, color=color, label=f"AUROC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=2)
    ax1.set_title(f"{subset_name} ROC Curve", fontsize=14)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()
    style_axes(ax1)

    # PR
    ax2.plot(recall, precision, lw=2, color=color, label=f"AUPRC = {pr_auc:.3f}")
    ax2.set_title(f"{subset_name} Precision–Recall Curve", fontsize=14)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()
    style_axes(ax2)

    fig.suptitle(f"{algorithm_name} – {subset_name} Performance", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()

    safe_subset = subset_name.lower().replace(" ", "_").replace("only", "")
    safe_algo = algorithm_name.lower().replace(" ", "_")

    out_path = os.path.join(output_dir, f"{safe_algo}_{safe_subset}_individual.png")
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"      -> Individual plot saved: {out_path}")

# --- Main Analysis Function ---
def analyze_and_plot_subset(title_prefix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    print(f"\n--- Analyzing Subset: {title_prefix} ---")

    plotted_anything = False

    for algorithm_name, file_path in files_to_analyze.items():
        try:
            full_data = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"  -> ERROR: Missing file '{file_path}'. Skipping.")
            continue

        # Subset
        if "Disordered" in title_prefix:
            subset_data = full_data[full_data["protein_type"] == "Disordered"]
            subset_name = "Disordered"
        elif "Folded" in title_prefix:
            subset_data = full_data[full_data["protein_type"] == "Folded"]
            subset_name = "Folded"
        else:
            subset_data = full_data
            subset_name = "Overall"

        if subset_data.empty or len(subset_data["true_label"].unique()) < 2:
            print(f"  -> Skipping '{algorithm_name}': Not enough data for this subset.")
            continue

        plotted_anything = True

        y_true = subset_data["true_label"]
        y_score = subset_data["prediction_score"]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)

        color = get_color_for(algorithm_name)

        # Add to combined panel figures
        ax1.plot(fpr, tpr, lw=2, color=color, label=f"{algorithm_name} (AUROC = {roc_auc:.3f})")
        ax2.plot(recall, precision, lw=2, color=color, label=f"{algorithm_name} (AUPRC = {pr_auc:.3f})")

        print(f"  Algorithm: {algorithm_name} -> AUROC: {roc_auc:.4f}, AUPRC: {pr_auc:.4f}")

        # --- Create individual plots ---
        save_individual_plot(
            subset_name=subset_name,
            algorithm_name=algorithm_name,
            color=color,
            fpr=fpr,
            tpr=tpr,
            precision=precision,
            recall=recall,
            roc_auc=roc_auc,
            pr_auc=pr_auc
        )

    if not plotted_anything:
        print(f" Skipping combined plot for '{title_prefix}': no valid data.")
        plt.close(fig)
        return

    # Finalize combined ROC
    ax1.plot([0, 1], [0, 1], "k--", lw=2)
    ax1.set_title(f"{title_prefix} ROC Curves", fontsize=14)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(fontsize="small")
    style_axes(ax1)

    # Finalize combined PR
    ax2.set_title(f"{title_prefix} Precision–Recall Curves", fontsize=14)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(fontsize="small")
    style_axes(ax2)

    fig.canvas.draw()

    # Save combined figure
    out_path = os.path.join(output_dir, f"performance_{title_prefix.lower().replace(' ', '_')}.png")
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"  -> Combined plot saved: {out_path}")


# --- Run Analyses ---
analyze_and_plot_subset("Overall")
analyze_and_plot_subset("Disordered Only")
analyze_and_plot_subset("Folded Only")

print("\n Stratified analysis complete!")