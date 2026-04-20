%matplotlib inline

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

SHOW_PLOTS = True

# --- Configuration ---
files_to_analyze = {
    'Molphase': 'Virus/Molphase_Virus.csv',
    'DeePhase': 'Virus/DeePhase_Virus.csv',
    'Fuzdrop': 'Virus/Fuzdrop_Virus.csv',
    'PSPHunter': 'Virus/PSPHunter_Virus.csv',
    'LLPhyScore': 'Virus/LLPhyScore_Virus.csv',
    'PSAP': 'Virus/PSAP_Virus.csv',
    'PSPire': 'Virus/PSPire_Virus.csv',
    'Phaseek': 'Virus/Phaseek_Virus.csv',
    'PICNIC': 'Virus/PICNIC_Virus.csv',
}

# --- Create an output directory for the plots ---
output_dir = "Virus/New_plots"
os.makedirs(output_dir, exist_ok=True)

# --- Style & colors ---
plt.style.use('seaborn-v0_8-whitegrid')
# Ensure consistent, visible, non-white colors everywhere
COLOR_CYCLE = list(plt.cm.tab10.colors)  # 10 distinct colors
algo_to_color = {}  # keep color consistent per algorithm across figures

def get_color_for(algo_name):
    if algo_name not in algo_to_color:
        algo_to_color[algo_name] = COLOR_CYCLE[len(algo_to_color) % len(COLOR_CYCLE)]
    return algo_to_color[algo_name]

def style_axes(ax):
    """Apply consistent axis styling and fixed limits (0..1)."""
    ax.grid(visible=False)
    for side in ('bottom', 'top', 'right', 'left'):
        ax.spines[side].set_color('black')
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.01)

print("--- LLPS Algorithm Performance Metrics ---")

# --- 1. Setup for the COMBINED Plot ---
fig_combined, (ax_comb1, ax_comb2) = plt.subplots(1, 2, figsize=(16, 7))
fig_combined.patch.set_facecolor('white')  # force white background

# --- Loop Through Each File ---
for algorithm_name, file_path in files_to_analyze.items():
    # Load data
    data = pd.read_csv(file_path)
    true_labels = data['true_label']
    prediction_scores = data['prediction_score']

    # Calculate metrics
    fpr, tpr, _ = roc_curve(true_labels, prediction_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(true_labels, prediction_scores)
    pr_auc = average_precision_score(true_labels, prediction_scores)

    # Choose a consistent, visible color
    color = get_color_for(algorithm_name)

    # --- 2. Add this algorithm's data to the COMBINED plot ---
    ax_comb1.plot(fpr, tpr, label=f'{algorithm_name} (AUROC = {roc_auc:.3f})', lw=2, color=color)
    ax_comb2.plot(recall, precision, label=f'{algorithm_name} (AUPRC = {pr_auc:.3f})', lw=2, color=color)

    # --- 3. Create and save an INDIVIDUAL plot for this algorithm ---
    fig_ind, (ax_ind1, ax_ind2) = plt.subplots(1, 2, figsize=(16, 7))
    fig_ind.patch.set_facecolor('white')  # force white background

    # Plot ROC (individual)
    ax_ind1.plot(fpr, tpr, label=f'AUROC = {roc_auc:.3f}', lw=2, color=color)
    ax_ind1.plot([0, 1], [0, 1], 'k--', lw=2)  # Random guess line
    ax_ind1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax_ind1.set_xlabel('False Positive Rate', fontsize=12)
    ax_ind1.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax_ind1.legend(loc='lower right')
    style_axes(ax_ind1)

    # Plot PR (individual)
    ax_ind2.plot(recall, precision, label=f'AUPRC = {pr_auc:.3f}', lw=2, color=color)
    ax_ind2.set_title('Precision-Recall (PR) Curve', fontsize=14)
    ax_ind2.set_xlabel('Recall', fontsize=12)
    ax_ind2.set_ylabel('Precision', fontsize=12)
    ax_ind2.legend(loc='lower left')
    style_axes(ax_ind2)

    # Layout, render, and save
    fig_ind.suptitle(f'Performance for: {algorithm_name}', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Ensure the canvas actually draws before saving (prevents blank/black artifacts)
    fig_ind.canvas.draw()

    # Sanitize filename
    safe_filename = algorithm_name.replace(' ', '_').lower()
    individual_plot_path = os.path.join(output_dir, f'performance_{safe_filename}.png')

    # Save with explicit facecolor to avoid backend quirks
    fig_ind.savefig(individual_plot_path, dpi=300, facecolor='white', bbox_inches='tight')

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_ind)

    # Print the scores to the terminal
    print(f"\nResults for: {algorithm_name}")
    print(f"  AUROC: {roc_auc:.4f}")
    print(f"  AUPRC: {pr_auc:.4f}")
    print(f"  -> Individual plot saved to: {individual_plot_path}")

# --- 4. Finalize and save the COMBINED plot ---
ax_comb1.set_title('Combined ROC Curves', fontsize=14)
ax_comb1.set_xlabel('False Positive Rate', fontsize=12)
ax_comb1.set_ylabel('True Positive Rate', fontsize=12)
ax_comb1.plot([0, 1], [0, 1], 'k--', lw=2) # Random guess line
ax_comb1.legend(loc='best', fontsize='x-small', frameon=True, edgecolor='black')
style_axes(ax_comb1)

ax_comb2.set_title('Combined Precision-Recall Curves', fontsize=14)
ax_comb2.set_xlabel('Recall', fontsize=12)
ax_comb2.set_ylabel('Precision', fontsize=12)
ax_comb2.legend(loc='best', fontsize='x-small', frameon=True, edgecolor='black')
style_axes(ax_comb2)

fig_combined.suptitle('LLPS Prediction Algorithm Comparison', fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Ensure combined figure is drawn before saving
fig_combined.canvas.draw()

combined_plot_path = os.path.join(output_dir, 'llps_performance_COMBINED.png')
fig_combined.savefig(combined_plot_path, dpi=300, facecolor='white', bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig_combined)

print(f"\n Analysis complete! All plots saved in the '{output_dir}' folder.")