import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import os

# --- Load and prepare data ---
df = pd.read_csv("Mutations_for_Heatmap.csv")
df_sorted = df.sort_values(by="Protein").reset_index(drop=True)
algo_cols = ['Molphase', 'DeePhase', 'Fuzdrop', 'PSPHunter', 'LLPhyScore', 'PSAP', 'PSPire', 'Phaseek', 'PICNIC']

# Convert "N/A" to NaN for numeric handling
df_sorted[algo_cols] = df_sorted[algo_cols].replace("N/A", np.nan).astype(float)

# --- Create custom diverging colormap ---
colors = ["#2166ac", "lightgrey", "#b2182b"]  # Blue → Grey → Red
custom_cmap = LinearSegmentedColormap.from_list("custom_bwr", colors)

# --- Compute  scaling limits to ignore extreme outliers ---
vmin, vmax = np.percentile(
    df_sorted[algo_cols].values[np.isfinite(df_sorted[algo_cols].values)],
    [2, 98]
)

# --- Use linear diverging normalization centered at 0 ---
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# --- Create output directory ---
output_dir = "heatmaps"
os.makedirs(output_dir, exist_ok=True)

# --- Split by Protein ---
proteins = df_sorted['Protein'].unique()
n_proteins = len(proteins)

# --- Setup subplot grid ---
fig, axes = plt.subplots(
    1, n_proteins, figsize=(5 * n_proteins, 6), sharey=False, constrained_layout=True
)
if n_proteins == 1:
    axes = [axes]

# --- Plot each protein heatmap ---
for ax, protein in zip(axes, proteins):
    subset = df_sorted[df_sorted['Protein'] == protein]
    matrix = subset[algo_cols].values
    row_labels = subset['Mutation']

    # Heatmap
    im = ax.imshow(matrix, cmap=custom_cmap, norm=norm, aspect='equal')

    # Add hatch for NaN cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch='////', edgecolor='gray', lw=0.0
                )
                ax.add_patch(rect)

    # Add text labels (skip NaNs)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=8)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(algo_cols)))
    ax.set_xticklabels(algo_cols, rotation=45, ha='left')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.invert_yaxis()  # M1 at top

    # Center protein name along left Y-axis
    ax.text(
        -1.7, matrix.shape[0] / 3.5,
        protein, fontsize=12, rotation=90, va='center', ha='center'
    )

    # Add gridlines
    ax.set_xticks(np.arange(-0.5, len(algo_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # --- Save Individual Heatmaps ---
    fig_ind, ax_ind = plt.subplots(figsize=(6, 5))
    im_ind = ax_ind.imshow(matrix, cmap=custom_cmap, norm=norm, aspect='equal')

    # Hatch + labels (same style)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch='////', edgecolor='gray', lw=0.0
                )
                ax_ind.add_patch(rect)
            else:
                ax_ind.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=8)

    # --- Axes formatting (same as combined) ---
    ax_ind.set_xticks(np.arange(len(algo_cols)))
    ax_ind.set_xticklabels(algo_cols, rotation=45, ha='left')
    ax_ind.xaxis.tick_top()
    ax_ind.xaxis.set_label_position("top")
    ax_ind.set_yticks(np.arange(len(row_labels)))
    ax_ind.set_yticklabels(row_labels)
    ax_ind.invert_yaxis()

    # Add protein label at side
    ax_ind.text(
        -1.7, matrix.shape[0] / 3.5,
        protein, fontsize=12, rotation=90, va='center', ha='center'
    )

    # Gridlines
    ax_ind.set_xticks(np.arange(-0.5, len(algo_cols), 1), minor=True)
    ax_ind.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax_ind.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax_ind.tick_params(which="minor", bottom=False, left=False)

    # Add shared colorbar (same scale as combined)
    cbar_ind = plt.colorbar(im_ind, ax=ax_ind, fraction=0.046, pad=0.04)
    cbar_ind.set_label("Algorithm Score", rotation=-90, va="bottom")

    tick_locs = np.linspace(vmin, vmax, 6)
    if 0 not in tick_locs:
        tick_locs = np.sort(np.append(tick_locs, 0))
    cbar_ind.set_ticks(tick_locs)
    cbar_ind.ax.set_yticklabels([f"{x:.1f}" for x in tick_locs])
    cbar_ind.ax.tick_params(size=4, labelsize=8)

    # Save file
    out_path = os.path.join(output_dir, f"heatmap_{protein.replace(' ', '_')}.png")
    fig_ind.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig_ind)

# --- Shared colorbar for combined figure ---
cbar = fig.colorbar(
    im, ax=axes, orientation="vertical",
    fraction=0.02, pad=0.02, shrink=0.8
)
cbar.set_label("Algorithm Score", rotation=-90, va="bottom")

tick_locs = np.linspace(vmin, vmax, 6)
if 0 not in tick_locs:
    tick_locs = np.sort(np.append(tick_locs, 0))
cbar.set_ticks(tick_locs)
cbar.ax.set_yticklabels([f"{x:.1f}" for x in tick_locs])
cbar.ax.tick_params(size=4, labelsize=8)

# --- Save combined figure ---
combined_path = os.path.join(output_dir, "heatmap_combined.png")
fig.savefig(combined_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved individual and combined heatmaps to: {os.path.abspath(output_dir)}")
