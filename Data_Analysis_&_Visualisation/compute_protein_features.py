import subprocess
import sys

REQUIRED_PACKAGES = ["biopython", "numpy"]

def install_missing(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"Failed to install {package}: {e}")
        sys.exit(1)

for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Package '{pkg}' not found. Installing...")
        install_missing(pkg)

from Bio import SeqIO
import numpy as np
import csv



# --- Configuration ---

POS_FASTA = "Virus_Positive_FINAL.fasta"
NEG_FASTA = "Virus_Negative_FINAL.fasta"
OUTPUT_FILE = "Virus_pos_neg_features.csv"



# --- Feature Calculation ---

# Kyte–Doolittle hydropathy scale
HYDROPATHY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7,
    'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
    'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5,
    'N': -3.5, 'K': -3.9, 'R': -4.5
}

POS_CHARGED = {'K', 'R', 'H'}
NEG_CHARGED = {'D', 'E'}
DISORDER_PROMOTING = {'G', 'S', 'P', 'E', 'D', 'K', 'R', 'Q'}
AROMATICS = {'Y', 'F', 'W'}


def calc_low_complexity_fraction(seq, window=12, complexity_threshold=2.2):
    """
    Approximate SEG low-complexity via Shannon entropy.
    Low complexity = entropy < threshold.
    """
    seq = str(seq)
    n = len(seq)
    if n < window:
        return 0.0

    low_positions = 0

    for i in range(n - window + 1):
        w = seq[i:i+window]
        freqs = {aa: w.count(aa) / window for aa in set(w)}
        entropy = -sum(p * np.log2(p) for p in freqs.values())

        if entropy < complexity_threshold:
            low_positions += window

    return min(low_positions, n) / n


def compute_features(fasta_file, label):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    results = []

    for rec in records:
        seq = str(rec.seq)
        seq_len = len(seq)
        if seq_len == 0:
            continue

        # Amino-acid composition
        comp = {aa: seq.count(aa) / seq_len for aa in HYDROPATHY.keys()}

        # Hydropathy
        mean_hydro = np.mean([HYDROPATHY.get(aa, 0) for aa in seq])

        # Charge
        pos = sum(seq.count(aa) for aa in POS_CHARGED)
        neg = sum(seq.count(aa) for aa in NEG_CHARGED)
        ncpr = (pos - neg) / seq_len

        # Disorder-promoting residues
        disorder_fraction = sum(seq.count(aa) for aa in DISORDER_PROMOTING) / seq_len

        # Aromatic residues
        aromatic_fraction = sum(seq.count(aa) for aa in AROMATICS) / seq_len

        # Low complexity
        lcr_fraction = calc_low_complexity_fraction(seq)

        results.append({
            "ID": rec.id,
            "Label": label,
            "Length": seq_len,
            "Hydropathy": mean_hydro,
            "NCPR": ncpr,
            "DisorderFraction": disorder_fraction,
            "AromaticFraction": aromatic_fraction,
            "LowComplexityFraction": lcr_fraction,
            **comp
        })

    return results



# --- Main Execution ---

if __name__ == "__main__":

    print("Computing features for LLPS-positive proteins...")
    pos_results = compute_features(POS_FASTA, "LLPS_Positive")

    print("Computing features for LLPS-negative proteins...")
    neg_results = compute_features(NEG_FASTA, "LLPS_Negative")

    all_results = pos_results + neg_results

    # Write CSV
    if len(all_results) == 0:
        print("No sequences processed. Check FASTA file paths.")
        sys.exit(1)

    fieldnames = list(all_results[0].keys())
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nFeature table successfully written to: {OUTPUT_FILE}")
    print("Done.")