import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from scipy.stats import linregress, entropy, ttest_ind

from utils import (
    load_and_preprocess,
    compute_fft_magnitude,
    radial_profile,
    ensure_dir
)

# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = "../data"
OUTPUT_DIR = "../outputs/statistics"

ensure_dir(OUTPUT_DIR)

classes = ["real", "gan", "diffusion"]

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 12

# -----------------------------
# Helper Functions
# -----------------------------

def compute_alpha(profile):
    """
    Fit power-law slope alpha from radial profile.
    """
    freq = np.arange(1, len(profile))
    energy = profile[1:]

    log_f = np.log(freq)
    log_e = np.log(energy)

    slope, _, _, _, _ = linregress(log_f, log_e)
    return -slope


def normalize_distribution(arr):
    arr = arr / (np.sum(arr) + 1e-8)
    return arr


# -----------------------------
# Extract Radial Profiles
# -----------------------------

class_profiles = {}
class_alphas = {}

for cls in classes:
    class_path = os.path.join(DATA_DIR, cls)
    profiles = []

    for img_name in sorted(os.listdir(class_path)):
        img_path = os.path.join(class_path, img_name)

        gray = load_and_preprocess(img_path)
        magnitude = compute_fft_magnitude(gray)
        profile = radial_profile(magnitude)

        profiles.append(profile)

    # Trim profiles to common length
    min_len = min(len(p) for p in profiles)
    profiles_trimmed = np.array([p[:min_len] for p in profiles])

    class_profiles[cls] = profiles_trimmed

    # Compute alpha per image
    alphas = np.array([compute_alpha(p) for p in profiles_trimmed])
    class_alphas[cls] = alphas

    print(f"{cls} mean alpha: {np.mean(alphas):.4f}")


# -----------------------------
# KL Divergence
# -----------------------------

mean_real = normalize_distribution(np.mean(class_profiles["real"], axis=0))
mean_gan = normalize_distribution(np.mean(class_profiles["gan"], axis=0))
mean_diff = normalize_distribution(np.mean(class_profiles["diffusion"], axis=0))

kl_real_gan = entropy(mean_real, mean_gan)
kl_real_diff = entropy(mean_real, mean_diff)

print(f"KL(Real || GAN): {kl_real_gan:.6f}")
print(f"KL(Real || Diffusion): {kl_real_diff:.6f}")


# -----------------------------
# Statistical Testing (Alpha)
# -----------------------------

t_gan = ttest_ind(class_alphas["real"], class_alphas["gan"])
t_diff = ttest_ind(class_alphas["real"], class_alphas["diffusion"])

print("T-test Real vs GAN:", t_gan)
print("T-test Real vs Diffusion:", t_diff)


# -----------------------------
# Publication-Grade Visualization
# -----------------------------

# 1️⃣ Alpha Distribution Boxplot

plt.figure(figsize=(6, 5))

data = [
    class_alphas["real"],
    class_alphas["gan"],
    class_alphas["diffusion"]
]

plt.boxplot(
    data,
    labels=["Real", "GAN", "Diffusion"],
    showmeans=True
)

plt.ylabel("Power-law Exponent (α)")
plt.title("Distribution of Spectral Decay Exponents")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "alpha_distribution.png"),
    dpi=300
)
plt.close()


# 2️⃣ KL Divergence Bar Plot

plt.figure(figsize=(6, 4))

kl_values = [kl_real_gan, kl_real_diff]
labels = ["Real vs GAN", "Real vs Diffusion"]

plt.bar(labels, kl_values)
plt.ylabel("KL Divergence")
plt.title("Spectral Distribution Divergence")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "kl_divergence.png"),
    dpi=300
)
plt.close()


# -----------------------------
# Results Table (CSV)
# -----------------------------

results = pd.DataFrame({
    "Class": ["Real", "GAN", "Diffusion"],
    "Mean Alpha": [
        np.mean(class_alphas["real"]),
        np.mean(class_alphas["gan"]),
        np.mean(class_alphas["diffusion"])
    ],
    "Std Alpha": [
        np.std(class_alphas["real"]),
        np.std(class_alphas["gan"]),
        np.std(class_alphas["diffusion"])
    ]
})

results.to_csv(os.path.join(OUTPUT_DIR, "spectral_statistics_table.csv"), index=False)

print("Statistical analysis complete.")
print("Outputs saved in:", OUTPUT_DIR)