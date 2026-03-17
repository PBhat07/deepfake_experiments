import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import (
    load_and_preprocess,
    compute_fft_magnitude,
    radial_profile,
    normalize_spectrum,
    ensure_dir
)

# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = "../data"
OUTPUT_PLOTS = "../outputs/plots"

ensure_dir(OUTPUT_PLOTS)

classes = ["real", "gan", "diffusion"]

# Publication-style font settings
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 12

# -----------------------------
# Containers
# -----------------------------

class_radial_profiles = {}
class_spectra_images = {cls: [] for cls in classes}

# -----------------------------
# Processing
# -----------------------------

for cls in classes:
    class_path = os.path.join(DATA_DIR, cls)
    profiles = []

    for img_name in sorted(os.listdir(class_path)):
        img_path = os.path.join(class_path, img_name)

        gray = load_and_preprocess(img_path)
        magnitude = compute_fft_magnitude(gray)

        # Store normalized spectrum for visualization
        class_spectra_images[cls].append(normalize_spectrum(magnitude))

        # Radial profile (use non-normalized for statistics)
        profile = radial_profile(magnitude)
        profiles.append(profile)

    # Trim to smallest length
    min_len = min(len(p) for p in profiles)
    profiles_trimmed = [p[:min_len] for p in profiles]
    mean_profile = np.mean(profiles_trimmed, axis=0)

    class_radial_profiles[cls] = mean_profile

# -----------------------------
# 1️⃣ Publication-Quality Spectrum Grid
# -----------------------------

min_images = min(len(class_spectra_images[cls]) for cls in classes)

fig, axes = plt.subplots(
    min_images,
    len(classes),
    figsize=(10, 3 * min_images),
    constrained_layout=True
)

for col, cls in enumerate(classes):
    for row in range(min_images):

        spectrum = class_spectra_images[cls][row]

        ax = axes[row, col] if min_images > 1 else axes[col]
        im = ax.imshow(spectrum, cmap="inferno")
        ax.axis("off")

        if row == 0:
            ax.set_title(cls.capitalize(), fontsize=14, fontweight="bold")

# Shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
cbar.set_label("Normalized Log-Magnitude")

plt.savefig(
    os.path.join(OUTPUT_PLOTS, "spectrum_grid_publication.png"),
    dpi=300
)
plt.close()

print("Spectrum grid saved.")

# -----------------------------
# 2️⃣ Radial Energy Comparison Plot
# -----------------------------

plt.figure(figsize=(8, 5))

for cls in classes:
    plt.semilogy(
        class_radial_profiles[cls],
        label=cls.capitalize(),
        linewidth=2
    )

plt.xlabel("Frequency Radius (f)")
plt.ylabel("Spectral Energy E(f)")
plt.title("Radial Frequency Energy Distribution")

# Add theoretical power-law equation
plt.text(
    0.60, 0.85,
    r"$E(f) \propto \frac{1}{f^{\alpha}}$",
    transform=plt.gca().transAxes,
    fontsize=14,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
)

plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_PLOTS, "radial_comparison_publication.png"),
    dpi=300
)
plt.close()

print("Radial comparison plot saved.")
print("Frequency analysis complete.")