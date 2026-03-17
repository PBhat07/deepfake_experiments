import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO_DIR = "../data/video/compressed"
ORIGINAL_VIDEO = "../data/video/original.mp4"
OUTPUT_DIR = "../outputs/compression_study"
FRAME_INDEX = 50

CRF_LEVELS = [18, 23, 28, 35, 40]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ---------- FFT ----------
def compute_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Normalize total spectral energy
    magnitude = magnitude / (np.sum(magnitude) + 1e-8)

    return magnitude

# ---------- Radial Profile ----------
def radial_profile(data):
    y, x = np.indices(data.shape)
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / (nr + 1e-8)

    return radialprofile

# ---------- High Frequency Energy ----------
def high_frequency_energy(radial, percentage=0.8):
    cutoff = int(len(radial) * percentage)
    return np.sum(radial[cutoff:])

# ---------- Directional Artifact Energy ----------
def directional_energy(magnitude):
    h, w = magnitude.shape
    center_y, center_x = h//2, w//2

    vertical_band = magnitude[:, center_x-5:center_x+5]
    horizontal_band = magnitude[center_y-5:center_y+5, :]

    return np.sum(vertical_band) + np.sum(horizontal_band)

def main():
    ensure_dir(OUTPUT_DIR)

    frames = []
    fft_images_display = []
    radial_curves = []
    hf_energies = []
    directional_scores = []

    titles = ["Original"] + [f"CRF {c}" for c in CRF_LEVELS]

    # ---------- ORIGINAL ----------
    original_frame = extract_frame(ORIGINAL_VIDEO, FRAME_INDEX)
    original_fft = compute_fft(original_frame)
    original_radial = radial_profile(original_fft)
    original_hf = high_frequency_energy(original_radial, 0.8)
    original_dir = directional_energy(original_fft)

    frames.append(original_frame)
    fft_images_display.append(20 * np.log(original_fft + 1e-8))
    radial_curves.append(original_radial)
    hf_energies.append(original_hf)
    directional_scores.append(original_dir)

    # ---------- COMPRESSED ----------
    for crf in CRF_LEVELS:
        path = os.path.join(VIDEO_DIR, f"crf_{crf}.mp4")
        frame = extract_frame(path, FRAME_INDEX)
        fft_img = compute_fft(frame)
        radial = radial_profile(fft_img)
        hf = high_frequency_energy(radial, 0.8)
        dir_energy = directional_energy(fft_img)

        frames.append(frame)
        fft_images_display.append(20 * np.log(fft_img + 1e-8))
        radial_curves.append(radial)
        hf_energies.append(hf)
        directional_scores.append(dir_energy)

    # ---------- QUALITATIVE GRID ----------
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    for i in range(6):
        axes[0, i].imshow(frames[i], cmap="gray")
        axes[0, i].set_title(titles[i])
        axes[0, i].axis("off")

        axes[1, i].imshow(fft_images_display[i], cmap="inferno")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qualitative_fft_grid.png"), dpi=300)
    plt.close()

    # ---------- RADIAL DISTRIBUTION ----------
    plt.figure(figsize=(8,6))
    for i, radial in enumerate(radial_curves):
        plt.plot(radial[:300], label=titles[i])

    plt.title("Normalized Radial Frequency Distribution")
    plt.xlabel("Frequency Radius")
    plt.ylabel("Normalized Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "radial_frequency_plot.png"), dpi=300)
    plt.close()

    # ---------- RELATIVE HIGH FREQUENCY ----------
    relative_hf = [hf / original_hf for hf in hf_energies]

    plt.figure(figsize=(8,6))
    plt.plot(titles, relative_hf, marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.title("Relative High-Frequency Energy vs Compression")
    plt.ylabel("HF Energy (Normalized to Original)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "relative_high_frequency_decay.png"), dpi=300)
    plt.close()

    # ---------- DIRECTIONAL ARTIFACT ENERGY ----------
    relative_dir = [d / original_dir for d in directional_scores]

    plt.figure(figsize=(8,6))
    plt.plot(titles, relative_dir, marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.title("Relative Axis-Aligned Energy (Block Artifact Indicator)")
    plt.ylabel("Axis Energy (Normalized to Original)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "directional_artifact_energy.png"), dpi=300)
    plt.close()

    print("All visualizations saved to outputs/compression_study/")

if __name__ == "__main__":
    main()