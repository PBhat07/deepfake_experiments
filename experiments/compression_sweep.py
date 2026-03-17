import os
import subprocess

INPUT_VIDEO = "../data/video/original.mp4"
OUTPUT_DIR = "../data/video/compressed"
CRF_LEVELS = [18, 23, 28, 35, 40]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compress_video(crf):
    output_path = os.path.join(OUTPUT_DIR, f"crf_{crf}.mp4")

    command = [
        "ffmpeg",
        "-y",
        "-i", INPUT_VIDEO,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        output_path
    ]

    subprocess.run(command, check=True)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)

    print("Starting H.264 Compression Sweep...\n")
    for crf in CRF_LEVELS:
        compress_video(crf)

    print("\nCompression sweep complete.")