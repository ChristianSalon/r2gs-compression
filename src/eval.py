import argparse
from pathlib import Path
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity


def load_volume(path: Path) -> Any:
    with open(path, "rb") as file:
        return np.load(file)


def compute_quality_metrics(reconstructed: Any, ground_truth: Any) -> tuple[float, float, float, float]:
    recon = reconstructed.astype(np.float64)
    gt = ground_truth.astype(np.float64)

    # Check if volumes have same shape
    if recon.shape != gt.shape:
        raise ValueError("Reconstructed and GT volume don't have same shape")

    mse = np.mean((recon - gt) ** 2)

    # Avoid division (PSNR) and square root (RMSE) of zero
    if mse == 0:
        return 0.0, 0.0, float("inf"), 1.0

    rmse = np.sqrt(mse)

    data_range = recon.max() - recon.min()
    psnr = 20 * np.log10(data_range / rmse)

    ssim = structural_similarity(recon, gt, data_range=data_range, channel_axis=None)

    return mse, rmse, psnr, ssim


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(prog="R2GS Volume Evaluation")
    parser.add_argument("-i", "--input", type=str, required=True, help="Reconstructed volume (.npy)")
    parser.add_argument("-gt", "--ground-truth", type=str, required=True, help="GT volume (.npy)")

    args = parser.parse_args()

    # Load 3D volumes
    reconstructed_volume = load_volume(Path(args.input))
    gt_volume = load_volume(Path(args.ground_truth))

    # Print reconstructed volume
    print(f"Volume shape: {reconstructed_volume.shape}")
    print(reconstructed_volume)
    print()

    mse, rmse, psnr, ssim = compute_quality_metrics(reconstructed_volume, gt_volume)

    # Print volume reconstruction quality metrics
    print(f"3D Volume MSE: {mse:.6f}")
    print(f"3D Volume RMSE: {rmse:.6f}")
    print(f"3D Volume PSNR: {psnr:.2f} dB")
    print(f"3D Volume SSIM: {ssim:.4f}")


if __name__ == "__main__":
    main()
