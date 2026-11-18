import argparse
import json
import os
import subprocess
import sys
from typing import Any

import numpy as np
import xrmreader
from scipy.ndimage import zoom


def get_scanner_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    dso = float(abs(metadata["source_isocenter_distance"]))
    dsd = float(dso + abs(metadata["detector_isocenter_distance"]))

    # Convert pixel size from micrometers to millimeters
    pixel_size_mm = float(metadata["pixel_size"]) / 1000.0
    n_det = [int(metadata["image_width"]), int(metadata["image_height"])]
    s_det = [round(n_det[0] * pixel_size_mm, 4), round(n_det[1] * pixel_size_mm, 4)]

    # Calculate angular span from theta
    thetas_deg = np.degrees(metadata["thetas"]) % 360.0
    total_angle = float(round(thetas_deg.max() - thetas_deg.min(), 2))
    start_angle = float(round(thetas_deg.min(), 2))

    # Do not generate noise since actual projections already contain noise

    return {
        "mode": "cone",
        "filter": metadata.get("filter", None),
        "DSD": round(dsd, 4),
        "DSO": round(dso, 4),
        "nDetector": n_det,
        "sDetector": s_det,
        "nVoxel": [256, 256, 256],
        "sVoxel": [2.0, 2.0, 2.0],
        "offOrigin": [0, 0, 0],
        "offDetector": [0, 0],
        "accuracy": 0.5,
        "totalAngle": total_angle,
        "startAngle": start_angle,
        "noise": False,
    }


def run_initialize_pcd(data_dir: str) -> None:
    script_path = os.path.abspath("thirdparty/r2_gaussian/initialize_pcd.py")
    cmd = [sys.executable, script_path, "--data", data_dir]

    try:
        subprocess.run(cmd, check=True, text=True)
        print(f"Point cloud initialization completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: initialize_pcd.py failed with exit code {e.returncode}")


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(description="Initialize HDTomo dataset")
    parser.add_argument("-p", "--projections", type=str, required=True, help=".txrm file")
    parser.add_argument("-v", "--volume", type=str, required=True, help=".txm file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    parser.add_argument("--train-projections", type=int, default=50, help="Number of train projections")
    parser.add_argument("--test-projections", type=int, default=100, help="Number of test projections")
    args = parser.parse_args()

    metadata = xrmreader.read_metadata(args.projections)

    # Preprocess projection data in individual steps
    projections = xrmreader.read_txrm(args.projections)
    projections = xrmreader.divide_by_reference(projections, metadata["reference"])
    projections = xrmreader.revert_shifts(projections, metadata["x-shifts"], metadata["y-shifts"])
    projections = xrmreader.negative_logarithm(projections)
    print(f"Projections shape: {projections.shape}")

    # Read the 3D reconstructed volume (Z, Y, X)
    volume = xrmreader.read_txrm(args.volume)

    # Read the corresponding angles in radians
    angles = metadata["thetas"]
    normalized_angles = angles % (2 * np.pi)

    # Sort projections based on angle
    angle_sorted_indices = np.argsort(normalized_angles)
    sorted_angles = normalized_angles[angle_sorted_indices]
    sorted_projections = projections[angle_sorted_indices, :, :]

    # Initialize train projections
    train_indices = np.linspace(0, len(sorted_angles) - 1, args.train_projections, dtype=int)
    train_angles = sorted_angles[train_indices]
    train_projections = sorted_projections[train_indices, :, :]

    # Remove train projections before selecting test projections
    sorted_angles = np.delete(sorted_angles, train_indices)
    sorted_projections = np.delete(sorted_projections, train_indices, axis=0)

    # Initialize test projections
    test_indices = np.linspace(0, len(sorted_angles) - 1, args.test_projections, dtype=int)
    test_angles = sorted_angles[test_indices]
    test_projections = sorted_projections[test_indices, :, :]

    # Normalize volume to [0, 1]
    v_min = volume.min()
    v_max = volume.max()
    volume_normalized = (volume - v_min) / (v_max - v_min + 1e-9)

    # Resize volume to 256 x 256 x 256
    target_shape = (256, 256, 256)
    zoom_factors = [t / s for t, s in zip(target_shape, volume_normalized.shape)]
    volume_resized = zoom(volume_normalized, zoom_factors, order=1)
    volume_resized = np.clip(volume_resized, 0.0, 1.0)

    os.makedirs(args.output, exist_ok=True)

    # Save train projections
    train_proj_out_dir = os.path.join(args.output, "proj_train")
    os.makedirs(train_proj_out_dir, exist_ok=True)
    for i, proj in enumerate(train_projections):
        proj_out_path = os.path.join(train_proj_out_dir, f"proj_train_{i}.npy")
        np.save(proj_out_path, proj.astype(np.float32))
    print(f"Saved training projections with shape {train_projections.shape} to {train_proj_out_dir}")

    # Save test projections
    test_proj_out_dir = os.path.join(args.output, "proj_test")
    os.makedirs(test_proj_out_dir, exist_ok=True)
    for i, proj in enumerate(test_projections):
        proj_out_path = os.path.join(test_proj_out_dir, f"proj_test_{i}.npy")
        np.save(proj_out_path, proj.astype(np.float32))
    print(f"Saved testing projections with shape {test_projections.shape} to {test_proj_out_dir}")

    # Save ground truth volume
    vol_out_path = os.path.join(args.output, "vol_gt.npy")
    np.save(vol_out_path, volume_resized.astype(np.float32))
    print(f"Saved GT 3D volume with shape {volume_resized.shape} to {vol_out_path}")

    r2_metadata = {
        "scanner": get_scanner_metadata(metadata=metadata),
        "bbox": [[-1, -1, -1], [1, 1, 1]],
        "proj_train": [
            {"file_path": f"proj_train/proj_train_{i}.npy", "angle": float(a)} for i, a in enumerate(train_angles)
        ],
        "proj_test": [
            {"file_path": f"proj_test/proj_test_{i}.npy", "angle": float(a)} for i, a in enumerate(test_angles)
        ],
        "vol": "vol_gt.npy",
    }

    # Save meta_data.json
    meta_data_path = os.path.join(args.output, "meta_data.json")
    with open(meta_data_path, "w", encoding="utf-8") as file:
        json.dump(r2_metadata, file, indent=2)
    print(f"Saved metadata to {meta_data_path}")

    # Initialize point cloud with FDK
    run_initialize_pcd(data_dir=args.output)
    print(f"Saved initial point cloud to {args.output} directory")


if __name__ == "__main__":
    main()
