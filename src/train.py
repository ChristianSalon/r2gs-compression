import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath("../thirdparty/r2_gaussian"))
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, initialize_gaussian, query
from r2_gaussian.utils.general_utils import t2a

from compaction.arguments import CompactionMethod, CompactionParams
from compaction.pup import PUP


def main() -> None:
    # Define training arguments
    parser = argparse.ArgumentParser(description="R2GS compaction and compression training script")
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    compaction_params = CompactionParams(parser)

    # Parse training arguments
    args = parser.parse_args()
    model_params = model_params.extract(args)
    pipeline_params = pipeline_params.extract(args)
    optimization_params = optimization_params.extract(args)
    compaction_params = compaction_params.extract(args)

    print(f"Started initializing R2GS scene")

    # Initialize scene with r2 gaussian
    scene = Scene(args=model_params, shuffle=False)

    volume_to_world = max(scene.scanner_cfg["sVoxel"])
    scale_bound = None
    if model_params.scale_min > 0 and model_params.scale_max > 0:
        scale_bound = np.array([model_params.scale_min, model_params.scale_max]) * volume_to_world

    gaussians = GaussianModel(scale_bound=scale_bound)
    # Load the maximum iteration of trained scene
    loaded_iteration = initialize_gaussian(gaussians, model_params, loaded_iter=-1)
    # Restore the spatial learning-rate scale
    # TODO: Load checkpoint and call restore() or leave as is
    gaussians.spatial_lr_scale = 1.0
    scene.gaussians = gaussians
    scene.gaussians.training_setup(optimization_params)

    print(f"Started compressing scene")

    # Select compaction and compression methods
    if compaction_params.compaction_method == CompactionMethod.PUP.value:
        # Perform compaction
        PUP(
            scene=scene,
            model_params=model_params,
            pipeline_params=pipeline_params,
            optimization_params=optimization_params,
            compaction_params=compaction_params,
        ).run()

    print(f"Done compressing scene")

    # Write compressed scene
    output_dir = os.path.join(
        model_params.model_path,
        "point_cloud",
        "iteration_" + str(loaded_iteration),
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save compressed gaussians
    gaussians_path = os.path.join(output_dir, compaction_params.scene_file_name)
    scene.gaussians.save_ply(gaussians_path)

    # Save compressed volume
    compressed_volume = query(
        scene.gaussians,
        scene.scanner_cfg["offOrigin"],
        scene.scanner_cfg["nVoxel"],
        scene.scanner_cfg["sVoxel"],
        pipeline_params,
    )["vol"]
    np.save(os.path.join(output_dir, compaction_params.volume_file_name), t2a(compressed_volume))


if __name__ == "__main__":
    main()
