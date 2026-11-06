import argparse
import os
import sys

sys.path.append(os.path("../thirdparty/r2_gaussian"))
from r2_gaussian.arguments import ModelParams
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, initialize_gaussian

from compaction.arguments import CompactionMethod, CompactionParams
from compaction.pup import PUP


def main() -> None:
    # Define training arguments
    parser = argparse.ArgumentParser(description="R2GS compaction and compression training script")
    model_params = ModelParams(parser)
    compaction_params = CompactionParams(parser)

    # Parse training arguments
    args = parser.parse_args()
    model_params.extract(args)
    compaction_params.extract(args)

    print(f"Started initializing R2GS scene")

    # Initialize scene with r2 gaussian
    scene = Scene(args=model_params, shuffle=False)
    gaussians = GaussianModel()
    # Load the maximum iteration of trained scene
    loaded_iteration = initialize_gaussian(gaussians, model_params, loaded_iter=-1)
    scene.gaussians = gaussians

    print(f"Started compressing scene")

    # Select compaction and compression methods
    if compaction_params.compaction_method == CompactionMethod.PUP:
        # Perform compaction
        PUP(scene=scene, compaction_params=compaction_params).prune()

    print(f"Done compressing scene")

    # Save compressed scene
    output_path = os.path.join(
        model_params.model_path,
        "point_cloud",
        "iteration_" + str(loaded_iteration),
        "point_cloud_compressed.pickle",
    )
    scene.gaussians.save_ply(output_path)


if __name__ == "__main__":
    main()
