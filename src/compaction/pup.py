import random

import torch
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.dataset import Scene
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.gaussian import query, render
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss

from compaction.arguments import CompactionParams


class PUP:

    REFINEMENT_ITERATIONS = 5000
    REFINEMENT_LOG_INTERVAL = 1000

    def __init__(
        self,
        scene: Scene,
        model_params: ModelParams,
        pipeline_params: PipelineParams,
        optimization_params: OptimizationParams,
        compaction_params: CompactionParams,
    ):
        self.scene = scene
        self.model_params = model_params
        self.pipeline_params = pipeline_params
        self.optimization_params = optimization_params
        self.compaction_params = compaction_params

        self.global_refinement_iteration = 0

    def run(self) -> None:
        print(f"Started PUP pipeline")

        for iteration in range(len(self.compaction_params.prune_ratios)):
            self.prune(iteration=iteration)

            if self.compaction_params.refine:
                self.refine(iteration=iteration)

            print(f"Done with {iteration} iteration of PUP pipeline")

        print(f"Done running PUP pipeline")

    def prune(self, iteration: int) -> None:
        print(f"Started pruning using PUP")

        # Load training views (x-ray projections)
        train_cameras: list[Camera] = self.scene.getTrainCameras()
        assert isinstance(train_cameras, list), "Training cameras have wrong type"
        print(f"Loaded {len(train_cameras)} training views")

        print(f"Started computing hessian")
        hessian = self.compute_hessian(cameras=train_cameras, iteration=iteration)

        # Compute sensitivity score
        num_gaussians = hessian.shape[0]
        damping = 1e-6 * torch.eye(6, device="cuda").unsqueeze(0).repeat(num_gaussians, 1, 1)
        stabilized_hessian = hessian + damping
        _, log_det = torch.slogdet(stabilized_hessian)

        # Prune gaussians in place
        print(f"Started pruning gaussians")
        prune_mask = self.compute_prune_mask(sensitivity_score=log_det, iteration=iteration)
        self.scene.gaussians.prune_points(prune_mask)

        print(f"Done performing pruning using PUP")
        print(f"Total number of gaussians: {self.scene.gaussians.get_xyz.shape[0]}")

    def compute_hessian(self, cameras: list[Camera], iteration: int) -> torch.Tensor:
        num_gaussians = self.scene.gaussians.get_xyz.shape[0]
        fisher_blocks = torch.zeros((num_gaussians, 6, 6), device="cuda")

        for camera in cameras:
            self.scene.gaussians.optimizer.zero_grad(set_to_none=True)

            # Ground truth x-ray projection for this camera pose
            gt_projection = camera.original_image.to("cuda").float()

            # Render x-ray projection using R2 gaussian
            rendered_result = render(camera, self.scene.gaussians, self.pipeline_params)
            rendered_projection = rendered_result["render"]

            # Downsample projections 16x (creates 4x4 patches)
            gt_downsampled = torch.nn.functional.interpolate(
                gt_projection.unsqueeze(0), scale_factor=0.0625, mode="bilinear", align_corners=False
            ).squeeze(0)
            rendered_downsampled = torch.nn.functional.interpolate(
                rendered_projection.unsqueeze(0), scale_factor=0.0625, mode="bilinear", align_corners=False
            ).squeeze(0)

            # Compute L2 loss
            loss = torch.nn.functional.mse_loss(rendered_downsampled, gt_downsampled)
            loss.backward()

            # Accumulate 6D spatial gradients
            grad_6d = torch.cat([self.scene.gaussians._xyz.grad, self.scene.gaussians._scaling.grad], dim=-1)
            # [N, 6, 6] = [N, 6, 1] * [N, 1, 6]
            fisher_blocks += torch.bmm(grad_6d.unsqueeze(-1), grad_6d.unsqueeze(1))

            # Free up memory
            self.scene.gaussians._xyz.grad = None
            self.scene.gaussians._scaling.grad = None
            del rendered_result, rendered_projection, gt_projection, gt_downsampled, rendered_downsampled, loss, grad_6d
            torch.cuda.empty_cache()

        self.scene.gaussians.optimizer.zero_grad(set_to_none=True)

        return fisher_blocks

    def compute_prune_mask(self, sensitivity_score: torch.Tensor, iteration: int) -> torch.Tensor:
        num_gaussians = sensitivity_score.shape[0]

        keep_count = int((1.0 - self.compaction_params.prune_ratios[iteration]) * num_gaussians)
        _, keep_indices = torch.topk(sensitivity_score, keep_count, largest=True)

        prune_mask = torch.ones(num_gaussians, dtype=torch.bool, device="cpu")
        prune_mask[keep_indices] = False

        return prune_mask

    def refine(self, iteration: int) -> None:
        print(f"Started refinement of gaussians")

        # Set up some parameters
        scanner_cfg = self.scene.scanner_cfg
        bbox = self.scene.bbox

        # Set up loss
        use_tv = self.optimization_params.lambda_tv > 0
        if use_tv:
            print("Use total variation loss")
            tv_vol_size = self.optimization_params.tv_vol_size
            tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
            tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

        # Train
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        viewpoint_stack = None
        for refine_iteration in range(self.REFINEMENT_ITERATIONS):
            iter_start.record()

            # Update learning rate
            self.global_refinement_iteration += 1
            self.scene.gaussians.update_learning_rate(self.global_refinement_iteration)

            # Get one camera for training
            if not viewpoint_stack:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))

            # Render X-ray projection
            render_pkg = render(viewpoint_cam, self.scene.gaussians, self.pipeline_params)
            image, _, _, _ = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            # Compute loss
            gt_image = viewpoint_cam.original_image.cuda()
            loss = {"total": 0.0}
            render_loss = l1_loss(image, gt_image)
            loss["render"] = render_loss
            loss["total"] += loss["render"]
            if self.optimization_params.lambda_dssim > 0:
                loss_dssim = 1.0 - ssim(image, gt_image)
                loss["dssim"] = loss_dssim
                loss["total"] = loss["total"] + self.optimization_params.lambda_dssim * loss_dssim
            # 3D TV loss
            if use_tv:
                # Randomly get the tiny volume center
                tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (bbox[1] - tv_vol_sVoxel - bbox[0]) * torch.rand(3)
                vol_pred = query(
                    self.scene.gaussians,
                    tv_vol_center,
                    tv_vol_nVoxel,
                    tv_vol_sVoxel,
                    self.pipeline_params,
                )["vol"]
                loss_tv = tv_3d_loss(vol_pred, reduction="mean")
                loss["tv"] = loss_tv
                loss["total"] = loss["total"] + self.optimization_params.lambda_tv * loss_tv

            loss["total"].backward()

            iter_end.record()
            torch.cuda.synchronize()

            # Optimization
            with torch.no_grad():
                self.scene.gaussians.optimizer.step()
                self.scene.gaussians.optimizer.zero_grad(set_to_none=True)

            if (refine_iteration + 1) % self.REFINEMENT_LOG_INTERVAL == 0:
                print(
                    f"Refinement round {iteration + 1}: "
                    f"iteration {refine_iteration + 1}/"
                    f"{self.REFINEMENT_ITERATIONS}, "
                    f"loss={loss['total'].item():.6f}"
                )

        print(f"Done performing refinement of gaussians")
