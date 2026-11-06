import torch
from r2_gaussian.arguments import PipelineParams
from r2_gaussian.dataset import Scene
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.gaussian import render

from compaction.arguments import CompactionParams


class PUP:
    def __init__(self, scene: Scene, compaction_params: CompactionParams):
        self.scene = scene
        self.params = compaction_params

    def prune(self) -> None:
        print(f"Stared compaction using PUP")

        # Load training views (x-ray projections)
        train_cameras: list[Camera] = self.scene.getTrainCameras()
        assert isinstance(train_cameras, list[Camera]), "Training cameras have wrong type"
        print(f"Loaded {len(train_cameras)} training views")

        print(f"Stared computing hessian")
        hessian = self.compute_hessian(cameras=train_cameras)

        # Compute sensitivity score
        num_gaussians = hessian.shape[0]
        damping = 1e-6 * torch.eye(6, device="cuda").unsqueeze(0).repeat(num_gaussians, 1, 1)
        stabilized_hessian = hessian + damping
        _, log_det = torch.slogdet(stabilized_hessian)

        # Prune gaussians in place
        print(f"Started pruning gaussians")
        prune_mask = self.compute_prune_mask(sensitivity_score=log_det)
        self.scene.gaussians.prune_points(prune_mask)

        print(f"Done performing compaction using PUP")

    def compute_hessian(self, cameras: list[Camera]) -> torch.Tensor:
        num_gaussians = self.scene.gaussians.get_xyz.shape[0]
        fisher_blocks = torch.zeros((num_gaussians, 6, 6), device="cuda")

        for camera in cameras:
            # Ground truth x-ray projection for this camera pose
            gt_projection = camera.original_image.to("cuda").float()

            # Track gradients for spatial mean and scale
            xyz = self.scene.gaussians.get_xyz
            scale = self.scene.gaussians.get_scaling
            xyz.requires_grad_(True)
            scale.requires_grad_(True)

            # Render x-ray projection using R2 gaussian
            rendered_result = render(camera, self.scene.gaussians, PipelineParams())
            rendered_projection = rendered_result["render"]

            # Compute L2 loss
            loss = torch.nn.functional.mse_loss(rendered_projection, gt_projection)
            loss.backward()

            # Accumulate 6D spatial gradients
            grad_6d = torch.cat([xyz.grad, scale.grad], dim=-1)
            # [N, 6, 6] = [N, 6, 1] * [N, 1, 6]
            fisher_blocks += torch.bmm(grad_6d.unsqueeze(-1), grad_6d.unsqueeze(1))

            xyz.grad.zero_()
            scale.grad.zero_()

        return fisher_blocks

    def compute_prune_mask(self, sensitivity_score: torch.Tensor) -> None:
        num_gaussians = sensitivity_score.shape[0]

        keep_count = int((1.0 - self.params.prune_ratio) * num_gaussians)
        _, keep_indices = torch.topk(sensitivity_score, keep_count, largest=True)

        prune_mask = torch.ones(num_gaussians, dtype=torch.bool, device="cuda")
        prune_mask[keep_indices] = False

        return prune_mask
