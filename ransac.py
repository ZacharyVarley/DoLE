import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from kornia.geometry import (
    find_fundamental,
    find_homography_dlt,
    find_homography_dlt_iterated,
    symmetrical_epipolar_distance,
)
from kornia.geometry.homography import symmetric_transfer_error

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE

from kornia.geometry.epipolar import normalize_points
from kornia.utils import safe_solve_with_mask

TupleTensor = Tuple[Tensor, Tensor]


def find_rigid(
        points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Compute the rigid transformation matrix using Procrustes Analysis.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed rigid transformation matrix with shape :math:`(B, 3, 3)`.
    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    if points1.shape[1] < 2:
        raise AssertionError(points1.shape)

    KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "N", "2"])

    # center the points
    points1_centroid = torch.mean(points1, dim=1, keepdim=True)
    points2_centroid = torch.mean(points2, dim=1, keepdim=True)

    points1_centered = points1 - points1_centroid
    points2_centered = points2 - points2_centroid

    # apply optional weighting
    if weights is not None:
        if not (len(weights.shape) == 2 and weights.shape == points1.shape[:2]):
            raise AssertionError(weights.shape)
        points2_centered = points2_centered * weights.unsqueeze(-1)

    # compute the covariance matrix
    H = torch.matmul(points1_centered.transpose(-2, -1), points2_centered)

    # SVD to compute rotation
    U, _, V = torch.svd(H)
    R = torch.matmul(V, U.transpose(-2, -1))

    # Correct reflection case
    det_R = torch.det(R)
    V[det_R < 0, :, -1] *= -1
    R = torch.matmul(V, U.transpose(-2, -1))

    # compute translation
    t = points2_centroid.squeeze() - (torch.bmm(R, points1_centroid.transpose(-2, -1)).squeeze(-1))

    # create 3x3 transformation matrix
    transform = torch.eye(3).expand(points1.shape[0], -1, -1).to(points1.device)
    transform[..., 0:2, 0:2] = R
    transform[..., 0:2, 2] = t

    return transform


def find_rigid_iterated(points1: Tensor, points2: Tensor, weights: Tensor, soft_inl_th: float = 3.0,
                        n_iter: int = 5) -> Tensor:
    r"""Compute the rigid matrix.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
        soft_inl_th: Soft inlier threshold used for weight calculation.
        n_iter: number of iterations.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    H: Tensor = find_rigid(points1, points2, weights)
    for _ in range(n_iter - 1):
        errors: Tensor = symmetric_transfer_error(points1, points2, H, False)
        weights_new: Tensor = torch.exp(-errors / (2.0 * (soft_inl_th ** 2)))
        H = find_rigid(points1, points2, weights_new)
    return H


def find_affine(
        points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None, solver: str = 'lu'
) -> torch.Tensor:
    r"""Compute the affine matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 3 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
        solver: variants: svd, lu.


    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    if points1.shape[1] < 3:
        raise AssertionError(points1.shape)
    KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
    KORNIA_CHECK_SHAPE(points2, ["B", "N", "2"])

    device, dtype = points1.device, points1.dtype

    eps: float = 1e-8
    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2], dim=-1)
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2], dim=-1)
    A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

    if weights is None:
        # All points are equally important
        A = A.transpose(-2, -1) @ A
    else:
        # We should use provided weights
        if not (len(weights.shape) == 2 and weights.shape == points1.shape[:2]):
            raise AssertionError(weights.shape)
        w_diag = torch.diag_embed(weights.unsqueeze(dim=-1).repeat(1, 1, 2).reshape(weights.shape[0], -1))
        A = A.transpose(-2, -1) @ w_diag @ A

    if solver == 'svd':
        try:
            _, _, V = torch.linalg.svd(A)
        except RuntimeError:
            return torch.empty((points1_norm.size(0), 2, 3), device=device, dtype=dtype)
        H = V[..., -1].view(-1, 2, 3)
    elif solver == 'lu':
        B = torch.ones(A.shape[0], A.shape[1], device=device, dtype=dtype)
        sol, _, _ = safe_solve_with_mask(B, A)
        H = sol.reshape(-1, 7)
    else:
        raise NotImplementedError
    # concatenate to make affine into homography
    homography = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).repeat(H.shape[0], 1, 1)
    homography[..., :2, :] = H[..., :-1].reshape(-1, 2, 3)
    homography[..., 2, 2] = H[..., -1]
    homography = transform2.inverse() @ (homography @ transform1)
    homography_norm = homography / (homography[..., -1:, -1:] + eps)
    return homography_norm


def find_affine_iterated(points1: Tensor, points2: Tensor, weights: Tensor, soft_inl_th: float = 3.0,
                         n_iter: int = 5) -> Tensor:
    r"""Compute the affine matrix using the direct linear transform (DLT) formulation.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
        soft_inl_th: Soft inlier threshold used for weight calculation.
        n_iter: number of iterations.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    H: Tensor = find_affine(points1, points2, weights)
    for _ in range(n_iter - 1):
        errors: Tensor = symmetric_transfer_error(points1, points2, H, False)
        weights_new: Tensor = torch.exp(-errors / (2.0 * (soft_inl_th ** 2)))
        H = find_affine(points1, points2, weights_new)
    return H


class RANSAC(nn.Module):
    """Module for robust geometry estimation with RANSAC.

    https://en.wikipedia.org/wiki/Random_sample_consensus

    Args:
        model_type: type of model to estimate, e.g. "homography" or "fundamental".
        inliers_threshold: threshold for the correspondence to be an inlier.
        batch_size: number of generated samples at once.
        max_iterations: maximum batches to generate. Actual number of models to try is ``batch_size * max_iterations``.
        confidence: desired confidence of the result, used for the early stopping.
        max_local_iterations: number of local optimization (polishing) iterations.
    """
    supported_models = ['rigid', 'affine', 'homography', 'fundamental']

    def __init__(self,
                 model_type: str = 'homography',
                 inl_th: float = 2.0,
                 batch_size: int = 2048,
                 max_iter: int = 100,
                 confidence: float = 0.999,
                 max_lo_iters: int = 5):
        super().__init__()
        self.inl_th = inl_th
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.model_type = model_type
        self.confidence = confidence
        self.max_lo_iters = max_lo_iters
        if model_type == 'rigid':
            self.error_fn = symmetric_transfer_error  # type: ignore
            self.minimal_solver = find_rigid  # type: ignore
            self.polisher_solver = find_rigid_iterated  # type: ignore
            self.minimal_sample_size = 2
        elif model_type == 'affine':
            self.error_fn = symmetric_transfer_error  # type: ignore
            self.minimal_solver = find_affine  # type: ignore
            self.polisher_solver = find_affine_iterated  # type: ignore
            self.minimal_sample_size = 3
        elif model_type == 'homography':
            self.error_fn = symmetric_transfer_error  # type: ignore
            self.minimal_solver = find_homography_dlt  # type: ignore
            self.polisher_solver = find_homography_dlt_iterated  # type: ignore
            self.minimal_sample_size = 4
        elif model_type == 'fundamental':
            self.error_fn = symmetrical_epipolar_distance  # type: ignore
            self.minimal_solver = find_fundamental  # type: ignore
            self.minimal_sample_size = 8
            self.polisher_solver = find_fundamental  # type: ignore
        else:
            raise NotImplementedError(f"{model_type} is unknown. Try one of {self.supported_models}")

    def sample(self,
               sample_size: int,
               pop_size: int,
               batch_size: int,
               device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Minimal sampler, but unlike traditional RANSAC we sample in batches to get benefit of the parallel
        processing, esp. on GPU
        """
        rand = torch.rand(batch_size, pop_size, device=device)
        _, out = rand.topk(k=sample_size, dim=1)
        return out

    @staticmethod
    def max_samples_by_conf(n_inl: int, num_tc: int, sample_size: int, conf: float) -> float:
        """Formula to update max_iter in order to stop iterations earlier
        https://en.wikipedia.org/wiki/Random_sample_consensus."""
        if n_inl == num_tc:
            return 1.0
        return math.log(1.0 - conf) / math.log(1. - math.pow(n_inl / num_tc, sample_size))

    def estimate_model_from_minsample(self,
                                      kp1: torch.Tensor,
                                      kp2: torch.Tensor) -> torch.Tensor:
        batch_size, sample_size = kp1.shape[:2]
        H = self.minimal_solver(kp1,
                                kp2,
                                torch.ones(batch_size,
                                           sample_size,
                                           dtype=kp1.dtype,
                                           device=kp1.device))
        if self.model_type == 'affine':
            # set perspective change to 0
            H[:, 2, 2] = 1.0
            H[:, 2, :2] = 0.0
        return H

    def verify(self,
               kp1: torch.Tensor,
               kp2: torch.Tensor,
               models: torch.Tensor, inl_th: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if len(kp1.shape) == 2:
            kp1 = kp1[None]
        if len(kp2.shape) == 2:
            kp2 = kp2[None]
        batch_size = models.shape[0]
        errors = self.error_fn(kp1.expand(batch_size, -1, 2),
                               kp2.expand(batch_size, -1, 2),
                               models)
        inl = (errors <= inl_th)
        models_score = inl.to(kp1).sum(dim=1)
        best_model_idx = models_score.argmax()
        best_model_score = models_score[best_model_idx].item()
        model_best = models[best_model_idx].clone()
        inliers_best = inl[best_model_idx]
        return model_best, inliers_best, best_model_score

    def remove_bad_samples(self, kp1: torch.Tensor, kp2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        # E.g. constraints on not to be a degenerate sample
        return kp1, kp2

    def remove_bad_models(self, models: torch.Tensor) -> torch.Tensor:
        # For now it is simple and hardcoded
        main_diagonal = torch.diagonal(models,
                                       dim1=1,
                                       dim2=2)
        mask = main_diagonal.abs().min(dim=1)[0] > 1e-4
        return models[mask]

    def polish_model(self,
                     kp1: torch.Tensor,
                     kp2: torch.Tensor,
                     inliers: torch.Tensor) -> torch.Tensor:
        kp1_inl = kp1[inliers][None]
        kp2_inl = kp2[inliers][None]
        num_inl = kp1_inl.size(1)
        model = self.polisher_solver(kp1_inl,
                                     kp2_inl,
                                     torch.ones(1,
                                                num_inl,
                                                dtype=kp1_inl.dtype,
                                                device=kp1_inl.device))
        return model

    def forward(self,
                kp1: torch.Tensor,
                kp2: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Main forward method to execute the RANSAC algorithm.

        Args:
            kp1 (torch.Tensor): source image keypoints :math:`(N, 2)`.
            kp2 (torch.Tensor): distance image keypoints :math:`(N, 2)`.
            weights (torch.Tensor): optional correspondences weights. Not used now

        Returns:
            - Estimated model, shape of :math:`(1, 3, 3)`.
            - The inlier/outlier mask, shape of :math:`(1, N)`, where N is number of input correspondences.
            """
        if not isinstance(kp1, torch.Tensor):
            raise TypeError(f"Input kp1 is not torch.Tensor. Got {type(kp1)}")
        if not isinstance(kp2, torch.Tensor):
            raise TypeError(f"Input kp2 is not torch.Tensor. Got {type(kp2)}")
        if not len(kp1.shape) == 2:
            raise ValueError(f"Invalid kp1 shape, we expect Nx2 Got: {kp1.shape}")
        if not len(kp2.shape) == 2:
            raise ValueError(f"Invalid kp2 shape, we expect Nx2 Got: {kp2.shape}")
        if not (kp1.shape[0] == kp2.shape[0]) or (kp1.shape[0] < self.minimal_sample_size):
            raise ValueError(f"kp1 and kp2 should be \
                             equal shape at at least [{self.minimal_sample_size}, 2], \
                             got {kp1.shape}, {kp2.shape}")

        best_score_total: float = float(self.minimal_sample_size)
        num_tc: int = len(kp1)
        best_model_total = torch.zeros(3, 3, dtype=kp1.dtype, device=kp1.device)
        inliers_best_total: torch.Tensor = torch.zeros(num_tc, 1, device=kp1.device, dtype=torch.bool)
        for i in range(self.max_iter):
            # Sample minimal samples in batch to estimate models
            idxs = self.sample(self.minimal_sample_size, num_tc, self.batch_size, kp1.device)
            kp1_sampled = kp1[idxs]
            kp2_sampled = kp2[idxs]

            kp1_sampled, kp2_sampled = self.remove_bad_samples(kp1_sampled, kp2_sampled)
            # Estimate models
            models = self.estimate_model_from_minsample(kp1_sampled, kp2_sampled)
            models = self.remove_bad_models(models)
            if (models is None) or (len(models) == 0):
                continue
            # Score the models and select the best one
            model, inliers, model_score = self.verify(kp1, kp2, models, self.inl_th)
            # Store far-the-best model and (optionally) do a local optimization
            if model_score > best_score_total:
                # Local optimization
                for lo_step in range(self.max_lo_iters):
                    model_lo = self.polish_model(kp1, kp2, inliers)
                    if (model_lo is None) or (len(model_lo) == 0):
                        continue
                    _, inliers_lo, score_lo = self.verify(kp1, kp2, model_lo, self.inl_th)
                    # print (f"Orig score = {best_model_score}, LO score = {score_lo} TC={num_tc}")
                    if score_lo > model_score:
                        model = model_lo.clone()[0]
                        inliers = inliers_lo.clone()
                        model_score = score_lo
                    else:
                        break
                # Now storing the best model
                best_model_total = model.clone()
                inliers_best_total = inliers.clone()
                best_score_total = model_score

                # Should we already stop?
                new_max_iter = int(self.max_samples_by_conf(int(best_score_total),
                                                            num_tc,
                                                            self.minimal_sample_size,
                                                            self.confidence))
                # print (f"New max_iter = {new_max_iter}")
                # Stop estimation, if the model is very good
                if (i + 1) * self.batch_size >= new_max_iter:
                    break
        # local optimization with all inliers for better precision
        return best_model_total, inliers_best_total
