from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.fft import rfft2, fftshift, irfft2
import kornia
from kornia.core import Tensor
from kornia.core import concatenate
from kornia.feature.laf import denormalize_laf, laf_is_inside_image, normalize_laf
from ransac import RANSAC


class EMIScalePyramid(nn.Module):
    def __init__(self, n_levels: int = 3, init_radius: float = 1.6, min_size: int = 15, radius_step: float = 1.6,
                 multiplicative: bool = False, bins: int = 8, mode: str = 'box'):
        super().__init__()
        self.extra_levels: int = 3
        self.n_levels = n_levels
        self.init_radius = init_radius
        self.min_size = min_size
        self.radius_step = radius_step
        self.multiplicative = multiplicative
        self.bins = bins
        self.mode = mode

        if self.multiplicative:
            assert math.sqrt(2) * init_radius * radius_step ** (n_levels - 1) < min_size
        else:
            assert math.sqrt(2) * (init_radius + radius_step * (n_levels - 1)) < min_size

    def __repr__(self) -> str:
        return (
                self.__class__.__name__
                + '(n_levels='
                + str(self.n_levels)
                + ', '
                + 'init_radius='
                + str(self.init_radius)
                + ', '
                + 'min_size='
                + str(self.min_size)
                + ', '
                + 'radius_step='
                + str(self.radius_step)
                + ')'
        )

    @staticmethod
    def circular_masks_like(img_shape, k_tensor):
        i = torch.arange(img_shape[-2], device=k_tensor.device) - img_shape[-2] // 2
        j = torch.arange(img_shape[-1], device=k_tensor.device) - img_shape[-1] // 2
        kk, ii, jj = torch.meshgrid(k_tensor.float(), i.float(), j.float())
        circular_mask = ((ii ** 2 + jj ** 2) <= kk ** 2).float()
        # circular_mask = torch.exp(-0.5 * (ii**2 + jj**2) / (kk**2))
        # circular_mask = circular_mask * ((ii**2 + jj**2) <= kk**2).float()
        return circular_mask

    @staticmethod
    def odd_pad(tensor, num_pix):
        # ensure resultant tensor is odd length in both image dimensions
        wild_card = len(tensor.shape) - 2
        h, w = tensor.shape[-2:]
        w_is_even = ((w + (2 * num_pix)) % 2 == 0)
        h_is_even = ((h + (2 * num_pix)) % 2 == 0)
        # padding arguments are ordered last dimension to first
        padding = (num_pix, num_pix + w_is_even, num_pix, num_pix + h_is_even) + (0, 0,) * wild_card
        padded = F.pad(tensor, padding, 'constant', 0.0)
        return padded

    def windowed_entropy(self, A, ksizes):

        pad = int(ksizes.max().item())

        level_sets = (A * self.bins).byte().repeat(1, self.bins, 1, 1) == torch.arange(self.bins, dtype=torch.uint8,
                                                                                       device=A.device)[None, :, None,
                                                                          None]
        level_sets = torch.cat([torch.ones_like(A), level_sets], dim=1)

        level_sets_padded = self.odd_pad(level_sets, pad)
        # get odd shaped zero padded tensors
        masks = self.circular_masks_like(level_sets_padded.shape[-2:], ksizes).float()[None]

        # take the level sets and masks into 2D fourier domain
        stackA_f = rfft2(level_sets_padded)
        stackB_f = rfft2(masks).conj()
        # cross correlate each pairwise choice of circular mask and levelset
        stackAll = torch.einsum('bihw, bjhw->bijhw', stackA_f, stackB_f)
        # Inverse 2D real valued fourier transform and shift and abs
        stackCC = torch.abs(fftshift(irfft2(stackAll), dim=(-2, -1)))
        # clamp to prevent normalizing by zero intersection norm and log of 0
        stackCC_clamp = torch.clamp(stackCC, min=1.0)
        stackCC_norm = stackCC_clamp / stackCC_clamp[:, [0]]
        # calc entropy
        chi_log_chi = stackCC_norm * torch.log(stackCC_norm)
        # calc entropy
        entropies = torch.sum(chi_log_chi[:, 1:, :, ...], dim=1)
        # crop out the original image size (they may have been odd zero padding)
        output = -1.0 * entropies[:, :, pad:pad + A.shape[-2], pad:pad + A.shape[-1]]
        # show_kornia(entropies[0, 0, pad:pad+A.shape[-2], pad:pad+A.shape[-1]], (512,512))
        # show_kornia(entropies[1, 0, pad:pad+A.shape[-2], pad:pad+A.shape[-1]], (512,512))
        return output

    def box_est_entropy(self, A, ksizes, uncounted_bin=0):
        B, C, H, W = A.size()

        pad = int(ksizes.max().item()) + 1

        level_sets = (A * self.bins).byte().repeat(1, self.bins, 1, 1) == torch.arange(self.bins, dtype=torch.uint8,
                                                                                       device=A.device)[None, :, None,
                                                                          None]
        level_sets = torch.cat([~level_sets[:, [uncounted_bin]], level_sets[:, (1+uncounted_bin):]], dim=1).float()
        # level_sets = torch.cat([torch.ones_like(A), level_sets[(1+uncounted_bin):]], dim=1).float()

        level_sets_padded = torch.nn.functional.pad(level_sets, (pad, pad, pad, pad), mode='constant', value=0.0)

        # cumulative sum along H and W dimensions
        level_sets_padded_cumsum = torch.cumsum(torch.cumsum(level_sets_padded, dim=-2), dim=-1)

        # bottom right minus top left
        areas = []
        for k in ksizes:
            k = int(k.item())
            bot_right = level_sets_padded_cumsum[:, :, int(pad + k):int(pad + k + A.size(-2)),
                        int(pad + k):int(pad + k + A.size(-1))]
            bot_left = level_sets_padded_cumsum[:, :, int(pad + k):int(pad + k + A.size(-2)),
                       int(pad - k):int(pad - k + A.size(-1))]
            top_right = level_sets_padded_cumsum[:, :, int(pad - k):int(pad - k + A.size(-2)),
                        int(pad + k):int(pad + k + A.size(-1))]
            top_left = level_sets_padded_cumsum[:, :, int(pad - k):int(pad - k + A.size(-2)),
                       int(pad - k):int(pad - k + A.size(-1))]
            area = bot_right - bot_left - top_right + top_left
            areas.append(area[:, :, None, :, :])
        areas = torch.concat(areas, dim=2)
        areas_clamp = torch.clamp(areas, min=1.0)
        areas_norm = areas_clamp[:, 1:] / areas_clamp[:, [0]]
        entropies = -1.0 * torch.sum(areas_norm * torch.log(areas_norm), dim=1)
        return entropies

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 1
        if self.multiplicative:
            radii_factors = torch.cumprod(
                (self.radius_step * torch.ones(B, 1 + self.n_levels).to(x.device).to(x.dtype)), 1)
            radii = ((self.init_radius / self.radius_step) * radii_factors)
        else:
            radii_factors = torch.cumsum((self.radius_step * torch.ones(B, 1 + self.n_levels).to(x.device).to(x.dtype)),
                                         1)
            radii = ((self.init_radius - self.radius_step) + radii_factors)
        pixel_dists = torch.ones(B, self.n_levels).to(x.device).to(x.dtype)
        pyr_list = []
        radii_list = []
        pixel_dists_list = []
        num_octaves = int((math.log(min(H, W) / self.min_size) / math.log(2)) + 1)
        current_base_img = x
        for i in range(num_octaves):
            if self.mode == 'box':
                current_mi = self.box_est_entropy(current_base_img, radii[0])
            elif self.mode == 'fft':
                current_mi = self.windowed_entropy(current_base_img, radii[0])
            else:
                raise ValueError(f"mode must be on of {'box', 'fft'}, but got {self.mode}")
            # add back in color channel (must be grayscale)
            pyr_list.append(current_mi[:, None])
            radii_list.append(2.0 * radii[:, :-1].contiguous())
            # full 2
            pixel_dists_list.append(2 ** (i) * pixel_dists)
            half_size = (int(current_base_img.size(-2) / 2), int(current_base_img.size(-1) / 2))
            current_base_img = F.interpolate(current_base_img, size=half_size, mode='nearest')
        return pyr_list, radii_list, pixel_dists_list


def _scale_index_to_scale(max_coords: Tensor, sigmas: Tensor, num_levels: int) -> Tensor:
    r"""Auxiliary function for ScaleSpaceDetector. Converts scale level index from ConvSoftArgmax3d to the actual
    scale, using the sigmas from the ScalePyramid output.

    Args:
        max_coords: tensor [BxNx3].
        sigmas: tensor [BxNxD], D >= 1

    Returns:
        tensor [BxNx3].
    """
    # depth (scale) in coord_max is represented as (float) index, not the scale yet.
    # we will interpolate the scale using pytorch.grid_sample function
    # Because grid_sample is for 4d input only, we will create fake 2nd dimension

    # Reshape for grid shape
    B, N, _ = max_coords.shape
    scale_coords = max_coords[:, :, 0].contiguous().view(-1, 1, 1, 1)
    # Replace the scale_x_y
    out = concatenate(
        [sigmas[0, 0] * torch.pow(2.0, scale_coords / float(num_levels)).view(B, N, 1), max_coords[:, :, 1:]], 2
    )
    return out


class MIScaleSpaceDetector(nn.Module):
    r"""Module for differentiable local feature detection, as close as possible to classical local feature detectors
    like Harris, Hessian-Affine or SIFT (DoG).

    It has 5 modules inside: scale pyramid generator, response ("cornerness") function,
    soft nms function, affine shape estimator and patch orientation estimator.
    Each of those modules could be replaced with learned custom one, as long, as
    they respect output shape.

    Args:
        num_features: Number of features to detect. In order to keep everything batchable,
          output would always have num_features output, even for completely homogeneous images.
        mr_size: multiplier for local feature scale compared to the detection scale.
          6.0 is matching OpenCV 12.0 convention for SIFT.
        scale_pyr_module: generates scale pyramid. See :class:`~kornia.geometry.ScalePyramid` for details.
          Default: ScalePyramid(3, 1.6, 10).
        resp_module: calculates ``'cornerness'`` of the pixel.
        nms_module: outputs per-patch coordinates of the response maxima.
          See :class:`~kornia.geometry.ConvSoftArgmax3d` for details.
        ori_module: for local feature orientation estimation. Default:class:`~kornia.feature.PassLAF`,
           which does nothing. See :class:`~kornia.feature.LAFOrienter` for details.
        aff_module: for local feature affine shape estimation. Default: :class:`~kornia.feature.PassLAF`,
            which does nothing. See :class:`~kornia.feature.LAFAffineShapeEstimator` for details.
        minima_are_also_good: if True, then both response function minima and maxima are detected
            Useful for symmetric response functions like DoG or Hessian. Default is False
    """

    def __init__(
            self,
            num_features,
            mr_size,
            scale_pyr_module,
            resp_module,
            nms_module,
            ori_module,
            aff_module,
            minima_are_also_good: bool = True,
            scale_space_response=False,
    ):
        super().__init__()
        self.mr_size = mr_size
        self.num_features = num_features
        self.scale_pyr = scale_pyr_module
        self.resp = resp_module
        self.nms = nms_module
        self.ori = ori_module
        self.aff = aff_module
        self.minima_are_also_good = minima_are_also_good
        self.scale_space_response = scale_space_response

    def __repr__(self):
        return (
                self.__class__.__name__ + '('
                                          'num_features='
                + str(self.num_features)
                + ', '
                + 'mr_size='
                + str(self.mr_size)
                + ', '
                + 'scale_pyr='
                + self.scale_pyr.__repr__()
                + ', '
                + 'resp='
                + self.resp.__repr__()
                + ', '
                + 'nms='
                + self.nms.__repr__()
                + ', '
                + 'ori='
                + self.ori.__repr__()
                + ', '
                + 'aff='
                + self.aff.__repr__()
                + ')'
        )

    def detect(self, img: Tensor, num_feats: int):
        dev = img.device
        dtype = img.dtype
        sp_no_dif, sigmas, _ = self.scale_pyr(img)
        # sp_no_dif, sigmas = [sp_no_dif[0]], [sigmas[0]]
        # sp_no_dif = [s / s.mean(dim=(-2,-1), keepdims=True) for s in sp_no_dif]
        # sp = [(s[:, :, 1:] - s[:, :, :-1]) for s in sp_no_dif]
        # sp = [(s[:, :, 1:] - s[:, :, :-1])**2 for s in sp_no_dif]
        sp = [(s[:, :, 1:] - s[:, :, :-1]).abs() for s in sp_no_dif]
        # sp = [(s[:, :, 1:] - s[:, :, :-1]).abs()**0.5 for s in sp_no_dif]
        # sp = [((s[:, :, :-1].abs() + 1) / (s[:, :, 1:].abs() + 1)) for s in sp_no_dif]

        all_responses: List[Tensor] = []
        all_lafs: List[Tensor] = []
        for oct_idx, octave in enumerate(sp):
            sigmas_oct = sigmas[oct_idx]
            B, CH, L, H, W = octave.size()
            # Run response function
            if self.scale_space_response:
                oct_resp = self.resp(octave, sigmas_oct.view(-1))
            else:
                oct_resp = self.resp(octave.permute(0, 2, 1, 3, 4).reshape(B * L, CH, H, W), sigmas_oct.view(-1)).view(
                    B, L, CH, H, W
                )
                # We want nms for scale responses, so reorder to (B, CH, L, H, W)
                oct_resp = oct_resp.permute(0, 2, 1, 3, 4)
                # 3rd extra level is required for DoG only
                if self.scale_pyr.extra_levels % 2 != 0:  # type: ignore
                    oct_resp = oct_resp[:, :, :-1]

            # Differentiable nms
            coord_max, response_max = self.nms(oct_resp)
            if self.minima_are_also_good:
                coord_min, response_min = self.nms(-oct_resp)
                take_min_mask = (response_min > response_max).to(response_max.dtype)
                response_max = response_min * take_min_mask + (1 - take_min_mask) * response_max
                coord_max = coord_min * take_min_mask.unsqueeze(2) + (1 - take_min_mask.unsqueeze(2)) * coord_max

            # Now, lets crop out some small responses
            responses_flatten = response_max.view(response_max.size(0), -1)  # [B, N]
            max_coords_flatten = coord_max.view(response_max.size(0), 3, -1).permute(0, 2, 1)  # [B, N, 3]

            if responses_flatten.size(1) > num_feats:
                resp_flat_best, idxs = torch.topk(responses_flatten, k=num_feats, dim=1)
                max_coords_best = torch.gather(max_coords_flatten, 1, idxs.unsqueeze(-1).repeat(1, 1, 3))
            else:
                resp_flat_best = responses_flatten
                max_coords_best = max_coords_flatten
            B, N = resp_flat_best.size()

            # Converts scale level index from ConvSoftArgmax3d to the actual scale, using the sigmas
            max_coords_best = _scale_index_to_scale(
                max_coords_best, sigmas_oct, self.scale_pyr.n_levels  # type: ignore
            )

            # Create local affine frames (LAFs)
            rotmat = torch.eye(2, dtype=dtype, device=dev).view(1, 1, 2, 2)
            current_lafs = concatenate(
                [
                    self.mr_size * max_coords_best[:, :, 0].view(B, N, 1, 1) * rotmat,
                    max_coords_best[:, :, 1:3].view(B, N, 2, 1),
                ],
                3,
            )

            # Zero response lafs, which touch the boundary
            good_mask = laf_is_inside_image(current_lafs, octave[:, 0])
            resp_flat_best = resp_flat_best * good_mask.to(dev, dtype)

            # Normalize LAFs
            current_lafs = normalize_laf(current_lafs, octave[:, 0])  # We don`t need # of scale levels, only shape

            all_responses.append(resp_flat_best)
            all_lafs.append(current_lafs)

        # Sort and keep best n
        responses = concatenate(all_responses, 1)
        lafs = concatenate(all_lafs, 1)
        responses, idxs = torch.topk(responses, k=num_feats, dim=1)
        lafs = torch.gather(lafs, 1, idxs.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 3))
        return responses, denormalize_laf(lafs, img), (sp[0]).mean(dim=2)
        # return responses, denormalize_laf(lafs, img), sp[0][:,:,-1]
        # return responses, denormalize_laf(lafs, img), sp_no_dif[0].mean(dim=2)
        # return responses, denormalize_laf(lafs, img), sp_no_dif[0][:,:,0]

    def forward(self, img: Tensor):
        """Three stage local feature detection. First the location and scale of interest points are determined by
        detect function. Then affine shape and orientation.

        Args:
            img: image to extract features with shape [BxCxHxW]

        Returns:
            lafs: shape [BxNx2x3]. Detected local affine frames.
            responses: shape [BxNx1]. Response function values for corresponding lafs
        """
        responses, lafs, nmi_maps = self.detect(img, self.num_features)
        lafs = self.aff(lafs, nmi_maps)
        lafs = self.ori(lafs, nmi_maps)
        return lafs, responses, nmi_maps


class Identity(nn.Module):
    r"""Module that does nothing to the input. For compatibility with current code.
    """

    def __init__(self, grads_mode='sobel') -> None:
        super().__init__()
        self.grads_mode: str = grads_mode
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + 'grads_mode=' + self.grads_mode + ')'

    def forward(self, input: Tensor, sigmas = None) -> Tensor:
        return input


def dole_match(images: torch.Tensor,
               patch_size: int = 61,
               ang_bins: int = 16,
               spatial_bins: int = 20,
               n_levels: int = 5,
               radius_start: float = 5.0,
               radius_step: float = 1.4,
               mr_size: float = 6.0,
               n_intensity_bins: int = 20,
               mode: str = 'fft',
               n_features: int = 5000,
               model_type: str = 'homography',
               match_thresh_ratio: float = 0.99,
               match_thresh_sq_dist: float = 20
               ):
    """
    DoLE matching algorithm

    Args:
        images: tensor of shape [2, C, H, W]
        patch_size: size of the patch for descriptor extraction
        ang_bins: number of bins for orientation histogram
        spatial_bins: number of bins for spatial histogram
        n_levels: number of levels in the scale space
        radius_start: radius of the first level in the scale space
        radius_step: multiplicative step for the radius
        mr_size: multiplier for local feature scale compared to the detection scale
        n_intensity_bins: number of bins for intensity histogram
        mode: mode for scale space generation, one of ['box', 'fft']
        n_features: number of features to detect
        model_type: type of the model to fit, one of ['homography', 'affine']
        match_thresh_ratio: threshold for the ratio test
        match_thresh_sq_dist: threshold for the squared distance test

    Returns:
        homography: tensor of shape [3, 3]
        src_pts: tensor of shape [N, 2]
        dst_pts: tensor of shape [N, 2]
        mask: tensor of shape [N]
        inliers: tensor of shape [M, 2]
        lafs: tensor of shape [2, N, 2, 3]
    """
    sift = kornia.feature.SIFTDescriptor(patch_size, ang_bins, spatial_bins, rootsift=True).to(images.device)
    resp = Identity()
    scale_pyr = EMIScalePyramid(n_levels,
                                init_radius=radius_start,
                                min_size=64,
                                radius_step=radius_step,
                                multiplicative=True,
                                bins=n_bins,
                                mode=mode)

    # use non-maximum suppression to get the maxima
    nms = kornia.geometry.ConvQuadInterp3d()

    detector = MIScaleSpaceDetector(n_features,
                                    mr_size=mr_size,
                                    resp_module=resp,
                                    nms_module=nms,
                                    scale_pyr_module=scale_pyr,
                                    ori_module=kornia.feature.LAFOrienter(patch_size),
                                    aff_module=kornia.feature.LAFAffineShapeEstimator(patch_size),
                                    minima_are_also_good=True,
                                    scale_space_response=True
                                    ).to(images.device)

    with torch.no_grad():
        lafs, resps, nmi_maps = detector(images)
        patches = kornia.feature.extract_patches_from_pyramid(nmi_maps, lafs, patch_size)
        B, N, CH, H, W = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        descriptors = sift(patches.view(B * N, CH, H, W)).view(B, N, -1)
        match_dists, match_idxs = kornia.feature.match_fginn(descriptors[0], descriptors[1], lafs[[0]], lafs[[1]], match_thresh_ratio, 10, mutual=True)
    
    src_pts = lafs[1, match_idxs[:, 1], :, 2]
    dst_pts = lafs[0, match_idxs[:, 0], :, 2]

    # use ransac to fit transformation
    ransac = RANSAC(model_type=model_type, inl_th=match_thresh_sq_dist, batch_size=32768, max_iter=100,
                              confidence=0.99999, max_lo_iters=20)
    homography, mask = ransac(src_pts, dst_pts, match_dists)

    mask = mask.cpu()
    inliers = match_idxs[mask.bool().squeeze(), :]

    return homography, (src_pts, dst_pts, mask, inliers, lafs)

