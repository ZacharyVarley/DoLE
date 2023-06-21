import torch
import math
from kornia.geometry import warp_perspective
from typing import List, Tuple


@torch.jit.script
def warp(images_tensor: torch.Tensor,
         homographs: torch.Tensor,
         size: Tuple[int, int]):
    grid = torch.meshgrid(
        torch.linspace(-1, 1, size[1], dtype=torch.float32, device=images_tensor.device),
        torch.linspace(-1, 1, size[0], dtype=torch.float32, device=images_tensor.device),
        indexing='ij'
    )
    grid = torch.stack(grid, dim=-1).unsqueeze(0).repeat(images_tensor.shape[0], 1, 1, 1)
    grid = torch.cat([grid, torch.ones_like(grid)[..., :1]], dim=-1)
    grid = torch.matmul(grid, homographs.transpose(-1, -2))
    grid = grid.view(images_tensor.shape[0], size[0], size[1], 3)
    grid = grid[:, :, :, :2] / grid[:, :, :, [2]]
    warped = torch.nn.functional.grid_sample(images_tensor, grid, align_corners=True, mode='bilinear')
    return warped


class MIND(torch.nn.Module):
    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MIND, self).__init__()
        self.nl_size = non_local_region_size
        self.p_size = patch_size
        self.n_size = neighbor_size
        self.sigma2 = gaussian_patch_sigma * gaussian_patch_sigma

        # calc shifted images in non local region
        self.image_shifter = torch.nn.Conv2d(in_channels=1, out_channels=self.nl_size * self.nl_size,
                                             kernel_size=(self.nl_size, self.nl_size),
                                             stride=1, padding=((self.nl_size - 1) // 2, (self.nl_size - 1) // 2),
                                             dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.nl_size * self.nl_size):
            t = torch.zeros((1, self.nl_size, self.nl_size))
            t[0, i % self.nl_size, i // self.nl_size] = 1
            self.image_shifter.weight.data[i] = t

        # patch summation
        self.summation_patcher = torch.nn.Conv2d(in_channels=self.nl_size * self.nl_size,
                                                 out_channels=self.nl_size * self.nl_size,
                                                 kernel_size=(self.p_size, self.p_size),
                                                 stride=1, padding=((self.p_size - 1) // 2, (self.p_size - 1) // 2),
                                                 dilation=1, groups=self.nl_size * self.nl_size, bias=False,
                                                 padding_mode='zeros')

        for i in range(self.nl_size * self.nl_size):
            # gaussian kernel
            t = torch.zeros((1, self.p_size, self.p_size))
            cx = (self.p_size - 1) // 2
            cy = (self.p_size - 1) // 2
            for j in range(self.p_size * self.p_size):
                x = j % self.p_size
                y = j // self.p_size
                d2 = torch.norm(torch.tensor([x - cx, y - cy]).float(), 2)
                t[0, x, y] = math.exp(-d2 / self.sigma2)

            self.summation_patcher.weight.data[i] = t

        # neighbor images
        self.neighbors = torch.nn.Conv2d(in_channels=1, out_channels=self.n_size * self.n_size,
                                         kernel_size=(self.n_size, self.n_size),
                                         stride=1, padding=((self.n_size - 1) // 2, (self.n_size - 1) // 2),
                                         dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.n_size * self.n_size):
            t = torch.zeros((1, self.n_size, self.n_size))
            t[0, i % self.n_size, i // self.n_size] = 1
            self.neighbors.weight.data[i] = t

        # neighbor patcher
        self.neighbor_summation_patcher = torch.nn.Conv2d(in_channels=self.n_size * self.n_size,
                                                          out_channels=self.n_size * self.n_size,
                                                          kernel_size=(self.p_size, self.p_size),
                                                          stride=1,
                                                          padding=((self.p_size - 1) // 2, (self.p_size - 1) // 2),
                                                          dilation=1, groups=self.n_size * self.n_size, bias=False,
                                                          padding_mode='zeros')

        for i in range(self.n_size * self.n_size):
            t = torch.ones((1, self.p_size, self.p_size))
            self.neighbor_summation_patcher.weight.data[i] = t

    def forward(self, orig):
        assert (len(orig.shape) == 4)
        assert (orig.shape[1] == 1)

        # get original image channel stack
        orig_stack = torch.stack([orig.squeeze(dim=1) for i in range(self.nl_size * self.nl_size)], dim=1)

        # get shifted images
        shifted = self.image_shifter(orig)

        # get image diff
        diff_images = shifted - orig_stack

        # diff's L2 norm
        Dx_alpha = self.summation_patcher(torch.pow(diff_images, 2.0))

        # calc neighbor's variance
        neighbor_images = self.neighbor_summation_patcher(self.neighbors(orig))

        Vx = neighbor_images.var(dim=1).unsqueeze(dim=1)

        # output mind
        nume = torch.exp(-Dx_alpha / (Vx + 1e-8))
        denomi = nume.sum(dim=1).unsqueeze(dim=1)
        mind = nume / denomi
        return mind


class MINDLoss(torch.nn.Module):

    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MINDLoss, self).__init__()
        self.nl_size = non_local_region_size
        self.MIND = MIND(non_local_region_size=non_local_region_size,
                         patch_size=patch_size,
                         neighbor_size=neighbor_size,
                         gaussian_patch_sigma=gaussian_patch_sigma)

    def forward(self, input, target):
        in_mind = self.MIND(input)
        tar_mind = self.MIND(target)
        mind_diff = in_mind - tar_mind
        l1 = torch.norm(mind_diff, 1)
        return l1 / (input.shape[2] * input.shape[3] * self.nl_size * self.nl_size)


class MINDPyramidPair:
    def __init__(self,
                 images: torch.Tensor,
                 scales: int = 3,
                 scale_factor: float = 2.0,
                 non_local_region_size: int = 9,
                 patch_size: int = 7,
                 neighbor_size: int = 3,
                 gaussian_patch_sigma: float = 3.0):
        device = images.device
        self.scale_factor = scale_factor
        self.nl_size = non_local_region_size
        mind = MIND(non_local_region_size=non_local_region_size,
                    patch_size=patch_size,
                    neighbor_size=neighbor_size,
                    gaussian_patch_sigma=gaussian_patch_sigma).to(device)

        # make an image pyramid
        self.pyramid = [mind(images)]
        for i in range(1, scales):
            rescaled = torch.nn.functional.interpolate(images,
                                                       scale_factor=1.0 / (scale_factor ** i),
                                                       mode='bicubic',
                                                       align_corners=True)
            mind_rescaled = mind(torch.clamp(rescaled, min=0.0, max=1.0))
            self.pyramid.append(mind_rescaled)

        self.homographs_image_to_canvas = []
        self.homographs_canvas_to_image = []
        for i in range(len(self.pyramid)):
            upscale = torch.eye(3, device=device)
            upscale[0, 0] = (self.scale_factor ** i)
            upscale[1, 1] = (self.scale_factor ** i)
            downscale = torch.eye(3, device=device)
            downscale[0, 0] = 1.0 / (self.scale_factor ** i)
            downscale[1, 1] = 1.0 / (self.scale_factor ** i)
            self.homographs_image_to_canvas.append(upscale)
            self.homographs_canvas_to_image.append(downscale)

    # overload the subtraction operator
    def loss(self, homograph):
        loss = 0.0
        norm = 0.0
        for i, images_at_size in enumerate(self.pyramid):
            composite = self.homographs_canvas_to_image[i] @ homograph[None] @ self.homographs_image_to_canvas[i]
            warped_1_to_0 = warp_perspective(images_at_size[[1]],
                                             composite,
                                             images_at_size[0].shape[-2:])
            warped_0_to_1 = warp_perspective(images_at_size[[0]],
                                             torch.inverse(composite),
                                             images_at_size[1].shape[-2:])
            loss += ((1.0 / self.scale_factor) ** i) * (0.5 * torch.norm(images_at_size[0] - warped_1_to_0, 1) +
                                                        0.5 * torch.norm(images_at_size[1] - warped_0_to_1, 1)) / \
                    (images_at_size[0].shape[-2] * images_at_size[0].shape[-1] * self.nl_size ** 2)
            norm += ((1.0 / self.scale_factor) ** i)
        return loss / norm
