import torch
import torch.nn as nn
from typing import Optional

# list of allowed groups
ALLOWED_GROUPS = ["so2",   # rotation
                  "se2",   # rotation + translation
                  "sim2",  # rotation + translation + scaling
                  "as2",   # rotation + translation + scaling + axial stretching
                  "aff2",  # affine transformation "as2" + shear
                  "sl3"]   # projective transformation


class LieGroupImageTransform(nn.Module):
    """
    Parameters
    ----------
    group : str
        The Lie group to use. Allowed groups are:
        - "so2": rotation
        - "se2": rotation + translation
        - "sim2": rotation + translation + scaling
        - "as2": rotation + translation + scaling + axial stretching
        - "aff2": affine transformation
        - "sl3": projective transformation

    bias : torch.Tensor, optional
        The bias to use for the linear combination of the lie algebra elements (favor less translation).
        If not given, the bias is set to ones.
        If given, the bias must be a 1D tensor of the same length as the number of elements in the group.
        The bias can also be used to constrain the homography to a specific group.
        For example, using "sl3" and setting bias to [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        makes the homograph affine again by setting the projective terms to zero.
    """
    def __init__(self, group: str,
                 bias: Optional[torch.Tensor] = None):
        super(LieGroupImageTransform, self).__init__()
        self.group = group

        if group == "so2":
            elements = torch.zeros((1, 3, 3))
            elements[0, 0, 1] = -1  # rotation
            elements[0, 1, 0] = 1   # rotation
        elif group == "se2":
            elements = torch.zeros((3, 3, 3))
            elements[0, 0, 2] = 1   # translation in x
            elements[1, 1, 2] = 1   # translation in y
            elements[2, 0, 1] = -1  # rotation
            elements[2, 1, 0] = 1   # rotation
        elif group == "sim2":
            elements = torch.zeros((4, 3, 3))
            elements[0, 0, 2] = 1   # translation in x
            elements[1, 1, 2] = 1   # translation in y
            elements[2, 0, 1] = -1  # rotation
            elements[2, 1, 0] = 1   # rotation
            elements[3, 2, 2] = -1   # isotropic scaling
        elif group == 'as2':
            # this is the shear-less affine group
            elements = torch.zeros((5, 3, 3))
            elements[0, 0, 2] = 1   # translation in x
            elements[1, 1, 2] = 1   # translation in y
            elements[2, 0, 1] = -1  # rotation
            elements[2, 1, 0] = 1   # rotation
            elements[3, 0, 0] = 1   # isotropic scaling
            elements[3, 1, 1] = 1   # isotropic scaling
            elements[4, 0, 0] = 1   # stretching
            elements[4, 1, 1] = -1  # stretching
        elif group == "aff2":
            elements = torch.zeros((6, 3, 3))
            elements[0, 0, 2] = 1   # translation in x
            elements[1, 1, 2] = 1   # translation in y
            elements[2, 0, 1] = -1  # rotation
            elements[2, 1, 0] = 1   # rotation
            elements[3, 0, 0] = 1   # isotropic scaling
            elements[3, 1, 1] = 1   # isotropic scaling
            elements[4, 0, 0] = 1   # stretching
            elements[4, 1, 1] = -1  # stretching
            elements[5, 0, 1] = 1   # shear
            elements[5, 1, 0] = 1   # shear
        elif group == "sl3":
            elements = torch.zeros((8, 3, 3))
            elements[0, 0, 2] = 1   # translation in x
            elements[1, 1, 2] = 1   # translation in y
            elements[2, 0, 1] = -1  # rotation
            elements[2, 1, 0] = 1   # rotation
            elements[3, 0, 0] = 1   # isotropic scaling
            elements[3, 1, 1] = 1   # isotropic scaling
            elements[3, 2, 2] = -2  # isotropic scaling
            elements[4, 0, 0] = 1   # stretching
            elements[4, 1, 1] = -1  # stretching
            elements[5, 0, 1] = 1   # shear
            elements[5, 1, 0] = 1   # shear
            elements[6, 2, 0] = 1   # projective keystone in x (I might have these swapped for x/y)
            elements[7, 2, 1] = 1   # projective keystone in y (I might have these swapped for x/y)
        else:
            raise NotImplementedError(f"Group {group} not implemented. Allowed groups are {ALLOWED_GROUPS}")

        # set elements buffer (without grad attribute)
        self.register_buffer('elements', elements)

        # set the parameters (the linear combination the lie algebra elements)
        self.weights = nn.Parameter(torch.zeros(len(elements), 1, 1))

        if bias is not None:
            # if bias are given, check if they are of the right shape
            assert len(bias.shape) == 1
            assert bias.shape[0] == len(self.elements), \
                f"Number of bias terms ({bias.shape[0]}) does not match number of elements ({len(self.elements)})"
            # set the bias buffer (no grad attribute)
            self.register_buffer('bias', bias[:, None, None])
        else:
            # if no bias are given, set them to ones
            self.register_buffer('bias', torch.ones(len(elements), 1, 1))

    def __call__(self):
        # multiply the basis components with the bias tensor and sum them up and take the matrix exponential
        return torch.linalg.matrix_exp((self.weights * self.elements * self.bias).sum(dim=0))