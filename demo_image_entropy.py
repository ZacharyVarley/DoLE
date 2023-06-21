import numpy as np
from PIL import Image
import torch
from dole import dole_match
import kornia
from mind import MINDPyramidPair
from tqdm import tqdm
from lie_homograph import LieGroupImageTransform

# set the device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the images using PIL
image1 = Image.open('images/Ti7_eCHORD_wd_10_tilt_3_rot_0.tif')
image2 = Image.open('images/Ti7_eCHORD_wd_10_tilt_3_rot_10.tif')

# convert to numpy arrays and check min/max values
image1_np = np.array(image1)
image2_np = np.array(image2)

# normalize the images with the largest max value across both images
image1_norm = image1_np / max([np.max(image1_np), np.max(image2_np)])
image2_norm = image2_np / max([np.max(image1_np), np.max(image2_np)])

# convert the images to tensors
image1_tensor = torch.tensor(image1_norm).float()[None, None].to(device)
image2_tensor = torch.tensor(image2_norm).float()[None, None].to(device)

# use Kornia to run CLAHE
image1_tensor_clahe = kornia.enhance.equalize_clahe(image1_tensor, clip_limit=4.0, grid_size=(4, 4))
image2_tensor_clahe = kornia.enhance.equalize_clahe(image2_tensor, clip_limit=4.0, grid_size=(4, 4))

# stack the images into a 2-batch, 1-channel, input tensor
img_input_dole = torch.concat([image1_tensor_clahe, image2_tensor_clahe], dim=0)

# run the DOLE algorithm to get the homography
dole_homography, inliers = dole_match(img_input_dole)

# apply the homography to the second image using Kornia
img2_warped_dole = kornia.geometry.warp_perspective(image2_tensor, dole_homography[None], dsize=image1_tensor.shape[-2:])

# save the warped image as png using PIL
dole_warped_pil = Image.fromarray((img2_warped_dole[0, 0].squeeze()* 255.0).byte().cpu().numpy())
dole_warped_pil.save('images/image_pre_refine.png')

# refine the homography with MIND + gradient descent
img_input_refine = torch.cat([image1_tensor, image2_tensor], dim=0)
pyramid_pair = MINDPyramidPair(img_input_refine,
                               scale_factor=1.4,
                               scales=3,
                               non_local_region_size=3,
                               patch_size=7,
                               neighbor_size=3)

# train a warp model to fine tune the homography (use weights to heavily diminish projective terms)
weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-4, 1e-4])  # * 1e-1
warp_model = LieGroupImageTransform("sl3", weights).to(device)

# # train a warp model to fine tune the affine
# weights_aff = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# warp_model = LieGroupImageTransform("aff2", weights_aff).to(device)

# set the optimizer
optimizer = torch.optim.Adam(warp_model.parameters(), lr=1e-3)

# set the number of epochs
n_epochs = 500

progress_bar = tqdm(range(n_epochs), colour='white')

for epoch in progress_bar:
    # zero the gradients
    optimizer.zero_grad()

    # find the composite homograph
    loss = pyramid_pair.loss(warp_model() @ dole_homography)

    # compute the gradients
    loss.backward(retain_graph=True)

    # update the parameters
    optimizer.step()

    # print the loss
    progress_bar.set_description(f'epoch: {(epoch + 1):03d}/{n_epochs}, '
                                 f'loss = {loss.item():.13f}')
    
# apply the homography to the second image using Kornia
img2_warped_refine = kornia.geometry.warp_perspective(image2_tensor,
                                                      warp_model() @ dole_homography[None], 
                                                      dsize=image1_tensor.shape[-2:])

# save the warped image as png using PIL
refine_warped_pil = Image.fromarray((img2_warped_refine[0, 0].squeeze()* 255.0).byte().cpu().numpy())
refine_warped_pil.save('images/image_post_refine.png')

# save the target image (image 1)
target_pil = Image.fromarray((image1_tensor[0, 0].squeeze()* 255.0).byte().cpu().numpy())
target_pil.save('images/image_target.png')
