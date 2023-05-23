import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from dole import dole_match
import kornia

# set the device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the images using PIL
image1 = Image.open('images/Ti7_eCHORD_wd_10_tilt_3_rot_0.tif')
image2 = Image.open('images/Ti7_eCHORD_wd_10_tilt_3_rot_10.tif')

# convert to numpy arrays and check min/max values
image1_np = np.array(image1)
image2_np = np.array(image2)
print('image1 min/max: ', np.min(image1_np), np.max(image1_np))
print('image2 min/max: ', np.min(image2_np), np.max(image2_np))

# normalize the images with the largest max value across both images
image1_norm = image1_np / max([np.max(image1_np), np.max(image2_np)])
image2_norm = image2_np / max([np.max(image1_np), np.max(image2_np)])

# save the norm images as png
plt.imsave('images/image1_norm.png', image1_norm.reshape(image1.size[::-1]), cmap='gray')
plt.imsave('images/image2_norm.png', image2_norm.reshape(image2.size[::-1]), cmap='gray')

# convert the images to tensors
image1_tensor = torch.tensor(image1_norm).float()[None, None].to(device)
image2_tensor = torch.tensor(image2_norm).float()[None, None].to(device)

# stack the images into a 2-batch, 1-channel, input tensor
img_input = torch.concat([image1_tensor, image2_tensor], dim=0)

# run the DOLE algorithm to get the homography
H, inliers = dole_match(img_input)

# apply the homography to the second image using Kornia
img2_warped = kornia.geometry.warp_perspective(image2_tensor, H[None], dsize=image1.size[::-1])

# save the warped image as png
plt.imsave('images/image2_warped.png', img2_warped.squeeze().cpu().numpy(), cmap='gray')
