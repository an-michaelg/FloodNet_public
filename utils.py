import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize

def vis_pair(img, label, unnorm=True):
    """
    Visualize an image-label pair using matplotlib

    Parameters
    ----------
    img : tensor of size (num_channels, h, w)
    label : tensor of size (h, w) with integer classes
    unnorm: bool, True if image needs to be unnormalized first

    """
    if unnorm:
        img = unnormalize(img)
    img_arr = img.permute(1,2,0).cpu().numpy()
    label_arr = label.permute(1,2,0).cpu().numpy()
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].imshow(img_arr)
    axs[0].set_title('Image')
    axs[1].imshow(label_arr, cmap='Set3')
    axs[1].set_title('Target')
    plt.show()
    
def vis_pred(img, pred, label, unnorm=True):
    """
    Visualize an image-pred-label triplet using matplotlib

    Parameters
    ----------
    img : tensor of size (num_channels, h, w)
    pred : tensor of size (h, w) with integer classes
    label: tensor of size (h, w) with integer classes
    unnorm: bool, True if image needs to be unnormalized first

    """
    if unnorm:
        img = unnormalize(img)
    img_arr = img.permute(1,2,0).cpu().numpy()
    pred_arr = pred.permute(1,2,0).cpu().numpy()
    label_arr = label.permute(1,2,0).cpu().numpy()
    fig, axs = plt.subplots(1,3, figsize=(18,6))
    axs[0].imshow(img_arr)
    axs[0].set_title('Image')
    axs[1].imshow(decode_segmap(pred_arr))
    axs[1].set_title('Prediction')
    axs[2].imshow(decode_segmap(label_arr))
    axs[2].set_title('Target')
    plt.show()
    

def label_to_one_hot(y, nc):
    """
    Convert label into one-hot encoding

    Parameters
    ----------
    y : tensor, (Bx1xHxW)
    nc : int, number of classes

    Returns
    -------
    y_oh : tensor, (BxCxHxW)

    """
    y_oh = F.one_hot(y, nc) # Bx1xHxWxC since OH always extends to the right
    return y_oh.squeeze().permute(0,3,1,2)

def unnormalize(img):
    """ un-normalize the image tensor with imagenet mean/std """
    inv_normalize = Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    inv_tensor = inv_normalize(img)
    return inv_tensor

def decode_segmap(image, nc=10):
    """
    Map the colors of categorical segmap to RGB values

    Parameters
    ----------
    image : tensor of size (h, w) with integer classes
    nc : number of classes The default is 10 (FloodNet task).

    Returns
    -------
    rgb : corresponding RGB image (h, w, 3)

    """
  
    label_colors = np.array([(0, 0, 0),  # 0=background
        # 1=building_flooded, 2=building_non_flooded
        (250, 0, 0), (150, 100, 100),
        # 3=road_flooded, 4=road_non_flooded, 5=water, 6=tree
        (175, 175, 50), (128, 128, 128), (0, 250, 250), (0, 0, 250),
        # 7=vehicle, 8=pool, 9=grass
        (250, 0, 250), (250, 128, 0), (0, 250, 0)])
    
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
      
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        
    rgb = np.stack([r, g, b], axis=2).squeeze()
    return rgb
        