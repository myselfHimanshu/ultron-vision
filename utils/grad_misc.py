import torch
import numpy as np

def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
        
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    import cv2
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().cpu().numpy()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    
    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result

def find_resnet_layer(arch, target_layer_name):
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split("_")
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num==1:
            target_layer = arch.layer1
        elif layer_num==2:
            target_layer = arch.layer2
        elif layer_num==3:
            target_layer = arch.layer3
        elif layer_num==4:
            target_layer = arch.layer4
        else:
            raise ValueError(f'unknown layer : {target_layer_name}')

        if len(hierarchy)>=2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]
        if len(hierarchy)>=3:
            target_layer = target_layer._modules[hierarchy[2]]
        if len(hierarchy)>=4:
            target_layer = target_layer._modules[hierarchy[3]]
        
    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer
