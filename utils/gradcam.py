"""
Gradcam implementation
"""

import torch
import torch.nn.functional as F
from .grad_misc import find_resnet_layer

class GradCam(object):
    """
    Calculate GradCam feature map
    
    :param model_dict (dict): (model_type, arch, layer_name, input_size)
    """
    def __init__(self, model_dict):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
        
        if 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        b,c,h,w = input.size()
        logit = self.model_arch(input)
        if class_idx == None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']

        b,k,u,v = gradients.size()

        alpha = gradients.view(b,k,-1).mean(2)
        weights = alpha.view(b,k,1,1)

        fmap = (weights*activations).sum(1, keepdim=True)
        fmap = F.relu(fmap)
        fmap = F.upsample(fmap, size=(h, w), mode='bilinear', align_corners=False)
        fmap_min, fmap_max = fmap.min(), fmap.max()
        fmap = (fmap - fmap_min).div(fmap_max - fmap_min).data

        return fmap, logit
    
    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)



        
