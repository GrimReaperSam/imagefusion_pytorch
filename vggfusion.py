import numpy as np
from sporco.util import tikhonov_filter

import torch
from torchvision.models.vgg import vgg19

def lowpass(s, lda, npad):
    return tikhonov_filter(s, lda, npad)

def c3(s):
    s3 = np.dstack([s, s, s])
    return np.rollaxis(s3, 2, 0)[None, :, :, :]

def l1_features(out):
    h, w, d = out.shape
    A_temp = np.zeros((h+2, w+2))
    
    l1_norm = np.sum(np.abs(out), axis=2)
    A_temp[1:h+1, 1:w+1] = l1_norm
    return A_temp

def fusion_strategy(feat_a, feat_b, source_a, source_b, unit):
    
    m, n = feat_a.shape
    m1, n1 = source_a.shape
    weight_ave_temp1 = np.zeros((m1, n1))
    weight_ave_temp2 = np.zeros((m1, n1))
    
    for i in range(1, m):
        for j in range(1, n):
            A1 = feat_a[i-1:i+1, j-1:j+1].sum() / 9
            A2 = feat_b[i-1:i+1, j-1:j+1].sum() / 9
            
            weight_ave_temp1[(i-2)*unit+1:(i-1)*unit+1, (j-2)*unit+1:(j-1)*unit+1] = A1 / (A1+A2)
            weight_ave_temp2[(i-2)*unit+1:(i-1)*unit+1, (j-2)*unit+1:(j-1)*unit+1] = A2 / (A1+A2)
            
    gen = source_a * weight_ave_temp1 + source_b * weight_ave_temp2
    
    return gen

def get_activation(model, layer_numbers, input_image):
    outs = []
    out = input_image
    for i in range(max(layer_numbers)+1):
        out = model.features[i](out)
        if i in layer_numbers:
            outs.append(np.rollaxis(out.detach().cpu().numpy()[0], 0, 3))
    return outs

def fuse(vis, ir, model=None):
    npad = 16
    lda = 5
    vis_low, vis_high = lowpass(vis.astype(np.float32)/255, lda, npad)
    ir_low, ir_high = lowpass(ir.astype(np.float32)/255, lda, npad)
    
    if model is None:
        model = vgg19(True)
    model.cuda().eval()
    relus = [2, 7, 12, 21]
    unit_relus = [1, 2, 4, 8]
    
    vis_in = torch.from_numpy(c3(vis_high)).cuda()
    ir_in = torch.from_numpy(c3(ir_high)).cuda()
    
    relus_vis = get_activation(model, relus, vis_in)
    relus_ir = get_activation(model, relus, ir_in)
    
    vis_feats = [l1_features(out) for out in relus_vis]
    ir_feats = [l1_features(out) for out in relus_ir]
    
    saliencies = []
    saliency_max = None
    for idx in range(len(relus)):
        saliency_current = fusion_strategy(vis_feats[idx], ir_feats[idx], vis_high, ir_high, unit_relus[idx])
        saliencies.append(saliency_current)

        if saliency_max is None:
            saliency_max = saliency_current
        else:
            saliency_max = np.maximum(saliency_max, saliency_current)

    low_fused = (vis_low + ir_low) / 2
    high_fused = saliency_max
    return low_fused + high_fused
    
    