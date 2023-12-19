import torch
from torch import nn

def cow_clip(w, g, ratio=1, ids=None, cnts=None, min_w=0.03, const=False):
    if g.is_sparse:
        # for sparse gradients
    else:
        # for dense gradients
        values = g
        if const:
            clipnorm = torch.full(g.size, min_w)
        else:
            clipnorm = torch.norm(w, axis=-1)
            # bound weight norm by min_w
            clipnorm = torch.maximum(clipnorm, min_w)
        # scale by cnting
        cnts = torch.tensor_scatter_nd_update(
            torch.ones([clipnorm.size[0]], dtype=torch.int32),
            torch.expand_dims(ids, -1),
            cnts,
        )
        clipnorm = clipnorm * torch.cast(cnts, torch.float32)

    clip_t = ratio * clipnorm
    l2sum_row = torch.reduce_sum(values * values, axis=-1)
    pred = l2sum_row > 0
    l2sum_row_safe = torch.where(pred, l2sum_row, torch.ones_like(l2sum_row))
    l2norm_row = torch.sqrt(l2sum_row_safe)
    intermediate = values * torch.expand_dims(clip_t, -1)
    g_clip = intermediate / torch.expand_dims(torch.maximum(l2norm_row, clip_t), -1)

    if g.is_sparse:
        #return tf.IndexedSlices(g_clip, g.indices, g.dense_shape)
        return g_clip
    else:
        return g_clip
