import numpy as np
import torch
import torch.nn.functional as F
from shapeless import liquid
from torch import nn


#UTILS 
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W),
        torch.linspace(0, H-1, H)
    )
    i = i.t()
    j = j.t()

    dirs = torch.stack(
        [(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1
    )

    #rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)

    #translate camera frame origin to the world frame it is the origin of all rays
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


@liquid
class Embedder:
    input_dim = None
    include_input: False
    max_freq_log2: None
    num_freqs: None
    log_sampling: False
    period_fns: None

    def create_emedding_fn(self):
        embed_fns = []
        d = self.input_dim
        out_dim = 0

        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d
        
        max_freq = self.max_freq_log2
        N = self.num_freqs

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(
                0.,
                max_freq,
                steps=N,
            )
        else:
            freq_bands = torch.linspace(
                2. ** 0.,
                2. ** max_freq, 
                steps=N,
            )
        
        for freq in freq_bands:
            for p_fn in self.period_fns:
                embed_fns.append(
                    lambda x,
                    p_fn=p_fn,
                    freq=freq: p_fn(x * freq)
                )
                out_dim += d
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    
    def embed(self, inputs):
        return torch.cat([
            fn(inputs) for fn in self.embed_fns
        ], -1)


#MODEL

class Nerf(nn.Module):

    def __init__(
        self,
        D = 8,
        W = 256,
        input_ch = 3,
        input_ch_views = 3,
        output_ch = 4,
        skips = [4],
        use_viewdirs = False,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linear = nn.ModuleList(
            [nn.Linear(
                self.input_ch, self.W
            )] + [nn.Linear(
                self.W, self.W
            ) if i not in self.skips else nn.Linear(
                self.W + self.input_ch, self.W
            ) for i in range(self.D - 1)]
        )

        self.views_linear = nn.ModuleList(
            [nn.Linear(self.input_ch_views, self.W, self.W // 2)]
        )
        
        if self.use_viewdirs:
            self.feature_linear = nn.Linear(self.W + self.W)
            self.alpha_linear = nn.Linear(self.W, 1)
            self.rgb_linear = nn.Linear(self.W // 2, 3)

        else:
            self.output_linear = nn.Linear(self.W, self.output_ch)

    def forward(self, x):
        
        ############
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=1
        )

        h = input_pts

        for i, layer in enumerate(self.pts_linear):
            h = self.pts_linear[i](h)
            h = F.relu(h)

            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, layer in enumerate(self.views_linear):
                h = self.views_linear[i](h)
                h = F.relu(h)
            
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        
        else:
            outputs = self.output_linear(h)
            return outputs
    
    def load_weights_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs is False"

        #load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears])
            )

            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1])
            )
        
        #load views linears
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear])
        )
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1])
        )

        #load views_linear
        idx_views_linears = 2 * self.D + 2
        self.views_linear[0].weight.data = torch.from_numpy(np.transpose(
            weights[idx_views_linears]
        ))
        self.views_linear[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1])
        )
