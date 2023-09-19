import torch 
from shapeless import liquid
from torch import nn
import torch.nn.functional as F


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

@liquid
class Nerf(nn.Module):
    D = 8
    W = 256
    input_ch = 3
    input_ch_views = 3
    output_ch = 4
    skips = [4]
    use_viewdirs = False

    def forward(self, x):
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