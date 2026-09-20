import torch
import torch.nn as nn
from accelerate import Accelerator

from einops import rearrange


class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x

class SLSTMCell(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = in_channels
        self.conv = nn.Conv2d(in_channels + in_channels//4, in_channels, kernel_size, padding=1)

    def forward(self, x, hidden_state):
        h_cur, c_cur = hidden_state
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels//4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

def SLSTM(seq):
    batch_size, channels, frames, height, width = seq.shape
    h = torch.zeros(batch_size, channels//4, height, width).to("cuda:0",dtype=seq.dtype)
    c = torch.zeros(batch_size, channels//4, height, width).to("cuda:0",dtype=seq.dtype)
    outputs = []
    if channels == 1280:
        for frame in range(frames):
            h, c = s_lstm_1280[frame](seq[:, :, frame], (h, c))
            outputs.append(out_1280(h))
    if channels == 640:
        for frame in range(frames):
            h, c = s_lstm_640[frame](seq[:, :, frame], (h, c))
            outputs.append(out_640(h))
    if channels == 320:
        for frame in range(frames):
            h, c = s_lstm_320[frame](seq[:, :, frame], (h, c))
            outputs.append(out_320(h))
            
    return torch.stack(outputs).transpose(1,0).transpose(2,1)