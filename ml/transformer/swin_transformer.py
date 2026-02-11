import torch

BxnWindows = 1
NHead = 1
NPatchesPerWindow = 2
ChannlesPerHead = 1

attention = torch.rand(BxnWindows, NHead, NPatchesPerWindow, NPatchesPerWindow)
attention = (attention + attention.transpose(-2, -1)) / 2
v = torch.rand(BxnWindows, NHead, NPatchesPerWindow, ChannlesPerHead) * 0 + 1
print(attention.shape, v.shape)
out = attention @ v
print(v.shape)

print(attention, "\n", v, "\n", out)
