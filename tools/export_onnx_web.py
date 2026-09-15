"""Export the XFeat backbone (+ softmax keypoint heatmap) to ONNX for onnxruntime-web.

Outputs (batch 1, grayscale input, H/W multiples of 32):
  feats     : (1, 64, H/8, W/8)   L2-normalized dense descriptors
  heatmap   : (1, 1, H, W)        keypoint probability map (pixel-unshuffled softmax)
  reliab    : (1, 1, H/8, W/8)    reliability map
NMS / top-k / descriptor sampling are done in JS (see public/js/xfeat.js).
"""
import sys, torch, torch.nn.functional as F
from modules.model import XFeatModel

def _unfold2d_static(self, x, ws=2):
    return F.pixel_unshuffle(x, ws)
XFeatModel._unfold2d = _unfold2d_static

class XFeatWeb(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        feats, kpts, heat = self.net(x)
        feats = F.normalize(feats, dim=1)
        sc = F.softmax(kpts, 1)[:, :64]
        B, _, H, W = sc.shape
        hm = sc.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        hm = hm.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return feats, hm, heat

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "xfeat_web.onnx"
    net = XFeatModel().eval()
    net.load_state_dict(torch.load("weights/xfeat.pt", map_location="cpu"))
    m = XFeatWeb(net).eval()
    x = torch.rand(1, 1, 480, 640) * 255
    torch.onnx.export(
        m, x, out, opset_version=17, dynamo=False,
        input_names=["image"], output_names=["feats", "heatmap", "reliab"],
        dynamic_axes={"image": {2: "H", 3: "W"}, "feats": {2: "h", 3: "w"},
                      "heatmap": {2: "H", 3: "W"}, "reliab": {2: "h", 3: "w"}},
        do_constant_folding=True,
    )
    print("exported", out)
