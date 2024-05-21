import torch
import torch.nn as nn

# MFAA: multi-scale feature alignment and aggregation
__all__ = ["MFAA"]


class MFAA(nn.Module):
    def __init__(self, inplanes, outplanes, instrides, outstrides):
        super(MFAA, self).__init__()

        assert isinstance(inplanes, list)
        assert isinstance(outplanes, list) and len(outplanes) == 1
        assert isinstance(outstrides, list) and len(outstrides) == 1
        assert outplanes[0] == sum(inplanes)  # concat
        self.inplanes = inplanes
        self.outplanes = inplanes
        self.instrides = instrides
        self.outstrides = outstrides
        self.align_stride = instrides[0]
        self.patch_size = outstrides[0] // instrides[0]
        self.scale_factors = [
            in_stride / self.align_stride for in_stride in instrides
        ]  # for resize
        self.upsample_list = [
            nn.Upsample(scale_factor=scale_factor)
            for scale_factor in self.scale_factors
        ]
        self.avg_pool = nn.AvgPool2d(self.patch_size, self.patch_size)

    def forward(self, input):
        features = input["features"]
        assert len(self.inplanes) == len(features)

        feature_list = []
        # resize & concatenate
        for i in range(len(features)):
            upsample = self.upsample_list[i]
            feature_resize = upsample(features[i])
            feature_list.append(feature_resize)

        feature_align = torch.cat(feature_list, dim=1)
        feature_align = self.avg_pool(feature_align)

        return {"feature_align": feature_align, "outplane": self.get_outplanes()}

    def get_outplanes(self):
        return self.outplanes

    def get_outstrides(self):
        return self.outstrides
