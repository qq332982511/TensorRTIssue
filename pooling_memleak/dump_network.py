from typing import Optional, List, Tuple

import torch
import torch.nn as M
import torch.nn.functional as F


class ConvNet(M.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = M.Conv2d(3, 6, 5)
        self.conv2 = M.Conv2d(6, 6, 5)
        self.pool = M.MaxPool2d(9, stride=5)
        self.relu = M.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv2(x)
        return x


def dump_mobileone(path):
    model = ConvNet()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(
        model,
        x,
        path,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


dump_mobileone("./xxx.onnx")
