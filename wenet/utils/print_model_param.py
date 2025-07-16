import os
import sys
import torch

model = torch.load(sys.argv[1])

with open(sys.argv[2],'w',encoding="utf-8") as file:
    for name,param in model.items():
        # 名称，形状，精度，参数
        file.write(f"{name}-{param.shape}-{param.dtype}\n{param}\n\n")
        pass
    pass




