import pandas as pd
import sys
import torch
import numpy as np 

pt = sys.argv[1]
out = sys.argv[2]

info = torch.load(pt)

request_type = sys.argv[3]
if request_type == "gene_sample":
   idx = int(sys.argv[4])
   obj = info["smpl"]["tx"].squeeze()[:,:,idx].t()
   fmt = "%.0f"
if request_type == "sm":
   obj = info["smpl"]["sm"].squeeze()[:,:,].mean(dim=0)
   fmt = "%.5f"
elif request_type == "param":
   obj = info["params"][sys.argv[4]]
   obj = obj.view(-1,obj.shape[-1])
   fmt = "%.5f"

pd.DataFrame(obj.numpy()).to_csv(
      out, # sys.argv[2]
      sep = "\t",
      header = False,
      index = False,
      float_format = fmt
)
