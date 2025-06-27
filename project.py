import torch
# print(torch.cuda.is_available())
import pandas as pd
import os
meta= "./symbols_valid_meta.csv"
symbol_meta= pd.read_csv(meta)
print(symbol_meta.info())