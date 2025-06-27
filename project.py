import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataset= pd.read_csv("Cleaned_data.csv")
X= dataset.drop("Beats_spy", axis=1)
y= dataset["Beats_spy"]
# print(X.info(), y.info())
# converting to np arrays because scikit learn expects those
X= np.array(X)
y= np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=16)
# print (len(X_train), len(X_test), len(y_train), len(y_test))

# 
class MLP(torch.nn.Module):
    def __init__()