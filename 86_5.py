# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

df = pd.read_table("WAL.txt")
df.columns = [0]

with open("message.txt","r") as f:
    data = f.read()
    f.close()
    
with open("symbols.txt","r") as f:
    symbols = f.read()
    f.close()
    
symbol_ls = []
for i in range(len(symbols)):
    if i%2 == 0:
        symbol_ls.append(symbols[i])

        
transition = np.zeros(len(symbol_ls)**2).reshape(len(symbol_ls),len(symbol_ls))
for i in range(len(symbol_ls)):
    for j in range(len(symbol_ls)):
        for k in range(df.shape[0]-1):
            for w in range(len(df[0][k])-1):
                if df[0][k][w] == symbol_ls[i] and df[0][k][w+1] == symbol_ls[j]:
                    transition[i][j]+=1
                