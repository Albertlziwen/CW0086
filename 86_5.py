# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:00:21 2021

@author: zczl625
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:46:32 2021

@author: Albert
"""

import numpy as np
import pandas as pd
import time
import random
import seaborn as sns
import matplotlib.pyplot as plt

with open('2.txt',encoding='utf-8') as f:
  book = f.read().lower()
  f.close()
  
with open("symbols.txt","r") as f:
  symbols = f.read()
  f.close()
 
#remove '\n'
symbol_ls = []
for i in range(len(symbols)):
  if i%2 == 0:
    symbol_ls.append(symbols[i])
    
with open("message.txt","r") as f:
  data = f.read()
  f.close()
  
#clean the data
clean_book = []
for i in range(len(book)):
  if book[i] in symbol_ls:
    clean_book.append(book[i])

def tran_matrix():
  transition = np.ones([len(symbol_ls),len(symbol_ls)])
  transition_original = np.zeros([len(symbol_ls),len(symbol_ls)])
  for i in range(len(clean_book)-1):
    cur = clean_book[i]
    ne = clean_book[i+1]
    try:
      transition[symbol_ls.index(cur)][symbol_ls.index(ne)]+= 1
      transition_original[symbol_ls.index(cur)][symbol_ls.index(ne)]+= 1
    except:
      continue
    
  for j in range(transition.shape[0]):
    transition[j] = transition[j]/np.sum(transition[j])
    
  return transition,transition_original

transition, transition_original = tran_matrix()

def stationary(transition):
  #initialise (uniform distribution)
  sta = (1/len(symbol_ls))*np.ones(len(symbol_ls))
  before = np.zeros(len(symbol_ls))
  t = 0
  thre = 1e-100
  while np.sum((sta-before)**2)>=thre:
    t = t+1
    before = sta
    sta = sta.dot(transition)
    
  return sta,t

sta,t = stationary(transition)
#create a dictionary to store the key
def create_cipher(encrypt,decrypt):
  cipher_key = {}
  for i in range(len(encrypt)):
    cipher_key[encrypt[i]] = decrypt[i]
    
  return cipher_key
  
def new_cipher(cipher_):
  cipher = cipher_.copy()
  keys = cipher.keys()
  keys = list(keys)
  a = random.randint(0,len(keys)-1)
  b = random.randint(0,len(keys)-1)
  t = cipher[keys[a]]
  cipher[keys[a]] = cipher[keys[b]]
  cipher[keys[b]] = t
  
  return cipher,cipher[keys[a]],cipher[keys[b]]

def initialization(set1,set2):
  #set = {a:1,b:2,c:3}
  init = {}
  keys = set1.keys()
  for i in range(len(keys)):
    key1 = max(set1,key = set1.get)
    key2 = max(set2,key = set2.get)
    set1.pop(key1)
    set2.pop(key2)
    init[key1] = key2
    
  return init
   
init = [0 for i in range(len(symbol_ls))]
set1 = create_cipher(symbol_ls,init)
for i in range(len(data)):
  set1[data[i]]+=1
  
set2 = create_cipher(symbol_ls,sta)
cipher = initialization(set1,set2)

def MH_sampler(cipher):
  P1 = sta[symbol_ls.index(cipher[data[0]])]
  P1 = np.log(P1)
  for i in range(len(data)-1):
    curr_real = cipher[data[i]]
    nex_real = cipher[data[i+1]]
    P = transition[symbol_ls.index(curr_real),symbol_ls.index(nex_real)]
    P1 += np.log(P)
  
  new,a,b = new_cipher(cipher)
  P2 = sta[symbol_ls.index(new[data[0]])]
  P2 = np.log(P2)
  for i in range(len(data)-1):
    curr_real = new[data[i]]
    nex_real = new[data[i+1]]
    P = transition[symbol_ls.index(curr_real),symbol_ls.index(nex_real)]
    P2 += np.log(P)
  tt = np.exp(P2-P1)
  u = np.random.uniform()
  
  if u<=min(1,tt):
    cipher_ = new.copy()
  else:
    cipher_ = cipher.copy()
  return cipher_.copy()

  
def run_MH(n,cipher):
  for i in range(n):
    cipher = MH_sampler(cipher)
    if i%100 == 0:
      real_text = ''
      for i in range(60):
        p = cipher[data[i]]
        real_text += p
      print(real_text)

run_MH(10000,cipher)
plt.subplot(111)
sns.heatmap(transition)
plt.show()