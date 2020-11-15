#!/usr/bin/env python
# coding: utf-8

# #NOV 2020 PLANS
# 
# ## Three peptide classes 
# 
# - AAGY (FULLY OPEN)
# - ABGY (FOLDED BUT!!)
# - AMV (NICELY FOLDED)
# 
# 
# ## Peptide sequences 
# ### Amino acid **sequence** of the L-Ala, **all-L AAGY** peptide
# 
# Ac-Ala- Ala 2-Ala-Lys-Ala- Ala 6-Lys-Ala-Lys-Ala- Ala 11-Gly-Gly-Tyr-NH2
# 
# ### Amino acid **sequence** of the Nα-acetylated Aib-based, **all-L ABGY** peptide
# 
# Ac-Ala-Aib2-Ala-Lys-Ala-Aib6-Lys-Ala-Lys-Ala-Aib11-Gly-Gly-Tyr-NH2
# 
# ### Amino acid sequence of its **all-L-AMV** peptide analog
# 
# Ac-Ala-AMV2-Ala-Lys-Ala- AMV6-Lys-Ala-Lys-Ala- AMV11-Gly-Gly-Tyr-NH2
# 
# ## For each peptide T-variation and TFE (helix inducing solvent) variation is avialable 
# ## Key observation  
# - AAGY and ABGY shows NO isosbestic point in water
# - AAGY at higher TFE and ABGY at lower TFE shows isosbestic point
# - AMV shows clear isosbestic point in all condition (water/TFE)
# - Sharpness of isosbestic point enhances with 
#   - increase of TFE for AAGY and ABGY
#   -For AMV sharpness in isosbestic point is moderate even in only water.
# 
# ## Questions asked 
# We want to explore
# - whether can we get any thermodynamic correlation between three states especially in water (as in drug design water is acting as only solvent) and also in TFE
# - can appearance of isosbestic point and its nature of sharpness indicates/add something more in its thermodynamic parameters
# - can  these three sets of peptides and their thermodynamic aspects be used as model/template for understanding the characteristics/behavior of protein system (e.g IDP, induced helical system or fully ordered).

# In[ ]:





# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import networkx as nx
from sklearn.preprocessing import StandardScaler


# # Let us first explore the spectral data character 
# - There are two ways of reading pandas files 
# - df=pd.read_excel(xlsfile,sheet_name=<sheetName>) 
# - df=pd.ExcelFile(<xlsfile>) . We can find out the sheet names by df.sheet_names
# - For simplicity we shall first explore all the 0% TFE data 

# In[3]:


os.chdir("/content/drive/My Drive/Raja_CD/")


# In[4]:


get_ipython().system('ls *.xlsx')


# # We may note that df11_amv contains the 0% TFE data for AMV 

# In[6]:


os.chdir("/content/drive/My Drive/Raja_CD/")
df1_amv=pd.ExcelFile('AMV_Data_All.xlsx')
n1_amv=df1_amv.sheet_names
print(n1_amv)
df11_amv=pd.read_excel('AMV_Data_All.xlsx',sheet_name=n1_amv[1])
df11_amv # 11 stands for 0 %TFE 


# In[ ]:


os.chdir("/content/drive/My Drive/Raja_CD/AAGY")
df1_aagy=pd.ExcelFile('AAGY_Function of temperature.xlsx')
n1_aagy=df1_aagy.sheet_names
n1_aagy


# #*df11_AAGY* 0% TFE for AAGY

# In[ ]:


df11_aagy=pd.read_excel('AAGY_Function of temperature.xlsx',sheet_name=n1_aagy[1])
df11_aagy


# # df11_ABGY 5% TFE for ABGY (0% TFE data not available)
# 

# In[ ]:


os.chdir("/content/drive/My Drive/Raja_CD/AAGY")
df1_abgy=pd.ExcelFile('ABGY_Function of temperature.xlsx')
n1_abgy=df1_abgy.sheet_names
n1_abgy

df11_abgy=pd.read_excel('ABGY_Function of temperature.xlsx',sheet_name=n1_abgy[1])

df11_abgy


# # Now let us have the temperature varying spectra for AMV, AAGY and ABGY 

# In[ ]:


def findT(df):
  T=[]
  TTCD=df
  ll=list(TTCD.columns.values)
  ll.remove(ll[0])
  return np.array(ll)


# # FOR AMV 

# In[ ]:


T_amv=findT(df11_amv)
T_amv


# ## For AAGY 

# In[ ]:


T_aagy=findT(df11_aagy)
T_aagy


# # Since the temperature is coming in a format that contains a string we have to strip the string C 

# In[ ]:


def tonumbers(s):
  # 20C=>20
  S=[]
  for i in s:
    S.append(int(re.search(r"\d+", i).group(0)))
  return(S)
T_aagyn=tonumbers(T_aagy)
T_aagyn


# ## For ABGY 

# In[ ]:


T_abgy=findT(df11_abgy)
T_abgy # Comes in numbers - no conversion required


# # Now Let us draw the spectra for each set (AMV,AAGY,ABGY)at 0% (or at 5% TFE, when 0% data is not available)  

# In[ ]:


def readCDT(df,T,pepname):# T must be in numbers 
  
  X=df.values
  lam=X[:,0]
  lam=np.delete(lam,0)
  X=np.delete(X, 0, axis=1)
  X=np.delete(X, 0, axis=0)
  r,c=X.shape
  plt.figure(figsize=(8, 7))
  for i in range(c):
      plt.plot(lam,X[:,i],label=str(T[i]))
      plt.ylabel(r'$[\theta ^o] cm^2 dmol^{-1}\cdot 10^{-3} $')
      plt.xlabel(r'$\lambda (nm)$')
      plt.title(['$T$ Variation in C',pepname])
      plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
  return T,lam,X 


# In[ ]:


T_amv,lam_amv,X_amv = readCDT(df11_amv,T_amv,'AMV')


# In[ ]:





# In[ ]:


T_aagy,lam_aagy_amv,X_aagy = readCDT(df11_aagy,T_aagyn,'AAGY')


# In[ ]:


T_abgy,lam_abgy_amv,X_abgy = readCDT(df11_abgy,T_abgy,'AbGY')

