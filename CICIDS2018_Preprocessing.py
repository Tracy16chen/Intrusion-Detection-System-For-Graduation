# coding: utf-8

import numpy as np
from glob import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler

InD = np.zeros((0,80),dtype=object)
for x in glob('./temp/*.csv'):
    print(x)
    #载入数据
    InD=np.vstack((InD,pd.read_csv(x, low_memory=False)))

for i in range(InD.shape[0]):
    temp = InD[i,2]
    temp = temp[3]+temp[4]+temp[0]+temp[1]+temp[11]+temp[12]+temp[14]+temp[15]+temp[17]+temp[18]
    InD[i,2] = float(temp)
    # print(temp)

# InD=InD[:,3:]
print(InD.shape)

#取lable外的特征
Dt=InD[:,:-1].astype(float)

# #选不为空的特征中的lable值
LNMV=InD[~np.isnan(InD).any(axis=1),-1]
# #选非lable值不为空的
DtNMV=Dt[~np.isnan(Dt).any(axis=1)]

LNMIV=LNMV[~np.isinf(DtNMV).any(axis=1)]
print(LNMIV.shape)
# print(LNMIV[:10])
DtNMIV=DtNMV[~np.isinf(DtNMV).any(axis=1)]
print(DtNMIV.shape)

del(DtNMV)

np.save('NBx', MinMaxScaler().fit_transform(DtNMIV))
np.save('NBy', (LNMIV!='Benign').astype(int))
# np.save('./DistKeras/NBy',(LNMIV=='BENIGN').astype(int).reshape(-1,1))