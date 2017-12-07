import pandas as pd 
import numpy as np 
import os
from scipy import sparse

'''
This code simply creates train validation test labels and feature numpy 
matrices.
'''
path="../../Data/SplitCSV/Clust/"
destPath="../../Data/npyFiles/Clust/"
fnList=["devDf.csv","testDf.csv","trainDf.csv"]
labList=["dev","test","train"]
for i in range(len(fnList)):
	df=pd.read_csv(path+fnList[i], error_bad_lines=True)
	#df=df.drop(['Unnamed: 0.1', 'Unnamed: 0.1',u'Unnamed: 0.1.1',u'Unnamed: 0'],1)
	print df.columns
	cols=[u'2P', u'2P%', u'2PA', u'3P', u'3P%', u'3PA', u'AST',
       u'Age', u'BLK', u'DRB', u'FG', u'FG%', u'FGA', u'FT', u'FT%', u'FTA',
       u'G', u'GS', u'MP', u'ORB', u'PF', u'STL', u'TOV', u'TRB',
       u'd_0', u'd_1', u'd_2', u'd_3', u'd_4', u'eFG%',
       u'o_0', u'o_1', u'o_2', u'o_3', u'o_4', u'result', u's_0', u's_1',
       u's_2', u's_3', u's_4', u'time', u'x', u'y']
	features=df.ix[:,df.columns !="result"]
	labels=df.ix[:,df.columns=="result"]
	features=features.as_matrix(columns=cols)
	print features[0,:]
	labels=labels.as_matrix(columns=["result"])
	labels=labels.ravel()
	#labels=labels.reshape((1,labels.shape[0]))
	print labels.shape
	print labels
	print features.shape
	np.save(destPath+labList[i]+"_features",features)
	np.save(destPath+labList[i]+"_labels",labels)


