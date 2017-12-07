import os
import pandas as pd
landing='/Users/nugthug/Documents/cmpsci/589/TermProject/Data/ClustLanding/'
dfList=[]
for fn in os.listdir(landing):
	if fn[0]!='.':
		df=pd.read_csv(landing+fn,error_bad_lines=False)
		dfList.append(df)
	
df=pd.concat(dfList).fillna(value=0)
print df
print df.size
df.to_csv('/Users/nugthug/Documents/cmpsci/589/TermProject/Data/Cluster/bigDataCleanClust2.csv',error_bad_lines=False,index=False)
print df

# df=pd.read_csv('bigDataClean3.csv',error_bad_lines=True)
# cols=df.columns

'''
This code simply creates on large csv
from all of the individual game files.

'''