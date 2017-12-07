import pandas as pd
import numpy as np 
import os

'''
Here is where the train validation test split is made
since lineups are frozen at a certain point in the season
and the validation and test set had to be future data I made sure
that the training set recieved at least one month of records that
would be in the validation and test sets.
I split the validation and test sets about half way between the reamaing
season in number of days.

'''
path='/Users/nugthug/Documents/cmpsci/589/TermProject/Data/Cluster/'
fn="bigDataCleanClust2.csv"

df=pd.read_csv(path+fn,error_bad_lines=True)
df=df.drop(["Unnamed: 0"],1)
#print list(df.columns.values)
print df.shape
trainDf=df[df.Month.isin([10,11,12,1,2])&df.Year.isin([2009,2010,2008])|df.Year.isin([2009,2008])]
devDf=df[(df.Month.isin([3])) & (df.Day.isin([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]))&df.Year.isin([2010])]
testDf1=df[(df.Month.isin([3]))& (df.Day.isin([19,20,21,22,23,24,25,26,27,28,29,30,31]))&df.Year.isin([2010])]
testDf2=df[df.Month.isin([4])& df.Year.isin([2010])]
testDf=pd.concat([testDf1,testDf2]).fillna(value=0)
trainDf=trainDf.drop(['Year','Month','Day'],1)
devDf=devDf.drop(['Year','Month','Day'],1)
testDf=testDf.drop(['Year','Month','Day'],1)

print trainDf.shape
print devDf.shape
print testDf.shape

trainDf.to_csv("../../Data/SplitCSV/Clust/trainDf.csv",index=False)
devDf.to_csv("../../Data/SplitCSV/Clust/devDf.csv",index=False)
testDf.to_csv("../../Data/SplitCSV/Clust/testDf.csv",index=False)