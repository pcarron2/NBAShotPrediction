
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
pd.set_option('display.max_columns', 40)
dataPath="/Users/nugthug/Documents/cmpsci/589/TermProject/Data/PlayerData/rookiePlayerData.csv"


df=pd.read_csv(dataPath,error_bad_lines=False)
df['Pid']=df['Player'].str.split('\\').str.get(1)
df['Player']=df['Player'].str.split('\\').str.get(0)
df['Player']=df['Player'].str.strip('*')
df['Year']=df['Season'].str[:4]


colList=['MP','FG','FGA','2P','2PA','3P','3PA','FT','FTA','ORB','DRB','TRB','AST','STL','TOV','PF','PTS']
for col in colList:
    df[col]=df[col]/df['G']
newDf=df.drop(['Rk','Season','Age','Tm','Lg'],1)

newDf=newDf.sort_values(by='Year')

func='mean'
rookPivot=pd.pivot_table(newDf,index=["Year"],aggfunc=[np.mean])
print list(rookPivot.columns.values)
'''
Below recalculates ratios for rookies

'''
rookPivot[(func,"2P%")]=rookPivot[(func,"2P")]/rookPivot[(func,"2PA")]
rookPivot[(func,"3P%")]=rookPivot[(func,"3P")]/rookPivot[(func,"3PA")]
rookPivot[(func,"FG%")]=rookPivot[(func,"FG")]/rookPivot[(func,"FGA")]
rookPivot[(func,"FT%")]=rookPivot[(func,"FT")]/rookPivot[(func,"FTA")]
rookPivot[(func,"Year")]=rookPivot.index
rookPivot.columns=rookPivot.columns.droplevel()

rookPivot=rookPivot.drop(['TS%','WS','eFG%','TRB'],1)

print rookPivot.index
print rookPivot.to_csv("rookiestats.csv")
#rookPivot.columns=rookPivot.columns.droplevel()
#['MP','FG','FGA','2P','2PA','3P','3PA','FT','FTA','ORB','DRB','TRB','AST','STL','TOV','PF','PTS']
#rookPivot.head()
newDf=newDf.drop(['G','GS','MP','FG','FGA','2P','2PA','3P','3PA','FT',
                  'FTA','ORB','DRB','TRB','AST','STL','BLK','TOV',
                  'PF','PTS','FG%','2P%','3P%','eFG%','FT%','TS%',
                 'WS','Pid'],1)

newDf=newDf.merge(rookPivot,how='left',right_on='Year',left_on='Year')

newDf['Player']="s_"+newDf['Player']
newDf['Player']=newDf['Player'].str.replace(" ",'_').replace("'",'_')
newDf['Year']=newDf['Year'].convert_objects(convert_numeric=True)


print newDf


# In[2]:

dataPath="/Users/nugthug/Documents/cmpsci/589/TermProject/Data/PlayerData/"

dfList=[]
for fn in os.listdir(dataPath):
    if fn[:4] not in ['rook','Rook','corr']:
        season=fn[:4]
        #print season
        fullDataPath=dataPath+fn
        df=pd.read_csv(fullDataPath,error_bad_lines=False)
        df['Pid']=df['Player'].str.split('\\').str.get(1)
        df['Player']=df['Player'].str.split('\\').str.get(0)
        df['Player']=df['Player'].str.strip('*')
        df['Year']=season
        df['Year']=df['Year'].convert_objects(convert_numeric=True)
        '''
        Below are a bunch of manual cleaning steps that are required
        because the player statistics and the play by play data used
        inconsistent naming conventions 
        '''
        df.ix[df.Player=="Metta World Peace","Player"]="Ron Artest"
        df.ix[df.Player=="Didier Ilunga-Mbenga","Player"]="D.J. Mbenga"
        df.ix[df.Player=="J.J. Barea","Player"]="Jose Juan Barea"
        df.ix[df.Player=="Amar'e Stoudemire","Player"]="Amare Stoudemire"
        df.ix[df.Player=="Lou Amundson","Player"]="Louis Amundson"
        df.ix[df.Player=="Wesley Matthews","Player"]="Wes Matthews"
        df.ix[df.Player=="Luc Mbah a Moute","Player"]="Luc Richard Mbah a Moute"
        df.ix[df.Player=="Henry Walker","Player"]="Bill Walker"
        df.ix[df.Player=="Danny Green","Player"]="Daniel Green"
        df['Player']="s_"+df['Player']
        df['Player']=df['Player'].str.replace(" ",'_')
        df['Player']=df['Player'].str.replace("'",'_')
        df=df.drop(['Rk','Tm','PS/G'],1)
        #df=df.merge(newDf,how="left", right_on=["Player","Year"],left_on=["Player","Year"])
        #df.loc[(df.Player==newDf.Player)) & (df.Year==newDf.Year)), ['MP','FG','FGA','2P','2PA','3P','3PA','FT','FTA','ORB','DRB','AST','STL','TOV','PF']] = newDf[['MP','FG','FGA','2P','2PA','3P','3PA','FT','FTA','ORB','DRB','AST','STL','TOV','PF']]
        x1 = df.set_index(['Player', 'Year'])
        x2 = newDf.set_index(['Player', 'Year'])
        x1.update(x2)
        #print x1
        #varList=['MP','FG','FGA','2P','2PA','3P','3PA','FT','FTA','ORB','DRB','AST','STL','TOV','PF']
        df=x1
        df=df.drop(['Pid'],1)
        #df=df.update()
#         for i in range(df.shape[0]):
#             if df.ix[i,"Year_x"]==df.ix[i,"Year_y"]:
#                 df["2P"]=df["2P_y"]
#                 df["2P%"]=df["2P%_y"]
        #df=df.drop(["Tm"],1)
        dfList.append(df)
df=pd.concat(dfList)

#playSet=set(df["Player"])
#print len(df['Player']), len(playSet)
'''
Increment year so that previous year data is used instead
'''
df.to_csv("correctedPlayerSeasonData.csv")
df=pd.read_csv("correctedPlayerSeasonData.csv")
df['Year']=df['Year']+1
df.to_csv("correctedPlayerSeasonData.csv",index=False)            


# In[5]:

#df.columns


# In[3]:

#get_ipython().magic(u'matplotlib inline')
'''
Below is kmeans clustering code over several K
values to create the chart where the clustering Number
is selected.
'''
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
destPath="../../Figures/"
rowsize, colsize=df.shape
df2=df.drop(['Pos','Player'],1)
df2=df2.fillna(value=0)
# print df2
distList=[]
for i in range(1,20):
    km=KMeans(n_clusters=i,
             init='k-means++',
             n_init=10,
             max_iter=500,
             tol=.00001,
             random_state=1)
    km.fit(df2)
    distList.append(km.inertia_)
plt.plot(range(1,20),distList , marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Within Cluster Sum of Squares")
plt.savefig(destPath+"playerClusters.pdf")
plt.show()


# In[4]:
'''
Below creates a file with the offensive defensive and shooting player
name along with their clustering given by the kmeans procedure

'''
#select 5 clusters
km=KMeans(n_clusters=5,
         init='k-means++',
         n_init=10,
         max_iter=300,
         tol=.0001,
         random_state=0)
labels=km.fit_predict(df2)
labelFrame=pd.DataFrame(labels.T,columns=["Cluster"])
labelFrame=pd.concat([df,labelFrame],axis=1)
labelFrame['OffPlayer']="o_"+labelFrame['Player']
labelFrame['OffPlayer']=labelFrame['OffPlayer'].str.replace("_s_","_")
labelFrame['DefPlayer']="d_"+labelFrame['Player']
labelFrame['DefPlayer']=labelFrame['DefPlayer'].str.replace("_s_","_")
cols=['Pos','Age','G','GS','MP','FG','FGA','FG%','3P','3PA',
 '3P%','2P','2PA','2P%','eFG%','FT','FTA','FT%','ORB','DRB','TRB','AST',
 'STL','BLK','TOV','PF']
labelFrame=labelFrame.drop(cols,1)
labelFrame.to_csv("clusterAssign.csv",index=False) 

print labelFrame


# In[ ]:



