'''
First step in cluster reperesentation processing
Before running this code be sure to run CleanRookieDataClust.py and place the output file in proper directory.
After this run is complete run createMassivFileClust.py
'''

import os
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import collections
import csv



dataPath="/Users/nugthug/Documents/cmpsci/589/TermProject/Data/"
playerFilePath="/Users/nugthug/Documents/cmpsci/589/TermProject/Data/PlayerData/correctedPlayerSeasonData.csv"
clusterPath="/Users/nugthug/Documents/cmpsci/589/TermProject/Data/Cluster/clusterAssign.csv"
offNameConvert={}
defNameConvert={}
shotNameConvert={}
'''
Below populates a dictionary to change the assigned player names to their correct cluster
'''
with open(clusterPath) as f:
	reader=csv.DictReader(f)
	for row in reader:
		#offNameConvert.setdefault((row['Year'],row['OffPlayer']),{}).update('o_'+str(row['Cluster']))
		offNameConvert[(row['Year'],row['OffPlayer'])]='o_'+str(row['Cluster'])
		defNameConvert[(row['Year'],row['DefPlayer'])]='d_'+str(row['Cluster'])
		shotNameConvert[(row['Year'],row['Player'])]='s_'+str(row['Cluster'])
#print defNameConvert


shootingPlayerDict={"2006-2007":"0506","2007-2008":"0607","2008-2009":"0708","2009-2010":"0809"}
landing='/Users/nugthug/Documents/cmpsci/589/TermProject/Data/ClustLanding/'
#seasonList=["2006-2007","2007-2008","2008-2009","2009-2010"]
# seasonList=["2006-2007"]
#seasonList=["2007-2008","2008-2009","2009-2010"]
'''
Loop below reads each game and each season in and cleans it of useless from_records
It then uses individual rows, which team is home and away, and which team is shooting
to assign offensive and defensive status to each player
Then the cluster assignments are joined and summed for each possession
Time is converted to a total second count instead of a combo of period and second
Finally, all the useless columns are removed.
'''
seasonList=["2009-2010"]
fullDataPathList=[]
dfList=[]
playerDf=pd.read_csv(playerFilePath,error_bad_lines=True)
for season in seasonList:
	fullDataPath=dataPath+season+".regular_season/"
	fileList=os.listdir(fullDataPath)
	yrtag=season[5:]
	for i in range(len(fileList)):
	# for i in range(2):
		if fileList[i][0]=='.':
			continue
		#if fileList[i].endswith(".csv"):
		year=fileList[i][:4]
		month=fileList[i][4:6]
		day=fileList[i][6:8]
		awayTeam=fileList[i][9:12]
		homeTeam=fileList[i][12:15]
		# print year, month, day, awayTeam, homeTeam
		# print fullDataPath+fileList[i]
		df=pd.read_csv(fullDataPath+fileList[i],error_bad_lines=True)
		df=df.loc[df['result'].isin(["made","missed"])]
		df=df.loc[df['etype'].isin(['shot'])]
		df=df.drop(['outof','points','possession','reason','steal','type','num','block','entered',
			'home','left','opponent','assist','away','etype'],1)
		df['Year']=year
		df['Month']=month
		df['Day']=day
		df['AwayTeam']=awayTeam
		df['HomeTeam']=homeTeam
		df['LastSeason']=shootingPlayerDict[season]
		print fileList[i], str(i)
		playDict={}
		df=df.reset_index(drop=True)
		for k in range(df.shape[0]):
			#playSet=set([])
			playList=[]
			if df.ix[k,'team']==df.ix[k,'HomeTeam']:
				for j in range(1,6):
					df.ix[k,'h{0}'.format(j)]=offNameConvert[(yrtag,"o_"+df.ix[k,'h{0}'.format(j)].replace("'",'_').replace(" ",'_'))]
					df.ix[k,'a{0}'.format(j)]=defNameConvert[(yrtag,"d_"+df.ix[k,'a{0}'.format(j)].replace("'",'_').replace(" ",'_'))]
					#df.ix[k,'h{0}'.format(j)]=offNameConvert[(df.ix[k,'Year'],df.ix[k,'h{0}'.format(j)])]
										
			else:
				for j in range(1,6):
					df.ix[k,'h{0}'.format(j)]=defNameConvert[(yrtag,"d_"+df.ix[k,'h{0}'.format(j)].replace("'",'_').replace(" ",'_'))]
					df.ix[k,'a{0}'.format(j)]=offNameConvert[(yrtag,"o_"+df.ix[k,'a{0}'.format(j)].replace("'",'_').replace(" ",'_'))]
			for j in range(1,6):
				playList.append(df.ix[k,'a{0}'.format(j)])
				playList.append(df.ix[k,'h{0}'.format(j)])
				#playSet=playSet.union(set([df.ix[k,'a{0}'.format(j)]]))
				#playSet=playSet.union(set([df.ix[k,'h{0}'.format(j)]]))
			df.ix[k,'player']="s_"+df.ix[k,'player'].replace("'",'_').replace(" ",'_')
			df.ix[k,'time']=(float(df.ix[k,'time'].split(":")[0])*60)+float(df.ix[k,'time'].split(":")[1])
			df.ix[k,'time']=df.ix[k,'time']+((4-df.ix[k,'period'])*60*12)
			if df.ix[k,'result']=='missed':
				df.ix[k,'result']=-1
			else:
				df.ix[k,'result']=1
			#playSet=playSet.union(set([df.ix[k,'player']]))
			playList.append(shotNameConvert[(yrtag,df.ix[k,'player'])])
			playDict['{0}'.format(k)]=Counter(playList)
		df2=pd.DataFrame.from_dict(
			{'players':playDict})
		df3=df2['players'].apply(collections.Counter)
		df4=pd.DataFrame.from_records(df3).fillna(value=0)
		result=pd.concat([df,df4],axis=1)
		result=result.drop(['a1','a2','a3','a4','a5','h1','h2','h3','h4','h5'],1)
		result["Year"]=result['Year'].convert_objects(convert_numeric=True)
		result['result']
		result=result.merge(playerDf,how='left',right_on=["Player","Year"],left_on=["player","Year"])
		result=result.fillna(value=0)
		result=result.drop(['player','team','Player','AwayTeam','HomeTeam','LastSeason','Pos','period'],1)
		result.to_csv(landing+season+"_"+str(i)+'.csv',index=False)







