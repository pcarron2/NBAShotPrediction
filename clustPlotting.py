'''
Plotting Code
Run after run_me_clust.py
Make sure you point to correct pickle files.

'''

import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy

destPath="../../Figures/"


def plotSDGAll(svmGrid):
	# loss=["hinge"]
	penalty=["l1"]
	alpha=[.00095,.001,.002,.003,.004,.005,.006,.007]
	whiten=[False,True]
	loss=["hinge","log","perceptron"]
	# penalty=["none","l1","l2"]
	scores=[feat[1] for feat in svmGrid]
	newMatrix=np.zeros((len(svmGrid),6),dtype=object)
	for i in range(len(svmGrid)):
		newMatrix[i,0]=svmGrid[i][1]
		newMatrix[i,1]=svmGrid[i][0]['clf__penalty']
		newMatrix[i,2]=svmGrid[i][0]['clf__loss']
		newMatrix[i,3]=svmGrid[i][0]['dim_reduction__whiten']
		newMatrix[i,4]=svmGrid[i][0]['clf__alpha']
		# newMatrix[i,5]=svmGrid[i][0]["clf__penalty"]
	metDict={}
	for met in loss:
		tupList=[]
		for i in range(len(svmGrid)):
		    if newMatrix[i,1]=="l1" and newMatrix[i,2]==met and newMatrix[i,3]==True:
		        # scoreArray.append(newMatrix[i,0])
		        # alphaList.append(newMatrix[i,4])
		        tupList.append((newMatrix[i,4],newMatrix[i,0]))
		#print tupList
		tupList=sorted(tupList,key=lambda x: x[0])
		#print tupList
		alphaList, scoreArray=zip(*tupList)
		metDict[met]=[scoreArray,alphaList]
	typeDict={"hinge":'b-',"log":'g-',"perceptron":'r-'}
	#fig, ax=plt.subplots(figsize=(10,5))
	#for met in metDict.keys():
	#print met, metDict[met]
	labelDict={"hinge":"SVM","log":"Logistic Regression","perceptron":"Perceptron"}
	#print str(labelDict["hinge"])
	for met in metDict.keys():
		print met, metDict[met]
		plt.plot(metDict[met][1],metDict[met][0],typeDict[met], linewidth=3)
    	#plt.plot(kList,testErrList,'sb-', linewidth=3)
	plt.grid(True) #Turn the grid on
	plt.ylabel("Validation Accuracy") #Y-axis label
	plt.xlabel("Alpha") #X-axis label
	plt.title("Classifiers with L1 penalty by Alpha") #Plot title
	plt.xlim(0,1000) #set x axis range
	plt.ylim(.45,.6) #Set yaxis range
	plt.legend(["SVM","Logistic Regrssion","Perceptron"],loc="best")
	#Make sure labels and titles are inside plot area
	plt.tight_layout()
	#Save the chart
	plt.savefig(destPath+"SDG_line_plot_bigRange.pdf")
	plt.show()
	plt.clf()




def plotSDGClose(svmGrid):
	alpha=[.00095,.001,.002,.003,.004,.005,.006,.007]
	whiten=[False,True]
	loss=["hinge","perceptron","log"]
	penalty=["none","l1","l2"]
	scores=[feat[1] for feat in svmGrid]
	newMatrix=np.zeros((len(svmGrid),6),dtype=object)
	for i in range(len(svmGrid)):
		newMatrix[i,0]=svmGrid[i][1]
		newMatrix[i,1]=svmGrid[i][0]['clf__penalty']
		newMatrix[i,2]=svmGrid[i][0]['clf__loss']
		newMatrix[i,3]=svmGrid[i][0]['dim_reduction__whiten']
		newMatrix[i,4]=svmGrid[i][0]['clf__alpha']
		# newMatrix[i,5]=svmGrid[i][0]["clf__penalty"]
	metDict={}
	for met in loss:
		tupList=[]
		scoreArray=[]
		alphaList=[]
		for i in range(len(svmGrid)):
		    if newMatrix[i,1]=="l1" and newMatrix[i,2]==met and newMatrix[i,3]==True:
		        scoreArray.append(newMatrix[i,0])
		        alphaList.append(newMatrix[i,4])
		        #tupList.append((newMatrix[i,4],newMatrix[i,0]))
		#print tupList
		#tupList=sorted(tupList,key=lambda x: x[0])
		#alphaList, scoreArray=zip(*tupList)
		metDict[met]=[scoreArray,alphaList]
	typeDict={"hinge":'b+',"log":'go',"perceptron":'r+'}
	#fig, ax=plt.subplots(figsize=(10,5))
	#for met in metDict.keys():
	#print met, metDict[met]
	labelDict={"hinge":"SVM","log":"Logistic Regression","perceptron":"Perceptron"}
	#print str(labelDict["hinge"])
	for met in metDict.keys():
		plt.plot(metDict[met][1],metDict[met][0],typeDict[met], linewidth=1,mew=7, ms=7)
    	#plt.plot(kList,testErrList,'sb-', linewidth=3)
	plt.grid(True) #Turn the grid on
	plt.ylabel("Validation Accuracy") #Y-axis label
	plt.xlabel("Alpha") #X-axis label
	plt.title("Classifiers with L1 penalty by Alpha") #Plot title
	plt.xlim(0.0001,.012) #set x axis range
	plt.ylim(.45,.6) #Set yaxis range
	plt.legend(["SVM","Perceptron","Logistic Regrssion"],loc="best")
	#Make sure labels and titles are inside plot area
	plt.tight_layout()
	#Save the chart
	plt.savefig(destPath+"SDG_line_plot_smallRange.pdf")
	plt.show()
	plt.clf()


def plotRandomForrestEst(svmGrid):
	clf__n_estimators=[25,30,35,36,37,38,39,40,41,42,43,44,45,50]
	clf__max_depth=[5,6,7,8,9,10,11,12,13,14,15]
	dim_reduction__whiten=[False,True]
	scores=[feat[1] for feat in svmGrid]
	newMatrix=np.zeros((len(svmGrid),5),dtype=object)
	for i in range(len(svmGrid)):
		newMatrix[i,0]=svmGrid[i][1]
		newMatrix[i,1]=svmGrid[i][0]['clf__n_estimators']
		newMatrix[i,2]=svmGrid[i][0]['clf__max_depth']
		newMatrix[i,3]=svmGrid[i][0]['dim_reduction__whiten']
		# newMatrix[i,5]=svmGrid[i][0]["clf__penalty"]
	tupList=[]
	scoreArray=[]
	estimatorList=[]
	for i in range(len(svmGrid)):
		   if newMatrix[i,3]==False and newMatrix[i,2]==10:
		        scoreArray.append(newMatrix[i,0])
		        estimatorList.append(newMatrix[i,1])
	plt.plot(estimatorList,scoreArray,"b-", linewidth=3,)
    	#plt.plot(kList,testErrList,'sb-', linewidth=3)
	plt.grid(True) #Turn the grid on
	plt.ylabel("Validation Accuracy") #Y-axis label
	plt.xlabel("Estimators") #X-axis label
	plt.title("Random Forrest with Tree Depth 10 \n Dev Set Accuracy Vs. Number of Estimators") #Plot title
	plt.xlim(25,50) #set x axis range
	plt.ylim(.58,.6) #Set yaxis range
	#Make sure labels and titles are inside plot area
	plt.tight_layout()
	#Save the chart
	plt.savefig(destPath+"RandomForrestEstimators.pdf")
	plt.show()
	plt.clf()


def plotRandomForrestDepth(svmGrid):
	clf__n_estimators=[25,30,35,36,37,38,39,40,41,42,43,44,45,50]
	clf__max_depth=[5,6,7,8,9,10,11,12,13,14,15]
	dim_reduction__whiten=[False,True]
	scores=[feat[1] for feat in svmGrid]
	newMatrix=np.zeros((len(svmGrid),5),dtype=object)
	for i in range(len(svmGrid)):
		newMatrix[i,0]=svmGrid[i][1]
		newMatrix[i,1]=svmGrid[i][0]['clf__n_estimators']
		newMatrix[i,2]=svmGrid[i][0]['clf__max_depth']
		newMatrix[i,3]=svmGrid[i][0]['dim_reduction__whiten']
		# newMatrix[i,5]=svmGrid[i][0]["clf__penalty"]
	tupList=[]
	scoreArray=[]
	estimatorList=[]
	for i in range(len(svmGrid)):
		   if newMatrix[i,3]==False and newMatrix[i,1]==40:
		        scoreArray.append(newMatrix[i,0])
		        estimatorList.append(newMatrix[i,2])
	plt.plot(estimatorList,scoreArray,"b-", linewidth=3,)
    	#plt.plot(kList,testErrList,'sb-', linewidth=3)
	plt.grid(True) #Turn the grid on
	plt.ylabel("Validation Accuracy") #Y-axis label
	plt.xlabel("Max Tree Depth") #X-axis label
	plt.title("Random Forrest with 40 Estimators \n Dev Set Accuracy Vs. Max Tree Depth") #Plot title
	plt.xlim(5,15) #set x axis range
	plt.ylim(.56,.6) #Set yaxis range
	#Make sure labels and titles are inside plot area
	plt.tight_layout()
	#Save the chart
	plt.savefig(destPath+"RandomForrestDepth.pdf")
	plt.show()
	plt.clf()


def loadFromPickle(filename):
	dictLoad=pickle.load(open(filename,"rb"))
	return copy.deepcopy(dictLoad)


# grid=loadFromPickle('gridCvSVMClustPCAAllInt913.p')
# plotSDGClose(grid)
grid=loadFromPickle('RandomForrestClust812.p')
plotRandomForrestDepth(grid)
plotRandomForrestEst(grid)
