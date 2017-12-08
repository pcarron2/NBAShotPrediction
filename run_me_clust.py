import numpy as np 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import PredefinedSplit
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from scipy import sparse
import pickle, copy
import time
import datetime

def loadData():
	path="../../Data/npyFiles/Clust/"
	dev_features=np.load(path+"dev_features.npy")
	dev_labels=np.load(path+"dev_labels.npy")
	test_features=np.load(path+"test_features.npy")
	test_labels=np.load(path+"test_labels.npy")
	train_features=np.load(path+"train_features.npy")
	train_labels=np.load(path+"train_labels.npy")
	return train_features, train_labels, dev_features, dev_labels, test_features, test_labels
'''
Below are basline Naive Bayes and SDG tests
'''
def NBbase():
	train_features, train_labels, dev_features, dev_labels, test_features, test_labels= loadData()
	clf=BernoulliNB(alpha=.001)
	clf.fit(train_features,train_labels.ravel())
	clf.predict(test_features)
	print clf.score(test_features,test_labels)

def SVMbase():
	train_features, train_labels, dev_features, dev_labels, test_features, test_labels= loadData()
	clf=SGDClassifier(loss='hinge',penalty='l2',n_jobs=-1)
	clf.fit(train_features,train_labels.ravel())
	clf.predict(test_features)
	print clf.score(test_features,test_labels)

def NBPipeLine():
	'''
	Below is the BernoulliNB Experiment Pipeline
	'''
	train_features, train_labels, dev_features, dev_labels, test_features, test_labels= loadData()
	trainSize=train_features.shape[0]
	devSize=dev_features.shape[0]
	print devSize
	test_fold=np.append(np.array([-1]*trainSize),np.array([0]*devSize))
	train_features=np.concatenate((train_features,dev_features))
	print dev_labels.shape
	print train_labels
	train_labels=np.hstack((train_labels,dev_labels))
	print train_features.shape
	print train_labels.shape
	ps=PredefinedSplit(test_fold=test_fold)
	print test_fold
	print  datetime.datetime.time(datetime.datetime.now())
	pipeFeat=[
			('addFeatures',PolynomialFeatures(2,interaction_only=False)),
			('dim_reduction',PCA()),
			('clf',BernoulliNB())]
	gridParam=dict(
			#clf__alpha=[.01,.001,1,.0001,10,100,800,1000,1100,10000])
			clf__alpha=[95,96,97,98,99,100,101,102,103,104],
			dim_reduction__whiten=[False,True])
	pipe=Pipeline(pipeFeat)
	ps=PredefinedSplit(test_fold=test_fold)
	gridCV=GridSearchCV(pipe,param_grid=gridParam,cv=ps,n_jobs=-1)
	print  datetime.datetime.time(datetime.datetime.now())
	gridCV.fit(train_features,train_labels)
	print  datetime.datetime.time(datetime.datetime.now())
	print "NAIVE BAYES TEST"
	print "Best Estimator: "+str(gridCV.best_estimator_)
	print "Best Params: "+ str(gridCV.best_params_)
	print "Score: "+str(gridCV.best_score_)
	predict=gridCV.predict(test_features)
	print "Test Accuracy:"+ str(gridCV.score(test_features,test_labels))
	return gridCV.grid_scores_



def SGDPipeLine():
	'''
	Below is the SGD Experiment Pipeline
	'''
	train_features, train_labels, dev_features, dev_labels, test_features, test_labels= loadData()
	trainSize=train_features.shape[0]
	devSize=dev_features.shape[0]
	test_fold=np.append(np.array([-1]*trainSize),np.array([0]*devSize))
	train_features=np.concatenate((train_features,dev_features))
	train_labels=np.hstack((train_labels,dev_labels))
	print train_features.shape
	print train_labels.shape
	ps=PredefinedSplit(test_fold=test_fold)
	print test_fold
	print  datetime.datetime.time(datetime.datetime.now())
	pipeFeat=[
			('addFeatures',PolynomialFeatures(2,interaction_only=False)),
			('dim_reduction', PCA()),
			('clf',SGDClassifier())]
	gridParam=dict(
			clf__loss=["hinge","log","perceptron"],
			clf__penalty=["l1"],
			clf__alpha=[.001,.002,.003,.004,.005,.006,.007,.008,.009,.01,.011,.012],
			dim_reduction__whiten=[True])
	pipe=Pipeline(pipeFeat)
	ps=PredefinedSplit(test_fold=test_fold)
	print  datetime.datetime.time(datetime.datetime.now())
	gridCV=GridSearchCV(pipe,param_grid=gridParam,cv=ps,n_jobs=-1)
	print  datetime.datetime.time(datetime.datetime.now())
	gridCV.fit(train_features,train_labels)
	print  datetime.datetime.time(datetime.datetime.now())
	print "Best Estimator: "+str(gridCV.best_estimator_)
	print "Best Params: "+ str(gridCV.best_params_)
	print "Score: "+str(gridCV.best_score_)
	predict=gridCV.predict(test_features)
	print "Test Accuracy:"+ str(gridCV.score(test_features,test_labels))
	print  datetime.datetime.time(datetime.datetime.now())
	return gridCV.grid_scores_




def randomForest():
	'''
	Below is the Random Forest Experiment Pipeline
	'''
	train_features, train_labels, dev_features, dev_labels, test_features, test_labels= loadData()
	trainSize=train_features.shape[0]
	devSize=dev_features.shape[0]
	test_fold=np.append(np.array([-1]*trainSize),np.array([0]*devSize))
	train_features=np.concatenate((train_features,dev_features))
	train_labels=np.hstack((train_labels,dev_labels))
	print train_features.shape
	print train_labels.shape
	ps=PredefinedSplit(test_fold=test_fold)
	print "Starting Random Forest experiment"
	print test_fold
	print  datetime.datetime.time(datetime.datetime.now())
	pipeFeat=[
			('addFeatures',PolynomialFeatures(2,interaction_only=False)),
			('dim_reduction', PCA()),
			('clf',RandomForestClassifier())]
	gridParam=dict(
			clf__n_estimators=[25,30,35,36,37,38,39,40,41,42,43,44,45,50],
			clf__max_depth=[5,6,7,8,9,10,11,12,13,14,15],
			dim_reduction__whiten=[False,True])
	pipe=Pipeline(pipeFeat)
	ps=PredefinedSplit(test_fold=test_fold)
	gridCV=GridSearchCV(pipe,param_grid=gridParam,cv=ps,n_jobs=-1)
	print  datetime.datetime.time(datetime.datetime.now())
	gridCV.fit(train_features,train_labels)
	print "Best Estimator: "+str(gridCV.best_estimator_)
	print "Best Params: "+ str(gridCV.best_params_)
	print "Score: "+str(gridCV.best_score_)
	predict=gridCV.predict(test_features)
	print "Test Accuracy:"+ str(gridCV.score(test_features,test_labels))
	print  datetime.datetime.time(datetime.datetime.now())
	return gridCV.grid_scores_





def saveToPickle(dict,name):
	pickle.dump(dict,open(name,"wb"))
	return

def loadFromPickle(filename):
	dictLoad=pickle.load(open(filename,"rb"))
	return copy.deepcopy(dictLoad)

capture=randomForest()
saveToPickle(capture,'RandomForrestClust912.p')

capture=NBPipeLine()
saveToPickle(capture,'NBPipeLinePCAAllIntFewAlpha812.p')

capture=SGDPipeLine()
saveToPickle(capture,'gridCvSVMClustPCAAllInt913.p')


#pcaPlot()
#capture=NBPipeLine()
#saveToPickle(capture,'NBPipeLinePCA20.p')
