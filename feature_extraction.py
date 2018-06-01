# coding=utf-8
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from scipy import sparse
import time
import os
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

ad_feature=pd.read_csv(".../tx/adFeature.csv")
user_feature=pd.read_csv('.../tx/userFeature.csv')
train=pd.read_csv('.../tx/train.csv')
predict=pd.read_csv('.../tx/test1.csv')

print('Preprocessing start1!')

def NULL_STA(data):
	
	total = data.isnull().sum().sort_values(ascending=False)
	percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
	missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
	return missing_data

user_feature['house']=user_feature['house'].fillna(0)
u_miss_data = NULL_STA(user_feature)
user_feature=user_feature.drop(u_miss_data.index[u_miss_data['Percent']>0.8],axis=1)
user_feature=user_feature.fillna('0')
del(u_miss_data)

one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct',
    'marriageStatus','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType']

one_hot_feature1=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct',
    'marriageStatus']
one_hot_feature2=['advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType']
vector_feature=['interest1','interest2', 'interest5','kw1','kw2','topic1','topic2']

print('Preprocessing start2!')

for feature in one_hot_feature1:
	try:
		user_feature[feature] = LabelEncoder().fit_transform(user_feature[feature].apply(int))
	except:
		user_feature[feature] = LabelEncoder().fit_transform(user_feature[feature])

for feature in one_hot_feature2:
	try:
		ad_feature[feature] = LabelEncoder().fit_transform(ad_feature[feature].apply(int))
	except:
		ad_feature[feature] = LabelEncoder().fit_transform(ad_feature[feature])

ad_feature['creativeSize']=ad_feature['creativeSize']/np.max(ad_feature['creativeSize'].values)

def Preprocess(ad_feature,user_feature,train,test,train_cv):
	train_x=train[['creativeSize']]
	test_x=test[['creativeSize']]
	train_x_cv=train_cv[['creativeSize']]
	feature_name=[['creativeSize']]

	print('OneHotEncoder start!')
	enc = OneHotEncoder()
	for feature in one_hot_feature1:
		enc.fit(user_feature[feature].values.reshape(-1, 1))
		feature_name.extend([[feature,'%d'%i] for i in range(enc.feature_indices_[-1])])
		train_a=enc.transform(train[feature].values.reshape(-1, 1))
		train_x= sparse.hstack((train_x, train_a))
		train_a=enc.transform(train_cv[feature].values.reshape(-1, 1))
		train_x_cv= sparse.hstack((train_x_cv, train_a))
		del(train_a)
		test_a = enc.transform(test[feature].values.reshape(-1, 1))
		test_x = sparse.hstack((test_x, test_a))
		del(test_a)
	for feature in one_hot_feature2:
		enc.fit(ad_feature[feature].values.reshape(-1, 1))
		feature_name.extend([[feature,'%d'%i] for i in range(enc.feature_indices_[-1])])
		train_a=enc.transform(train[feature].values.reshape(-1, 1))
		train_x= sparse.hstack((train_x, train_a))
		train_a=enc.transform(train_cv[feature].values.reshape(-1, 1))
		train_x_cv= sparse.hstack((train_x_cv, train_a))
		del(train_a)
		test_a = enc.transform(test[feature].values.reshape(-1, 1))
		test_x = sparse.hstack((test_x, test_a))
		del(test_a)
	print('one-hot prepared !')
	cv=CountVectorizer(token_pattern=r"\d+")
	for feature in vector_feature:
		cv.fit(user_feature[feature])
		feature_name.extend([[feature,'%s'%i] for i in cv.get_feature_names()])
		train_a = cv.transform(train[feature])
		train_x = sparse.hstack((train_x, train_a))
		train_a = cv.transform(train_cv[feature])
		train_x_cv = sparse.hstack((train_x_cv, train_a))
		del(train_a)
		test_a = cv.transform(test[feature])
		test_x = sparse.hstack((test_x, test_a))
		del(test_a)
	print('cv prepared !')
	Name = pd.DataFrame(feature_name)
	Name.to_csv('.../NAME_9.csv',index=None)
	return train_x,test_x,train_x_cv

print('Preprocessing start3!')

train.loc[train['label']==-1,'label']=0
predict['label']=-1

res=predict[['aid','uid']]
data = pd.concat([train,predict])
data = pd.merge(data,ad_feature,on='aid',how='left')
data = pd.merge(data,user_feature,on='uid',how='left')
del(predict)

data=data.fillna('-1')

train_pos=data[data.label==1]
train_neg=data[data.label==0]
test=data[data.label==-1]
del(data)
train=pd.concat([train_pos[0:200000],train_neg[0:400000]],ignore_index=True)
train_cv=pd.concat([train_pos[200000:300000],train_neg[400000:600000]],ignore_index=True)
del(train_pos,train_neg)
test=test.drop('label',axis=1)
test=test[0:100]
train_y=train.pop('label')
train_cv_y=train_cv.pop('label')


train_x, test_x, train_x_cv=Preprocess(ad_feature,user_feature,train,test,train_cv)
del(train,test,ad_feature,user_feature)
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

LGM = lgb.LGBMClassifier(
    boosting_type='gbdt', n_jobs=15, max_depth=-1, subsample_freq=1,
    min_child_weight=50, num_leaves=256, #63  7456   127 748 
    subsample=0.9, colsample_bytree=0.9, reg_alpha=0, reg_lambda=1,
    objective='binary',learning_rate=0.06, 
    n_estimators=1000, random_state=2018
	)

LGM.fit(train_x,train_y, eval_set=[(train_x_cv, train_cv_y)],
	eval_metric='auc',early_stopping_rounds=100) 

print(LGM.n_features_)
print(LGM.feature_importances_.shape)
print(LGM.feature_importances_)
Fea=pd.DataFrame(LGM.feature_importances_)
Fea.to_csv('.../feature_importances.csv',index=None)
