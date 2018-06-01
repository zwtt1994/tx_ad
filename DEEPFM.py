# coding=utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from scipy import sparse
import time
import os

all_feature=['house','age','carrier','consumptionAbility','education','gender','os',
        'ct','marriageStatus','creativeSize','advertiserId','campaignId', 'creativeId',
        'adCategoryId', 'productId', 'productType','interest1','interest2', 'interest5',
        'kw1','kw2','topic1','topic2']
one_hot_feature1=['house','age','carrier','consumptionAbility','education','gender','os']
one_hot_feature2=['ct','marriageStatus']
vector_feature=['interest1','interest2', 'interest5','kw1','kw2','topic1','topic2']
aid_feature=['creativeSize','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType']
uid_feature=['house','age','carrier','consumptionAbility','education','gender','os',
        'ct','marriageStatus','interest1','interest2', 'interest5','kw1','kw2','topic1','topic2']

def read_feature():
    
    user_feature=pd.read_csv('.../tx/userFeature.csv')
    user_feature['house']=user_feature['house'].fillna(0)
    user_feature=user_feature.drop(['interest3','interest4','appIdAction','appIdInstall','kw3','topic3'],axis=1)
    user_feature=user_feature.fillna('0')
    
    for feature in one_hot_feature1:
        try:
            user_feature[feature] = LabelEncoder().fit_transform(user_feature[feature].apply(int))
        except:
            user_feature[feature] = LabelEncoder().fit_transform(user_feature[feature])
    
    ad_feature=pd.read_csv('.../ad_x.csv')
    NAME=pd.read_csv('.../NAME_9.csv')
    F_importance=pd.read_csv('.../feature_importances.csv')
    NAME=NAME[F_importance['0']>100][1:] #225
    NAME['1']=NAME['1'].apply(int)

    return user_feature,ad_feature,NAME

def prepare_data(data,user_feature,ad_feature,NAME,Siz_Ind=True,Test=False):
    print('Preprocessing start!')
    if(not Test):
        data.loc[data['label']==-1,'label']=0
    data=pd.merge(data,user_feature,on='uid',how='left')
    data_x=data[['aid']]
    data_x=pd.merge(data_x,ad_feature,on='aid',how='left')
    data_x=data_x.drop('aid',axis=1)

    print('OneHot start!')
    enc = OneHotEncoder()
    for feature in one_hot_feature1:
        enc.fit(user_feature[feature].values.reshape(-1, 1))
        Dummy=pd.DataFrame(enc.transform(data[feature].values.reshape(-1, 1)).toarray(),
            columns=[feature+str(i) for i in range(enc.n_values_[0])])
        data_x=pd.concat([data_x,Dummy],axis=1)
    del(enc,Dummy)
    print('CV start!')

    cv=CountVectorizer(token_pattern=r"\d+")
    for feature in one_hot_feature2:
        cv.fit(user_feature[feature])
        Dummy=pd.DataFrame(cv.transform(data[feature]).toarray(),
            columns=[feature+str(i) for i in range(len(cv.get_feature_names()))])
        data_x=pd.concat([data_x,Dummy],axis=1)
    if(not Test):
        data_y=data.pop('label')
    del(cv,Dummy)

    print('Vector start!')
    for feature in vector_feature:
        feature_name=NAME['1'][NAME['0']==feature]
        for fn in feature_name:
            data_x[feature+'_'+str(fn)]=data[feature].apply(lambda x: 1 if str(fn) in x else 0)
    del(data)
    print('Preprocessing end!')
    if(Test):
        return data_x
    if(Siz_Ind):
        return data_x,data_y

    Siz={}
    Ind={}
    for feature in all_feature:
        index = [True if feature in data_x.columns[i] else False 
            for i in range(len(data_x.columns))]
        Siz[feature]=np.sum(index)
        Ind[feature]=index
    return data_x,data_y,Siz,Ind


def DeepFM(EMB_input):
    EMB_layer={}
    for feature in all_feature:
        EMB_layer[feature]=tf.layers.dense(inputs=EMB_input[feature],units=10,activation=tf.nn.elu)

    FM_layer={}
    for aidF in aid_feature:
        for uidF in uid_feature:
            FM_layer[aidF+uidF]=tf.reshape(tf.diag_part(tf.matmul(EMB_layer[aidF], EMB_layer[uidF],transpose_b=True)),[-1,1])

    FM_out=tf.reduce_sum([FM_V for FM_V in FM_layer.values()],0)
    EMB_all=tf.concat([EMB_layer[feature] for feature in all_feature],1)
    Hiden_1=tf.layers.dense(inputs=EMB_all,units=10,activation=tf.nn.elu)
    Hiden_2=tf.layers.dense(inputs=Hiden_1,units=10,activation=tf.nn.elu)
    net_out=tf.concat([FM_out,Hiden_2],1)
    output=tf.layers.dense(inputs=net_out,units=1,activation=tf.nn.sigmoid)
    return output

if __name__ == "__main__":
    TRAIN_NUMBER=8798814
    read_batch=430000
    Round=20
    batch_size=86
    batch_round=5000

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print('Program start!')

    reader=pd.read_table('C:/Users/ZWT/Desktop/mofan/tx/train.csv', sep=',', iterator=True)
    user_feature,ad_feature,NAME=read_feature()
    train=reader.get_chunk(1000)
    train_xcv,train_ycv,Siz,Ind=prepare_data(train,user_feature,ad_feature,NAME,Siz_Ind=False)
    EMB_input={}

    for feature in all_feature:
        EMB_input[feature]=tf.placeholder(tf.float32, shape=(None,Siz[feature]))
    tf_y=tf.placeholder(tf.float32, shape=(None,1))
    y_pred=DeepFM(EMB_input)

    LR = 0.0001
    loss = tf.losses.log_loss(tf_y,y_pred)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    cv_emb={}
    for feature in all_feature:
        cv_emb[feature] = train_xcv.loc[:,Ind[feature]]
    del(train_xcv)

    feedcv={EMB_input[feature]:cv_emb[feature]
        for feature in all_feature}
    feedcv[tf_y]=train_ycv.values.reshape(-1,1)

    for b_i in range(Round):
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('Round %s!'%b_i)
        train = reader.get_chunk(read_batch)
        train_x,train_y = prepare_data(train,user_feature,ad_feature,NAME)

        train_emb={}
        for feature in all_feature:
            train_emb[feature] = train_x.loc[:,Ind[feature]]
        del(train_x)
        print('Trainning start!')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        for batch_i in range(batch_round):
            feed={EMB_input[feature]:train_emb[feature][batch_i*batch_size:(batch_i+1)*batch_size]
                for feature in all_feature}
            feed[tf_y]=train_y[batch_i*batch_size:(batch_i+1)*batch_size].values.reshape(-1,1)
            sess.run(train_op,feed_dict=feed)
            if(batch_i%1000==0):
                print(sess.run(loss,feed_dict=feedcv))

    print('Predict start!')
    TEST_SHAPE=2265989
    test_batch=453000
    read_round=5
    pred_batch=1000
    test_round=453
    reader_test=pd.read_table('C:/Users/ZWT/Desktop/mofan/tx/test1.csv', sep=',', iterator=True)
    res=np.array([]).reshape(-1,1)
    for b_i in range(read_round):
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('Round %s!'%b_i)
        test = reader_test.get_chunk(test_batch)
        test_x = prepare_data(test,user_feature,ad_feature,NAME,Test=True)

        emb_test={}
        for feature in all_feature:
            emb_test[feature] = test_x.loc[:,Ind[feature]]
        del(test_x)
        for batch_i in range(test_round):
            feed_test={EMB_input[feature]:emb_test[feature][batch_i*pred_batch:(batch_i+1)*pred_batch]
                for feature in all_feature}
            res=np.vstack([res,sess.run(y_pred,feed_dict=feed_test)])
        print(res.shape)
    test = reader_test.get_chunk(test_batch)
    test_x = prepare_data(test,user_feature,ad_feature,NAME,Test=True)

    emb_test={}
    for feature in all_feature:
        emb_test[feature] = test_x.loc[:,Ind[feature]]
    del(test_x)
    feed_test={EMB_input[feature]:emb_test[feature] for feature in all_feature}
    res=np.vstack([res,sess.run(y_pred,feed_dict=feed_test)])
    print(res.shape)

    RES_pd=pd.read_csv('.../tx/test1.csv')
    RES_pd['score'] = res
    RES_pd.to_csv('.../submission.csv',index=None)
        


        



