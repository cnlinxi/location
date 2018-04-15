# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
import lightgbm as lgb    

df=pd.read_csv('~/data/train_ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv('~/data/ccf_first_round_shop_info.csv')
test=pd.read_csv('~/data/AB_test_evaluation_public.csv')
df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')
df['time_stamp']=pd.to_datetime(df['time_stamp'])
train=pd.concat([df,test])
mall_list=list(set(list(shop.mall_id)))
result=pd.DataFrame()
for mall in mall_list:
    train1=train[train.mall_id==mall].reset_index(drop=True)       
    l=[]#l是一个存储字典的list，字典形式:{'bssid':'signal'}
    wifi_dict = {}#wifi_dict是一个字典，字典形式：{'bssid':'出现次数'}
    for index,row in train1.iterrows():
        r = {}
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            r[i[0]]=int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]]=1
            else:
                wifi_dict[i[0]]+=1
        l.append(r)    
    delate_wifi=[]#delate_wifi是一个存储bssid的list，存储出现次数不超过15次的bssid
    for i in wifi_dict:
        if wifi_dict[i]<15:
            delate_wifi.append(i)
    m=[]#m是一个存储字典的list，字典形式：{'bssid':'signal'}，该list存储出现次数超过15次的bssid形成的字典
    for row in l:
        new={}
        for n in row.keys():
            if n not in delate_wifi:
                new[n]=row[n]
        m.append(new)
    train1 = pd.concat([train1,pd.DataFrame(m)], axis=1)#另一种形式的one-hot，不过出现的bssid标出现的次数，该bssid必须出现15次以上
    df_train=train1[train1.shop_id.notnull()]
    df_test=train1[train1.shop_id.isnull()]
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))    
    num_class=df_train['label'].max()+1    
    params = {
            'objective': 'multi:softmax',
            'eta': 0.1,
            'max_depth': 9,
            'eval_metric': 'merror',
            'seed': 0,
            'missing': -999,
            'num_class':num_class,
            'silent' : 1
            }
    #特征中只存在经纬度和wifi_info形成的'one-hot'
    feature=[x for x in train1.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos']]    
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    xgbtest = xgb.DMatrix(df_test[feature])
    watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
    num_rounds=100
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
    df_test['label']=model.predict(xgbtest)
    df_test['shop_id']=df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
    r=df_test[['row_id','shop_id']]
    result=pd.concat([result,r])
    result['row_id']=result['row_id'].astype('int')
    result.to_csv('~/data/sub.csv',index=False)
