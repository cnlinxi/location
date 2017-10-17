# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

malls=pd.read_csv('mall_test1.csv')#mall_test1.csv
malls=malls['mall_id'].values.tolist()
print('malls type:',type(malls))

for mall in malls:
    train=pd.read_csv('3train_'+mall+'.csv')
    train=train[['bssid','signal','shop_id']]
    test=pd.read_csv('3test_'+mall+'.csv')
    test=test[['row_id','bssid','signal']]

    train_x=train[['bssid','signal']]
    train_y=train[['shop_id']]
    test_x=test[['bssid','signal']]
    test_pred=test[['row_id']]

    model=GradientBoostingRegressor(n_estimators=102,learning_rate=0.1,min_samples_leaf=12,max_depth=4,subsample=0.77,verbose=2)
    model.fit(train_x,train_y)
    test_pred['shop_id']=model.predict(test_x)
    test_pred.drop_duplicates(inplace=True)
    test_pred.to_csv('gart_pred_'+mall+'.csv',index=None)
    #break#用于调试

ls_pred=[]
for mall in malls:
    pred=pd.read_csv('gart_pred_'+mall+'.csv')
    pred['row_id']=pred.row_id.astype('str')
    pred['shop_id']=pred.shop_id.apply(lambda x:'s_'+str(round(x)))
    ls_pred.append(ls_pred)

pred=pd.concat(ls_pred,ignore_index=True).reset_index()#合并结果集

pred.sort(['row_id'])
pred.to_csv('result.csv',index=None)
print('pred shape',pred.shape)