# -*- coding:utf-8 -*-

import pandas as pd
import xgboost as xgb

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

	train=xgb.DMatrix(train_x,label=train_y)
	test=xgb.DMatrix(test_x)
	params={'booster':'gbtree',
		'objective': 'rank:pairwise',
		'eval_metric':'auc',
		'gamma':0.1,
		'min_child_weight':1.1,
		'max_depth':5,
		'lambda':10,
		'subsample':0.7,
		'colsample_bytree':0.7,
		'colsample_bylevel':0.7,
		'eta': 0.01,
		'tree_method':'exact',
		'seed':0,
		'nthread':12
		}
	watchlist=[(train,'train')]
	model=xgb.train(params,train,num_boost_round=3500,evals=watchlist)
	test_pred['shop_id']=model.predict(test_x)
	test_pred.to_csv('xgb_pred_'+mall+'.csv',index=None)
	#break#用于调试

ls_pred=[]
for mall in malls:
    pred=pd.read_csv('xgb_pred_'+mall+'.csv')
    pred['row_id']=pred.row_id.astype('str')
    pred['shop_id']=pred.shop_id.apply(lambda x:'s_'+str(round(x)))
    ls_pred.append(ls_pred)

pred=pd.concat(ls_pred,ignore_index=True).reset_index()#合并结果集

pred.sort(['row_id'])
pred.to_csv('result.csv',index=None)
print('pred shape',pred.shape)