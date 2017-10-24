# -*- coding:utf-8 -*-
import pandas as pd

bssid_frame=None#每一个mall的bssid_frame
train=None#每一个mall的train
min_signal=-100#设定的最小信号量

def get_distance(x):
    train_wifis=str(x[0])
    dict_test=eval(x[1])
    wifi_infos=train_wifis.split(';')
    dict_train={}
    for wifi in wifi_infos:
        wifi_info=wifi.split('|')
        bssid=str(wifi_info[0])
        signal=float(wifi_info[1])
        dict_train[bssid]=signal
    
    #计算两字典的距离
    key_sample=dict_train.keys()&dict_test.keys()
    key_test_more=dict_test.keys()-dict_train.keys()
    key_train_more=dict_train.keys()-dict_test.keys()

    dist=0
    for key in key_sample:
        dist=dist+((float(dict_test[key])-float(dict_train[key]))**2)
    for key in key_test_more:
        dist=dist+((float(dict_test[key])-min_signal)**2)
    for key in key_train_more:
        dist=dist+((float(dict_train[key])-min_signal)**2)

    return dist

def get_shop(s):
    wifi_infos=s.split(';')
    dict_test={}
    for wifi in wifi_infos:
        wifi_info=s.split('|')
        bssid=str(wifi_info[0])
        signal=float(wifi_info[1])
        dict_test[bssid]=signal

    #对test的wifi按照强度排序
    ls_dict_test=sorted(dict_test.items(),key=lambda x:x[1],reverse=True)
    head_signal_shop_num=3
    ls_str_shops=[]
    counter=0
    for k,v in ls_dict_test:
        bssid=k
        shops=bssid_frame[bssid_frame.bssid==bssid].shop_id.values
        if len(shops)>0:
            shops=str(shops[0])
            ls_str_shops.append(shops)
            counter+=1
        if counter==head_signal_shop_num:
            break
    ls_shops=[]
    for shop in ls_str_shops:
        ls_shops.extend(shop.split(','))
    ls_shops=list(set(ls_shops))#去重

    ls_train_t=[]
    #train_t=train[train.shop_id in ls_shops][['shop_id','wifi_infos']]
    for shop_t in ls_shops:
        ls_train_t.append(train[train.shop_id==shop_t][['shop_id','wifi_infos']])

    shop='null'
    if len(ls_train_t)>0:
        train_t=pd.concat(ls_train_t,ignore_index=True).reset_index()
        #print(train_t.head())
        train_t['test_wifi']=str(dict_test)
        train_t['distance']=train_t[['wifi_infos','test_wifi']].apply(get_distance,axis=1)

        min_distance=min(train_t['distance'])
        #####此处可扩展为knn（直接赋值最小距离所在列的distance为INF），在此为最近邻#####

        shop=train_t[train_t.distance==min_distance][['shop_id']].values
        if len(shop)>0:
            #print(shop[0][0])
            #print(type(shop[0][0]))
            shop=','.join(shop[0])
            shops=shop.split(',')
            dict_shops=dict((shops.count(i),i) for i in shops)
            shop=dict_shops[max(dict_shops.keys())]

    return shop

malls=pd.read_csv('mall_test2.csv')#mall_test2.csv
malls=malls.mall_id.values.tolist()
print('malls type:',type(malls[0]))

for mall in malls:
    bssid_frame=pd.read_csv('bssid_frames//bssid_frame_'+mall+'.csv')
    train=pd.read_csv('2//2train_'+mall+'.csv')#shop_id,wifi_infos,以mall_id分
    test=pd.read_csv('2//2test_mall_'+mall+'.csv')
    test['shop_id']=test['wifi_infos'].apply(get_shop)
    test=test[['row_id','shop_id']]
    test.to_csv('result//knn_pred_'+mall+'.csv',index=None)
    print(mall)

ls_pred=[]
for mall in malls:
    pred=pd.read_csv('result//knn_pred_'+mall+'.csv')
    pred['row_id']=pred.row_id.astype('str')
    pred['shop_id']=pred.shop_id.astype('str')
    ls_pred.append(pred)

pred=pd.concat(ls_pred,ignore_index=True).reset_index()#合并结果集
pred.to_csv('result.csv',index=None)

print('pred shape',pred.shape)