# -*- coding:utf-8 -*-

import pandas as pd

malls=pd.read_csv('data/ccf_first_round_shop_info.csv')#ccf_first_round_shop_info.csv
malls=malls[['mall_id']]
malls.drop_duplicates(inplace=True)
print('malls columns',malls.columns)
malls.to_csv('mall_test2.csv',index=None)