# 天池比赛：商场中精确定位用户所在店铺解决方案
第一次参加天池比赛，没有任何经验，最后还没有上传最好成绩，初赛惨被淘汰：171/2845。尝试了几种方法，取得最高成绩的方案为wifi_simple.py，直接运行即可。

另外的一个解决方案：
- 先运行get_mall.py获得test_mall2.csv,这是所有mall_id;
- 然后运行extract_feature_test2.py获得特征;
- 然后运行xgb_test2.py或者gart_test2.py获得结果


可以最后将两者的结果做一个stacking，应会进一步提高准确率，比赛时没有尝试
