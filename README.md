#赛题介绍
赛题以预测二手车的交易价格为任务，数据集报名后可见并可下载，该数据来自某交易平台的二手车交易记录，总数据量超过40w，包含31列变量信息，其中15列为匿名变量。为了保证比赛的公平性，将会从中抽取15万条作为训练集，5万条作为测试集A，5万条作为测试集B，同时会对name、model、brand和regionCode等信息进行脱敏。评价标准为MAE(Mean Absolute Error)。
#项目说明
该项目使用LightGBM与CatBoost加权融合的方案，线上成绩为440.458948(MAE)。测试集使用used_car_testB_20200421.csv，下载地址为https://tianchi.aliyun.com/competition/entrance/231784/information
