DATA_PATH = "../data/"
SAVE_PATH = "../prediction_result/"

drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'price', 'price_log', 'seller', 'offerType', 'regDate_fix', 'creatDate_fix']

cat_feature = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode', 'brand_age']

lgb_params = {
    'n_estimators': 10000,  # 迭代次数\树的数量
    'learning_rate': 0.01,
    'num_leaves': 45,  # 叶结点数
    'max_depth': 7,  # 最大树深
    'reg_alpha': 1,  # L1正则
    'reg_lambda': 1.2,  # L2正则
    'random_state': 42,
    'objective': 'mae',
    'metric': 'mae',
    'min_child_samples': 20,
    'n_jobs': -1  # 调用CPU
}

cb_params = {
    'iterations': 10000,
    'learning_rate': 0.02,
    'depth': 6,
    'l2_leaf_reg': 10,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'loss_function': 'MAE',  # 直接优化 MAE
    'eval_metric': 'MAE',
    'random_seed': 42,
    'early_stopping_rounds': 100,
    'task_type': 'GPU',  # 如果有 GPU 可以改为 'GPU'
    'verbose': 200
}