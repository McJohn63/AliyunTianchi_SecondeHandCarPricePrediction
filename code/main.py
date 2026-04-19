import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
output_dir = os.path.join(ROOT_DIR, r'prediction_result')
save_path = os.path.join(output_dir, 'result.csv')

from feature.cleaning import cleaning_pipline
from feature.generation import feature_generation_pipeline

from model.training import train_lgb, train_catboost
from model.ensemble import find_best_weights

from config import lgb_params, cb_params, drop_cols, cat_feature


def main():
    # 1. 加载
    train = pd.read_csv(os.path.join(ROOT_DIR, r'data/used_car_train_20200313.csv'), sep=' ')
    test = pd.read_csv(os.path.join(ROOT_DIR, r'data/used_car_testB_20200421.csv'), sep=' ')
    

    # 2. 清洗 
    train, test = cleaning_pipline(train, test)

    # 3. 特征工程
    train = feature_generation_pipeline(train)
    test = feature_generation_pipeline(test)

    features = [col for col in train.columns if col not in drop_cols]
    X = train[features]
    Y = train['price_log']
    test_X = test[features]

    # 4. 训练单个模型
    print("Training LightGBM...")
    oof_lgb, test_preds_lgb = train_lgb(X, Y, test_X, lgb_params, cat_feature)

    print("Training CatBoost...")
    oof_cb, test_preds_cb = train_catboost(X, Y, test_X, cb_params, cat_feature)

    # 5. 寻找最佳融合权重
    print("Optimizing weights...")
    weights = find_best_weights(Y, [oof_lgb, oof_cb])
    print(f"Best Weights: LGB={weights[0]:.4f}, CB={weights[1]:.4f}")

    # 6. 生成最终预测
    final_preds = np.expm1(weights[0] * test_preds_lgb + weights[1] * test_preds_cb)

    submission = pd.DataFrame({
    'SaleID': test['SaleID'],
    'price': final_preds
    })

    submission.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()