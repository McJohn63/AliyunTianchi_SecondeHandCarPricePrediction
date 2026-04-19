import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

def train_lgb(X, y, test_X, params, cat_feature): 
    X_lgb = X.copy()
    test_X_lgb = test_X.copy()

    for col in cat_feature:
        if col in X_lgb.columns:
            X_lgb[col] = X_lgb[col].astype('category')
            test_X_lgb[col] = test_X_lgb[col].astype('category')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    preds = np.zeros(len(test_X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_feature,
            callbacks=[lgb.early_stopping(100)]
        )
        
        oof[val_idx] = model.predict(X_val)
        preds += model.predict(test_X_lgb) / kf.n_splits
    
    return oof, preds

def train_catboost(X, y, test_X, params, cat_feature):
    X_cb = X.copy()
    test_X_cb = test_X.copy()

    for col in cat_feature:
        X_cb[col] = X_cb[col].astype(str)
        test_X_cb[col] = test_X_cb[col].astype(str)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    preds = np.zeros(len(test_X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_feature,
            early_stopping_rounds=100
        )
        
        oof[val_idx] = model.predict(X_val)
        preds += model.predict(test_X_cb) / kf.n_splits
        
    return oof, preds