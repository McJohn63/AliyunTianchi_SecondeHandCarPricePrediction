import numpy as np
import pandas as pd

# 将creatDate和regDate转化为车龄
def to_used_time(df):
    def fix_date(date):
        date_str = str(int(date))
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:8]
        if month == '00':
            month = '01'
        return f"{year}-{month}-{day}"

    df['regDate_fix'] = pd.to_datetime(df['regDate'].apply(fix_date), errors='coerce')
    df['creatDate_fix'] = pd.to_datetime(df['creatDate'].apply(fix_date), errors='coerce')
    df['used_time'] = (df['creatDate_fix'] - df['regDate_fix']).dt.days
    df['used_time'].fillna(df['used_time'].median())
    return df

def create_business_features(df):
    # 年均功率
    df['power_decay'] = df['power'] / (df['used_time'] / 365 + 1)
    
    # 品牌+车龄组合 (分桶特征)
    df['brand_age'] = df['brand'].astype(str) + '_' + pd.qcut(df['used_time'], 5, labels=False).astype(str)
    
    # 年均里程
    df['annual_mileage'] = df['kilometer'] / (df['used_time'] / 365 + 1)
    return df

def create_cross_features(df):
    anon_features = ['v_14', 'v_0', 'v_13', 'v_3', 'v_1']
    for i in range(len(anon_features)):
        for j in range(len(anon_features)):
            feat1, feat2 = anon_features[i], anon_features[j]
            # 乘法交叉
            df[f"{feat1}_mult_{feat2}"] = df[feat1] * df[feat2]
            # 加法交叉
            df[f"{feat1}_plus_{feat2}"] = df[feat1] + df[feat2]
    return df

def feature_generation_pipeline(df):
    print("Generating time features...")
    df = to_used_time(df)
    print("Generating business features...")
    df = create_business_features(df)
    print("Generating cross features...")
    df = create_cross_features(df)
    return df