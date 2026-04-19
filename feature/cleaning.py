import numpy as np
import pandas as pd

def cleaning_pipline(train_df, test_df):
    # 将 '-' 替换为 NaN，并转为 float
    for df in [train_df, test_df]:
        df['notRepairedDamage'].replace('-', np.nan)
        df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')

    # 处理power列中的极端大值
    # 处理power列中的极端小值，由于值为0的数目过多，认为是缺失值
    # 用np.nan填充power列中的0,将数据集按照model切分，计算对应power中位数，填充对应power的nan值
    train_df['power'] = train_df['power'].replace(0, np.nan)
    test_df['power'] = test_df['power'].replace(0, np.nan)

    for df in [train_df, test_df]:  # 截断power
        df['power'] = df['power'].clip(30, 600)
    median = train_df.groupby('model')['power'].median()
    train_df['power'] = train_df['power'].fillna(train_df['model'].map(median))
    test_df['power'] = test_df['power'].fillna(test_df['model'].map(median))

    # 处理price中的极端小值,对price进行对数变换
    train_df = train_df[train_df['price'] > 100]
    train_df = train_df.copy()
    train_df['price_log'] = np.log1p(train_df['price'])
    return train_df, test_df
