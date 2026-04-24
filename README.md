# 🚗 二手车价格预测

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.x-green.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.x-yellow.svg)](https://catboost.ai/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

基于 **LightGBM + CatBoost 加权融合**的二手车交易价格预测方案，线上 MAE 得分 **440.46**。

---

## 📖 项目简介

本项目来源于**阿里云天池二手车交易价格预测**赛题。数据集来自某交易平台的真实二手车交易记录，总数据量超过 **40 万条**，包含 **31 列变量信息**（其中 15 列为匿名特征）。赛题从中抽取约 15 万条作为训练集、5 万条作为测试集，评价指标为 **MAE（Mean Absolute Error）**。

项目覆盖了机器学习项目的完整流程：**数据清洗 → 特征工程 → 模型训练 → 模型融合 → 结果输出**，适合作为结构化数据回归任务的入门与进阶学习参考。

---

## 🛠 技术栈

| 类别 | 技术/库 |
|------|---------|
| **编程语言** | Python 3.8+ |
| **数据处理** | Pandas, NumPy |
| **机器学习框架** | LightGBM, CatBoost, Scikit-learn |
| **优化工具** | SciPy (SLSQP 权重优化) |
| **实验管理** | K-Fold 交叉验证, Early Stopping |

---

## 🧱 核心功能模块说明

### 1. 数据清洗 (`feature/cleaning.py`)

| 处理步骤 | 说明 |
|----------|------|
| `notRepairedDamage` 清洗 | 将 `-` 替换为 NaN 并转为数值型 |
| `power` 缺失值处理 | 将无效值（0）替换为 NaN，按 `model` 分组取中位数填充 |
| `power` 截尾 | 限制在 `[30, 600]` 区间，剔除极端异常值 |
| `price` 过滤 | 剔除价格 ≤ 100 的异常样本 |
| `price` 对数变换 | 对目标变量做 `log1p` 变换，使其更接近正态分布 |

### 2. 特征工程 (`feature/generation.py`)

| 特征类别 | 生成特征 | 说明 |
|----------|----------|------|
| **时间特征** | `used_time` | 注册日期 (`regDate`) 到创建日期 (`creatDate`) 的时间差（天数） |
| **业务特征** | `power_decay` | 年均功率：`power / (used_time/365 + 1)` |
| | `brand_age` | 品牌 × 车龄分桶组合特征（5 分位数） |
| | `annual_mileage` | 年均里程：`kilometer / (used_time/365 + 1)` |
| **交叉特征** | `v_i_mult_v_j` | Top 5 匿名特征的**乘法交叉**（共 15 个） |
| | `v_i_plus_v_j` | Top 5 匿名特征的**加法交叉**（共 15 个） |

> 匿名特征选取：`v_0`, `v_1`, `v_3`, `v_13`, `v_14`

### 3. 模型训练 (`model/training.py`)

- **LightGBM 模型** (`train_lgb`)
  - 支持类别特征原生处理（`category` dtype）
  - 5 折交叉验证 + Early Stopping（100 轮）
  - 优化目标：MAE

- **CatBoost 模型** (`train_catboost`)
  - 类别特征以字符串形式传入
  - 5 折交叉验证 + Early Stopping（100 轮）
  - 支持 GPU 加速（可配置）
  - 采用 Bernoulli Bootstrap 采样

### 4. 模型融合 (`model/ensemble.py`)

- 使用 **SciPy SLSQP 优化器**寻找最佳加权系数
- 约束条件：权重 ∈ [0, 1]，且权重之和 = 1
- 优化目标：在 OOF（Out-of-Fold）预测上最小化原始空间 MAE（`expm1` 还原后）

### 5. 配置管理 (`code/config.py`)

集中管理所有超参数与常量：
- `lgb_params`：LightGBM 超参数（`n_estimators=10000`, `learning_rate=0.01`, `max_depth=7` 等）
- `cb_params`：CatBoost 超参数（`iterations=10000`, `learning_rate=0.02`, `depth=6` 等）
- `drop_cols`：训练时需丢弃的列（ID、原始标签、日期等）
- `cat_feature`：类别特征列表

---

## 📁 目录结构

```
.
├── code/                       # 核心代码
│   ├── main.py                 # 主入口：串联数据加载 → 清洗 → 特征 → 训练 → 融合 → 输出
│   └── config.py               # 全局配置（模型参数、特征列表等）
│
├── feature/                    # 特征工程模块
│   ├── cleaning.py             # 数据清洗流水线
│   └── generation.py           # 特征生成流水线
│
├── model/                      # 模型模块
│   ├── training.py             # LightGBM / CatBoost 训练器（含 K-Fold）
│   └── ensemble.py             # 加权融合权重优化
│
├── data/                       # 📥 数据目录（需自行下载，已 gitignore）
│   ├── used_car_train_20200313.csv
│   └── used_car_testB_20200421.csv
│
├── prediction_result/          # 📤 预测结果输出
│   └── result.csv
│
├── catboost_info/              # CatBoost 训练日志（可清理）
├── .gitignore
└── README.md
```

---

## 🚀 快速启动

### 前置要求

- Python **3.8** 或更高版本
- 建议使用虚拟环境（`venv` / `conda`）

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd <project-directory>
```

### 2. 安装依赖

```bash
pip install pandas numpy lightgbm catboost scikit-learn scipy
```

> 若需 GPU 加速 CatBoost，请将 `code/config.py` 中 `cb_params['task_type']` 改为 `'GPU'`，并安装对应 CUDA 版本的 CatBoost。

### 3. 下载数据集

前往 [阿里云天池赛题页面](https://tianchi.aliyun.com/competition/entrance/231784/information) 下载以下文件，放入 `data/` 目录：

- `used_car_train_20200313.csv` — 训练集
- `used_car_testB_20200421.csv` — 测试集

```
mkdir -p data
# 将下载的 CSV 文件移动至 data/ 目录下
```

### 4. 运行

```bash
cd code
python main.py
```

程序将依次执行：
1. 加载原始数据
2. 数据清洗
3. 特征工程（时间/业务/交叉特征）
4. LightGBM 5 折交叉验证训练
5. CatBoost 5 折交叉验证训练
6. 优化融合权重
7. 输出最终预测结果至 `prediction_result/result.csv`

### 5. 查看结果

```bash
cat ../prediction_result/result.csv
```

输出格式：

| SaleID | price |
|--------|-------|
| 0 | 12500.5 |
| 1 | 8900.3 |
| ... | ... |

---

## 📊 模型配置速览

| 参数 | LightGBM | CatBoost |
|------|----------|----------|
| 迭代次数 | 10,000 | 10,000 |
| 学习率 | 0.01 | 0.02 |
| 最大深度 | 7 | 6 |
| 叶子数 | 45 | — |
| L2 正则 | 1.2 | 10 |
| Early Stopping | 100 轮 | 100 轮 |
| 交叉验证 | 5-Fold | 5-Fold |

---

## 📝 License

本项目仅供学习交流使用。数据集版权归阿里云天池平台所有。

---

## 🙏 致谢

- [阿里云天池](https://tianchi.aliyun.com/) — 提供赛题与数据
- [LightGBM](https://lightgbm.readthedocs.io/) & [CatBoost](https://catboost.ai/) — 卓越的 GBDT 框架
- [SciPy](https://scipy.org/) — 优化工具支持"
