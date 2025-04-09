# -*- coding: utf-8 -*-
"""
训练 LightGBM 模型
"""

import matplotlib
import joblib
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from typing import Final

matplotlib.rcParams['font.family'] = 'SimSun'
MODEL_SRC: Final[str] = './model'
CHARTS_SRC: Final[str] = './charts'


def load_data():
    """加载数据集并进行标准化处理"""
    df = pd.read_csv("final.csv")
    X = df[['reported_bps', 'reported_pps']]
    y = df['optimal_prd_m']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, rf'{MODEL_SRC}/lgb_scaler.pkl')  # 单独保存标准化器

    return train_test_split(X_scaled, y, test_size=0.2)


def objective(trial):
    """贝叶斯优化目标函数"""
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 15, 60),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5),
        'verbose': -1
    }

    scores = []
    for fold in range(3):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all, y_train_all, test_size=0.2, random_state=fold)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model = lgb.train(params,
                          lgb_train,
                          valid_sets=[lgb_val],
                          num_boost_round=1000,
                          callbacks=[lgb.early_stopping(stopping_rounds=50)])

        pred = model.predict(X_val)
        scores.append(mean_absolute_error(y_val, pred))

    return np.mean(scores)


if __name__ == "__main__":
    X_train_all, X_test, y_train_all, y_test = load_data()

    # 贝叶斯优化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\n最佳参数组合:")
    for key, value in study.best_params.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    # 最终模型训练
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'mae',
        'verbose': -1,
        'num_boost_round': 1000
    })

    lgb_train = lgb.Dataset(X_train_all, y_train_all)
    eval_result = {}  # 用于存储评估记录
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.record_evaluation(eval_result)  # 新增记录器
    ]

    final_model = lgb.train(best_params, lgb_train, valid_sets=[lgb_train], callbacks=callbacks)  # 传入回调函数

    # 性能评估
    y_pred = final_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n测试集MAE: {mae:.1f}ms")
    print(f"测试集RMSE: {rmse:.1f}ms")

    # 可视化
    plt.figure(figsize=(10, 5))
    lgb.plot_metric(eval_result,
                    title='训练过程指标变化',
                    xlabel='Boosting Rounds')
    plt.savefig(rf'{CHARTS_SRC}/lgb_training_curve.png', dpi=300)

    plt.figure(figsize=(8, 4))
    lgb.plot_importance(final_model, importance_type='gain', title='特征重要性 (Gain)')
    plt.tight_layout()
    plt.savefig(rf'{CHARTS_SRC}/lgb_feature_importance.png', dpi=300)

    # 模型保存
    final_model.save_model(rf'{MODEL_SRC}/lgb_model.txt')
