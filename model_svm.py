# -*- coding: utf-8 -*-
"""
训练 SVM 模型
"""

import joblib
import numpy as np
import optuna
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn.pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from typing import Final
from typing import Sequence

matplotlib.rcParams['font.family'] = 'SimSun'
MODEL_SRC: Final[str] = './model'
CHARTS_SRC: Final[str] = './charts'


def load_data():
    """加载并标准化数据集"""
    df: pd.DataFrame = pd.read_csv("dataset.csv")
    X = df[['reported_bps', 'reported_pps']]
    y = df['optimal_prd_m']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, rf'{MODEL_SRC}/svm_scaler.pkl')

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def objective(trial) -> float | Sequence[float]:
    """Optuna 优化目标函数"""
    params = {
        'C': trial.suggest_float('C', 1e-2, 1e3, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
        'gamma': trial.suggest_float('gamma', 1e-4, 1e1, log=True),
        'degree': trial.suggest_int('degree', 2, 5)  # 仅 poly 核有效
    }

    # 构建带标准化的 Pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(**params))
    ])

    #  3 折交叉验证
    scores = -cross_val_score(model, X_train_all, y_train_all,
                              cv=3, scoring='neg_mean_absolute_error')
    return np.mean(scores)


if __name__ == "__main__":
    X_train_all, X_test, y_train_all, y_test = load_data()

    # 超参数优化
    study: Final[optuna.study.Study] = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\n最佳参数组合:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # 训练最终模型
    final_model: Final[sklearn.pipeline.Pipeline] = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('svr', SVR(**study.best_params))
    ])
    final_model.fit(X_train_all, y_train_all)

    # 性能评估
    y_pred = final_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n测试集 MAE: {mae:.1f}ms")
    print(f"测试集 RMSE: {rmse:.1f}ms")

    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w')
    plt.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], 'r--', lw=2)
    plt.title('SVM 回归预测结果散点图')
    plt.xlabel('真实阈值 (ms)')
    plt.ylabel('预测阈值 (ms)')
    plt.grid(True)
    plt.savefig(rf'{CHARTS_SRC}/svm_prediction_scatter.png', dpi=300)
    plt.show()

    # 误差分布直方图
    errors = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='k')
    plt.title('SVM 预测误差分布')
    plt.xlabel('预测误差 (ms)')
    plt.ylabel('频数')
    plt.grid(True)
    plt.savefig(rf'{CHARTS_SRC}/svm_error_distribution.png', dpi=300)
    plt.show()

    # 模型保存（使用 joblib 兼容 pipeline）
    joblib.dump(final_model, rf'{MODEL_SRC}/svm_model.pkl')
