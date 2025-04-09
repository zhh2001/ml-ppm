# -*- coding: utf-8 -*-
"""
训练 XGBoost 模型
"""

import optuna
import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Final

matplotlib.rcParams['font.family'] = 'SimSun'
MODEL_SRC: Final[str] = './model'
CHARTS_SRC: Final[str] = './charts'


def load_data():
    """加载数据集并进行标准化处理"""
    df = pd.read_csv("dataset.csv")
    X = df[['reported_bps', 'reported_pps']]
    y = df['optimal_prd_m']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, rf'{MODEL_SRC}/xgboost_scaler.pkl')

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def objective(trial):
    """定义需要优化的超参数空间"""
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5),
        'random_state': 42,
    }

    # 交叉验证防止过拟合
    scores = []
    for fold in range(3):  #  3 折交叉验证
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all, y_train_all, test_size=0.2, random_state=fold)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        pred = model.predict(X_val)
        scores.append(mean_absolute_error(y_val, pred))

    return np.mean(scores)


if __name__ == "__main__":
    X_train_all, X_test, y_train_all, y_test = load_data()

    # 贝叶斯优化搜索
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\n最佳参数组合:")
    for key, value in study.best_params.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    # 使用最佳参数训练最终模型
    best_params = study.best_params
    best_params.update({'n_jobs': 8})  # 启用多核加速

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train_all, y_train_all,
                    eval_set=[(X_train_all, y_train_all), (X_test, y_test)],
                    eval_metric='mae',
                    early_stopping_rounds=50,
                    verbose=10)

    # 性能评估
    y_pred = final_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n最终模型测试集MAE: {mae:.1f}ms")
    print(f"最终模型测试集RMSE: {rmse:.1f}ms")

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(final_model.evals_result()['validation_0']['mae'], label='Train')
    plt.plot(final_model.evals_result()['validation_1']['mae'], label='Test')
    plt.title('优化后模型训练过程监控')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('MAE (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig(rf'{CHARTS_SRC}/xgboost_optimized_training_curve.png', dpi=300)
    plt.show()

    xgb.plot_importance(final_model, height=0.8, importance_type='weight')
    plt.title('原始特征重要性 (Weight)')
    plt.tight_layout()
    plt.savefig(rf'{CHARTS_SRC}/xgboost_feature_importance.png', dpi=300)
    plt.show()

    # 模型保存
    final_model.save_model(rf'{MODEL_SRC}/xgboost_model.json')
    joblib.dump(final_model, rf'{MODEL_SRC}/xgboost_model.pkl')
