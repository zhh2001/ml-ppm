# -*- coding: utf-8 -*-
"""
本地测试模型性能
"""

import time
import joblib
import warnings
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from pympler import asizeof
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
matplotlib.rcParams['font.family'] = 'SimSun'


class SVMPredictor:
    def __init__(self,
                 model_path='./model/svm_model.pkl',
                 scaler_path='./model/svm_scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_order = ['reported_bps', 'reported_pps']  # 定义特征顺序

    def preprocess(self, new_data):
        df = pd.DataFrame([new_data], columns=self.feature_order)  # 转换为 DataFrame 确保特征顺序正确
        scaled_data = self.scaler.transform(df)  # 应用标准化
        return scaled_data

    def predict(self, new_data):
        try:
            scaled_data = self.preprocess(new_data)  # 数据预处理
            pred = self.model.predict(scaled_data)[0]  # 模型预测
            return pred
        except Exception as e:
            print(f"[ERROR] 预测失败: {str(e)}")
            return None


class XGBPredictor:
    def __init__(self,
                 model_path='./model/xgboost_model.pkl',
                 scaler_path='./model/xgboost_scaler.pkl'):
        self.model = joblib.load(model_path)  # 加载训练好的XGBoost模型
        self.scaler = joblib.load(scaler_path)  # 加载标准化器
        self.feature_order = ['reported_bps', 'reported_pps']

    def preprocess(self, new_data):
        df = pd.DataFrame([new_data], columns=self.feature_order)
        scaled_data = self.scaler.transform(df)
        return scaled_data

    def predict(self, new_data):
        try:
            scaled_data = self.preprocess(new_data)
            pred = self.model.predict(scaled_data)[0]
            return pred
        except Exception as e:
            print(f"[ERROR] 预测失败: {str(e)}")
            return None


class LGBPredictor:
    def __init__(self,
                 model_path='./model/lgb_model.txt',
                 scaler_path='./model/lgb_scaler.pkl'):
        """初始化加载模型资源"""
        self.model = lgb.Booster(model_file=model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_order = ['reported_bps', 'reported_pps']

    def _validate_input(self, data_dict):
        """验证输入数据完整性"""
        missing = [f for f in self.feature_order if f not in data_dict]
        if missing:
            raise ValueError(f"缺少必要特征: {missing}")

    def predict(self, input_data):
        """
        输入格式示例：
        {
            'reported_bps': 1200000,  # 单位：bps
            'reported_pps': 850        # 单位：包/秒
        }
        """
        try:
            self._validate_input(input_data)
            df = pd.DataFrame([input_data], columns=self.feature_order)
            scaled_data = self.scaler.transform(df)
            pred = self.model.predict(scaled_data)[0]
            return pred
        except Exception as e:
            print(f"[ERROR] 预测失败: {str(e)}")
            return None


if __name__ == "__main__":

    new_network_data = {
        # 'reported_byte': 1500,
        'reported_bps': 1.2e6,
        'reported_pps': 100
    }

    xgb_predictor = XGBPredictor()
    print(f"XGBoost 总内存占用: {asizeof.asizeof(xgb_predictor) / 1024 ** 1:.2f} KB")

    lgb_predictor = LGBPredictor()
    print(f"LGB 总内存占用: {asizeof.asizeof(lgb_predictor) / 1024 ** 1:.2f} KB")

    svm_predictor = SVMPredictor()
    print(f"SVM 总内存占用: {asizeof.asizeof(svm_predictor) / 1024 ** 1:.2f} KB")

    loop = 10000

    xgboost_perfs = []
    for _ in range(loop):
        start = time.perf_counter_ns()
        optimal_prd = xgb_predictor.predict(new_network_data)
        end = time.perf_counter_ns()
        xgboost_perfs.append(end - start)
    xgboost_perfs = [perf / 1000 / 1000 for perf in xgboost_perfs]
    print(f'平均耗时: {sum(xgboost_perfs) / loop:.6f}ms')

    lgb_perfs = []
    for _ in range(loop):
        start = time.perf_counter_ns()
        optimal_prd = lgb_predictor.predict(new_network_data)
        end = time.perf_counter_ns()
        lgb_perfs.append(end - start)
    lgb_perfs = [perf / 1000 / 1000 for perf in lgb_perfs]
    print(f'平均耗时: {sum(lgb_perfs) / loop:.6f}ms')

    svm_perfs = []
    for _ in range(loop):
        start = time.perf_counter_ns()
        optimal_prd = svm_predictor.predict(new_network_data)
        end = time.perf_counter_ns()
        svm_perfs.append(end - start)
    svm_perfs = [perf / 1000 / 1000 for perf in svm_perfs]
    print(f'平均耗时: {sum(svm_perfs) / loop:.6f}ms')

    # 性能折线图
    plt.figure(figsize=(12, 6))
    plt.plot(xgboost_perfs, color='blue', linestyle='-', label='XGBoost 预测耗时', alpha=0.7)
    plt.plot(lgb_perfs, color='orange', linestyle='-', label='LightGBM 预测耗时', alpha=0.7)
    plt.plot(svm_perfs, color='green', linestyle='-', label='SVM 预测耗时', alpha=0.7)
    plt.title('预测耗时对比: XGBoost vs LightGBM vs SVM', fontsize=14)
    plt.xlabel('预测次数', fontsize=12)
    plt.ylabel('耗费时间（毫秒）', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
