# -*- coding: utf-8 -*-
"""
本地测试模型性能
"""

import time
import timeit
import gc
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
matplotlib.rcParams['font.family'] = 'Times New Roman'


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
    print(f"XGB 总内存占用: {asizeof.asizeof(xgb_predictor) / 1024 ** 1:.2f} KB")

    lgb_predictor = LGBPredictor()
    print(f"LGB 总内存占用: {asizeof.asizeof(lgb_predictor) / 1024 ** 1:.2f} KB")

    svm_predictor = SVMPredictor()
    print(f"SVM 总内存占用: {asizeof.asizeof(svm_predictor) / 1024 ** 1:.2f} KB")

    loop = 1000

    gc.disable()
    xgb_predictor.predict(new_network_data)
    times = timeit.repeat(lambda: xgb_predictor.predict(new_network_data), repeat=loop, number=1)
    xgboost_perfs = [t * 1000 for t in times]
    gc.enable()

    gc.disable()
    lgb_predictor.predict(new_network_data)
    times = timeit.repeat(lambda: lgb_predictor.predict(new_network_data), repeat=loop, number=1)
    lgb_perfs = [t * 1000 for t in times]
    gc.enable()

    gc.disable()
    svm_predictor.predict(new_network_data)
    times = timeit.repeat(lambda: svm_predictor.predict(new_network_data), repeat=loop, number=1)
    svm_perfs = [t * 1000 for t in times]
    gc.enable()

    # 性能折线图
    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(xgboost_perfs, color='blue', linestyle='-', label='XGBoost', alpha=0.7)
    plt.plot(lgb_perfs, color='orange', linestyle='-', label='LightGBM', alpha=0.7)
    plt.plot(svm_perfs, color='green', linestyle='-', label='SVM', alpha=0.7)
    plt.xlabel('Prediction Count', fontsize=28)
    plt.ylabel('Prediction Time (ms)', fontsize=28)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=24)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)
    plt.xlim(0, loop)
    plt.tight_layout()
    plt.savefig('./charts/PredictionTimeCostsLine.pdf')
    plt.show()

    # 箱型图
    plt.figure(figsize=(12, 6), dpi=300)
    data_to_plot = [xgboost_perfs, lgb_perfs, svm_perfs]
    labels = ['XGBoost', 'LightGBM', 'SVM']
    plt.boxplot(data_to_plot, patch_artist=True, tick_labels=labels, showmeans=True, showfliers=False)
    plt.ylabel('Prediction Time (ms)', fontsize=28)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)
    plt.tight_layout()
    plt.savefig('./charts/PredictionTimeCostsBox.pdf')
    plt.show()

    print(f'XGB平均耗时: {sum(xgboost_perfs) / loop:.6f}ms')
    print(f'XGB最大耗时: {max(xgboost_perfs):.6f}ms')
    print(f'LGB平均耗时: {sum(lgb_perfs) / loop:.6f}ms')
    print(f'LGB最大耗时: {max(lgb_perfs):.6f}ms')
    print(f'SVM平均耗时: {sum(svm_perfs) / loop:.6f}ms')
    print(f'SVM最大耗时: {max(svm_perfs):.6f}ms')
