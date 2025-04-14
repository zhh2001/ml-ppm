# -*- coding: utf-8 -*-

import logging
import time
import warnings
from typing import Final

import joblib
import lightgbm as lgb
import pandas as pd
from pympler import asizeof

from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_p4runtime_API import SimpleSwitchP4RuntimeAPI
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

warnings.filterwarnings(action='ignore')
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

MICRO_SECOND: Final[int] = 1
MILLI_SECOND: Final[int] = 1000 * MICRO_SECOND
SECOND: Final[int] = 1000 * MILLI_SECOND

topo = load_topo(json_path='topology.json')
controller = SimpleSwitchP4RuntimeAPI(
    device_id=1,
    grpc_port=9559,
    grpc_ip='10.120.130.59',
    p4rt_path='/home/p4/Desktop/PPM/mlppm_p4rt.txt',
    json_path='/home/p4/Desktop/PPM/mlppm.json'
)
client = SimpleSwitchThriftAPI(
    thrift_port=9090,
    thrift_ip='10.120.130.59',
    json_path='/home/p4/Desktop/PPM/mlppm.json'
)


def convert_bytes(size_in_bytes: float) -> str:
    """
    将字节数转换为更易读的格式（B, KB, MB, GB）
    :param size_in_bytes: 字节数
    :return: 转换后的字符串
    """
    units: list[str] = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index: int = 0

    while size_in_bytes >= 1024 and unit_index < len(units) - 1:
        size_in_bytes /= 1024
        unit_index += 1

    return f"{size_in_bytes:.2f} {units[unit_index]}"


def format_time(seconds: float) -> str:
    """
    根据浮点秒数自动选择最合适的单位（s, ms, μs）并返回格式化字符串。
    :param seconds: 浮点秒数
    :return: 格式化后的时间字符串，带有合适的单位
    """
    if not isinstance(seconds, (int, float)):
        raise ValueError("输入必须是一个浮点数或整数")

    if seconds >= 1:
        return f"{seconds:.3f} s"
    elif seconds >= 0.001:
        return f"{seconds * 1000:.3f} ms"
    else:
        return f"{seconds * 1_000_000:.3f} μs"


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
            'reported_pps': 850       # 单位：pps
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


def main():
    table_name: str = 'ipv4_lpm'
    action_name: str = 'ipv4_forward'
    controller.table_clear(table_name=table_name)
    for neigh in topo.get_neighbors(name='s1'):
        if topo.isHost(name=neigh):
            dst_ip: str = f'{topo.get_host_ip(name=neigh)}/32'
            dst_mac: str = topo.get_host_mac(name=neigh)
            dst_port: int = topo.node_to_node_port_num(node1='s1', node2=neigh)
            logging.info(f'[添加表项] {table_name} {action_name} {dst_ip} => {dst_mac} {dst_port}')
            controller.table_add(table_name, action_name, [dst_ip], [dst_mac, str(dst_port)])

    digest_name = 'reported_data'
    if controller.digest_get_conf(digest_name) is None:
        controller.digest_enable(digest_name)

    client.register_write('thld_prd_m', 0, 500 * MILLI_SECOND)

    lgb_predictor = LGBPredictor()
    logging.info(f"[模型加载] {asizeof.asizeof(lgb_predictor) / 1024 ** 1:.2f} KB")

    while True:
        digest = controller.get_digest_list()
        counter_data = digest.data[0].struct.members
        counter_data = (int.from_bytes(counter.bitstring, 'big', signed=False)
                        for counter in counter_data)
        pkt_count, byte_count, diff_ts = counter_data
        diff_ts /= SECOND
        reported_bps = byte_count / diff_ts
        reported_pps = pkt_count / diff_ts
        logging.info(f'[接收消息] 包数：{pkt_count}，'
                     f'字节数：{convert_bytes(byte_count)}，'
                     f'时间间隔：{format_time(diff_ts)}，'
                     f'速率：{convert_bytes(byte_count / diff_ts)}/s')

        network_data = {
            'reported_bps': reported_bps,
            'reported_pps': reported_pps
        }
        start = time.perf_counter_ns()
        prd_m = lgb_predictor.predict(network_data)
        end = time.perf_counter_ns()
        t = (end - start) / 1_000_000
        logging.info(f'[阈值计算] 耗时：{t}ms')

        client.register_write('thld_prd_m', 0, int(prd_m * MILLI_SECOND))
        logging.info(f'[调整阈值] thld_prd_m.write(0, {prd_m});')


if __name__ == '__main__':
    main()
