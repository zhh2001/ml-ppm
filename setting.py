# -*- coding: utf-8 -*-
"""
配置文件
"""

from ipaddress import IPv4Network
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Final
from typing import AnyStr

# 文件路径
HOME_PATH: Final[Path] = Path.home()  # HOME目录：~/
DESKTOP_PATH: Final[Path] = HOME_PATH.joinpath('Desktop')  # 桌面的根目录：~/Desktop/
PPM_PATH: Final[Path] = DESKTOP_PATH.joinpath('PPM')  # 项目的根目录：~/Desktop/PPM/
P4_JSON_PATH: Final[Path] = PPM_PATH.joinpath('ppm.json')
P4RUNTIME_PATH: Final[Path] = PPM_PATH.joinpath('ppm_p4rt.txt')

CONTROLLER_PROCESS_NAME: Final[AnyStr] = 'controller.py'  # 控制器进程名称

# P4 控制器的 IP 信息
SERVER_IP: Final[IPv4Network] = IPv4Network('10.120.130.59')
SERVER_IP_STR: Final[AnyStr] = SERVER_IP.network_address.__str__()

# gRPC 配置
GRPC_OPTIONS: Final[Dict[AnyStr, Any]] = {
    'device_id': 1,
    'grpc_ip': SERVER_IP_STR,
    'grpc_port': 9559,
    'p4rt_path': P4RUNTIME_PATH,
    'json_path': P4_JSON_PATH
}

# Thrift 配置
THRIFT_OPTIONS: Final[Dict[AnyStr, Any]] = {
    'thrift_port': 9090,
    'thrift_ip': SERVER_IP_STR,
    'json_path': P4_JSON_PATH
}
