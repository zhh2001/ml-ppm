# -*- coding: utf-8 -*-
"""
用来给 h2 发送数据，通过 iPerf 命令生成流量
记得在 Mininet 中先在 h2 上开启 iPref 服务器用于接收流量：`h2 iperf -s &`
"""

import subprocess
import time

total_time = 30
target_host_ip = '10.0.1.2'

#  iPerf 的命令参数
iperf_format = 'A'  # -f, --format [bkmaBKMA]
iperf_time = 5  # -t, --time [n]


def get_bw(t: float) -> float:
    """
    获取当前阶段的流量速率（即 iPerf 带宽限制）

    :param t: 当前秒数
    :return: 带宽数值，单位为 Mbps
    """
    if 0 <= t < 5:
        base = 5  # 5 Mbps
    elif 5 <= t < 10:
        base = 6.5  # 6.5 Mbps
    elif 10 <= t < 15:
        base = 10  # 10 Mbps
    elif 15 <= t < 20:
        base = 8  # 8 Mbps
    elif 20 <= t < 25:
        base = 6  # 6 Mbps
    else:
        base = 9  # 9 Mbps
    return base


def main():
    start_time = time.time()
    elapsed_time = time.time() - start_time
    while elapsed_time < total_time:
        bw = get_bw(elapsed_time)
        options = []
        options.append(f'-c {target_host_ip}')
        options.append(f'-b {bw}M')
        options.append(f'-t 5')  # 每个阶段持续 5 秒
        options.append(f'-f {iperf_format}')
        cmd = f'iperf {' '.join(options)}'
        subprocess.run(cmd, shell=True)
        elapsed_time = time.time() - start_time


if __name__ == '__main__':
    main()
