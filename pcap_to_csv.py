# -*- coding: utf-8 -*-
"""将 pcap 文件提取为 csv 文件"""

import csv
import sys
from scapy.all import *


def packet_to_dict(packet):
    """将单个数据包转换为字典格式"""
    packet_dict = {
        # 基础信息
        'timestamp': round(packet.time, 6),  # 时间戳（保留 6 位小数）
        'packet_len': len(packet)
    }

    # 协议分层解析
    layers = {
        'Ether': {},
        'IP': {},
        'TCP': {},
        'UDP': {},
        'ICMP': {},
        'ARP': {},
        'DHCP': {},
        'DNS': {},
        'HTTP': {}
    }

    # 遍历各协议层
    for layer in packet.layers():
        layer_name = layer.__name__
        if layer_name in layers:
            # 获取标准字段
            for field in layer.fields_desc:
                field_name = field.name
                # 处理特殊字段（如IPv4/IPv6兼容）
                try:
                    value = getattr(packet, field_name)
                    if isinstance(value, (list, tuple)):
                        value = ','.join(map(str, value))
                    elif isinstance(value, bytes):
                        value = value.hex().upper()
                    layers[layer_name][field_name] = str(value)
                except AttributeError:
                    continue

    # 合并所有层信息
    for layer_name, fields in layers.items():
        for key, value in fields.items():
            full_key = f"{layer_name}.{key}"
            packet_dict[full_key] = value

    return packet_dict


def pcap_to_csv(input_pcap, output_csv):
    """主转换函数"""
    try:
        packets = rdpcap(input_pcap)
        print(f"成功加载 {len(packets)} 个数据包")

        # 收集所有字段
        fieldnames = set()
        packet_list = []

        for pkt in packets:
            if IP in pkt:
                # 处理 IPv4/IPv6
                ip_layer = 'IPv4' if pkt[IP].version == 4 else 'IPv6'
                fieldnames.add(ip_layer)
            packet_dict = packet_to_dict(pkt)
            packet_list.append(packet_dict)
            fieldnames.update(packet_dict.keys())

        # 写入 CSV 文件
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(fieldnames))
            writer.writeheader()
            for row in packet_list:
                writer.writerow(row)

        print(f"成功导出到 {output_csv}，包含 {len(packet_list)} 条记录")

    except Exception as e:
        print(f"处理失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    pcap_file = './data/switch.pcap'
    csv_file = './data/packets.csv'
    pcap_to_csv(pcap_file, csv_file)
