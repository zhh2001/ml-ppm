/**
 * P4_16
 * 机器学习模型动态调整阈值
 */

#define V1MODEL_VERSION 20200408

#include <core.p4>
#include <v1model.p4>

typedef bit<9>  egressSpec_t;
typedef bit<16> etherType_t;
typedef bit<32> ipAddr_t;
typedef bit<48> macAddr_t;
typedef bit<32> reportedCount_t;
typedef bit<48> timestamp_t;

const etherType_t TYPE_IPV4 = 16w0x0800;

header ethernet_t {
    macAddr_t   dstAddr;
    macAddr_t   srcAddr;
    etherType_t etherType;
}

header ipv4_t {
    bit<4>   version;
    bit<4>   ihl;
    bit<8>   diffserv;
    bit<16>  totalLen;
    bit<16>  identification;
    bit<3>   flags;
    bit<13>  fragOffset;
    bit<8>   ttl;
    bit<8>   protocol;
    bit<16>  hdrChecksum;
    ipAddr_t srcAddr;
    ipAddr_t dstAddr;
}

struct metadata { }

struct headers {
    ethernet_t ethernet;
    ipv4_t     ipv4;
}

struct reported_data {
    reportedCount_t reported_pkts;
    reportedCount_t reported_bytes;
    timestamp_t     timestamp_delta;
}

const bit<1> PKT_INDEX  = 0;
const bit<1> BYTE_INDEX = 1;

register<reportedCount_t, bit<1>>(2) statistical_data;  // 统计数据：[包数, 字节数]
register<timestamp_t, bit<1>>(1) prev_timestamp;        // 上次上报的时间戳
register<timestamp_t, bit<1>>(1) thld_prd_m;            // 时间阈值

/**
 * 上报数据到控制器
 *
 * :param pkt_count: 数据包的数量
 * :param byte_count: 字节数量
 * :param timestamp_delta: 距离上次报告数据的时间
 */
void report(in reportedCount_t pkt_count,
            in reportedCount_t byte_count,
            in timestamp_t timestamp_delta) {
    reported_data data;
    data.reported_pkts = pkt_count;
    data.reported_bytes = byte_count;
    data.timestamp_delta = timestamp_delta;
    digest<reported_data>(1, data);
}

/**
 * 重置统计数据
 */
void reset_statistical_data() {
    statistical_data.write(PKT_INDEX, 0);
    statistical_data.write(BYTE_INDEX, 0);
}

/**
 * 获取阈值
 *
 * :return: 时间阈值
 */
timestamp_t get_prd_m() {
    timestamp_t prd_m;
    thld_prd_m.read(prd_m, 0);
    return prd_m;
}

parser MLPPMParser(packet_in packet,
                   out headers hdr,
                   inout metadata meta,
                   inout standard_metadata_t standard_metadata) {

    state start {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            default:   accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition accept;
    }

}

control MLPPMVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply { }
}

control MLPPMIngress(inout headers hdr,
                     inout metadata meta,
                     inout standard_metadata_t standard_metadata) {

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action ipv4_forward(
        macAddr_t    dstAddr,
        egressSpec_t port
    ) {
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = dstAddr;
        standard_metadata.egress_spec = port;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }

    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    /**
     * 更新统计数据
     *
     * :param pkt_count: 数据包的数量
     * :param byte_count: 字节数量
     */
    action increase_statistical_data(inout reportedCount_t pkt_count,
                                     inout reportedCount_t byte_count) {
        statistical_data.read(pkt_count, PKT_INDEX);
        statistical_data.read(byte_count, BYTE_INDEX);
        pkt_count = pkt_count + 1;
        byte_count = byte_count + standard_metadata.packet_length;
        statistical_data.write(PKT_INDEX, pkt_count);
        statistical_data.write(BYTE_INDEX, byte_count);
    }

    apply {
        if (hdr.ipv4.isValid()) {
            ipv4_lpm.apply();
        }

        reportedCount_t pkt_count;
        reportedCount_t byte_count;
        increase_statistical_data(pkt_count, byte_count);

        // 获取时间差
        timestamp_t prev_reported_timestamp;
        prev_timestamp.read(prev_reported_timestamp, 0);
        timestamp_t timestamp_delta = standard_metadata.ingress_global_timestamp - prev_reported_timestamp;

        // 是否超过阈值
        if (timestamp_delta >= get_prd_m()) {
            report(pkt_count, byte_count, timestamp_delta);
            prev_timestamp.write(0, standard_metadata.ingress_global_timestamp);
            reset_statistical_data();
        }
    }

}

control MLPPMEgress(inout headers hdr,
                    inout metadata meta,
                    inout standard_metadata_t standard_metadata) {
    apply { }
}

control MLPPMComputeChecksum(inout headers hdr, inout metadata meta) {

    apply {
	update_checksum(
	    hdr.ipv4.isValid(),
	    {
	        hdr.ipv4.version,
	        hdr.ipv4.ihl,
	        hdr.ipv4.diffserv,
	        hdr.ipv4.totalLen,
	        hdr.ipv4.identification,
	        hdr.ipv4.flags,
	        hdr.ipv4.fragOffset,
	        hdr.ipv4.ttl,
	        hdr.ipv4.protocol,
	        hdr.ipv4.srcAddr,
	        hdr.ipv4.dstAddr,
	    },
	    hdr.ipv4.hdrChecksum,
	    HashAlgorithm.csum16
	);
    }

}

control MLPPMDeparser(packet_out packet, in headers hdr) {

    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
    }

}

V1Switch(
    MLPPMParser(),
    MLPPMVerifyChecksum(),
    MLPPMIngress(),
    MLPPMEgress(),
    MLPPMComputeChecksum(),
    MLPPMDeparser()
) main;
