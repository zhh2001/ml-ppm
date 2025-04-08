/**
 * P4_16
 * 固定阈值的监控
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

struct metadata {

}

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
register<timestamp_t, bit<1>>(1) prev_timestamp;  // 上次上报的时间戳

parser FixedParser(
    packet_in packet,
    out headers hdr,
    inout metadata meta,
    inout standard_metadata_t standard_metadata
) {

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

control FixedVerifyChecksum(
    inout headers  hdr,
    inout metadata meta
) {

    apply {

    }

}

control FixedIngress(
    inout headers             hdr,
    inout metadata            meta,
    inout standard_metadata_t standard_metadata
) {

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

    apply {
        if (hdr.ipv4.isValid()) {
            ipv4_lpm.apply();
        }

        // 读取先前的数据
        reportedCount_t pkt_count;
        reportedCount_t byte_count;
        statistical_data.read(pkt_count, PKT_INDEX);
        statistical_data.read(byte_count, BYTE_INDEX);

        // 更新统计数据
        pkt_count = pkt_count + 1;
        byte_count = byte_count + standard_metadata.packet_length;
        statistical_data.write(PKT_INDEX, pkt_count);
        statistical_data.write(BYTE_INDEX, byte_count);

        timestamp_t prev_reported_timestamp;
        prev_timestamp.read(prev_reported_timestamp, 0);

        timestamp_t timestamp_delta;
        timestamp_delta = standard_metadata.ingress_global_timestamp - prev_reported_timestamp;

        if (timestamp_delta >= 500_000) {
            // 上报统计信息
            reported_data data;
            data.reported_pkts = pkt_count;
            data.reported_bytes = byte_count;
            data.timestamp_delta = timestamp_delta;
            digest<reported_data>(1, data);

            // 记录上报时间
            prev_timestamp.write(0, standard_metadata.ingress_global_timestamp);

            // 重置统计数据
            statistical_data.write(PKT_INDEX, 0);
            statistical_data.write(BYTE_INDEX, 0);
        }
    }

}

control FixedEgress(inout headers             hdr,
                    inout metadata            meta,
                    inout standard_metadata_t standard_metadata) {
    apply { }
}

control FixedComputeChecksum(inout headers  hdr,
                             inout metadata meta) {

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

control FixedDeparser(packet_out packet,
                      in headers hdr) {

    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
    }

}

V1Switch(
    FixedParser(),
    FixedVerifyChecksum(),
    FixedIngress(),
    FixedEgress(),
    FixedComputeChecksum(),
    FixedDeparser()
) main;
