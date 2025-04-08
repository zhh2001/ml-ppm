/**
 * P4_16
 * 直接转发，无监控
 */

#define V1MODEL_VERSION 20200408

#include <core.p4>
#include <v1model.p4>

typedef bit<9>  egressSpec_t;
typedef bit<16> etherType_t;
typedef bit<32> ipAddr_t;
typedef bit<48> macAddr_t;

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

parser UnmonitoredParser(
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

control UnmonitoredVerifyChecksum(
    inout headers  hdr,
    inout metadata meta
) {

    apply {

    }

}

control UnmonitoredIngress(
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
    }

}

control UnmonitoredEgress(
    inout headers             hdr,
    inout metadata            meta,
    inout standard_metadata_t standard_metadata
) {

    apply {

    }

}

control UnmonitoredComputeChecksum(
    inout headers  hdr,
    inout metadata meta
) {

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

control UnmonitoredDeparser(packet_out packet,
                   in headers hdr) {

    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
    }

}

V1Switch(
    UnmonitoredParser(),
    UnmonitoredVerifyChecksum(),
    UnmonitoredIngress(),
    UnmonitoredEgress(),
    UnmonitoredComputeChecksum(),
    UnmonitoredDeparser()
) main;
