topo: nx.Graph = load_topo(json_path='topology.json')
controller = SimpleSwitchP4RuntimeAPI(
    device_id=1,
    grpc_port=9559,
    grpc_ip='10.120.130.59',
    p4rt_path='/home/p4/Desktop/PPM/unmonitored_p4rt.txt',
    json_path='/home/p4/Desktop/PPM/unmonitored.json'
)

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