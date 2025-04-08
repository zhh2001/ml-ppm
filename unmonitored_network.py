net = NetworkAPI()

net.setCompiler(p4rt=True)
net.addP4RuntimeSwitch(name='s1')
net.setP4SourceAll(p4_src='unmonitored.p4')

net.addHost(name='h1')
net.addHost(name='h2')

net.addLink(node1='s1', node2='h1')
net.addLink(node1='s1', node2='h2')
net.setBwAll(10)

net.mixed()
net.setLogLevel(logLevel='info')
net.enablePcapDumpAll(pcap_dir='./pcap')
net.enableLogAll(log_dir='./log')
net.enableCli()
net.startNetwork()