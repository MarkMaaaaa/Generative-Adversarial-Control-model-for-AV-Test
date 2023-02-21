import random

nodes_num = 2
road_length = 1000
speed = random.randint(70, 80)
lane_num = random.randint(1, 4)

# 生成Node文件, 从(0,0)开始生成的
with open('example.nod.xml', 'w') as file:
    file.write('<nodes> \n\n')
    file.write('\t<node id="node%d" x="%d" y="%d" type="priority" /> \n' % (0, 0, 0))
    file.write('\t<node id="node%d" x="%d" y="%d" type="priority" /> \n' % (1, 0 + road_length, 0))
    file.write('\n</nodes>')

# 生成edge文件
with open('example.edg.xml', 'w') as file:
    file.write('<edges> \n\n')
    file.write(
        '\t<edge id="edge%d" from="node%d" to="node%d" priority="75" numLanes="%d" speed="%d"/> \n'
        % (0, 0, 1, 5, speed))
    file.write('\n</edges>')

# 生成route文件
with open('example.rou.xml', 'w') as file:
    file.write('<routes> \n\n')
    # 第一种车型：普通BV
    file.write(
        '\t<vType id="type%d" accel="%.1f" decel="%.1f" sigma="%.1f" length="%d" color="1,0,0"/>\n'
        % (0, 0.8, 4.5, 0.5, 5))
    # 第二种车型：pov
    file.write(
        '\t<vType id="type%d" accel="%.1f" decel="%.1f" sigma="%.1f" length="%d" color="0,1,0"/>\n\n'
        % (1, 0.8, 4.5, 0.5, 5))
    # 路径
    file.write('\t<route id="route0" edges="edge0"/>\n\n')
    # 车辆
    file.write(
        '\t<vehicle id="%d" type="type%d" route="route0" depart="0" departLane="%d" departPos="%d"/>\n'
        % (0, 0, 0, 100))
    file.write(
        '\t<vehicle id="%d" type="type%d" route="route0" depart="0" departLane="%d" departPos="%d"/>\n'
        % (2, 1, 1, 50))
    file.write('\n</routes>')
