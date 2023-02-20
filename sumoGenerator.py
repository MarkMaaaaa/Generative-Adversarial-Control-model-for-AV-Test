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
        % (0, 0, 1, lane_num, speed))
    file.write('\n</edges>')
