""" 模拟一辆车前进，另一辆车使用ACC模型跟进的情况
    在模拟中两辆车一起前进，遇到碰撞事件直接退出程序
    """
import sys

k1 = 0.23  # parameters
k2 = 0.07
thw = 0.0
v1_ini = 20  # initial speed
v2_ini = 20
h_ini = 40  # Initial Headway
T = 30  # Total Simulation Time
Tstep = 0.1  # Time step for Simulation

v1_2 = 0  # 行驶状态参数
v1_3 = 20
a1_ac = 4
a1_de = -4
t1 = 10
t2 = 15
t3 = 21

t = []  # Time of Traveling
for i in range(0, int(T / Tstep)):
    t.append(i * 0.1)

# Initialization of Deceleration Matrix, Speed Matrix, Movement Matrix
a = [[], []]
v = [[], []]
d = [[], []]


# 第二辆车按ACC前进
def v2_go(step):
    v[1].append(v[1][step - 1] + a[1][step - 1] * 0.1)
    d[1].append(d[1][step - 1] + (v[1][step - 1] + v[1][step]) / 2 * Tstep)
    a[1].append(k1 * (d[0][step] - d[1][step] - thw * v[1][step]) + k2 * (v[0][step] - v[1][step]))


# 检测是否发生碰撞，若发生则
def crash_test(step):
    if (d[0][step] <= d[1][step]):
        print("crash at time step: %.2f" % (step / 10))
        output_data(step)
        sys.exit(0)


# Output to File for Plotting
def output_data(step=int(T / Tstep)):
    f = open('CarFollowing.txt', 'w')
    for i in range(0, step):
        f.write('time %.1f:\t' % t[i])
        f.write('car 0: %.2f\t%.2f\t%.2f\t' % (d[0][i], v[0][i], a[0][i]))
        f.write('car 1: %.2f\t%.2f\t%.2f\t' % (d[1][i], v[1][i], a[1][i]))
        f.write('\n')
    f.close()


v[0].append(v1_ini)
d[0].append(h_ini)
a[0].append(0)
v[1].append(v2_ini)
d[1].append(0)
a[1].append(0)
# 不变速10秒
for i in range(1, int(t1 / Tstep)):
    v[0].append(v1_ini)
    d[0].append(d[0][i - 1] + (v[0][i - 1] + v[0][i]) / 2 * Tstep)
    a[0].append(0)
    v2_go(i)
    crash_test(i)
# 用a1_de减速到v1_2
for i in range(int(t1 / Tstep), int(t2 / Tstep)):
    v[0].append(v[0][i - 1] + a[0][i - 1] * Tstep)
    d[0].append(d[0][i - 1] + (v[0][i - 1] + v[0][i]) / 2 * Tstep)
    a[0].append(a1_de)
    v2_go(i)
    crash_test(i)
# 用a1_ac加速到v1_3
for i in range(int(t2 / Tstep), int(t3 / Tstep)):
    v[0].append(v[0][i - 1] + a[0][i - 1] * Tstep)
    d[0].append(d[0][i - 1] + (v[0][i - 1] + v[0][i]) / 2 * Tstep)
    a[0].append(a1_ac)
    v2_go(i)
    crash_test(i)
# 不变速10秒
for i in range(int(t3 / Tstep), int(T / Tstep)):
    v[0].append(v[0][i - 1])
    d[0].append(d[0][i - 1] + (v[0][i - 1] + v[0][i]) / 2 * Tstep)
    a[0].append(0)
    v2_go(i)
    crash_test(i)

output_data()
