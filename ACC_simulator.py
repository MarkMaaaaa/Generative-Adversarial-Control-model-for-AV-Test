""" this is a simulation for the ACC model
    in this scenario, the first car c_0 will first go in a constant speed(stage0),
    then decelerate(stage1), accelerate(stage2), and finally go at a constant speed again(stage3)
    the second car will move according to the ACC model

    result shows that when thw >= 2.4, there won't be any crash.
    """
import sys
import matplotlib.pyplot as plt
import numpy as np

# parameters for ACC model
k1 = 0.23  # (s^-1)
k2 = 0.07  # (s^-2)
thw = 2.4  # (s)
# simulation parameters
T = 30  # simulation time horizen (s)
Tstep = 0.1  # time step (s)
T1 = 10  # time points for each stage (s)
T2 = 15
T3 = 21
# driving states parameters for c_0 and c_1
v0_s0 = 20  # subscript order: car id, stage (m/s)
a0_s1 = -4  # (m/s^2)
a0_s2 = 4  # (m/s^2)
v0_s3 = 20  # (m/s)
v1_ini = 20  # (m/s)
# other parameters
h_ini = 40  # initial headway (m)
car_len = 5  # car length for crash_test (m)

# Initialization of Deceleration Matrix, Speed Matrix, Movement Matrix, Time of Traveling
a = [[], []]
v = [[], []]
d = [[], []]
t = []
for i in range(0, int(T / Tstep)):
    t.append(i * Tstep)


def cal_v(id, step):
    v[id].append(v[id][step - 1] + a[id][step - 1] * Tstep)


def cal_dis(id, step):
    d[id].append(d[id][step - 1] + (v[id][step - 1] + v[id][step]) / 2 * Tstep)


# car_id move forward according to ACC model
def ACC_move(id, step):
    cal_v(id, step)
    cal_dis(id, step)
    a[id].append(k1 * (d[id - 1][step] - d[id][step] - thw * v[id][step]) + k2 * (v[id - 1][step] - v[id][step]))


# test if there is any crash happened, stop the simulation if happened
def crash_test(step, exist=True):
    if (d[0][step] - d[1][step] <= car_len):
        print("crash at time step: %.2f" % (step / 10))
        if (exist):
            output_data(step + 1)
            sys.exit(0)


# output to file and plot
def output_data(step=int(T / Tstep)):
    pos_0 = np.array(d[0])
    pos_1 = np.array(d[1])
    del_v = np.array(v[0]) - np.array(v[1])
    del_a = np.array(a[0]) - np.array(a[1])
    # output file
    f = open('CarFollowing.txt', 'w')
    for i in range(0, step):
        f.write('time %.1f:\t%.1f\t%.1f\t%.1f\t%.1f' % (t[i], pos_0[i], pos_1[i], del_v[i], del_a[i]))
        f.write('\n')
    f.close()
    # time-pos
    x = np.linspace(0, step * Tstep, step)
    plt.figure(0)
    plt.xlabel('time')
    plt.ylabel('position')
    plt.plot(x, pos_0, linestyle='--')
    plt.plot(x, pos_1)
    plt.savefig('time-pos.png', dpi=300)
    plt.show()
    # time-delta_v
    plt.figure(1)
    plt.xlabel('time')
    plt.ylabel('delta v')
    plt.plot(x, del_v)
    plt.savefig('time-delta v.png', dpi=300)
    plt.show()
    # time-delta_a
    plt.figure(2)
    plt.xlabel('time')
    plt.ylabel('delta a')
    plt.plot(x, del_a)
    plt.savefig('time-delta a.png', dpi=300)
    plt.show()


# initial state
v[0].append(v0_s0)
d[0].append(h_ini)
a[0].append(0)
v[1].append(v1_ini)
d[1].append(0)
a[1].append(0)


def simulate_in_stage(start_time, end_time, a0):
    for i in range(int(start_time / Tstep), int(end_time / Tstep)):
        # update car_0
        cal_v(0, i)
        cal_dis(0, i)
        a[0].append(a0)
        # update car_1
        ACC_move(1, i)
        # test crash
        crash_test(i)


# start simulation
simulate_in_stage(0.1, T1, 0)  # stage0: c_0 constant speed
simulate_in_stage(T1, T2, a0_s1)  # stage1: c_0 decelerate
simulate_in_stage(T2, T3, a0_s2)  # stage2: c_0 accelerate
simulate_in_stage(T3, T, 0)  # stage3: c_0 constant speed

output_data()
