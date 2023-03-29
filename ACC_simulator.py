""" this is a simulation for the ACC model
    in this scenario, the first car c_0 will first go in a constant speed(stage0),
    then decelerate(stage1), accelerate(stage2), and finally go at a constant speed again(stage3)
    the second car will move according to the ACC model

    result shows that when thw >= 2.4, there won't be any crash.
    """
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# parameters for ACC model
k1 = 0.23  # (s^-1)
k2 = 0.07  # (s^-2)
thw = 2.4  # (s)
ttc = 1  # (s)
# simulation parameters
T = 0  # simulation time horizen (s)
Tstep = 0.1  # time step (s)
# other parameters
h_ini = 0  # initial headway (m)
car_len = 3  # car length for crash_test (m)

# Initialization of Deceleration Matrix, Speed Matrix, Movement Matrix, Time of Traveling
a = [[], []]
v = []
d = [[], []]
t = []
ttcs = []


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
def crash_test(step, exit=True):
    # if (d[0][step] - d[1][step] <= car_len):
    #     print("crash at time step: %.2f" % (step / 10))
    # record TTC
    t2c = (d[0][step] - d[1][step] - car_len) / (v[0][step] - v[1][step])
    t2c = max(0, t2c)
    t2c = min(10, t2c)
    ttcs.append(t2c)
    # TTC collision test
    if (0 <= (d[0][step] - d[1][step] - car_len) / (v[0][step] - v[1][step]) <= ttc):
        print("crash at time step: %.2f" % (step / 10))
        if (exit):
            output_data(step + 1, delta_a=False)
            sys.exit(0)


# output to file and plot
def output_data(step, outText=True, pos=True, delta_v=True, delta_a=True):
    out_path = "./output/"
    v[0] = v[0][0:step]
    pos_0 = np.array(d[0])
    pos_1 = np.array(d[1])
    del_v = np.array(v[0][0:step]) - np.array(v[1])
    # del_a = np.array(a[0]) - np.array(a[1])
    # output file
    if outText:
        f = open(out_path + 'CarFollowing.txt', 'w')
        for i in range(0, step):
            f.write('time %.1f:\t%.1f\t%.1f\t%.1f' % (t[i], pos_0[i], pos_1[i], del_v[i]))  # , del_a[i]
            f.write('\n')
        f.close()
    x = np.linspace(0, step * Tstep, step)
    # time-pos
    if pos:
        plt.figure(0)
        plt.xlabel('time')
        plt.ylabel('position')
        plt.plot(x, pos_0, linestyle='--')
        plt.plot(x, pos_1)
        plt.savefig(out_path + 'time-pos.png', dpi=300)
        plt.show()
    # time-v
    if delta_v:
        plt.figure(1)
        plt.xlabel('time')
        plt.ylabel('v')
        plt.plot(x, np.array(v[0]), linestyle='--')
        plt.plot(x, np.array(v[1]))
        plt.savefig(out_path + 'time-v.png', dpi=300)
        plt.show()
    # time-delta_v
    if delta_v:
        plt.figure(2)
        plt.xlabel('time')
        plt.ylabel('delta v')
        plt.plot(x, del_v)
        plt.savefig(out_path + 'time-delta_v.png', dpi=300)
        plt.show()
    # time-delta_a
    if delta_a:
        plt.figure(3)
        plt.xlabel('time')
        plt.ylabel('delta a')
        plt.plot(x, del_a)
        plt.savefig(out_path + 'time-delta_a.png', dpi=300)
        plt.show()
    # time-ttc
    x = np.linspace(0, 1000, step)
    plt.figure(3)
    plt.xlabel('time')
    plt.ylabel('ttc')
    # plt.plot(x, np.array(ttcs))
    plt.scatter(x, np.array(ttcs[:1000]), marker='o', color='green', s=40, label='ttc')
    plt.savefig(out_path + 'time-ttc.png', dpi=300)
    plt.show()


def compare_ACC(step, v_ACC, dgap_ACC):
    out_path = "./output/"
    x = np.linspace(0, step * Tstep, step)
    # delta_v_acc
    plt.figure(4)
    plt.xlabel('time')
    plt.ylabel('delta v acc')
    plt.plot(x, np.array(v[1]) - v_ACC)
    plt.savefig(out_path + 'time-delta_v_acc.png', dpi=300)
    plt.show()
    # v_acc
    plt.figure(5)
    plt.xlabel('time')
    plt.ylabel('v acc')
    plt.plot(x, np.array(v[1]), linestyle='--')
    plt.plot(x, v_ACC)
    plt.savefig(out_path + 'time-v_acc.png', dpi=300)
    plt.show()
    # delta_d_acc
    plt.figure(6)
    plt.xlabel('time')
    plt.ylabel('delta d acc')
    plt.plot(x, np.array(d[1]) - (np.array(d[0]) - dgap_ACC))
    plt.savefig(out_path + 'time-delta_d_acc.png', dpi=300)
    plt.show()
    # d_acc
    plt.figure(5)
    plt.xlabel('time')
    plt.ylabel('pos acc')
    plt.plot(x, np.array(d[1]), linestyle='--')
    plt.plot(x, np.array(d[0]) - dgap_ACC)
    plt.savefig(out_path + 'time-pos_acc.png', dpi=300)
    plt.show()


# read data from OpenACC and initalize data, output simulation time
def input_data(file):
    # read data in OpenACC
    data = pd.read_csv(file, header=5, usecols=['Time', 'Speed1', 'Speed2', 'Speed3', 'IVS1', 'IVS2'])
    T = round(data.iloc[-1, 0] - data.iloc[0, 0], 1)
    # T = 1000.0  # debug
    h_ini = round(data.loc[0, 'IVS1'], 1)
    v1_ini = round(data.loc[0, 'Speed2'], 1)
    v.append(data.loc[:T * 10 - 1, 'Speed1'].to_numpy(dtype=float))
    v_ACC = data.loc[:T * 10 - 1, 'Speed2'].to_numpy(dtype=float)  # reference v for ACC (m/s)
    dgap_ACC = data.loc[:T * 10 - 1, 'IVS1'].to_numpy(dtype=float)  # reference distance gap between vehicle and AV (m)
    # initialize
    for i in range(0, int(round(T / Tstep))):
        t.append(i * Tstep)
    v.append([])
    d[0].append(h_ini)
    v[1].append(v1_ini)
    d[1].append(0)
    a[1].append(0)
    return [T, v_ACC, dgap_ACC]


def simulate_in_stage(start_time, end_time):
    for i in range(int(round(start_time / Tstep)), int(round(end_time / Tstep))):
        # update car_0
        # cal_v(0, i)
        cal_dis(0, i)
        # a[0].append(a0)
        # update car_1
        ACC_move(1, i)
        # test crash
        crash_test(i)


file_list = [["Cher-JRC_240918_rec2_part1.csv"],
             ["VC-JRC_280219_part2.csv", "VC-JRC_280219_part3.csv"]]
file_path = ["./data/Cherasco/",
             "./data/Vicolungo/"]
file_name = file_path[1] + file_list[1][1]
[T, v_ACC, dgap_ACC] = input_data(file_name)
simulate_in_stage(0.1, T)  # start simulation
crash_test(0)
output_data(step=int(round(T / Tstep)), delta_a=False)
compare_ACC(int(round(T / Tstep)), v_ACC, dgap_ACC)

print("simulation finish!")
