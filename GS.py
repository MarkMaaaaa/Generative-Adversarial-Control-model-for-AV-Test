""" Generator-SimuNet network to generate testing scenarios for AV test
    """

import os
import tensorflow as tf
import numpy as np
from d2l import tensorflow as d2l
import sys

# from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Generator():
    """ generate scenarios in car following model.
        input: random variables
        output: scenarios, which are described by the following messages:
            x: distance between leading car and AV (m)
            v0: v for leading car (m/s)
            v1: v for AV (m/s)
            a0, t: a for leading car in t time (m/s^2, s)
    """

    def __init__(self):
        self.net = self.build_net()
        self.loss = tf.keras.losses.MeanSquaredError()  # binary crossentropy
        # self.trainer = tf.keras.optimizers.SGD()
        self.trainer = tf.keras.optimizers.Adam(0.0005, 0.9)
        self.input_size = 3  # size for input random variable

    class MyLayer(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.weight = tf.constant([10, 10, 10, 6, 2], shape=[1, 5], dtype=float)
            self.bias = tf.constant([10, 20, 20, -3, 0], shape=[1, 5], dtype=float)
            # 10 <= x <= 20 (m)
            # 20 <= v0 <= 30 (m/s)
            # 20 <= v1 <= 30 (m/s)
            # -3 <= a0 <= 3 (m/s^2)
            # 0 <= t <= 2 (s)

        def call(self, inputs):
            inputs = inputs * self.weight + self.bias
            # inputs_1 = tf.round(inputs[:, :4] * 100) / 100  # TODO: No gradients provided for any variable
            # inputs_2 = tf.round(inputs[:, 4:5] * 10) / 10
            # return tf.concat([inputs_1, inputs_2], 1)
            return inputs

    def build_net(self):
        net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(16, activation='relu', name="D1"),
            tf.keras.layers.Dense(64, activation='relu', name="D2"),
            tf.keras.layers.Dense(16, activation='relu', name="D3"),
            tf.keras.layers.Dense(5, activation='sigmoid', name="D4"),
            self.MyLayer()  # [x, v0, v1, a0, t]
        ])
        return net


class Simulator():
    """ simulation for car following model,
            where AV is moved according to ACC model.
            simulation will stop when collision happened, or it will continue T time.
        input: scenarios genearted by Generator
        output: the min TTC in the simulation
    """

    def __init__(self, inputs):
        # parameters for ACC model
        self.k1 = 0.23  # (s^-1)
        self.k2 = 0.07  # (s^-2)
        self.thw = 2.4  # (s)
        self.M = 10.0  # a big number of ttc TODO
        # simulation parameters
        self.T = 4  # simulation time horizen (s)
        self.Tstep = 0.1  # time step (s)
        # other parameters
        self.car_len = 3  # car length for crash_test (m)
        # input parameters
        self.h_ini = float(inputs[0, 0])  # initial headway (m)
        self.v0 = float(inputs[0, 1])
        self.v1 = float(inputs[0, 2])
        self.a0 = float(inputs[0, 3])
        self.t = float(inputs[0, 4])
        # record info
        self.a = [[], []]
        self.v = [[], []]
        self.d = [[], []]
        self.TTCs = []

    def cal_v(self, id, step):
        self.v[id].append(self.v[id][step - 1] + self.a[id][step - 1] * self.Tstep)

    def cal_dis(self, id, step):
        self.d[id].append(self.d[id][step - 1] + (self.v[id][step - 1] + self.v[id][step]) / 2 * self.Tstep)

    # car_id move forward according to ACC model
    def ACC_move(self, id, step):
        self.cal_v(id, step)
        self.cal_dis(id, step)
        self.a[id].append(self.k1 * (self.d[id - 1][step] - self.d[id][step] - self.thw * self.v[id][step])
                          + self.k2 * (self.v[id - 1][step] - self.v[id][step]))

    # test if there is any crash happened, stop the simulation if happened
    def crash_test(self, step, exit=True):
        # record TTC
        del_d = self.d[0][step] - self.d[1][step] - self.car_len
        del_v = self.v[1][step] - self.v[0][step]
        if del_d <= 0:
            ttc = 0.0
        elif del_v <= 0:
            ttc = self.M
        else:
            ttc = round(del_d / del_v, 1)
            ttc = min(self.M, ttc)
        # TTC collision test
        if (0 <= ttc <= 1):  # TODO setting collision standard
            # print("crash at time step: %.2f" % (step / 10))
            if (exit):
                while self.TTCs.__len__() < int(round(self.T / self.Tstep)):
                    self.TTCs.append(0)
                return True
        self.TTCs.append(ttc)
        return False

    def initial(self):
        self.d[0].append(self.h_ini)
        self.v[0].append(self.v0)
        self.a[0].append(self.a0)
        self.d[1].append(0)
        self.v[1].append(self.v1)
        self.a[1].append(0)
        self.crash_test(0)

    def simulate_in_stage(self, start_time, end_time, accelerate):
        for i in range(int(round(start_time / self.Tstep)), int(round(end_time / self.Tstep))):
            # update car_0
            self.cal_v(0, i)
            self.cal_dis(0, i)
            self.a[0].append(accelerate)
            # update car_1
            self.ACC_move(1, i)
            # test crash
            if self.crash_test(i):
                return True

    def simulate(self):
        self.initial()
        if self.simulate_in_stage(self.Tstep, self.t, self.a0):
            return self.TTCs
        self.simulate_in_stage(self.t, self.T, 0)
        return self.TTCs


class SimuNet():
    """ a neural network simulator for car following model.
        input: scenarios genearted by Generator
        output: the min TTC in the simulation
    """

    def __init__(self):
        self.net = self.build_net()
        self.loss = self.loss_msegap # tf.keras.losses.MeanSquaredError()
        # self.trainer = tf.keras.optimizers.SGD()
        self.trainer = tf.keras.optimizers.Adam(0.0005, 0.9)
        self.M = 10.0

    def build_net(self):
        net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1, activation='relu'),
        ])
        return net

    # test the accuracy of simu net
    def prediction_mse(self, y_true, y_pred):
        sqr = tf.square(y_true - y_pred)
        return tf.reduce_sum(sqr)

    def prediction_msegap(self, y_true, y_pred):
        x = 1.0
        sqr = tf.square(y_true - y_pred)
        nume = tf.fill(dims=y_true.shape, value=self.M) - y_true + x
        deno = tf.fill(dims=y_true.shape, value=self.M) + x
        gap = sqr * nume / deno
        return tf.reduce_sum(gap)

    def loss_msegap(self, y_true, y_pred):
        x = 1.0
        y_true = tf.cast(y_true, dtype=tf.float32)
        sqr = tf.square(y_true - y_pred)
        nume = tf.fill(dims=y_true.shape, value=self.M) - y_true + x
        deno = tf.fill(dims=y_true.shape, value=self.M) + x
        gap = sqr * nume / deno  # gap = (M-y_true+x)/(M+x), where x is a parameter
        return tf.reduce_mean(tf.reduce_sum(gap, axis=-1))


# random generate a scenario from uniform distribution
def random_generate(num):
    scenarios = []
    for _ in range(num):
        scenario = tf.random.uniform([1, 5], minval=0, maxval=1, dtype=tf.float32)
        weight = tf.constant([10, 10, 10, 6, 2], shape=[1, 5], dtype=float)
        bias = tf.constant([10, 20, 20, -3, 0], shape=[1, 5], dtype=float)
        scenario = scenario * weight + bias
        scenarios.append(scenario)
    return scenarios


# only train simu-net
def trainSN(epochs, sample_interval, simu_net):
    for epoch in range(epochs):
        # generate scenarios
        scenarios = random_generate(1)
        # train the simu net, make ttc close to real ttc (from simulator)
        d_loss_sum = [0, 0, 0]
        for scenario in scenarios:
            sim = Simulator(scenario)
            risk = np.array(sim.simulate())
            min_ttc = risk.min().reshape(1, -1)
            d_loss = simu_net.net.train_on_batch(scenarios, min_ttc)
            d_loss_sum = np.add(np.array(d_loss) / len(scenarios), d_loss_sum)

        if epoch % sample_interval == 0:
            print("%d [D loss: %f, MSE: %.2f, MSEG: %.2f]" % (
                epoch, d_loss_sum[0], d_loss_sum[1], d_loss_sum[2]))
            with open('train_log.txt', 'a') as f:
                f.write("%d [D loss: %f, MSE: %.2f, MSEG: %.2f]\n" % (
                    epoch, d_loss_sum[0], d_loss_sum[1], d_loss_sum[2]))
            simu_net.net.save('./simu_net.h5')
            simu_net.net.save('./simu_net_weight.h5')

    print("train complete!")


# train simu-net and generator together
def train(epochs, sample_interval, gen, simu_net, combined):
    for epoch in range(epochs):
        # generate scenarios
        scenarios = random_generate(0)
        inputs = tf.random.uniform([1, gen.input_size], minval=-10, maxval=10, dtype=tf.float32)
        scenarios.append(gen.net(inputs))
        # train the simu net, make ttc close to real ttc (from simulator)
        d_loss_sum = [0, 0, 0]
        for scenario in scenarios:
            sim = Simulator(scenario)
            risk = np.array(sim.simulate())
            min_ttc = risk.min().reshape(1, -1)
            d_loss = simu_net.net.train_on_batch(scenario, min_ttc)
            d_loss_sum = np.add(np.array(d_loss) / len(scenarios), d_loss_sum)
        # train the generator, make ttc close to 0
        g_loss = combined.train_on_batch(inputs, np.zeros((1, 1)))

        if epoch % sample_interval == 0:
            min_ttc = risk.min()
            print("%d [D loss: %f, MSE: %.2f, MSEG: %.2f] [G loss: %f, risk: %.2f]" % (
                epoch, d_loss_sum[0], d_loss_sum[1], d_loss_sum[2], g_loss, min_ttc))
            with open('train_log.txt', 'a') as f:
                f.write("%d [D loss: %f, MSE: %.2f] [G loss: %f, risk: %.2f]\n" % (
                    epoch, d_loss_sum[0], 100 * d_loss_sum[1], g_loss, min_ttc))
            simu_net.net.save('./simu_net.h5')
            simu_net.net.save('./simu_net_weight.h5')
            combined.save('./combined_model.h5')
            combined.save('./combined_model_weight.h5')

    print("train complete!")


def build_GS_model(gen, simu_net):
    # compile the simu net
    simu_net.net.compile(loss=simu_net.loss, optimizer=simu_net.trainer,
                         metrics=[simu_net.prediction_mse, simu_net.prediction_msegap])
    # The generator takes noise as input and generates scenarios
    gen_in = tf.keras.layers.Input(shape=(gen.input_size,), batch_size=1)
    scenario = gen.net(gen_in)
    # For the combined model we will only train the simu net
    simu_net.net.trainable = False
    # The simu net takes generated scenarios as input and determines risk level for this scenario
    risk = simu_net.net(scenario)
    # The combined model
    combined = tf.keras.models.Model(gen_in, risk)
    combined.compile(loss=gen.loss, optimizer=gen.trainer)

    # combined.summary()
    print("build complete!")

    return combined


# generate scenario for testing the model
def generate_scenario():
    combined = tf.keras.models.load_model('./combined_model.h5', custom_objects={'MyLayer': Generator.MyLayer})
    gen = tf.keras.Model(inputs=combined.layers[1].input, outputs=combined.layers[1].output)
    inputs = tf.random.uniform([1, 3], minval=-10, maxval=10, dtype=tf.float32)
    scenario = gen(inputs)
    print(scenario)
    inputs = tf.random.uniform([1, 3], minval=-10, maxval=10, dtype=tf.float32)
    scenario = gen(inputs)
    print(scenario)
    inputs = tf.random.uniform([1, 3], minval=-10, maxval=10, dtype=tf.float32)
    scenario = gen(inputs)
    print(scenario)
    inputs = tf.random.uniform([1, 3], minval=-10, maxval=10, dtype=tf.float32)
    scenario = gen(inputs)
    print(scenario)


# compare the risk value calculated by simu-net and simulator for testing
def cal_risk():
    # scenario = tf.constant([10, 20, 27, -1, 2], shape=[1, 5], dtype=float)
    # scenario = tf.constant([10, 25, 22, 0, 2], shape=[1, 5], dtype=float)
    # scenario = tf.constant([10, 20, 20, 3, 2], shape=[1, 5], dtype=float)
    scenario = tf.constant([15, 21, 28, -3, 4], shape=[1, 5], dtype=float)

    sim = Simulator(scenario)
    ttc = sim.simulate()
    print("ttc in simulator:")
    print(ttc)

    simu_net = tf.keras.models.load_model('./simu_net.h5', custom_objects={'prediction_mse': SimuNet.prediction_mse,
                                                                           'prediction_msegap': SimuNet.prediction_msegap,
                                                                           'loss_msegap': SimuNet.loss_msegap})
    Y = simu_net(scenario)
    print("ttc in simu-net:")
    print(Y)


if __name__ == '__main__':
    # gen = Generator()
    # simu_net = SimuNet()
    # combined = build_GS_model(gen, simu_net)
    # train(10000, 10, gen, simu_net, combined)

    # simu_net = SimuNet()
    # simu_net.net.compile(loss=simu_net.loss, optimizer=simu_net.trainer,
    #                      metrics=[simu_net.prediction_mse, simu_net.prediction_msegap])
    # trainSN(100000, 100, simu_net)

    cal_risk()
    # generate_scenario()
    # random_generate(5)
