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
        self.loss = tf.keras.losses.MeanSquaredError()  # binary crossentropy?
        # self.trainer = tf.keras.optimizers.SGD()
        self.trainer = tf.keras.optimizers.Adam(0.0002, 0.5)
        self.input_size = 2  # size for input random variable

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
            # inputs_1 = tf.round(inputs[:, :4] * 100) / 100  TODO: kerastensor can't pass tf.round
            # inputs_2 = tf.round(inputs[:, 4:5] * 10) / 10
            # return tf.concat([inputs_1, inputs_2], 1)
            return inputs

    def build_net(self):
        net = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation='relu', name="D1"),
            tf.keras.layers.Dense(64, activation='relu', name="D2"),
            tf.keras.layers.Dense(5, activation='sigmoid', name="D3"),
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
        self.M = 10  # a big number of ttc
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
        self.t = 2  # float(inputs[0, 4]) TODO default t=2
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
            ttc = 0
        elif del_v <= 0:
            ttc = self.M
        else:
            ttc = round(del_d / del_v, 2)
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
            use a RNN model to simulate the MDP process.
        input: scenarios genearted by Generator
        output: the min TTC in the simulation
    """

    def __init__(self):
        self.net = self.build_net()
        self.loss = tf.keras.losses.MeanSquaredError()
        # self.trainer = tf.keras.optimizers.SGD()
        self.trainer = tf.keras.optimizers.Adam(0.0002, 0.5)

    # using functional API to build RNN network
    def build_net(self):
        time_steps = 40  # how many time steps in simulation
        input_size = 1  # how many inputs in each time step
        units = 3  # the dimonsion of each state

        inputs = tf.keras.layers.Input(shape=(time_steps, input_size), batch_size=1)
        initial_state = tf.keras.layers.Input(shape=(units,), batch_size=1)
        rnn = tf.keras.layers.SimpleRNN(units, return_sequences=True, return_state=False)
        hidden = rnn(inputs, initial_state=initial_state)
        hidden = tf.reshape(hidden, (-1, hidden.shape[-1]))  # hidden is (40,3)
        output = tf.keras.layers.Dense(1, activation='relu')(hidden)  # output (40,1), where 1 comes from 3
        output = tf.reshape(output, (1, -1, 1))  # change to (1,40,3)
        model = tf.keras.models.Model([inputs, initial_state], output)

        # model.summary()
        return model

    # test the accuracy of simu net
    def prediction_gap(sellf, y_true, y_pred):
        reduce = tf.abs(y_true - y_pred)
        gap = reduce / y_true
        gap = tf.reduce_mean(gap)
        return gap

    # transfer output in generator to input and initial state
    def trans2inputs(self, scenario):
        # tmp = np.zeros((scenario.shape[0], 100, 1))
        # for i in range(scenario.shape[0]):
        #     for j in range(int(float(scenario[i, 4]) * 10)):
        #         tmp[i, j] = float(scenario[i, 3])
        # Xt = tf.convert_to_tensor(tmp)

        Xt = tf.reshape(scenario[:, 3:4], (-1, 1, 1))
        # Xt = tf.tile(Xt, [1, int(float(scenario[0, 4]) * 10), 1])  # TODO: keras tensor can't get specific value
        Xt = tf.tile(Xt, [1, 20, 1])  # default t=2
        Xt = tf.concat([Xt, tf.zeros((1, 20, 1))], 1)
        begin_state = tf.reshape(scenario[:, :3], (-1, 3))
        return [Xt, begin_state]


def train(epochs, sample_interval, gen, simu_net, combined):
    for epoch in range(epochs):
        # generate random inputs
        inputs = tf.random.normal([1, gen.input_size], mean=0.0, stddev=1.0, dtype=tf.float32)
        scenario = gen.net(inputs)
        # train the simu net, make ttc close to real ttc (from simulator)
        sim = Simulator(scenario)
        risk = np.array(sim.simulate()).reshape(1, -1)
        Xt, begin_state = simu_net.trans2inputs(scenario)
        d_loss = simu_net.net.train_on_batch([Xt, begin_state], risk)
        # train the generator, make ttc close to 0
        g_loss = combined.train_on_batch(inputs, np.zeros((1, 1)))

        if epoch % sample_interval == 0:
            min_ttc = risk.min()
            print("%d [D loss: %f, gap: %.2f%%] [G loss: %f, risk: %.2f]" % (
            epoch, d_loss[0], 100 * d_loss[1], g_loss, min_ttc))
            with open('train_log.txt', 'a') as f:
                f.write("%d [D loss: %f, gap: %.2f%%] [G loss: %f, risk: %.2f]\n" % (
                epoch, d_loss[0], 100 * d_loss[1], g_loss, min_ttc))
            simu_net.net.save('./simu_net.h5')
            simu_net.net.save('./simu_net_weight.h5')
            combined.save('./combined_model.h5')
            combined.save('./combined_model_weight.h5')

    print("train complete!")


def build_GS_model(gen, simu_net):
    # compile the simu net
    simu_net.net.compile(loss=simu_net.loss, optimizer=simu_net.trainer, metrics=[simu_net.prediction_gap])
    # The generator takes noise as input and generates scenarios
    gen_in = tf.keras.layers.Input(shape=(gen.input_size,), batch_size=1)
    gen_out = gen.net(gen_in)
    Xt, begin_state = simu_net.trans2inputs(gen_out)
    # For the combined model we will only train the simu net
    simu_net.net.trainable = False
    # The simu net takes generated scenarios as input and determines risk level for this scenario
    risk = tf.reduce_min(simu_net.net([Xt, begin_state]), axis=1)
    # The combined model
    combined = tf.keras.models.Model(gen_in, risk)
    combined.compile(loss=gen.loss, optimizer=gen.trainer)

    # combined.summary()
    print("build complete!")

    return combined


def generate_scenario():
    combined = tf.keras.models.load_model('./combined_model.h5', custom_objects={'MyLayer': Generator.MyLayer})
    gen = tf.keras.Model(inputs=combined.layers[1].input, outputs=combined.layers[1].output)
    inputs = tf.random.normal([1, 2], mean=0.0, stddev=1.0, dtype=tf.float32)
    scenario = gen(inputs)
    print(scenario)


def cal_risk():
    scenario = tf.constant([10, 23, 27, -1, 2], shape=[1, 5], dtype=float)

    sim = Simulator(scenario)
    ttc = sim.simulate()
    print("ttc in simulator:")
    print(ttc)

    simu_net = tf.keras.models.load_model('./simu_net.h5', custom_objects={'prediction_gap': SimuNet.prediction_gap})
    X, begin_state = SimuNet.trans2inputs(None, scenario)
    Y = simu_net([X, begin_state])
    print("ttc in simu-net:")
    print(Y)

    gap = SimuNet.prediction_gap(None, ttc, Y)
    print(gap)


if __name__ == '__main__':
    # gen = Generator()
    # simu_net = SimuNet()
    # combined = build_GS_model(gen, simu_net)
    # train(100000, 100, gen, simu_net, combined)

    # cal_risk()
    generate_scenario()
