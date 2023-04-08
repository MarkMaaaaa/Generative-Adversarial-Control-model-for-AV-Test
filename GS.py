""" Generator-SimuNet network for generating testing scenarios for AV test
    """

import os
import tensorflow as tf
import numpy as np
from d2l import tensorflow as d2l

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
        self.input_size = 10  # size for input random variable

    class FinalLayer(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.weight = tf.constant([10, 10, 10, 6, 5], shape=[1, 5], dtype=float)
            self.bias = tf.constant([10, 20, 20, -3, 0], shape=[1, 5], dtype=float)
            # 10 <= x <= 20 (m)
            # 20 <= v0 <= 30 (m/s)
            # 20 <= v1 <= 30 (m/s)
            # -3 <= a0 <= 3 (m/s^2)
            # 0 <= t <= 5 (s)

        def call(self, inputs):
            inputs = inputs * self.weight + self.bias
            return tf.round(inputs * 100) / 100

    def build_net(self):
        net = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(5, activation='sigmoid'),
            self.FinalLayer()  # [x, v0, v1, a0, t]
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
        self.T = 10  # simulation time horizen (s)
        self.Tstep = 0.1  # time step (s)
        # other parameters
        self.car_len = 3  # car length for crash_test (m)
        # input parameters
        self.h_ini = int(inputs[0, 0])  # initial headway (m)
        self.v0 = int(inputs[0, 1])
        self.v1 = int(inputs[0, 2])
        self.a0 = int(inputs[0, 3])
        self.t = int(inputs[0, 4])
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
            ttc = del_d / del_v
        # TTC collision test
        if (0 <= ttc <= 1):
            print("crash at time step: %.2f" % (step / 10))
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

    def build_net(self):
        time_steps = 100  # how many time steps in simulation
        input_size = 1  # how many inputs in each time step
        units = 3 # the dimonsion of each state
        output_size = time_steps

        inputs = tf.keras.layers.Input((time_steps, input_size))
        # initial state as Keras Input
        initial_state = tf.keras.layers.Input((units,))
        rnn = tf.keras.layers.SimpleRNN(units)
        hidden = rnn(inputs, initial_state=initial_state)
        output = tf.keras.layers.Dense(output_size, activation='relu')(hidden)
        model = tf.keras.models.Model([inputs, initial_state], output)

        # print(model.summary())
        return model

    def trans2inputs(self, scenario):
        Xt = tf.reshape(scenario[:, 4:5], (1, 1, -1))  # batch size = 1
        Xt = tf.tile(Xt, [1, 100, 1])  # number of step (equals to simulation T/Tstep)
        begin_state = tf.reshape(scenario[:, :3], (1, -1))
        return [Xt, begin_state]


def train(epochs, gen, simu_net):
    # compile the simu net
    simu_net.net.compile(loss=simu_net.loss, optimizer=simu_net.trainer, metrics=['accuracy'])
    # The generator takes noise as input and generates scenarios
    gen_in = tf.keras.layers.Input(shape=(gen.input_size,))
    gen_out = gen.net(gen_in)
    Xts, begin_states = simu_net.trans2inputs(gen_out)
    # For the combined model we will only train the simu net
    # after the simu_net has been compiled it is still trained during simu_net.train_on_batch but since it's set to non-trainable before the combined model is compiled it's not trained during combined.train_on_batch
    simu_net.net.trainable = False
    # The simu net takes generated scenarios as input and determines risk level for this scenario
    risks = simu_net.net([Xts, begin_states])
    # The combined model  (stacked generator and simu net)
    combined = tf.keras.models.Model(gen_in, risks)
    combined.compile(loss=gen.loss, optimizer=gen.trainer)

    print("build complete!")

    for epoch in range(epochs):
        # generate random inputs
        inputs = tf.random.normal([1, gen.input_size], mean=0.0, stddev=1.0, dtype=tf.float32)
        scenario = gen.net(inputs)
        # train the simu net, make ttc close to real ttc (from simulator)
        sim = Simulator(scenario)
        risk = np.array(sim.simulate()).reshape(1,-1)
        Xt, begin_state = simu_net.trans2inputs(scenario)
        # print(Xt.shape)
        # print(begin_state.shape)
        # print(risk.shape)
        d_loss = simu_net.net.train_on_batch([Xt, begin_state], risk)
        # train the generator, make ttc close to 0
        g_loss = combined.train_on_batch(inputs, np.zeros((1, 1)))
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss, g_loss))


def debug():
    # inputs = tf.random.normal([1, 10], mean=0.0, stddev=1.0, dtype=tf.float32)
    inputs = tf.constant(range(10), shape=[1, 10], dtype=float)
    print("input:")
    print(inputs)
    gen = Generator()
    scenario = gen.net(inputs)
    # scenario = tf.constant([10, 20, 25, -1, 5], shape=[1, 5], dtype=float)
    print("scenario:")
    print(scenario)
    sim = Simulator(scenario)
    ttc = sim.simulate()
    print("min ttc in simulator:")
    print(min(ttc))
    print(ttc.__len__())
    simu_net = SimuNet()
    X, begin_state = simu_net.trans2inputs(scenario)
    print("scenario transfer:")
    print(X.shape)
    print(begin_state.shape)
    Y = simu_net.net([X] + [begin_state])
    print("min ttc in simu-net:")
    print(float(tf.reduce_min(Y)))
    print(Y.shape)


if __name__ == '__main__':
    # debug()
    gen = Generator()
    simu_net = SimuNet()
    train(100, gen, simu_net)
