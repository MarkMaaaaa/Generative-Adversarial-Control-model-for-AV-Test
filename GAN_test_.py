# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 2023
@author: Ke, Hang
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models


class GAN():
    def __init__(self):
        self.road_dim = 2
        self.traj_dim = 10
        self.task_dim = 10
        self.scen_dim = 100

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer='adam', loss='mse')

        self.generator = self.build_generator()

        gan_input = layers.Input(shape=(self.road_dim, self.traj_dim, self.task_dim))
        scenario = self.generator(gan_input)
        validity = self.discriminator(scenario)
        self.combined = models.Model(gan_input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer='adam')

    def build_generator(self):
        model = models.Sequential()
        model.add(layers.Dense(100, input_dim=self.road_dim + self.traj_dim + self.task_dim))
        # todo
        road = layers.Input(shape=(self.road_dim,))
        trajectory = layers.Input(shape=(self.traj_dim,))
        AV_task = layers.Input(shape=(self.task_dim,))
        scenario = model([road, trajectory, AV_task]) # todo 处理多输入

        return models.Model([road, trajectory, AV_task], scenario)

    def build_discriminator(self):
        validity = 1
        # todo
        scenario = layers.Input(shape=self.scen_dim)
        return models.Model(scenario, validity)

    def train(self, epochs):
        for epoch in range(epochs):
            speed, lane_num = get_road_data()
        return 1

    def get_road_data(self):
        speed = random.randint(70, 80)
        lane_num = random.randint(1, 4)
        return speed, lane_num
