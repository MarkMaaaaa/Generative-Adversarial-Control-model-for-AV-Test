# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:04:30 2023

@author: Ke
"""

#pip install tensorflow
#pip install keras
#pip install matplotlib

import keras
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,100,30)
y=3*x+7+np.random.randn(30)

model = keras.Sequential()
from keras import layers
model.add(layers.Dense(1,input_dim=1))#Dense y=ax+b
model.summary()


model.compile(optimizer='adam',loss='mse')#mse


model.fit(x,y,epochs=30000)

z=model.predict([150])
print(z)
plt.scatter(x,y,c="r")
plt.plot(x,model.predict(x))
plt.show()
