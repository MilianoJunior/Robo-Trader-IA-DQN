# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:28:56 2020

@author: jrmfi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np


from tf_agents.trajectories import time_step as ts
from comunica import  Comunica
tf.compat.v1.enable_v2_behavior()
media = np.array([ 1.33793200e+03, -1.88003760e-02,  2.28691574e+01,  2.12422248e+01,
        5.03584330e+01,  2.91188164e+00,  5.82291246e+00, -2.91102462e+00,
        3.01221810e+01,  8.34974547e+01,  1.00003686e+02, -1.06897873e+03,
        6.47077044e+03,  2.37574951e-01, -3.97979894e+01,  4.25147569e+01,
        5.03948285e+01,  5.03950239e+01, -4.95499296e+01,  9.00340577e+01,
        4.96373217e+01,  1.03066813e+05,  1.03246881e+05,  1.02886744e+05])
std = np.array([2.61870414e+02, 5.33931614e+01, 2.05532357e+01, 1.99331333e+01,
       1.23527957e+01, 8.51901293e+01, 1.46821124e+02, 6.53311300e+01,
       1.18854135e+01, 3.04180846e+01, 2.36491257e-01, 2.00994580e+05,
       4.64809354e+03, 1.07799520e+02, 1.05157367e+02, 1.02117350e+02,
       2.40411806e+01, 2.22333844e+01, 2.83496587e+01, 7.57608266e+01,
       1.77987175e+01, 6.27292393e+03, 6.26327234e+03, 6.28621400e+03])
# batch_size = 3
saved_policy = tf.saved_model.load('policy3')
# policy_state = saved_policy.get_initial_state(batch_size=batch_size)

HOST = ''    # Host
PORT = 8888  # Porta
R = Comunica(HOST,PORT)
s = R.createServer()

while True:
    p,addr = R.runServer(s)
    jm = np.array((p-media)/std)
    jm = np.array(jm, dtype=np.float32)
    observations = tf.constant([[jm]])
    # print(observations)
    time_step = ts.restart(observations,1)
    # print(time_step)
    action = saved_policy.action(time_step)
    previsao2 = action.action.numpy()[0]
    d3 = p[0]
    print('recebido: ',p[0])
    print('previsao: ',previsao2)
    if previsao2 == 0:
        print('Sem operacao')
    if previsao2 == 1:
        flag = "compra-{}".format(d3)
        # flag ="compra"
        print('compra: ',previsao2)
        R.enviaDados(flag,s,addr)
    if previsao2 == 2:
        flag = "venda-{}".format(d3)
        # flag = "venda"
        print('venda: ',previsao2)
        R.enviaDados(flag,s,addr)


