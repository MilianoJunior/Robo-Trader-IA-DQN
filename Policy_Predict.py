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
import tensorflow_probability as tfp
import numpy as np

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.networks import network

from tf_agents.policies import py_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import scripted_py_policy

from tf_agents.policies import tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import actor_policy
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy

from tf_agents.trajectories import time_step as ts
# import pandas as pd
# import chardet
# import tensorflow as tf
import numpy as np
tf.compat.v1.enable_v2_behavior()


c1 = [919.0,60,55,0,69.58,115.43,179.51,-64.08,31.38,94.29,100.25,1185.0]

c2 = [-1.5621375506983564,1.0170146957413337,1.151379423051654,
 -0.8240188550530413, 1.9436058065119814, 1.15459996673134,
 1.1976462984199698, -1.248256181130577, 0.226830428685124,
 -0.050548391458088315, 0.5699084640448159, 0.2824263833205345]

c3 = [920.0 ,0 ,25 ,25 ,69.22, 117.95, 184.43, -66.48, 34.76, 88.93, 100.08, 1085.0]

c4 = [-1.5582745566819385, 0.04436943504060721, -0.03576076619457791,
 0.17340377053792586, 1.9134733684408487, 1.172374662245547,
 1.2202930019443057, -1.2793022587790714, 0.5362163052449721,
 -0.1499669030651435 ,0.33079696039159334, 0.27592727727991867]

c5 =[921.0, 30 ,15 ,0 ,70.05, 120.76, 189.74, -68.98, 38.44, 77.5, 100.23, 576.92]

c6 =[-1.5544115626655208, 0.5306920653909704 ,-0.43147416260998855,
 -0.8240188550530413, 1.9829453784381823, 1.192194858434008,
 1.2447348709919124, -1.3116419229962533, 0.8730624667065819,
 -0.3619731694212333, 0.5417776989091485, 0.24290661930875776]


batch_size = 3
saved_policy = tf.saved_model.load('policy')
policy_state = saved_policy.get_initial_state(batch_size=batch_size)

observations = tf.constant([[c2]])
print(observations)
time_step = ts.restart(observations,1)
print(time_step)
action2 = saved_policy.action(time_step)
print('recompensa: ',time_step.reward.numpy()[0],' action: ',action2.action.numpy()[0])
