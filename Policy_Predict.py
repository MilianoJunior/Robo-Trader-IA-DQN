# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:28:56 2020

@author: jrmfi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
# import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
# import PIL.Image

import pandas as pd
import chardet
import tensorflow as tf

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import policy_saver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common