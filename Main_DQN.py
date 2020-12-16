# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:40:04 2020

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


from CardTrader import CardGameEnv

" 1 passo: importar os dados"
try:
    with open('M3.csv', 'rb') as f:
        result = chardet.detect(f.read())  # or readline if the file is large
    
    base = pd.read_csv('M3.csv', encoding=result['encoding'])
    
except:
    print('Erro, é preciso fazer o download dos dados OHLC em csv')

#configuracoes iniciais - Hiperparametros
tf.compat.v1.enable_v2_behavior()
env_name = "CartPole-v1" # @param {type:"string"}
num_iterations = 150# @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

fc_layer_params = (100,)

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = 200  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = -200  # @param {type:"integer"}
max_q_value = 200  # @param {type:"integer"}
n_step_update = 20  # @param {type:"integer"}

num_eval_episodes = 100  # @param {type:"integer"}
eval_interval = 100  # @param {type:"integer"}

#Meio ambiente
train_py_env = CardGameEnv(base,num_iterations)
eval_py_env = CardGameEnv(base,num_iterations)
# environment = tf_py_environment.TFPyEnvironment(env_train)
# train_py_env = suite_gym.load(env_name)
# eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# train_env = tf_py_environment.TFPyEnvironment(env_train)
# eval_env = tf_py_environment.TFPyEnvironment(env_eval)
# #Agente

categorical_q_net = categorical_q_network.CategoricalQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    num_atoms=num_atoms,
    fc_layer_params=fc_layer_params)



optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network=categorical_q_net,
    optimizer=optimizer,
    min_q_value=min_q_value,
    max_q_value=max_q_value,
    n_step_update=n_step_update,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=train_step_counter)
agent.initialize()

    
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):
    time_step = environment.reset()

    episode_return = 0.0

    while not time_step.is_last():
      # print('-----------------------')
      # print('formato',time_step)
      # print('-----------------------')
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  # print('       ')
  # print(' media de retorno : ',avg_return)
  # print('       ')
  return avg_return.numpy()[0]


random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

compute_avg_return(eval_env, random_policy, num_eval_episodes)

# Please also see the metrics module for standard implementations of different
# metrics.

# coleta de dados

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

def collect_step(environment, policy):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)

for _ in range(initial_collect_steps):
  collect_step(train_env, random_policy)

# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1.
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)



# Treinando o agente

try:
  %%time
except:
  pass

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience)

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
    returns.append(avg_return)
    
# #Vizualizacao dos erros

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
# plt.ylim(top=550)

# my_policy = agent.collect_policy
# saver = policy_saver.PolicySaver(my_policy, batch_size=None)
# saver.save('policy')

# saved_policy = tf.compat.v2.saved_model.load('policy')



# time_step = clsTimeStep(step_type=<tf.Tensor: shape=(1,), dtype=int32, numpy=array([1])>, reward=<tf.Tensor: shape=(1,), dtype=float32, numpy=array([205.], dtype=float32)>, discount=<tf.Tensor: shape=(1,), dtype=float32, numpy=array([1.], dtype=float32)>, observation=<tf.Tensor: shape=(1, 1, 12), dtype=float32, numpy=array([[[-1.5582745 ,  0.04436944, -0.03576077,  0.17340377,1.9134734 ,  1.1723746 ,  1.220293  , -1.2793022 ,0.5362163 , -0.14996691,  0.33079696,  0.27592728]]],dtype=float32)>)
# observation=<tf.Tensor: shape=(1, 1, 12), dtype=float32, 
