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
try:
  # Specify an invalid GPU device
  with tf.device('/device:GPU:0'):
    " 1 passo: importar os dados"
    try:
        with open('M3.csv', 'rb') as f:
            result = chardet.detect(f.read())  # or readline if the file is large
        
        base = pd.read_csv('M3.csv', encoding=result['encoding'])
        
    except:
        print('Erro, é preciso fazer o download dos dados OHLC em csv')
    
    #configuracoes iniciais - Hiperparametros
    tf.compat.v1.enable_v2_behavior()
    
    num_iterations = 25000# @param {type:"integer"}
    
    # variavel que faz coleta de dados
    initial_collect_steps = 50000  # @param {type:"integer"} 
    # faz interações aleatorias para explorar
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    # memoria de dados
    replay_buffer_capacity = 100000  # @param {type:"integer"}
    
    fc_layer_params = (256,256)
    
    batch_size = 256  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    gamma = 0.99
    log_interval = 200  # @param {type:"integer"}
    
    num_atoms = 51  # @param {type:"integer"}
    min_q_value = -1000  # @param {type:"integer"}
    max_q_value = 1000  # @param {type:"integer"}
    n_step_update = 2  # @param {type:"integer"}
    
    num_eval_episodes = 1  # @param {type:"integer"}
    eval_interval = 100  # @param {type:"integer"}
    
    #Meio ambiente
    train_py_env = CardGameEnv(base,num_iterations)
    eval_py_env = CardGameEnv(base,num_iterations)
    
    
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    
    
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
        # print(time_step)
        episode_return = 0.0
    
        while not time_step.is_last():
          action_step = policy.action(time_step)
          time_step = environment.step(action_step.action)
          episode_return += time_step.reward
        total_return += episode_return
    
      avg_return = total_return / num_episodes
    
      return avg_return.numpy()[0]
    
    
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())
    
    teste = compute_avg_return(eval_env, random_policy, num_eval_episodes)
    # teste = compute_avg_return(eval_env, random_policy, 1)
    print(teste)
    
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
    
    #armazenar dados no buffer
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
    print(avg_return)
    # quantidade de vezes que havera treinameto
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
    
    my_policy = agent.collect_policy
    saver = policy_saver.PolicySaver(my_policy, batch_size=None)
    saver.save('policy3')
    
    saved_policy = tf.compat.v2.saved_model.load('policy3')
    colunas = ['Hora','dif', 'retacao +','retracao -', 'RSI',
                 'M22M44', 'M22M66', 'M66M44', 'ADX', 'ATR',
                'Momentum', 'Force']
    
    colunas1 = ['Hora', 'open', 'high', 'low', 'close']
    dados1 = pd.DataFrame(data=base[-150000:-100000].values,columns=base.columns)      
    dados2 = pd.DataFrame(data=base[-150000:-100000].values,columns=base.columns)
    dados1 = dados1[colunas1]
    dados2 = dados2[colunas]
    index = 0
    for i in dados2.values:
        base1 = i[0].split(':')
        dados2.at[index, 'Hora'] = float(base1[0])*100 + float(base1[1])
        index += 1
    train_mean = dados2.mean(axis=0)
    train_std = dados2.std(axis=0)
    dados2 = (dados2 - train_mean) / train_std
    
    
    from Trade import Trade
    from tf_agents.trajectories import time_step as ts
    
    trader = Trade()
    
    import random
    stop = -500
    gain = 500
    trader.reset()
    action = 0
    for i in range(len(dados1)):
        
        compra,venda,neg,ficha,comprado,vendido,recompensa= trader.agente(dados1.values[i],action,stop,gain,0)
        # print('estado: ',dados2.values[i])
        observations = tf.constant([[dados2.values[i]]])
        time_step = ts.restart(observations,1)
        action2 = saved_policy.action(time_step)
        # time_step = ts.transition(observations,1)
        # action2 = agent.policy.action(time_step)
        action = action2.action.numpy()[0]
        
        print(i,'------------------')
        print('acao: ',action)
        print('comprado: ',comprado)
        print('vendido: ',vendido)
        print('recompensa: ',recompensa)
        
        print('recompensa: ',time_step.reward.numpy(),' action: ',action2.action.numpy()[0])
    
    print(sum(neg.ganhofinal))
except RuntimeError as e:
  print(e)
# 2162.0
# variavel que faz coleta de dados
# initial_collect_steps = 10000  # @param {type:"integer"} 