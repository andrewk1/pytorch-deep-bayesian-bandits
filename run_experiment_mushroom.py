# from bandits.data.mnist import get_raw_features, get_vae_features, construct_dataset_from_features
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.core.hyperparams import HyperParams
from bandits.core.contextual_bandit import run_contextual_bandit
from bandits.data.data_sampler import sample_mushroom_data
import numpy as np
import torch
import time


file_name = "bandits/data/mushroom.data"

def display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed after {} seconds.'.format(
    name, time.time() - t_init))
  print('---------------------------------------------------')

  performance_pairs = []
  for j, a in enumerate(algos):
    performance_pairs.append((a.name, np.sum(h_rewards[:, j])))
  performance_pairs = sorted(performance_pairs,
                             key=lambda elt: elt[1],
                             reverse=True)
  for i, (name, reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))

  print('---------------------------------------------------')
  print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
  print('Frequency of optimal actions (action, frequency):')
  print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
  print('---------------------------------------------------')
  print('---------------------------------------------------')

def main():
    data_type = 'mushroom'

    # vae_data = get_vae_features()
    # features, rewards, opt_vals = construct_dataset_from_features(vae_data)
    # dataset = np.hstack((features, rewards))

    num_contexts = 2000
    dataset, opt_mushroom = sample_mushroom_data(file_name, num_contexts)
    opt_rewards, opt_actions = opt_mushroom

    context_dim=117
    num_actions = 2

    # dataset, opt_rewards, opt_actions, num_actions, context_dim

    # hyperparams
    hp_nlinear = HyperParams(num_actions=num_actions,
                             context_dim=context_dim,
                             init_scale=0.3,
                             layer_sizes=[50],
                             batch_size=512,
                             activate_decay=True,
                             initial_lr=0.1,
                             max_grad_norm=5.0,
                             show_training=False,
                             freq_summary=1000,
                             buffer_s=-1,
                             initial_pulls=2,
                             reset_lr=True,
                             lr_decay_rate=0.5,
                             training_freq=1,
                             training_freq_network=50,
                             training_epochs=100,
                             a0=6,
                             b0=6,
                             lambda_prior=0.25,
                             keep_prob=0.5,
                             global_step=1)

    algos = [NeuralLinearPosteriorSampling('NeuralLinear', hp_nlinear)]

    t_init = time.time()

    # run contextual bandit experiment
    print(context_dim, num_actions)
    results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
    _, h_rewards = results

    display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, data_type)


if __name__ == '__main__':
    main()
