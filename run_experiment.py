from bandits.data.mnist import get_raw_features, get_vae_features, construct_dataset_from_features
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.core.hyperparams import HyperParams
from bandits.core.contextual_bandit import run_contextual_bandit
import numpy as np
import torch


def main():
    data_type = 'mnist'

    vae_data = get_vae_features()
    features, rewards, opt_vals = construct_dataset_from_features(vae_data)
    dataset = np.hstack((features, rewards))

    context_dim = features.shape[1]
    num_actions = 10

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

    # run contextual bandit experiment
    print(context_dim, num_actions)
    results = run_contextual_bandit(context_dim, num_actions, dataset, algos)


if __name__ == '__main__':
    main()
