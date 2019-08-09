from bandits.data.mnist import get_raw_features, get_vae_features, construct_dataset_from_features
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.core.hyperparams import HyperParams
from bandits.core.contextual_bandit import run_contextual_bandit
from multiprocessing.pool import Pool
import numpy as np
import torch
import sys

np.set_printoptions(threshold=sys.maxsize)

def main():
    global num_processes, mode, dataset, features, rewards, context_dim, num_actions, combos
    data_type = 'mnist'
    num_processes = 4

    vae_data = get_vae_features()
    features, rewards, opt_vals = construct_dataset_from_features(vae_data)
    dataset = np.hstack((features, rewards))[:100]

    context_dim = features.shape[1]
    num_actions = 10

    mode = "triangular"

    init_lrs = [0.001, 0.0025, 0.005, 0.01]
    base_lrs = [0.0005, 0.001]
    batch_sizes = [32, 128, 512]
    layer_sizes = [[50, 50], [100, 50], [100]]
    training_freqs = [50, 100]
    idx = 0
    combos = []
    for init_lr in init_lrs:
        for base_lr in base_lrs:
            for batch_size in batch_sizes:
                for lz in layer_sizes:
                    for tf in training_freqs:
                        params = {"init_lr": init_lr,
                                  "base_lr": base_lr,
                                  "batch_size": batch_size,
                                  "layer_size": lz,
                                  "training_freq": tf}
                        combos.append(params)

    p = Pool(processes=num_processes)
    p.map(run_trial, range(num_processes))
    p.close()
    p.join()

def run_trial(process):
    for idx, combo in enumerate(combos):
        if idx % num_processes == process:
            print('running combo %d: %s', idx, combo)
            # hyperparams
            hp_nlinear = HyperParams(num_actions=num_actions,
                                     context_dim=context_dim,
                                     init_scale=0.3,
                                     layer_sizes=combo["layer_size"],
                                     batch_size=combo["batch_size"],
                                     activate_decay=True,
                                     initial_lr=combo["init_lr"],
                                     base_lr=combo["base_lr"],
                                     max_grad_norm=5.0,
                                     show_training=False,
                                     freq_summary=1000,
                                     buffer_s=-1,
                                     initial_pulls=2,
                                     reset_lr=True,
                                     lr_decay_rate=0.5,
                                     training_freq=1,
                                     training_freq_network=combo["training_freq"],
                                     training_epochs=100,
                                     a0=6,
                                     b0=6,
                                     lambda_prior=0.25,
                                     keep_prob=1.0,
                                     global_step=1,
                                     mode=mode)
            algos = [NeuralLinearPosteriorSampling('NeuralLinear', hp_nlinear)]

            # run contextual bandit experiment
            print(context_dim, num_actions)
            results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
            actions, rewards = results
            np.save(mode + "results" + str(idx) + ".npy", rewards)


if __name__ == '__main__':
    main()
