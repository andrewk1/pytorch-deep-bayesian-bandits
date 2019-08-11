import requests
import pandas as pd
import numpy as np

from bandits.data.mnist import get_raw_features, get_vae_features, construct_dataset_from_features
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.core.hyperparams import HyperParams
from bandits.core.contextual_bandit import run_contextual_bandit
import torch
import sys


class textVecData():

    def __init__(self):
        """
        Initialize model parameters.
        Apply for two embedding layers.
        Initialize layer weight
        Args:
            emb_size: Embedding size.
            emb_dimention: Embedding dimention, typically from 50 to 500.
        Returns:
            None
        """

        def download_file_from_google_drive(id, destination):
            URL = "https://docs.google.com/uc?export=download"

            session = requests.Session()

            response = session.get(URL, params={'id': id}, stream=True)
            token = get_confirm_token(response)

            if token:
                params = {'id': id, 'confirm': token}
                response = session.get(URL, params=params, stream=True)

            save_response_content(response, destination)

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        download_file_from_google_drive('14GNjfpVWeSDMEtqZHnXNeyY0hQIqeFO6',
                                        'data1.parquet')
        download_file_from_google_drive('1l59vzX-D0UeC-BgHsachau26sgBAT77O',
                                        'data2.parquet')

    def getLabelFeature(self):
        df_1 = pd.read_parquet('data1.parquet')
        df_2 = pd.read_parquet('data2.parquet')
        finaldf = pd.concat([df_1, df_2])
        finaldf = finaldf[['averagePooledValues', 'label']].copy()
        returnData = [tuple(r) for r in finaldf.values]
        return returnData


def main():
    nlp = textVecData()
    nlp_data = nlp.getLabelFeature()
    features, rewards, opt_vals = construct_dataset_from_features(nlp_data)
    dataset = np.hstack((features, rewards))

    context_dim = features.shape[1]
    num_actions = 2
    print(context_dim)

    init_lrs = [0.001, 0.0025, 0.005, 0.01]
    base_lrs = [0.0005, 0.001]
    modes = ["triangular", "triangular2", "exp_range"]
    mode = ['triangular']
    # batch_sizes = [32, 128, 512]
    # layer_sizes = [[50, 50], [100, 100], [100]]
    # # hyperparams
    hp_nlinear = HyperParams(num_actions=num_actions,
                             context_dim=context_dim,
                             init_scale=0.3,
                             layer_sizes=[50, 50],
                             batch_size=32,
                             activate_decay=True,
                             # initial_lr=0.1,
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
                             keep_prob=1.0,
                             global_step=1,
                             initial_lr=init_lrs[0],
                             base_lr=base_lrs[0],
                             mode=modes[0])

    algos = [NeuralLinearPosteriorSampling('NeuralLinear', hp_nlinear)]

    # run contextual bandit experiment
    print(context_dim, num_actions)
    results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
    actions, rewards = results
    np.save("results.npy", rewards)

if __name__ == '__main__':
    main()
