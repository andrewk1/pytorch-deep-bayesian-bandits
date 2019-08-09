from bandits.data.mnist import get_raw_features, get_vae_features, construct_dataset_from_features



def main():
    data_type = 'mnist'

    vae_data = get_vae_features()
    features, rewards, opt_vals = construct_dataset_from_features(vae_data)
    dataset = np.hstack((features, rewards))

    context_dim = vae_features.shape[1]
    num_actions = 10
    # hyperparams

    # TODO
    algos = []

    # run contextual bandit experiment
    results = run_contextual_bandit(context_dim, num_actions, dataset, algos)


if __name__ == '__main__':
    main()
