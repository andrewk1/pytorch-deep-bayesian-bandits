"""Functions to create bandit problems from datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd


def one_hot(df, cols):
    """Returns one-hot encoding of DataFrame df including columns in cols."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df


def sample_mushroom_data(file_name,
                         num_contexts,
                         r_noeat=0,
                         r_eat_safe=5,
                         r_eat_poison_bad=-35,
                         r_eat_poison_good=5,
                         prob_poison_bad=0.5):

    # first two cols of df encode whether mushroom is edible or poisonous
    df = pd.read_csv(file_name, header=None)
    df = one_hot(df, df.columns)
    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True)

    contexts = df.iloc[ind, 2:]
    no_eat_reward = r_noeat * np.ones((num_contexts, 1))
    random_poison = np.random.choice(
        [r_eat_poison_bad, r_eat_poison_good],
        p=[prob_poison_bad, 1 - prob_poison_bad],
        size=num_contexts)
    eat_reward = r_eat_safe * df.iloc[ind, 0]
    eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
    eat_reward = eat_reward.values.reshape((num_contexts, 1))

    # compute optimal expected reward and optimal actions
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
    exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad)
    opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]

    if r_noeat > exp_eat_poison_reward:
        # actions: no eat = 0 ; eat = 1
        opt_actions = df.iloc[ind, 0]  # indicator of edible
    else:
        # should always eat (higher expected reward)
        opt_actions = np.ones((num_contexts, 1))
    opt_vals = (opt_exp_reward.values, opt_actions.values)

    return np.hstack((contexts, no_eat_reward, eat_reward)), opt_vals
