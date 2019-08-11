from mnist import get_vae_features
import numpy as np

features = get_vae_features()

np.save('vae_features', np.array(features))
