## pytorch-deep-bayesian-bandits
PyTorch port and extension of the [Deep Bayesian Bandits Library](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits) (Work in Progress)

### Motivation
Recent advances in deep unsupervised learning allow for learning concise yet rich representations of images, audio, natural language, and more. Integrating these representations into sequential decision-making paradigms such as reinforcement learning is an essential step to creating general-purpose agents that can robustly incorporate diverse unstructured sources of data. We consider the contextual bandit setting as a tractable and real-world applicable version of reinforcement learning.

### What it does

We base our work off of recent work from Google Brain: [Deep Bayesian Bandits Showdown](https://arxiv.org/pdf/1802.09127.pdf). This paper (and accompanying [TensorFlow code](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits)) implements a simple MLP-based method for learning contexts from hand-crafted features via contextual bandit feedback. 

Our contribution: we extended this work to include a novel unsupervised representation learning step. Specifically, we pre-train an unsupervised model, and use the learned embedding as an input to the context encoding MLP. We re-implemented contextual bandit algorithms with deep Thompson sampling in PyTorch, and test our algorithm on several tasks, including the Mushroom dataset, MNIST, and polarized Yelp reviews.

This project started from the PyTorch Summer Hackathon. Check out our [DevPost Submission](https://devpost.com/software/unsupervised-representation-learning-for-contextual-bandits).

### Run

To run MNIST:

`python run_experiment_multithreaded.py`
