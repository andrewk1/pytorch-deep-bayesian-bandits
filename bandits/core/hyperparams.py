class HyperParams(object):

    def __init__(self, num_actions, context_dim, init_scale, layer_sizes,
                 batch_size, activate_decay, initial_lr, max_grad_norm,
                 show_training, freq_summary, buffer_s, initial_pulls, reset_lr,
                 lr_decay_rate, training_freq, training_freq_network, training_epochs,
                 a0, b0, lambda_prior, keep_prob, global_step):
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.init_scale = init_scale
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.activate_decay = activate_decay
        self.initial_lr = initial_lr
        self.max_grad_norm = max_grad_norm
        self.show_training = show_training
        self.freq_summary = freq_summary
        self.buffer_s = buffer_s
        self.initial_pulls = initial_pulls
        self.reset_lr = reset_lr
        self.lr_decay_rate = lr_decay_rate
        self.training_freq = training_freq
        self.training_freq_network = training_freq_network
        self.training_epochs = training_epochs
        self.a0 = a0
        self.b0 = b0
        self.lambda_prior = lambda_prior
        self.keep_prob = keep_prob
        self.global_step = global_step
