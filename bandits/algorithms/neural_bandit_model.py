"""Implement basic MLP model per hyperparams"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch import optim
from torch.nn.modules.normalization import LayerNorm
torch.set_printoptions(profile='full')

class NeuralBanditModel():
    """Implements a neural network for bandit problems."""
    def __init__(self, optimizer, hparams, name):
        self.name = name
        self.hparams = hparams
        self.times_trained = 0
        self.build_model()

    def build_layer(self, input_dim, output_dim, last=False):
        """Builds a fc layer with num_inputs and num_outputs"""
        layer = [nn.Linear(input_dim, output_dim)]
        torch.nn.init.uniform_(layer[0].weight, -0.3, 0.3)
        if not last: layer += [nn.ReLU()]
        layer += [LayerNorm(output_dim)]
        if self.hparams.keep_prob < 1.:
            layer.append(nn.Dropout(p=(1 - self.hparams.keep_prob)))
        return layer

    def build_model(self):
        self.net = []
        input_dim = self.hparams.context_dim

        for output_dim in self.hparams.layer_sizes:
            self.net += self.build_layer(input_dim, output_dim)
            input_dim = output_dim

        self.fc_nn = nn.Sequential(*self.net)

        self.net += self.build_layer(input_dim, self.hparams.num_actions, True)
        self.net = nn.Sequential(*self.net)
        self.lossCriterion = nn.MSELoss(reduction='none')
        self.optimizer = optim.RMSprop(self.net.parameters(),
                                       lr=self.hparams.initial_lr)
        # self.assign_lr()

    def assign_lr(self):
        # TODO: This isn't 1-1 the same
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   self.hparams.global_step)

    def train(self, data, num_steps):
        print("Training {} for {} steps...".format(self.name, num_steps))

        for step in range(num_steps):
            self.optimizer.zero_grad()

            # Get data, perform forward prop
            x, y, w = data.get_batch_with_weights(self.hparams.batch_size)
            x = torch.Tensor(x)
            y = torch.Tensor(y)
            w = torch.Tensor(w)
            y_pred = self.net.forward(x)

            # Compute loss
            loss = self.lossCriterion(y_pred, y)
            loss = loss * w  # This pulls out loss only on true y
            cost = loss.sum() / self.hparams.batch_size
            cost.backward()
            self.optimizer.step()

            print('cost', cost)

        # Anneal LR
        # self.scheduler.step()
        self.times_trained += 1

    def forward(self, x):
        return self.net.forward(torch.Tensor(x)).detach().numpy()

    def forward_z(self, x):
        return self.fc_nn.forward(torch.Tensor(x)).detach().numpy()
