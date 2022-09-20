import numpy as np
import math
import torch 
import torch.nn as nn

class MIEstimator(nn.Module): # Update 04/27/2022
    '''
    This class provide a CLUB estimator to calculate MI upper bound between vector-like embeddings and categorical labels.
    Estimate I(X,Y), where X is continuous vector and Y is discrete label.
    '''
    def __init__(self, input_dim, label_num, hidden_size=None):
        '''
        input_dim : the dimension of input embeddings
        label_num : the number of categorical labels 
        '''
        super(MIEstimator, self).__init__()
        
        if hidden_size is None:
            self.variational_net = nn.Linear(input_dim, label_num)
        else:
            self.variational_net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, label_num)
            )
            
    def forward(self, inputs, labels):
        '''
        inputs : shape [batch_size, input_dim], a batch of embeddings
        labels : shape [batch_size], a batch of label index
        '''
        logits = self.variational_net(inputs)  #[sample_size, label_num]
        
        # log of conditional probability of positive sample pairs
        #positive = - nn.functional.cross_entropy(logits, labels, reduction='none')    
        sample_size, label_num = logits.shape
        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)  # shape [sample_size, sample_size, label_num]
        labels_extend = labels.unsqueeze(0).repeat(sample_size, 1)     # shape [sample_size, sample_size]

        # log of conditional probability of negative sample pairs
        log_mat = - nn.functional.cross_entropy(
            logits_extend.reshape(-1, label_num),
            labels_extend.reshape(-1, ),
            reduction='none'
        )
        
        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat).mean()
        negative = log_mat.mean()
        return positive - negative

    def loglikeli(self, inputs, labels):
        logits = self.variational_net(inputs)
        return - nn.functional.cross_entropy(logits, labels)
    
    def learning_loss(self, inputs, labels):
        return - self.loglikeli(inputs, labels)
