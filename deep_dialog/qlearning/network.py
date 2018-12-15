import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

def init_weight(m, gain=1):
    for name, param in m.named_parameters():
        if name.find('weight') != -1:
            torch.nn.init.xavier_uniform_(param, gain)
        elif name.find('bias') != -1:
            param.data.fill_(0)

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):
            nn.init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            nn.init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            nn.init.constant(self.sigma_weight, self.sigma_init)
            nn.init.constant(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.cuda(), self.bias + self.sigma_bias * self.epsilon_bias.cuda())

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noisy=False):
        super(Network, self).__init__()
        output_class = NoisyLinear if noisy else nn.Linear
        self.qf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', output_class(hidden_size, output_size))]))
        init_weight(self.qf.w1, 1)
        init_weight(self.qf.w2, 0.1)
        self.noisy = noisy

    def forward(self, inputs, testing=False):
        return self.qf(inputs)

    def sample_noise(self):
        if self.noisy:
            self.qf.w2.sample_noise()
    
    def remove_noise(self):
        if self.noisy:
            self.qf.w2.remove_noise()
        

class DuelNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noisy=False):
        super(DuelNetwork, self).__init__()
        output_class = NoisyLinear if noisy else nn.Linear
        self.adv = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', output_class(hidden_size, output_size))]))
        self.vf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', output_class(hidden_size, 1))]))
        #init_weight(self.adv.w1, 1)
        #init_weight(self.adv.w2, 0.1)
        #init_weight(self.vf.w1, 1)
        #init_weight(self.vf.w2, 0.1)
        self.noisy = noisy

    def forward(self, inputs, testing=False):
        v = self.vf(inputs)
        adv = self.adv(inputs)
        return v.expand(adv.size()) + adv - adv.mean(-1).unsqueeze(1).expand(adv.size())

    def sample_noise(self):
        if self.noisy:
            self.adv.w2.sample_noise()
            self.vf.w2.sample_noise()
    
    def remove_noise(self):
        if self.noisy:
            self.adv.w2.remove_noise()
            self.vf.w2.remove_noise()

class CategoricalDuelNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, atoms=51):
        super(CategoricalDuelNetwork, self).__init__()
        self.atoms = atoms
        self.output_size = output_size
        self.adv = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, output_size * atoms))]))
        self.vf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, atoms))]))

    def forward(self, inputs, testing=False, log_prob=False):
        v = self.vf(inputs).view(-1, 1, self.atoms)
        adv = self.adv(inputs).view(-1, self.output_size, self.atoms)
        q = adv
        #q = v + adv - adv.mean(1, keepdim=True)
        if log_prob:
            return F.log_softmax(q, -1)
        else:
            return F.softmax(q, -1)

class CategoricalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, atoms=51):
        super(CategoricalNetwork, self).__init__()
        self.atoms = atoms
        self.output_size = output_size
        self.qf = nn.Sequential(OrderedDict([
            ('w1', nn.Linear(input_size, hidden_size)), 
            ('relu', nn.ReLU()), 
            ('w2', nn.Linear(hidden_size, output_size * atoms))]))

    def forward(self, inputs, testing=False, log_prob=False):
        v = self.qf(inputs).view(-1, 1, self.atoms)
        q = self.qf(inputs).view(-1, self.output_size, self.atoms)
        if log_prob:
            return F.log_softmax(q, -1)
        else:
            return F.softmax(q, -1)

