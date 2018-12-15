import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from network import *

use_cuda = torch.cuda.is_available()

class DistributionalDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, duel=True):
        super(DistributionalDQN, self).__init__()
        
        network = CategoricalDuelNetwork if duel else CategoricalNetwork
        self.v_min = -50
        self.v_max = 100
        self.atoms = 51
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms)
        self.delta = (self.v_max - self.v_min) / float(self.atoms - 1)
        if use_cuda:
            self.support = self.support.cuda()
        self.model = network(input_size, hidden_size, output_size, self.atoms)
        self.target_model = network(input_size, hidden_size, output_size, self.atoms)
        self.target_model.load_state_dict(self.model.state_dict())

        # hyper parameters
        self.max_norm = 1e-3
        lr = 1e-3
        self.tau = 1e-2
        self.regc = 1e-3
        self.bacth_count = 0
        self.update_target = 100

        #self.optimizer = optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)

        if use_cuda:
            self.cuda()

    def update_fixed_target_network(self):
        #self.target_model.load_state_dict(self.model.state_dict())
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def Variable(self, x):
        x = x.detach()
        if use_cuda:
            x = x.cuda()
        return x
        #return Variable(x, requires_grad=False).cuda() if use_cuda else Variable(x, requires_grad=False)

    def singleBatch(self, raw_batch, params):
        gamma = params.get('gamma', 0.9)
        batch_size = len(raw_batch)
        batch = [np.vstack(b) for b in zip(*raw_batch)]
        
        # each example in a batch: [s, a, r, s_prime, term]
        s = self.Variable(torch.FloatTensor(batch[0]))
        a = self.Variable(torch.LongTensor(batch[1]))
        r = self.Variable(torch.FloatTensor(batch[2]))
        s_prime = self.Variable(torch.FloatTensor(batch[3]))
        done = self.Variable(torch.FloatTensor(np.array(batch[4]).astype(np.float32)))
        #r = r.clamp(-1, 1)

        with torch.no_grad():
            prob_next = self.target_model(s_prime).detach()
            q_next = (prob_next * self.support).sum(-1)
            a_next = torch.argmax(q_next, -1)
            #prob_next = self.target_model(s_prime).detach()
            prob_next = prob_next[range(batch_size), a_next, :]
            
            atom_next = r + gamma * (1 - done) * self.support.unsqueeze(0)
            atom_next.clamp_(self.v_min, self.v_max)
            b = (atom_next - self.v_min) / self.delta
            l, u = b.floor(), b.ceil()
            d_m_l = (u + (l == u).float() - b) * prob_next
            d_m_u = (b - l) * prob_next
            target_prob = self.Variable(torch.zeros(prob_next.size()))
            for i in range(target_prob.size(0)):
                target_prob[i].index_add_(0, l[i].long(), d_m_l[i])
                target_prob[i].index_add_(0, u[i].long(), d_m_u[i])

        log_prob = self.model(s, log_prob=True)
        log_prob = log_prob[range(batch_size), a, :]
        loss = -(target_prob * log_prob).sum(-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(self.model.parameters(), self.max_norm)
        self.optimizer.step()
        self.update_fixed_target_network()
        return {'cost': {'loss_cost': loss.item(), 'total_cost': loss.item()}, 'error': 0, 'intrinsic_reward': 0}

    def predict(self, inputs, a, predict_model):
        inputs = self.Variable(torch.from_numpy(inputs).float())
        prob = self.model(inputs)
        q = (prob * self.support).sum(-1)
        return q.max(-1)[1].item()

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print "model saved."

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print "model loaded."
