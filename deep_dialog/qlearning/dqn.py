import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from network import Network, DuelNetwork

use_cuda = torch.cuda.is_available()

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, duel=True, double=True, use_icm=True, noisy=True):
        super(DQN, self).__init__()

        network = DuelNetwork if duel else Network

        self.model = network(input_size, hidden_size, output_size, noisy)
        self.target_model = network(input_size, hidden_size, output_size, noisy)
        self.target_model.load_state_dict(self.model.state_dict())


        # hyper parameters
        self.max_norm = 1e-3
        lr = 0.001
        self.tau = 1e-2
        self.regc = 1e-3

        self.icm = Network(hidden_size + input_size, hidden_size, input_size)
        self.action_emb = nn.Embedding(input_size, hidden_size)
        self.icm_optim = optim.Adam(self.icm.parameters(), lr=lr)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=lr)
        self.double = double
        self.use_icm = use_icm

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
        
        batch = [np.vstack(b) for b in zip(*raw_batch)]
        # each example in a batch: [s, a, r, s_prime, term]
        s = self.Variable(torch.FloatTensor(batch[0]))
        a = self.Variable(torch.LongTensor(batch[1]))
        r = self.Variable(torch.FloatTensor(batch[2]))
        s_prime = self.Variable(torch.FloatTensor(batch[3]))
        done = self.Variable(torch.FloatTensor(np.array(batch[4]).astype(np.float32)))
        i_r = self.Variable(torch.zeros(1)) 
        if self.use_icm:
            s_pred = self.icm(torch.cat([s.detach(), self.action_emb(a.detach()).squeeze()], -1))
            icm_loss = F.mse_loss(s_pred, s_prime.detach(), reduce=False)
            i_r = icm_loss.sum(-1).detach()
            i_r_norm = (i_r - i_r.mean()) / i_r.std()
            r = r + i_r_norm.unsqueeze(1)
            icm_loss = icm_loss.mean()
            self.icm_optim.zero_grad()
            icm_loss.backward()
            self.icm_optim.step()
        
        q = self.model(s)
        if self.double:
            q_prime = self.model(s_prime).detach()
            a_prime = q_prime.max(1)[1]
            q_target_prime = self.target_model(s_prime).detach()
            q_target_prime = q_target_prime.gather(1, a_prime.unsqueeze(1))
            q_target = r + gamma * q_target_prime * (1 - done) 
        else:
            q_prime = self.target_model(s_prime).detach()
            q_prime = q_prime.max(1)[0].unsqueeze(1)
            q_target = r + gamma * q_prime * (1 - done)
        q_pred = torch.gather(q, 1, a)
        loss = F.mse_loss(q_pred, q_target)
        err = torch.abs(q_pred - q_target).detach()
    

        reg_loss = 0 
        '''
        # L2 regularization
        for name, p in self.model.named_parameters():
            if name.find('weight') != -1:
                reg_loss += self.regc * 0.5 * p.norm(2) / s.size(0)
        '''
        self.update_fixed_target_network()
        self.optimizer.zero_grad()
        (loss + reg_loss).backward()
        clip_grad_norm(self.model.parameters(), self.max_norm)
        self.optimizer.step()
        
        self.model.sample_noise()
        self.target_model.sample_noise()


        return {'cost': {'reg_cost': reg_loss, 'loss_cost': loss.item(), 
            'total_cost': (loss + reg_loss).item()}, 'error':err.cpu().numpy(),
            'intrinsic_reward': i_r.mean().cpu().numpy()}

    def get_intrinsic_reward(self, state, next_state, action):
        state = self.Variable(torch.from_numpy(state.astype(np.float32)))
        next_state = self.Variable(torch.from_numpy(next_state.astype(np.float32)))
        action = self.Variable(torch.from_numpy(action.astype(np.int64))).view(1, 1)
        state_pred = self.icm(torch.cat([state, self.action_emb(action).squeeze(0)], -1))
        icm_loss = F.mse_loss(state_pred, next_state.detach(), reduce=False)
        i_r = icm_loss.sum(-1).detach()
        i_r = (i_r - i_r.mean()) / (i_r.std() + 1e-10)
        return i_r.cpu().numpy()


    def predict(self, inputs, a, predict_model):
        inputs = self.Variable(torch.from_numpy(inputs).float())
        return torch.max(self.model(inputs, True), 1)[1].cpu().data.numpy()[0]

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print "model saved."

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print "model loaded."
