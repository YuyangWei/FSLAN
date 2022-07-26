from embedding import *
from hyper_embedding import *
from collections import OrderedDict
import torch
import json
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)

class ELMO(nn.Module):
    def __init__(self, embed_size=100, out_size=100):
        super(ELMO, self).__init__()
        self.embed_size = embed_size
        self.out_size = out_size
        
        self.set_elmo_for1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_for2 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False) 
        self.set_elmo_for_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back2 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)

        self.set_att_W = nn.Linear(self.embed_size * 2, self.embed_size)
        self.set_att_u = nn.Linear(self.embed_size, 1)
        self.softmax = nn.Softmax(dim=2)
        self.out = nn.Linear(self.embed_size * 2, self.out_size)

    def reverse(self, encoder_tensor, batch, few):
        reverse_tensor = torch.zeros(batch, few, self.embed_size * 2).cuda()
        for i in range(encoder_tensor.size(1)):
            reverse_tensor[:,i,:] = encoder_tensor[:,few-i-1,:]
        return reverse_tensor

    def forward(self, inputs):
        size = inputs.shape
        #1024,3,200
        inputs = inputs.contiguous().view(size[0], size[1], -1)

        support_g_for = inputs
        support_g_for_encoder, support_g_for_state0 = self.set_elmo_for1(support_g_for)
        support_g_for_encoder = support_g_for_encoder + support_g_for
        support_g_for_encoder, support_g_for_state1 = self.set_elmo_for2(support_g_for_encoder)

        support_g_for_decoder = support_g_for_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_for_set=[]
        support_g_decoder_for_state = support_g_for_state1
        for idx in range(size[1]):
            support_g_for_decoder, support_g_decoder_for_state = self.set_elmo_for_decoder(support_g_for_decoder, support_g_decoder_for_state)
            decoder_for_set.append(support_g_for_decoder)
        decoder_for_set = torch.cat(decoder_for_set, dim=1)

        #print("support_g_for",support_g_for.shape)
        #print("decoder_for_set",decoder_for_set.shape)
        ae_for_loss = nn.MSELoss()(support_g_for, decoder_for_set.detach())        

        #backward
        support_g_back = self.reverse(inputs, size[0], size[1])
        support_g_back_encoder, support_g_back_state0 = self.set_elmo_back1(support_g_back)
        support_g_back_encoder = support_g_back_encoder + support_g_back
        support_g_back_encoder, support_g_back_state1 = self.set_elmo_back2(support_g_back_encoder)
        support_g_back_decoder = support_g_back_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_back_set=[]
        support_g_decoder_back_state = support_g_back_state1
        for idx in range(size[1]):
            support_g_back_decoder, support_g_decoder_state = self.set_elmo_back_decoder(support_g_back_decoder, support_g_decoder_back_state)
            decoder_back_set.append(support_g_back_decoder)
        decoder_back_set = torch.cat(decoder_back_set, dim=1)

        ae_back_loss = nn.MSELoss()(support_g_back, decoder_back_set.detach())
        support_g_back_encoder = self.reverse(support_g_back_encoder, size[0], size[1])

        #1024,3,200
        support_g_encoder = inputs + support_g_for_encoder + support_g_back_encoder
        
        support_g_att = self.set_att_W(support_g_encoder).tanh()
        att_w = self.set_att_u(support_g_att)
        att_w = self.softmax(att_w) #1024,3,1
        support_g_encoder = torch.bmm(support_g_encoder.transpose(1, 2), att_w) #1024, 200, 1
        support_g_encoder = support_g_encoder.transpose(1, 2) #1024, 1, 200
    
        outputs = self.out(support_g_encoder)

        return outputs.view(size[0], 1, 1, self.out_size), ae_for_loss + ae_back_loss

class ELMO_without(nn.Module):
    def __init__(self, embed_size=100, out_size=100):
        super(ELMO_without, self).__init__()
        self.embed_size = embed_size
        self.out_size = out_size
        
        self.set_elmo_for1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_for2 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False) 
        self.set_elmo_for_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back2 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)

        self.set_att_W = nn.Linear(self.embed_size * 2, self.embed_size)
        self.set_att_u = nn.Linear(self.embed_size, 1)
        self.softmax = nn.Softmax(dim=2)
        self.out = nn.Linear(self.embed_size * 2, self.out_size)

    def reverse(self, encoder_tensor, batch, few):
        reverse_tensor = torch.zeros(batch, few, self.embed_size * 2).cuda()
        for i in range(encoder_tensor.size(1)):
            reverse_tensor[:,i,:] = encoder_tensor[:,few-i-1,:]
        return reverse_tensor

    def forward(self, inputs):
        size = inputs.shape
        #1024,3,200
        inputs = inputs.contiguous().view(size[0], size[1], -1)

        support_g_for = inputs
        support_g_for_encoder, support_g_for_state0 = self.set_elmo_for1(support_g_for)
        support_g_for_encoder, support_g_for_state1 = self.set_elmo_for2(support_g_for_encoder)

        support_g_for_decoder = support_g_for_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_for_set=[]
        support_g_decoder_for_state = support_g_for_state1
        for idx in range(size[1]):
            support_g_for_decoder, support_g_decoder_for_state = self.set_elmo_for_decoder(support_g_for_decoder, support_g_decoder_for_state)
            decoder_for_set.append(support_g_for_decoder)
        decoder_for_set = torch.cat(decoder_for_set, dim=1)

        #print("support_g_for",support_g_for.shape)
        #print("decoder_for_set",decoder_for_set.shape)
        ae_for_loss = nn.MSELoss()(support_g_for, decoder_for_set.detach())        

        #backward
        support_g_back = self.reverse(inputs, size[0], size[1])
        support_g_back_encoder, support_g_back_state0 = self.set_elmo_back1(support_g_back)
        support_g_back_encoder, support_g_back_state1 = self.set_elmo_back2(support_g_back_encoder)
        support_g_back_decoder = support_g_back_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_back_set=[]
        support_g_decoder_back_state = support_g_back_state1
        for idx in range(size[1]):
            support_g_back_decoder, support_g_decoder_state = self.set_elmo_back_decoder(support_g_back_decoder, support_g_decoder_back_state)
            decoder_back_set.append(support_g_back_decoder)
        decoder_back_set = torch.cat(decoder_back_set, dim=1)

        ae_back_loss = nn.MSELoss()(support_g_back, decoder_back_set.detach())
        support_g_back_encoder = self.reverse(support_g_back_encoder, size[0], size[1])

        #1024,3,200
        support_g_encoder = inputs + support_g_for_encoder + support_g_back_encoder
        
        support_g_att = self.set_att_W(support_g_encoder).tanh()
        att_w = self.set_att_u(support_g_att)
        att_w = self.softmax(att_w) #1024,3,1
        support_g_encoder = torch.bmm(support_g_encoder.transpose(1, 2), att_w) #1024, 200, 1
        support_g_encoder = support_g_encoder.transpose(1, 2) #1024, 1, 200
    
        outputs = self.out(support_g_encoder)

        return outputs.view(size[0], 1, 1, self.out_size), ae_for_loss + ae_back_loss

class ELMO_T(nn.Module):
    def __init__(self, embed_size=100, out_size=100):
        super(ELMO_T, self).__init__()
        self.embed_size = embed_size
        self.out_size = out_size
        
        self.set_elmo_for1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_for2 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False) 
        self.set_elmo_for3 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_for_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back2 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back3 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)

        self.set_att_W = nn.Linear(self.embed_size * 2, self.embed_size)
        self.set_att_u = nn.Linear(self.embed_size, 1)
        self.softmax = nn.Softmax(dim=2)
        self.out = nn.Linear(self.embed_size * 2, self.out_size)

    def reverse(self, encoder_tensor, batch, few):
        reverse_tensor = torch.zeros(batch, few, self.embed_size * 2).cuda()
        for i in range(encoder_tensor.size(1)):
            reverse_tensor[:,i,:] = encoder_tensor[:,few-i-1,:]
        return reverse_tensor

    def forward(self, inputs):
        size = inputs.shape
        #1024,3,200
        inputs = inputs.contiguous().view(size[0], size[1], -1)

        support_g_for = inputs
        support_g_for_encoder, support_g_for_state0 = self.set_elmo_for1(support_g_for)
        support_g_for_encoder = support_g_for_encoder + support_g_for
        support_g_for_encoder, support_g_for_state1 = self.set_elmo_for2(support_g_for_encoder)
        support_g_for_encoder = support_g_for_encoder + support_g_for
        support_g_for_encoder, support_g_for_state2 = self.set_elmo_for3(support_g_for_encoder)
        support_g_for_decoder = support_g_for_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_for_set=[]
        support_g_decoder_for_state = support_g_for_state2
        for idx in range(size[1]):
            support_g_for_decoder, support_g_decoder_for_state = self.set_elmo_for_decoder(support_g_for_decoder, support_g_decoder_for_state)
            decoder_for_set.append(support_g_for_decoder)
        decoder_for_set = torch.cat(decoder_for_set, dim=1)

        #print("support_g_for",support_g_for.shape)
        #print("decoder_for_set",decoder_for_set.shape)
        ae_for_loss = nn.MSELoss()(support_g_for, decoder_for_set.detach())        

        #backward
        support_g_back = self.reverse(inputs, size[0], size[1])
        support_g_back_encoder, support_g_back_state0 = self.set_elmo_back1(support_g_back)
        support_g_back_encoder = support_g_back_encoder + support_g_back
        support_g_back_encoder, support_g_back_state1 = self.set_elmo_back2(support_g_back_encoder)
        support_g_back_encoder = support_g_back_encoder + support_g_back
        support_g_back_encoder, support_g_back_state2 = self.set_elmo_back3(support_g_back_encoder)
        support_g_back_decoder = support_g_back_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_back_set=[]
        support_g_decoder_back_state = support_g_back_state2
        for idx in range(size[1]):
            support_g_back_decoder, support_g_decoder_state = self.set_elmo_back_decoder(support_g_back_decoder, support_g_decoder_back_state)
            decoder_back_set.append(support_g_back_decoder)
        decoder_back_set = torch.cat(decoder_back_set, dim=1)

        ae_back_loss = nn.MSELoss()(support_g_back, decoder_back_set.detach())
        support_g_back_encoder = self.reverse(support_g_back_encoder, size[0], size[1])

        #1024,3,200
        support_g_encoder = inputs + support_g_for_encoder + support_g_back_encoder
        
        support_g_att = self.set_att_W(support_g_encoder).tanh()
        att_w = self.set_att_u(support_g_att)
        att_w = self.softmax(att_w) #1024,3,1
        support_g_encoder = torch.bmm(support_g_encoder.transpose(1, 2), att_w) #1024, 200, 1
        support_g_encoder = support_g_encoder.transpose(1, 2) #1024, 1, 200
    
        outputs = self.out(support_g_encoder)

        return outputs.view(size[0], 1, 1, self.out_size), ae_for_loss + ae_back_loss

class ELMO_DAN(nn.Module):
    def __init__(self, embed_size=100, out_size=100):
        super(ELMO_DAN, self).__init__()
        self.embed_size = embed_size
        self.out_size = out_size
        
        self.set_elmo_for1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_for_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)

        self.set_att_W = nn.Linear(self.embed_size * 2, self.embed_size)
        self.set_att_u = nn.Linear(self.embed_size, 1)
        self.softmax = nn.Softmax(dim=2)
        self.out = nn.Linear(self.embed_size * 2, self.out_size)

    def reverse(self, encoder_tensor, batch, few):
        reverse_tensor = torch.zeros(batch, few, self.embed_size * 2).cuda()
        for i in range(encoder_tensor.size(1)):
            reverse_tensor[:,i,:] = encoder_tensor[:,few-i-1,:]
        return reverse_tensor

    def forward(self, inputs):
        size = inputs.shape
        #1024,3,200
        inputs = inputs.contiguous().view(size[0], size[1], -1)

        support_g_for = inputs
        support_g_for_encoder, support_g_for_state0 = self.set_elmo_for1(support_g_for)
        support_g_for_decoder = support_g_for_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_for_set=[]
        support_g_decoder_for_state = support_g_for_state0
        for idx in range(size[1]):
            support_g_for_decoder, support_g_decoder_for_state = self.set_elmo_for_decoder(support_g_for_decoder, support_g_decoder_for_state)
            decoder_for_set.append(support_g_for_decoder)
        decoder_for_set = torch.cat(decoder_for_set, dim=1)

        #print("support_g_for",support_g_for.shape)
        #print("decoder_for_set",decoder_for_set.shape)
        ae_for_loss = nn.MSELoss()(support_g_for, decoder_for_set.detach())        

        #backward
        support_g_back = self.reverse(inputs, size[0], size[1])
        support_g_back_encoder, support_g_back_state0 = self.set_elmo_back1(support_g_back)
        support_g_back_decoder = support_g_back_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_back_set=[]
        support_g_decoder_back_state = support_g_back_state0
        for idx in range(size[1]):
            support_g_back_decoder, support_g_decoder_state = self.set_elmo_back_decoder(support_g_back_decoder, support_g_decoder_back_state)
            decoder_back_set.append(support_g_back_decoder)
        decoder_back_set = torch.cat(decoder_back_set, dim=1)

        ae_back_loss = nn.MSELoss()(support_g_back, decoder_back_set.detach())
        support_g_back_encoder = self.reverse(support_g_back_encoder, size[0], size[1])

        #1024,3,200
        support_g_encoder = inputs + support_g_for_encoder + support_g_back_encoder
        
        support_g_att = self.set_att_W(support_g_encoder).tanh()
        att_w = self.set_att_u(support_g_att)
        att_w = self.softmax(att_w) #1024,3,1
        support_g_encoder = torch.bmm(support_g_encoder.transpose(1, 2), att_w) #1024, 200, 1
        support_g_encoder = support_g_encoder.transpose(1, 2) #1024, 1, 200
    
        outputs = self.out(support_g_encoder)

        return outputs.view(size[0], 1, 1, self.out_size), ae_for_loss + ae_back_loss

class ELMO_Bi(nn.Module):
    def __init__(self, embed_size=100, out_size=100):
        super(ELMO_Bi, self).__init__()
        self.embed_size = embed_size
        self.out_size = out_size
        
        self.set_elmo_for1 = nn.LSTM(2 * self.embed_size, self.embed_size, 1, batch_first = True, bidirectional = True)
        self.set_elmo_for_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)

        self.set_att_W = nn.Linear(2 * self.embed_size, 2 * self.embed_size)
        self.set_att_u = nn.Linear(2 * self.embed_size, 1)
        self.softmax = nn.Softmax(dim=2)
        self.out = nn.Linear(self.embed_size * 2, self.out_size)

    def forward(self, inputs):
        size = inputs.shape
        #1024,3,200
        inputs = inputs.contiguous().view(size[0], size[1], -1)

        support_g_for = inputs
        support_g_for_encoder, support_g_for_state0 = self.set_elmo_for1(support_g_for)
        support_g_for_decoder = support_g_for_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size) #[1024,1,200]
        decoder_for_set=[]
        support_g_decoder_for_state = torch.cat((support_g_for_state0[0].view(1, -1, self.embed_size),support_g_for_state0[1].view(1, -1, self.embed_size)),dim=-1)
        for idx in range(size[1]):
            support_g_for_decoder, support_g_decoder_for_state = self.set_elmo_for_decoder(support_g_for_decoder)
            decoder_for_set.append(support_g_for_decoder)
        decoder_for_set = torch.cat(decoder_for_set, dim=1)

        ae_for_loss = nn.MSELoss()(support_g_for, decoder_for_set.detach())        

        #1024,3,200
        support_g_encoder = inputs + support_g_for_encoder
        
        support_g_att = self.set_att_W(support_g_encoder).tanh()
        att_w = self.set_att_u(support_g_att)
        att_w = self.softmax(att_w) #1024,3,1
        support_g_encoder = torch.bmm(support_g_encoder.transpose(1, 2), att_w) #1024, 200, 1
        support_g_encoder = support_g_encoder.transpose(1, 2) #1024, 1, 200

        outputs = self.out(support_g_encoder)
        return outputs.view(size[0], 1, 1, self.out_size), ae_for_loss


class ELMO_Mean(nn.Module):
    def __init__(self, embed_size=100, out_size=100):
        super(ELMO_Mean, self).__init__()
        self.embed_size = embed_size
        self.out_size = out_size
        
        self.set_elmo_for1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_for2 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False) 
        self.set_elmo_for_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back2 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)

        self.set_att_W = nn.Linear(self.embed_size * 2, self.embed_size)
        self.set_att_u = nn.Linear(self.embed_size, 1)
        self.softmax = nn.Softmax(dim=2)
        self.out = nn.Linear(self.embed_size * 2, self.out_size)

    def reverse(self, encoder_tensor, batch, few):
        reverse_tensor = torch.zeros(batch, few, self.embed_size * 2).cuda()
        for i in range(encoder_tensor.size(1)):
            reverse_tensor[:,i,:] = encoder_tensor[:,few-i-1,:]
        return reverse_tensor

    def forward(self, inputs):
        size = inputs.shape
        #1024,3,200
        inputs = inputs.contiguous().view(size[0], size[1], -1)

        support_g_for = inputs
        support_g_for_encoder, support_g_for_state0 = self.set_elmo_for1(support_g_for)
        support_g_for_encoder = support_g_for_encoder + support_g_for
        support_g_for_encoder, support_g_for_state1 = self.set_elmo_for2(support_g_for_encoder)

        support_g_for_decoder = support_g_for_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_for_set=[]
        support_g_decoder_for_state = support_g_for_state1
        for idx in range(size[1]):
            support_g_for_decoder, support_g_decoder_for_state = self.set_elmo_for_decoder(support_g_for_decoder, support_g_decoder_for_state)
            decoder_for_set.append(support_g_for_decoder)
        decoder_for_set = torch.cat(decoder_for_set, dim=1)

        #print("support_g_for",support_g_for.shape)
        #print("decoder_for_set",decoder_for_set.shape)
        ae_for_loss = nn.MSELoss()(support_g_for, decoder_for_set.detach())        

        #backward
        support_g_back = self.reverse(inputs, size[0], size[1])
        support_g_back_encoder, support_g_back_state0 = self.set_elmo_back1(support_g_back)
        support_g_back_encoder = support_g_back_encoder + support_g_back
        support_g_back_encoder, support_g_back_state1 = self.set_elmo_back2(support_g_back_encoder)
        support_g_back_decoder = support_g_back_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_back_set=[]
        support_g_decoder_back_state = support_g_back_state1
        for idx in range(size[1]):
            support_g_back_decoder, support_g_decoder_state = self.set_elmo_back_decoder(support_g_back_decoder, support_g_decoder_back_state)
            decoder_back_set.append(support_g_back_decoder)
        decoder_back_set = torch.cat(decoder_back_set, dim=1)

        ae_back_loss = nn.MSELoss()(support_g_back, decoder_back_set.detach())
        support_g_back_encoder = self.reverse(support_g_back_encoder, size[0], size[1])



        #1024,3,200
        support_g_encoder = inputs + support_g_for_encoder + support_g_back_encoder

        support_g_encoder = torch.mean(support_g_encoder,dim=1)

        outputs = self.out(support_g_encoder)

        return outputs.view(size[0], 1, 1, self.out_size), ae_for_loss + ae_back_loss

class ELMO_Max(nn.Module):
    def __init__(self, embed_size=100, out_size=100):
        super(ELMO_Max, self).__init__()
        self.embed_size = embed_size
        self.out_size = out_size
        
        self.set_elmo_for1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_for2 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False) 
        self.set_elmo_for_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back1 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back2 = nn.LSTM(self.embed_size * 2, self.embed_size * 2, 1, batch_first = True, bidirectional = False)
        self.set_elmo_back_decoder = nn.LSTM(2 * self.embed_size, 2 * self.embed_size, 1, batch_first = True, bidirectional = False)

        self.set_att_W = nn.Linear(self.embed_size * 2, self.embed_size)
        self.set_att_u = nn.Linear(self.embed_size, 1)
        self.softmax = nn.Softmax(dim=2)
        self.out = nn.Linear(self.embed_size * 2, self.out_size)

    def reverse(self, encoder_tensor, batch, few):
        reverse_tensor = torch.zeros(batch, few, self.embed_size * 2).cuda()
        for i in range(encoder_tensor.size(1)):
            reverse_tensor[:,i,:] = encoder_tensor[:,few-i-1,:]
        return reverse_tensor

    def forward(self, inputs):
        size = inputs.shape
        #1024,3,200
        inputs = inputs.contiguous().view(size[0], size[1], -1)

        support_g_for = inputs
        support_g_for_encoder, support_g_for_state0 = self.set_elmo_for1(support_g_for)
        support_g_for_encoder = support_g_for_encoder + support_g_for
        support_g_for_encoder, support_g_for_state1 = self.set_elmo_for2(support_g_for_encoder)

        support_g_for_decoder = support_g_for_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_for_set=[]
        support_g_decoder_for_state = support_g_for_state1
        for idx in range(size[1]):
            support_g_for_decoder, support_g_decoder_for_state = self.set_elmo_for_decoder(support_g_for_decoder, support_g_decoder_for_state)
            decoder_for_set.append(support_g_for_decoder)
        decoder_for_set = torch.cat(decoder_for_set, dim=1)

        #print("support_g_for",support_g_for.shape)
        #print("decoder_for_set",decoder_for_set.shape)
        ae_for_loss = nn.MSELoss()(support_g_for, decoder_for_set.detach())        

        #backward
        support_g_back = self.reverse(inputs, size[0], size[1])
        support_g_back_encoder, support_g_back_state0 = self.set_elmo_back1(support_g_back)
        support_g_back_encoder = support_g_back_encoder + support_g_back
        support_g_back_encoder, support_g_back_state1 = self.set_elmo_back2(support_g_back_encoder)
        support_g_back_decoder = support_g_back_encoder[:,-1,:].view(-1, 1, 2 * self.embed_size)
        decoder_back_set=[]
        support_g_decoder_back_state = support_g_back_state1
        for idx in range(size[1]):
            support_g_back_decoder, support_g_decoder_state = self.set_elmo_back_decoder(support_g_back_decoder, support_g_decoder_back_state)
            decoder_back_set.append(support_g_back_decoder)
        decoder_back_set = torch.cat(decoder_back_set, dim=1)

        ae_back_loss = nn.MSELoss()(support_g_back, decoder_back_set.detach())
        support_g_back_encoder = self.reverse(support_g_back_encoder, size[0], size[1])



        #1024,3,200
        support_g_encoder = inputs + support_g_for_encoder + support_g_back_encoder

        support_g_encoder, _ = torch.max(support_g_encoder,dim=1)

        outputs = self.out(support_g_encoder)

        return outputs.view(size[0], 1, 1, self.out_size), ae_for_loss + ae_back_loss



class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1):
        super(LSTM_attn, self).__init__()
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.lstm = nn.LSTM(self.embed_size*2, self.n_hidden, self.layers, bidirectional=True)
        self.out = nn.Linear(self.n_hidden*2*self.layers, self.out_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden*2, self.layers)
        attn_weight = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weight = F.softmax(attn_weight, 1)
        context = torch.bmm(lstm_output.transpose(1,2), soft_attn_weight)
        context = context.view(-1, self.n_hidden*2*self.layers)
        return context

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        input = inputs.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(self.layers*2, size[0], self.n_hidden)).cuda()
        cell_state = Variable(torch.zeros(self.layers*2, size[0], self.n_hidden)).cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_hidden_state)

        outputs = self.out(attn_output)
        return outputs.view(size[0], 1, 1, self.out_size)


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num, norm):
        #norm = norm[:,:1,:,:]						# revise
        #h = h - torch.sum(h * norm, -1, True) * norm
        #t = t - torch.sum(t * norm, -1, True) * norm
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


def save_grad(grad):
    global grad_norm
    grad_norm = grad


class MetaR(nn.Module):
    def __init__(self, dataset, parameter, num_symbols, co_entities = None, embed = None):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.rel2id = dataset['rel2id']
        self.num_rel = len(self.rel2id)
        self.embedding = Embedding(dataset, parameter)
        self.h_embedding = H_Embedding(dataset, parameter)
        self.few = parameter['few']
        self.dropout = nn.Dropout(parameter['dropout'])
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.embed_dim, padding_idx = num_symbols)
        self.max_neighbor = parameter['max_neighbor']
        self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))

        self.h_emb = nn.Embedding(self.num_rel, self.embed_dim)
        init.xavier_uniform_(self.h_emb.weight)

        self.co_entities = torch.from_numpy(co_entities)
        print("self.co_entities",self.co_entities.shape)

        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.attn_w = nn.Linear(self.embed_dim, 1)

        self.gate_w = nn.Linear(self.embed_dim, 1)
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.attn_w.weight)

        #self.symbol_emb.weight.requires_grad = False
        self.h_norm = None
        '''
        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = ELMO_Mean(embed_size=50, out_size=50)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = ELMO_Mean(embed_size=100, out_size=100)
        elif parameter['dataset'] == 'FB15k-One':
            self.relation_learner = ELMO_Mean(embed_size=100, out_size=100)
        '''

        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = ELMO_without(embed_size=50, out_size=50)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = ELMO_without(embed_size=100, out_size=100)
        elif parameter['dataset'] == 'FB15k-One':
            self.relation_learner = ELMO_without(embed_size=100, out_size=100)
        
        '''
        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = ELMO_Max(embed_size=50, out_size=50)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = ELMO_Max(embed_size=100, out_size=100)
        elif parameter['dataset'] == 'FB15k-One':
            self.relation_learner = ELMO_Max(embed_size=100, out_size=100)
        '''            
                    
        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.norm_q_sharing = dict()

    def get_co_weight(self, entities, targets):
        entities = entities.cpu()
        targets = targets.cpu()

        co_weight = torch.zeros([entities.shape[0],entities.shape[1]],dtype=torch.float)      
        
        for i in range(entities.shape[0]):
            co_weight[i,:] = self.co_entities[entities[i,:],targets[i,:]]
        return co_weight

    def neighbor_encoder(self, connections, num_neighbors, target):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:,:,1].squeeze(-1)
        entities = connections[:,:,2].squeeze(-1)
        entself = connections[:,0,0].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations)) # (batch, 200, embed_dim)
        ent_embeds = self.dropout(self.symbol_emb(entities)) # (batch, 200, embed_dim)
        entself_embeds = self.dropout(self.symbol_emb(entself))
        target_ent = target[:,0,0].view(-1, 1).repeat(1, self.max_neighbor)
        
        co_weight = self.get_co_weight(entities, target_ent).cuda().view(entities.size()[0], 1, self.max_neighbor)
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1) # (batch, 200, 2*embed_dim)

        out = self.gcn_w(concat_embeds) + self.gcn_b
        out = F.leaky_relu(out)
        attn_out = self.attn_w(out)
        attn_weight = F.softmax(attn_out, dim=1)
        out_attn = torch.bmm(out.transpose(1,2), attn_weight)
        out_attn = out_attn.squeeze(2)
        gate_tmp = self.gate_w(out_attn) + self.gate_b
        gate = torch.sigmoid(gate_tmp)
        out_neigh = torch.mul(out_attn, gate)
     
        neighbor_emb= torch.bmm(co_weight,ent_embeds).view(entities.size()[0], -1)
        out_neighbor = out_neigh + torch.mul(entself_embeds,1.0-gate) + neighbor_emb

        return out_neighbor

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2


    def forward(self, task, target_pairs, iseval=False, curr_rel='', support_meta=None, istest=False):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]
        norm_vector = self.h_embedding(task[0])
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        target_left = target_pairs[:,:,0].view(-1,self.few,1).repeat(1, 1, self.max_neighbor)
        target_right = target_pairs[:,:,1].view(-1,self.few,1).repeat(1, 1, self.max_neighbor)

        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta[0]
        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees, support_right_connections)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees, support_left_connections)
        support_few = torch.cat((support_left, support_right), dim=-1)
        support_few = support_few.view(support_few.shape[0], 2, self.embed_dim)

        for i in range(self.few-1):
            support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta[i+1]
            support_left = self.neighbor_encoder(support_left_connections, support_left_degrees, support_right_connections)
            support_right = self.neighbor_encoder(support_right_connections, support_right_degrees, support_left_connections)
            support_pair = torch.cat((support_left, support_right), dim=-1)  # tanh
            support_pair = support_pair.view(support_pair.shape[0], 2, self.embed_dim)
            support_few = torch.cat((support_few, support_pair), dim=1)
        support_few = support_few.view(support_few.shape[0], self.few, 2, self.embed_dim)
        rel, ae_loss = self.relation_learner(support_few)
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1)
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]

        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few, norm_vector)	# revise norm_vector

                y = torch.Tensor([1]).to(self.device)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)
                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta
                norm_q = norm_vector - self.beta*grad_meta				# hyper-plane update
            else:
                rel_q = rel
                norm_q = norm_vector

            self.rel_q_sharing[curr_rel] = rel_q
            self.h_norm = norm_vector.mean(0)
            self.h_norm = self.h_norm.unsqueeze(0)

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        if iseval:
            norm_q = self.h_norm
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q, norm_q)

        return p_score, n_score, ae_loss
