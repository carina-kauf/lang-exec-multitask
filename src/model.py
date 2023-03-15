import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#####################
# Discrete time multitask RNN model
#####################
class Multitask_RNNModel(nn.Module):
    """Discrete-time RNN.
    Based on Pytorch Word Language model https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    Container module with a task-specific encoder module list, a recurrent module, and a task-specific decoder module list
    The proper encoders and decoders are indexed when the model is called via on the 'mode' argument
    (see MODE2TASK dictionary)

    Args (different from base model):
        TASK2DIM: dictionary mapping from tasks to the respective input & output sizes
        MODE2TASK: dictionary mapping from modes (indices) to tasks
    """

    def __init__(self, TASK2DIM, MODE2TASK, rnn_type, ninp, hidden_size, nlayers, pretrained_emb_weights=None,
                 dropout=0.5, tie_weights=False):
        super(Multitask_RNNModel, self).__init__()

        self.modes = list(MODE2TASK.keys())
        self.TASK2DIM = TASK2DIM
        self.MODE2TASK = MODE2TASK

        self.drop = nn.Dropout(dropout)
        self.encoders, self.decoders = [], []

        # define task-specific encoders and decoders
        for key in TASK2DIM.keys():
            if any(x in key for x in ["lang", "contrib."]):
                if pretrained_emb_weights is not None:
                    self.encoders.append(nn.Embedding.from_pretrained(pretrained_emb_weights[key], freeze=True))
                    self.decoders.append(nn.Linear(hidden_size, TASK2DIM[key]["output_size"]))
                else:
                    self.encoders.append(nn.Embedding(TASK2DIM[key]["input_size"], hidden_size))
                    self.decoders.append(nn.Linear(hidden_size, TASK2DIM[key]["output_size"]))
            else:
                self.encoders.append(nn.Linear(TASK2DIM[key]["input_size"], hidden_size))
                self.decoders.append(nn.Linear(hidden_size, TASK2DIM[key]["output_size"]))

        # define model
        self.encoders = nn.ModuleList(self.encoders)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, hidden_size, nlayers, dropout=dropout)
        elif rnn_type in ['RNN_TANH', 'RNN_RELU']:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            self.rnn = nn.RNN(ninp, hidden_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        elif rnn_type in ['RNN_SOFTPLUS', 'RNN_RELU_TEST']:
            try:
                nonlinearity = {'RNN_SOFTPLUS': nn.Softplus(), 'RNN_RELU_TEST': nn.ReLU()}[rnn_type]
                self.rnn = RNNLayer(ninp, hidden_size, nonlinearity)
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU' or 'RNN_SOFTPLUS]""")
        else:
            raise NotImplementedError("Unknown!")

        self.rnn_type = rnn_type
        self.decoders = nn.ModuleList(self.decoders)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_size != ninp:
                raise ValueError('When using the tied flag, hidden_size must be equal to emsize')
            for i in self.modes:
                curr_mode = self.modes[i]
                self.decoders[curr_mode].weight = self.encoders[curr_mode].weight

        self.init_weights()

        self.nhid = hidden_size
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        for i in self.modes:
            curr_mode = self.modes[i]
            nn.init.uniform_(self.encoders[curr_mode].weight, -initrange, initrange)
            nn.init.zeros_(self.decoders[curr_mode].weight)
            nn.init.uniform_(self.decoders[curr_mode].weight, -initrange, initrange)

    def forward(self, input, hidden, mode):
        emb = self.encoders[mode](input) #input is (100, 20, 53) for cog (35, 20) for wikitext
        emb = self.drop(emb)
        rnn_activity, hidden = self.rnn(emb, hidden)
        output = self.drop(rnn_activity)
        decoded = self.decoders[mode](output)
        decoded = decoded.view(-1, self.TASK2DIM[self.MODE2TASK[mode]]["output_size"])
        return decoded, hidden, rnn_activity
        #return F.log_softmax(decoded, dim=1), hidden, rnn_activity #for NLLLoss

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


#####################
# Yang19 CTRNN model
#####################
class CTRNN(nn.Module):
    """Continuous-time RNN.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, hidden_size, nonlinearity, sigma_rec=None, dt=None, mask=None, **kwargs):
        super().__init__()
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha
        self.mask = mask

        self._sigma = np.sqrt(2 / alpha) * sigma_rec  # recurrent unit noise
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.reset_parameters()

        # initialize hidden to hidden weight matrix using the mask
        if mask is not None:
            self.h2h.weight.data = self.h2h.weight.data * torch.nn.Parameter(mask)

        self.hidden_size = hidden_size

        ACTIVATION_FN_DICT = {
            'relu': torch.nn.ReLU(),
            'softplus': torch.nn.Softplus(),
            'tanh': torch.nn.Tanh()
        }
        self.activation_fn = ACTIVATION_FN_DICT[nonlinearity]

    def reset_parameters(self):
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= 0.5

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.hidden_size).to(input.device)

    def recurrence(self, input, hidden, input2hs, mode):
        """Recurrence helper."""
        try:
            pre_activation = input2hs[mode](input) + self.h2h(hidden)
        except:
            print(f"Need float tensor here but got tensor of type #{input.dtype}#, converting input format")
            pre_activation = input2hs[mode](input.to(torch.float32)) + self.h2h(hidden)

        # add recurrent unit noise
        mean = torch.zeros_like(pre_activation)
        std = self._sigma
        noise_rec = torch.normal(mean=mean, std=std)
        pre_activation += noise_rec

        #h_new = hidden * self.oneminusalpha + torch.relu(pre_activation) * self.alpha
        h_new = hidden * self.oneminusalpha + self.activation_fn(pre_activation) * self.alpha
        return h_new

    def forward(self, input, input2hs, mode, hidden=None):
        """Propagate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden, input2hs, mode)
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, hidden


class Yang19_RNNNet(nn.Module):
    """Recurrent network model.
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """

    def __init__(self, TASK2DIM, MODE2TASK, pretrained_emb_weights, hidden_size, nonlinearity, sigma_rec=0.05, mask=None, **kwargs):
        super().__init__()

        self.input2hs = []
        self.fcs = []

        for task in TASK2DIM.keys():
            if any(x in task for x in ["lang", "contrib."]):
                if pretrained_emb_weights is not None:
                    input2h = nn.Embedding.from_pretrained(pretrained_emb_weights[task], freeze=True)
                else:
                    input2h = nn.Embedding(TASK2DIM[task]["input_size"], hidden_size)
            else:
                input2h = nn.Linear(TASK2DIM[task]["input_size"], hidden_size)
            self.input2hs.append(input2h)
            self.fcs.append(nn.Linear(hidden_size, TASK2DIM[task]["output_size"]))

        self.input2hs = nn.ModuleList(self.input2hs)
        # Continuous time RNN
        self.rnn = CTRNN(hidden_size, nonlinearity, sigma_rec=sigma_rec, mask=mask, **kwargs)
        self.fcs = nn.ModuleList(self.fcs)

        self.TASK2DIM = TASK2DIM
        self.MODE2TASK = MODE2TASK

    def forward(self, x, mode):
        rnn_activity, _ = self.rnn(x, self.input2hs, mode)
        out = self.fcs[mode](rnn_activity)
        out = out.view(-1, self.TASK2DIM[self.MODE2TASK[mode]]["output_size"])
        return out, rnn_activity


#####################################
## RNN MODEL WITH SOFTPLUS ACTIVATION
#####################################
#adapted from https://github.com/benhuh/RNN_multitask/blob/master/RNN_rate_dynamics.py

""" Custom RNN implementation """

import torch
from torch import nn
import math


class RNNCell_base(nn.Module):

    def __init__(self, input_size, hidden_size, nonlinearity, bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))  # , nonlinearity=nonlinearity)
        nn.init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))  # , nonlinearity=nonlinearity)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_ih)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

class RNNCell(RNNCell_base):
    def __init__(self, input_size, hidden_size, nonlinearity=None, bias=True):
        super().__init__(input_size, hidden_size, nonlinearity, bias)

    def forward(self, input, hidden):
        hidden = self.nonlinearity(input @ self.weight_ih.t() + hidden @ self.weight_hh.t() + self.bias)
        return hidden

class RNNLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.rnncell = RNNCell(*args)

    def forward(self, input, hidden_init):
        inputs = input.unbind(0)  # inputs has dimension [Time, batch, n_input]
        hidden = hidden_init[0]  # initial state has dimension [1, batch, n_input]
        outputs = []
        for i in range(len(inputs)):  # looping over the time dimension
            hidden = self.rnncell(inputs[i], hidden)
            outputs += [hidden]  # vanilla RNN directly outputs the hidden state
        return torch.stack(outputs), hidden

#####################################
## TRANSFORMER MODEL
#####################################
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module): #TODO CK: not integrated
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
