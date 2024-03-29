import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

#####################
# Yang19 CTRNN model
#####################
class CTRNN(nn.Module):
    """Continuous-time RNN.

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, args, TRAINING_TASK_SPECS, mask=None, **kwargs):
        """
        Initialize a Continuous-Time RNN.

        Description of relevant parameters:
        args.dt: Δt = time-discretization step
            * default is args.dt=20 ms
            * If None, dt equals time constant tau (turns into discrete-time RNN)
        tau (float): τ = 100 ms is the neuronal time constant.
            * Real neurons typically have shorter time constants around 20 ms
            * here the 100 ms time constant mimics the slower synaptic dynamics on the basis of NMDA receptors (Yang et al., 2019).
        alpha (float): A scalar parameter that determines the rate of change of the hidden state (Δt/τ).
            * If alpha=1, corresponds to a discrete-time RNN dynamic (set when args.dt=None), i.e., the hidden state
            update depends on both the previous hidden state and the current input.
            * If alpha=0, the new hidden state depends entirely on the previous hidden state
        """
        super().__init__()
        self.tau = 100
        if args.dt is None:
            alpha = 1
        else:
            alpha = args.dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        self.h2h = nn.Linear(args.hidden_size, args.hidden_size)
        self.reset_parameters()
        self.hidden_size = args.hidden_size

        # TODO: take out notes once this is settled!
        # recurrent unit noise (Source: https://github.com/gyyang/multitask/blob/master/network.py#L259)
        # Based on this: https://github.com/gyyang/multitask/blob/master/network.py#L259
        #self._sigma = np.sqrt(2 / alpha) * args.sigma_rec

        # BUT NOTE according to the Yang paper: self._sigma = np.sqrt(2 * self.tau) * args.sigma_rec
        # BUT according to the original paper, it should indeed be alpha, but as a factor:
        # Source: https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1004792&type=printable, equation 7
        self._sigma = np.sqrt(2 * alpha) * args.sigma_rec

        # initialize hidden to hidden weight matrix using the mask
        if mask is not None:
            self.h2h.weight.data = self.h2h.weight.data * torch.nn.Parameter(mask)

        # define activation function
        ACTIVATION_FN_DICT = {
            'relu': torch.nn.ReLU(),
            'softplus': torch.nn.Softplus(),
            'tanh': torch.nn.Tanh()
        }
        self.activation_fn = ACTIVATION_FN_DICT[args.nonlinearity]
        self.TRAINING_TASK_SPECS = TRAINING_TASK_SPECS

    def reset_parameters(self):
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= 0.5

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.hidden_size).to(input.device)

    def recurrence(self, input, hidden, input2hs, task):
        """
        Recurrence helper function for a Continuous-Time RNN.
        It computes the next hidden state of the RNN using a Continuous-Time RNN update equation with an additional Gaussian noise term.

        Parameters:
            input (torch.Tensor): Input tensor of shape (batch_size, input_dim) containing the input sequence at the current timestep.
            hidden (torch.Tensor): Hidden state tensor of shape (batch_size, hidden_dim) containing the hidden state of the RNN at the previous timestep.
            input2hs (nn.Module): Input-to-hidden linear layer of the RNN.
            task (int): Task index specifying which task the RNN is currently being used for.

        Returns:
            h_new (torch.Tensor): Updated hidden state tensor of shape (batch_size, hidden_dim) containing the hidden state of the RNN at the current timestep.

        """
        taskname = self.TRAINING_TASK_SPECS[task]["taskname"]

        try:
            pre_activation = input2hs[taskname](input) + self.h2h(hidden)
        except:
            print(f"Need torch.float32 tensor for cog. tasks but got tensor of type #{input.dtype}#,"
                  f"converting input format")
            pre_activation = input2hs[taskname](input.to(torch.float32)) + self.h2h(hidden)

        # TODO take out notes once this is settled!
        # add recurrent unit noise (Source: https://github.com/gyyang/multitask/blob/master/network.py#L210)
        # Based on: https://github.com/xjwanglab/pycog/blob/5398bb35ccf2749e1b5dc233f1eda77fb64b6004/pycog/rnn.py#LL347C1-L356C68
        # noise_rec = torch.normal(mean=torch.zeros_like(pre_activation), std=self._sigma)
        # BUT according to both papers:
        noise_rec = self._sigma * torch.randn_like(pre_activation)
        pre_activation += noise_rec

        # this implements the discretized version of the CTRNN update equation:
        # [(1 - (Δt/τ)) * r(t)] + [(Δt/τ) * f(W_x * x(t) + W_r * r(t) + b_r)]
        h_new = self.oneminusalpha * hidden + self.alpha * self.activation_fn(pre_activation)
        return h_new

    def forward(self, input, input2hs, task, hidden=None):
        """Propagate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden, input2hs, task)
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, hidden


class Yang19_CTRNNModel(nn.Module):
    """RNN model with CTRNN module.
    Args:
        TRAINING_TASK_SPECS: dictionary of training task specifications
        hidden_size: Number of hidden neurons
        nonlinearity: activation function
        sigma_rec: recurrent unit noise
        mask: optional mask for recurrent weight matrix
    Inputs:
        x: (seq_len, batch_size, input_size), network input
        task: task index
    Returns:
        out: (seq_len * batch_size, output_size), network output
        rnn_activity: (hidden_size, hidden_size), hidden activity
    """

    def __init__(self, args, TRAINING_TASK_SPECS, mask=None, **kwargs):
        super().__init__()
        self.TRAINING_TASK_SPECS = TRAINING_TASK_SPECS
        self.hidden_size = args.hidden_size

        # define task-specific encoders and decoders
        encoders, decoders = nn.ModuleDict(), nn.ModuleDict()
        for task in TRAINING_TASK_SPECS.keys():
            taskname = TRAINING_TASK_SPECS[task]["taskname"]
            if TRAINING_TASK_SPECS[task]["using_pretrained_emb"]:
                #FIXME take out the two options after this is settled
                if not args.add_emb2hid_layer: # frozen embeddings are directly fed to RNN
                    encoders.add_module(taskname,
                                        nn.Embedding.from_pretrained(TRAINING_TASK_SPECS[task]["pretrained_emb_weights"],
                                                                     freeze=True))
                else: # frozen embeddings are fed to a linear layer before being fed to RNN
                    encoders.add_module(taskname,
                                        nn.Sequential(
                                            *[nn.Embedding.from_pretrained(TRAINING_TASK_SPECS[task]["pretrained_emb_weights"], freeze=True),
                                            nn.Linear(args.emsize, args.hidden_size)]
                                        ))
            else:
                if TRAINING_TASK_SPECS[task]["dataset"] == "lang":
                    if not args.add_emb2hid_layer:
                        encoders.add_module(taskname,
                                            nn.Embedding.from_pretrained(
                                                TRAINING_TASK_SPECS[task]["pretrained_emb_weights"]))
                    else:
                        encoders.add_module(taskname,
                                        nn.Sequential(
                                            *[nn.Embedding.from_pretrained(
                                                TRAINING_TASK_SPECS[task]["pretrained_emb_weights"]),
                                              nn.Linear(args.emsize, args.hidden_size)]
                                        ))
                else:
                    assert TRAINING_TASK_SPECS[task]["dataset"] == "cog"
                    encoders.add_module(taskname, nn.Linear(TRAINING_TASK_SPECS[task]["input_size"], args.hidden_size))
            decoders.add_module(taskname, nn.Linear(args.hidden_size, TRAINING_TASK_SPECS[task]["output_size"]))

        # build model
        self.encoders = encoders
        # Continuous time RNN
        self.rnn = CTRNN(args=args, TRAINING_TASK_SPECS=TRAINING_TASK_SPECS, mask=mask, **kwargs)
        self.decoders = decoders

    def forward(self, x, task):
        taskname = self.TRAINING_TASK_SPECS[task]["taskname"]
        rnn_activity, _ = self.rnn(input=x, input2hs=self.encoders, task=task)
        out = self.decoders[taskname](rnn_activity)
        out = out.view(-1, self.TRAINING_TASK_SPECS[task]["output_size"])
        return out, rnn_activity


#####################
# Discrete time multitask RNN model
#####################
class Multitask_RNNModel(nn.Module):
    """Discrete-time RNN.
    Based on Pytorch Word Language model https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    Container module with a task-specific encoder module list, a recurrent module, and a task-specific decoder module list
    The proper encoders and decoders are indexed when the model is called via on the 'task' argument
    (see MODE2TASK dictionary)

    Args (different from base model):
        TRAINING_TASK_SPECS: dictionary of training task specifications (including vocab size, etc.)
    """

    def __init__(self, args, TRAINING_TASK_SPECS, dropout=0.5):
        super(Multitask_RNNModel, self).__init__()

        self.TRAINING_TASK_SPECS = TRAINING_TASK_SPECS

        self.tasks = list(TRAINING_TASK_SPECS.keys())
        self.drop = nn.Dropout(dropout)
        encoders, decoders = nn.ModuleDict(), nn.ModuleDict()

        # define task-specific encoders and decoders
        for task in self.tasks:
            taskname = TRAINING_TASK_SPECS[task]["taskname"]
            if TRAINING_TASK_SPECS[task]["pretrained_emb_weights"] is not None:
                if not args.add_emb2hid_layer: #TODO take out the two options after this is settled
                    encoders.add_module(taskname,
                                        nn.Embedding.from_pretrained(
                                            TRAINING_TASK_SPECS[task]["pretrained_emb_weights"],
                                            freeze=True))
                else:
                    encoders.add_module(taskname,
                                    nn.Sequential(
                                        *[nn.Embedding.from_pretrained(
                                            TRAINING_TASK_SPECS[task]["pretrained_emb_weights"], freeze=True),
                                          nn.Linear(args.emsize, args.hidden_size)]
                                    ))
            else:
                if TRAINING_TASK_SPECS[task]["dataset"] == "lang":
                    if not args.add_emb2hid_layer: #TODO take out the two options after this is settled
                        encoders.add_module(taskname,
                                            nn.Embedding.from_pretrained(
                                                TRAINING_TASK_SPECS[task]["pretrained_emb_weights"],
                                                freeze=True))
                    else:
                        encoders.add_module(taskname,
                                        nn.Sequential(
                                            *[nn.Embedding.from_pretrained(
                                                TRAINING_TASK_SPECS[task]["pretrained_emb_weights"], freeze=True),
                                              nn.Linear(args.emsize, args.hidden_size)]
                                        ))
                else:
                    assert TRAINING_TASK_SPECS[task]["dataset"] == "cog"
                    encoders.add_module(taskname, nn.Linear(TRAINING_TASK_SPECS[task]["input_size"], args.hidden_size))
            decoders.add_module(taskname, nn.Linear(args.hidden_size,TRAINING_TASK_SPECS[task]["output_size"]))

        # define model
        self.encoders = encoders

        if args.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, args.rnn_type)(args.hidden_size, args.hidden_size, args.nlayers, dropout=dropout)
        elif args.rnn_type in ['RNN_TANH', 'RNN_RELU']:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[args.rnn_type]
            self.rnn = nn.RNN(args.hidden_size, args.hidden_size, args.nlayers, nonlinearity=args.nonlinearity, dropout=dropout)
        elif args.rnn_type in ['RNN_SOFTPLUS', 'RNN_RELU_TEST']:
            try:
                nonlinearity = {'RNN_SOFTPLUS': nn.Softplus(), 'RNN_RELU_TEST': nn.ReLU()}[args.rnn_type]
                self.rnn = RNNLayer(args.hidden_size, args.hidden_size, nonlinearity)
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU' or 'RNN_SOFTPLUS]""")
        else:
            raise NotImplementedError("Unknown!")

        self.rnn_type = args.rnn_type
        self.decoders = decoders

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if args.tie_weights:
            for task in self.tasks:
                taskname = TRAINING_TASK_SPECS[task]["taskname"]
                self.decoders[taskname].weight = self.encoders[taskname].weight

        self.init_weights()

        self.nhid = args.hidden_size
        self.nlayers = args.nlayers

    def init_weights(self):
        initrange = 0.1
        for task in self.tasks:
            taskname = self.TRAINING_TASK_SPECS[task]["taskname"]
            nn.init.uniform_(self.encoders[taskname].weight, -initrange, initrange)
            nn.init.zeros_(self.decoders[taskname].weight)
            nn.init.uniform_(self.decoders[taskname].weight, -initrange, initrange)

    def forward(self, input, hidden, task):
        taskname = self.TRAINING_TASK_SPECS[task]["taskname"]
        emb = self.encoders[taskname](input) #input is (100, 20, 53) for cog (35, 20) for wikitext
        emb = self.drop(emb)
        rnn_activity, hidden = self.rnn(emb, hidden)
        output = self.drop(rnn_activity)
        decoded = self.decoders[taskname](output)
        decoded = decoded.view(-1, self.TRAINING_TASK_SPECS[task]["output_size"])
        return decoded, hidden, rnn_activity
        #return F.log_softmax(decoded, dim=1), hidden, rnn_activity #for NLLLoss

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


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
