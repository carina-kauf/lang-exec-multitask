import logging

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule

from tqdm import tqdm
import time

import numpy as np

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.


def batchify(data, bsz):
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    batch_seq_len = data.size(0) // bsz
    data = data[:batch_seq_len * bsz]
    data = data.view(bsz, batch_seq_len).t().contiguous() # .t() is transpose
    # Turning the data over to CUDA at this point may lead to more OOM errors
    # if args.cuda:
    #    data = data.cuda()
    return data


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, bptt):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len_lang = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len_lang]
    target = source[i+1:i+1+seq_len_lang].reshape(-1)
    # This is where data should be CUDA-fied to lessen OOM errors
    if torch.cuda.is_available():
        return data.cuda(), target.cuda()
    else:
        return data, target


def get_splits(dataset, batch_size, return_only_vocab=False):
    """
    Approach is used to get wikitext dataset!

    Args:
        dataset: Name of torchtext dataset
        batch_size
        return_only_vocab

    Returns:
        if return_only_vocab:
            vocabulary dictionary: index to word
        else:
            vocab_size: int, size of vocabulary
            train_data: Tensor, shape [full_seq_len, batch_size], i.e., [N // bsz, bsz]
            val_data: Tensor, shape [full_seq_len, eval_batch_size (here:10)], i.e., [N // 10, 10]
            test_data: Tensor, shape [full_seq_len, eval_batch_size (here:10)], i.e., [N // 10, 10]
    """

    train_iter = dataset(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    if return_only_vocab:
        return vocab

    def data_process(raw_text_iter):
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # train_iter was "consumed" by the process of building the vocab, so we have to create it again
    train_iter, val_iter, test_iter = dataset()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    eval_batch_size = 10
    train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)
    vocab_size = len(vocab)

    return vocab, vocab_size, train_data, val_data, test_data


### For other way of loading data (de_wiki, etc)
import os
import torch
from collections import defaultdict

class Dictionary(object):
    """
    Approach is used for loading all non torchtext datasets (e.g. German wikitext)
    #FIXME align dataloading for all text datasets
    """
    def __init__(self, path, args):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)

        vocab_path = os.path.join(path, 'vocab.txt') #got this from prep_wiki_corpus_pipeline.sh
        try:
            vocab = open(vocab_path, encoding="utf8").read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        except FileNotFoundError:
            _logger.info("Wikitext vocab file not found, creating new vocab file.")
            self.create_vocab(os.path.join(path, 'train.txt'))
            open(os.path.join(path, 'vocab.txt'), "w").write("\n".join([w for w in self.idx2word]))

        if args.glove_emb:
            _logger.info("Using pretrained emb. vocab")
            vocab_path = os.path.join(path, 'final_vocab.txt')

            try:
                vocab = open(vocab_path, encoding="utf8").read()
                self.word2idx = {w: i for i, w in enumerate(vocab.split())}
                self.idx2word = [w for w in vocab.split()]
                self.vocab_file_exists = True
            except FileNotFoundError:
                _logger.info("Aligned vocab file not found, creating new vocab file.")
                from pathlib import Path
                glove_path = Path('../data/de_wiki/pretrained_embeddings/glove').resolve() #FIXME generalize path to all languages
                vocab, _ = align_vocab_emb(glove_path, path)

                vocab = open(vocab_path, encoding="utf8").read()
                self.word2idx = {w: i for i, w in enumerate(vocab.split())}
                self.idx2word = [w for w in vocab.split()]

    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        #return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word)


def align_vocab_emb(glove_path, vocab_path):
    import pickle
    try:
        aligned_vocab = open(os.path.join(vocab_path, 'final_vocab.txt'), encoding="utf8").read()
        with open(os.path.join(vocab_path, "de_glove_300d.txt"), "rb") as f:
            german_embeddings = pickle.load(f)
        _logger.info(f"LOADED PRETRAINED EMBEDDINGS & VOCAB FROM STORAGE")

    except FileNotFoundError:
        _logger.info("Creating aligned vocab and embeddings file!")
        # this converts vocab into other format
        wiki_vocab = make_torchtext_vocab(os.path.join(vocab_path, 'vocab.txt'))
        # load glove embeddings
        german_pretrained_vectors = get_embedding_matrix(os.path.join(glove_path, 'glove_vectors.txt'))
        # align glove embeddings with vocab
        german_embeddings, aligned_vocab = get_vecs_by_tokens(wiki_vocab.get_itos(), german_pretrained_vectors)
        # save
        import pickle
        with open(os.path.join(vocab_path, 'final_vocab.txt'), 'w') as f:
            f.writelines(["%s\n" % item for item in aligned_vocab])
        with open(os.path.join(vocab_path, "de_glove_300d.txt"), "wb") as f: #FIXME generalize for other languages
            pickle.dump(german_embeddings, f)

    return aligned_vocab, german_embeddings


def make_torchtext_vocab(vocab_textfile):
    from collections import OrderedDict
    from torchtext.vocab import vocab

    with open(vocab_textfile, 'r') as f:
        tokens = f.readlines()
        tokens = [token.split()[0] for token in tokens]
    default_index = -1
    unk_token = "<unk>"
    eos_token = "<eos>"
    torch_vocab = vocab(OrderedDict([(token, 1) for token in tokens]), specials=[unk_token, eos_token])
    torch_vocab.set_default_index(default_index)
    # print(torch_vocab['<unk>'])  # prints 0
    # print(torch_vocab['out of vocab'])  # prints -1
    # make default index same as index of unk_token
    torch_vocab.set_default_index(torch_vocab[unk_token])
    # torch_vocab['out of vocab'] is torch_vocab[unk_token] > now prints True
    return torch_vocab


def get_embedding_matrix(glove_vector_path):
    embeddings_matrix = {}
    f = open(glove_vector_path, encoding="utf8")
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_matrix[word] = coefs
    embeddings_matrix["<unk>"] = np.zeros(300) #set unk token to 0-embedding
    f.close()
    print(f"Found {len(embeddings_matrix)} vectors.")
    return embeddings_matrix


def get_vecs_by_tokens(tokens, vectors):
    """Look up embedding vectors of tokens.
    Source: https://pytorch.org/text/stable/_modules/torchtext/vocab/vectors.html#Vectors.get_vecs_by_tokens

    Arguments:
        tokens: a token or a list of tokens. if `tokens` is a string,
            returns a 1-D tensor of shape `self.dim`; if `tokens` is a
            list of strings, returns a 2-D tensor of shape=(len(tokens),
            self.dim).
    """
    to_reduce = False

    if not isinstance(tokens, list):
        tokens = [tokens]
        to_reduce = True

    glove_keys = vectors.keys()

    indices = [torch.from_numpy(vectors[token]).float() for token in tokens if token in glove_keys]
    aligned_vocab = [token for token in tokens if token in glove_keys]

    vecs = torch.stack(indices)

    if to_reduce:
        return vecs[0], aligned_vocab
    else:
        return vecs, aligned_vocab


class Corpus(object):
    def __init__(self, path, args):
        self.dictionary = Dictionary(path, args)
        try:
            self.train = torch.load(os.path.join(path, f'glove={args.glove_emb}_train.pt'))
            self.valid = torch.load(os.path.join(path, f'glove={args.glove_emb}_valid.pt'))
            self.test = torch.load(os.path.join(path, f'glove={args.glove_emb}_test.pt'))

        except FileNotFoundError:
            _logger.info("Tokenized corpus files not found, creating new files.")
            self.train = tokenize(self.dictionary, os.path.join(path, 'train.txt'))
            self.valid = tokenize(self.dictionary, os.path.join(path, 'valid.txt'))
            self.test = tokenize(self.dictionary, os.path.join(path, 'test.txt'))

            for (set_name, ids) in list(zip(["train", "valid", "test"], [self.train, self.valid, self.test])):
                torch.save(ids, os.path.join(path, f'glove={args.glove_emb}_{set_name}.pt'))


def tokenize(dictionary, path):
    """Tokenizes a text file for training or testing to a sequence of indices format """
    assert os.path.exists(path)

    with open(path, 'r', encoding="utf8") as f:
        ntokens = 0
        for line in f:
            words = line.split()
            ntokens += len(words)

    # Tokenize file content
    _logger.info(f"Tokenizing file {path}")
    with open(path, 'r', encoding="utf8") as f:
        ids = torch.LongTensor(ntokens)
        token = 0
        for line in tqdm(f):
            words = line.split()
            for word in words:
                if word in dictionary.word2idx:
                    ids[token] = dictionary.word2idx[word]
                else:
                    ids[token] = dictionary.word2idx["<unk>"]
                token += 1

    return ids


def build_dataset_lang(datapath, args):
    # Load data
    print("Loading corpus")
    start = time.time()
    # import os
    # import hashlib
    # fn = 'corpus.{}.data'.format(hashlib.md5(datapath.encode()).hexdigest())
    # if os.path.exists(fn):
    #     print('Loading cached dataset...')
    #     corpus = torch.load(fn)
    # else:
    #     print('Producing dataset...')
    #     corpus = Corpus(datapath, args)
    #     torch.save(corpus, fn)

    corpus = Corpus(datapath, args)
    from pathlib import Path
    vocab = corpus.dictionary
    print("( %.2f )" % (time.time() - start))
    vocab_size = len(vocab)
    print("Vocab size %d", vocab_size)

    print("Batchying..")
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, args.eval_batch_size)
    test_data = batchify(corpus.test, args.eval_batch_size)
    print("( %.2f )" % (time.time() - start))
    print("Done batchifying!")

    return vocab, vocab_size, train_data, val_data, test_data
###### End for other way of loading data


def build_cognitive_dataset(training_tasks, batch_size, seq_len):
    # Environment specs
    kwargs = {'dt': 100}

    # cases
    if any(x == "yang19" for x in training_tasks):
        tasks = ngym.get_collection('yang19')  # TODO check where fn is called. check if this should indeed be if, elif! tasks are being overwritten if trained on both in this way.
        yang = True
    elif any(x == "yang19Dim32" for x in training_tasks):
        tasks = ngym.get_collection('yang19Dim32')
        yang = True
    elif any(x == "khonaChandra22" for x in training_tasks):
        tasks = ngym.get_collection('khonaChandra22')
        yang = True
    else:
        tasks = training_tasks  # used when specific tasks from the collection are used, e.g., yang19.dms-v0
        yang = False

    _logger.info(f"Building dataset/env for: {tasks}")

    # Make cog environment
    envs = [gym.make(task, **kwargs) for task in tasks]
    schedule = RandomSchedule(len(envs))
    if yang:
        env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
    else:
        env = ScheduleEnvs(envs, schedule=schedule, env_input=False)  # Don't add extra rule input

    dataset_cog = ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)
    env = dataset_cog.env

    try:
        ob_size = env.observation_space.shape[0]
    except:
        ob_size = 1  # TODO check added for contrib > overwritten later
    act_size = env.action_space.n

    return dataset_cog, env, ob_size, act_size
