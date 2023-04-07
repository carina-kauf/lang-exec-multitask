from datasets import list_datasets, load_dataset, concatenate_datasets
import torchtext
import torch
from tqdm import tqdm
import datasets
import os
import glob
from utils_pretrained_embeddings import align_vocab_emb
import pickle
import logging
from transformers import HfArgumentParser
from args import TaskArguments, CTRNNModelArguments, RNNModelArguments, SharedModelArguments, TrainingArguments
import random

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_train_val_test_splits(dataset, test_size=0.1):
    """Get or create train, validation, and test splits from a given dataset.

    Args:
    - dataset (dict): The dataset with certain splits
    - test_size (float): The proportion of the dataset to use for testing.

    Returns:
    - dataset (dict) with values:
        - "test" : (list or numpy array): The training set.
        - "valid" : val_set (list or numpy array): The validation set.
        - "test" (list or numpy array): The test set.
    """
    # input validation
    if (not isinstance(test_size, float)) or (test_size < 0 or test_size > 1):
        raise ValueError("test_size must be a float between 0 and 1")
    if not isinstance(dataset, dict):
        raise TypeError("dataset must be a dictionary")
    if not all(isinstance(x, datasets.arrow_dataset.Dataset) for x in dataset.values()):
        raise TypeError("dataset must contain only datasets.arrow_dataset.Dataset objects")

    # get names of splits
    split_names = list(dataset.keys())

    # create splits
    if split_names == ["train"]:
        _logger.debug('No validation or test set found, creating them')
        # create train-valid-test splits: 90% train, 10% test + validation
        train_testvalid = dataset["train"].train_test_split(test_size=test_size, shuffle=True)
        dataset["train"] = train_testvalid['train']
        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=True)
        dataset["valid"] = test_valid['train']
        dataset["test"] = test_valid['test']

    elif not any(x in dataset for x in ["valid", "validation"]):
        _logger.debug('No test set found, creating them')
        # create validation split
        train_valid_data = dataset["train"].train_test_split(test_size=test_size, shuffle=True)
        dataset["train"], dataset["valid"] = train_valid_data['train'], train_valid_data['test']

    elif all(x in dataset for x in ["train", "valid", "test"]) or all(x in dataset for x in ["train", "validation", "test"]):
        _logger.debug('Found train, validation, and test set')
        if "validation" in dataset:
            _logger.debug('Renaming "validation" to "valid" for consistency')
            dataset["valid"] = dataset.pop("validation")
        pass

    else:
        raise NotImplementedError("Dataset splits not recognized")

    return dataset


def get_vocabulary(args, dataset_identifier, tokenized_dataset, min_freq=3, max_tokens=30000):
    """Builds a vocabulary from the training data.
    Args:
        - tokenized_dataset (dict): The dataset with a 'train' split and a 'tokens' column.
        - min_freq (int): The minimum frequency of a token to be included in the vocabulary.
        - max_size (int): The maximum size of the vocabulary.
    Returns:
        - vocab (torchtext.vocab.Vocab): The vocabulary.
    """
    # input verification
    if 'train' not in tokenized_dataset:
        raise ValueError("tokenized_dataset must contain a 'train' split")
    if 'tokens' not in tokenized_dataset['train'].column_names:
        raise ValueError("tokenized_dataset must contain a 'tokens' column")

    special_tokens = ['<unk>', '<eos>']
    _logger.info(f"Creating a vocabulary from the training data with minimum frequency {min_freq} and maximal vocab size {max_tokens}...")
    vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset['train']['tokens'],
                                                      min_freq=min_freq,
                                                      specials=special_tokens,
                                                      max_tokens=max_tokens)

    if not args.glove_emb: # if not using pretrained embeddings (e.g. GloVe) then we can just return the vocab
        # set default index to <unk>
        _logger.debug(f"Setting '{vocab['<unk>']}' as the default index")
        vocab.set_default_index(vocab['<unk>'])
        return vocab, None
    else: # if using pretrained embeddings (e.g. GloVe) then we need to align the vocab with the pretrained embeddings
        aligned_vocab, aligned_embeddings = align_vocab_emb(vocab, dataset_identifier)

        return aligned_vocab, aligned_embeddings


def prep_and_numericalize_data(example, vocab):
    """ Does final data preprocessing and numericalizes the tokens in the dataset.
    Args:
        - example (dict): A dictionary containing a 'tokens' key.
        - vocab (dict): A mapping from tokens to integers.
    Returns:
        List of integers.
    """
    tokens = example['tokens']
    if not tokens: #exclude empty lines
        return None
    if tokens and tokens[-1] != '<eos>':  # append <eos> token if not there
        tokens.append('<eos>')
    token_ids = [vocab[token] for token in tokens]
    return token_ids


def preprocess_dataset(args, dataset_identifier, dataset, vocab_min_frequency=3,
                       vocab_max_tokens=30000, return_only_vocab=False):
    """Preprocesses the dataset by tokenizing and numericalizing the data.
    Args:
        - dataset (list): List of dictionaries, where each dictionary contains a 'text' key with a string.
    Returns:
        - List of dictionaries, where each dictionary contains a 'is' key with a list of integers.
        - vocab (torchtext.vocab.Vocab): The vocabulary.
    """
    # tokenize
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example['text'])}
    _logger.info("Tokenizing the data...")
    tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer})

    # get vocabulary from training data
    _logger.info("Building the vocabulary...")
    vocab, aligned_embeddings = get_vocabulary(args, dataset_identifier, tokenized_dataset, min_freq=vocab_min_frequency, max_tokens=vocab_max_tokens)
    if return_only_vocab:
        return vocab, aligned_embeddings
    # final prepocessing of data (exclude empty lines, append <eos> token) and numericalize
    _logger.info("Numericalizing the data...")
    preprocessed_dataset = tokenized_dataset.map(lambda example: {"ids": prep_and_numericalize_data(example, vocab)})
    preprocessed_dataset = preprocessed_dataset.filter(lambda example: example['ids'] is not None)

    # shuffle preprocessed_dataset training split at the sentence level
    _logger.info("Shuffling the training data at the sentence level before passing it to the batching function...")
    # make a copy for comparison
    comparison_df = preprocessed_dataset.copy()
    preprocessed_dataset['train'] = preprocessed_dataset['train'].shuffle()
    # assert that shuffling worked
    assert len(preprocessed_dataset['train']) == len(comparison_df['train'])
    assert preprocessed_dataset['train'][0]['ids'] != comparison_df['train'][0]['ids'], "Shuffling did not work"

    return preprocessed_dataset, vocab, aligned_embeddings


def concatenate_datasets(datasets): #TODO not currently used - maybe use during training?
    """Concatenate multiple datasets.
    Args:
        - datasets (list): A list of datasets to concatenate.
    Returns:
        - concatenated_dataset (dict): The concatenated dataset (for each split, a concatenated dataset is returned).
    """
    concatenated_dataset = {}
    for split in datasets[0].keys():
        concatenated_datasets_for_split = []
        for dataset in datasets:
            concatenated_datasets_for_split.append(dataset[split])
        concatenated_dataset[split] = concatenate_datasets(concatenated_datasets_for_split)
    return concatenated_dataset


def batchify(preprocessed_dataset_split, bsz):
    """
    Given a preprocessed dataset split and a batch size, returns the split as a single tensor of shape (seq_len, bsz),
    where each column represents a sequence of length seq_len.

    The function first concatenates all the sequences in the split into a single tensor, then splits the tensor into
    bsz equal-sized parts. If the length of the tensor is not evenly divisible by bsz, the extra elements are trimmed
    off.

    Args:
        - preprocessed_dataset_split (dict): The preprocessed dataset split.
        - bsz (int): The size of the batches to create.

    Returns:
        - Tensor of shape [N // batch_size, batch_size] (i.e., seq_len, bsz).

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
    """
    _logger.debug(f"Batchifying with batch size {bsz}...")
    data = [torch.tensor(example['ids'], dtype=torch.long) for example in preprocessed_dataset_split]
    data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()  # transposition (.t()) turns batch_first to False
    return data


def get_batch(source, i, bptt):
    """Get a batch of data from the source data.
     Given a source tensor, an index i, and a sequence length bptt, returns a tuple of two tensors: a tensor of input
    sequences of length seq_len starting from index i, and a tensor of target sequences of length seq_len shifted by one
    word relative to the input sequences.
    Args:
        source (Tensor): The source data, shape [full_seq_len of source data, batch_size]
        i (int): The index of the batch to get.
        bptt (int): The length of the sequence to get.
    Returns:
        A tuple of (data, target) where data is a Tensor of shape (bptt, bsz) and target is a Tensor of shape (bptt * bsz).

    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # predict the sequences shifted by one word
    target = source[i+1:i+1+seq_len].view(-1)

    # This is where data should be CUDA-fied to lessen OOM errors
    if torch.cuda.is_available():
        return data.cuda(), target.cuda()
    else:
        return data, target


def get_local_data_files(local_datapath, identifier):
    """Get the local data files for a given dataset identifier.
    Args:
        - identifier (str): The dataset identifier.
    Returns:
        - data_files (dict): A dictionary mapping the data split to the local file path.
    """
    data_path = os.path.join(local_datapath, identifier)
    data_files = {
        "train": f"{data_path}/train.txt",
        "valid": f"{data_path}/valid.txt",
        "test": f"{data_path}/test.txt",
    }
    return data_files


def build_text_dataset(args, dataset_identifier, return_only_vocab=False):
    """Build a text dataset.
    # FIXME implement shuffling for the training dataset
    # TODO note that this loads from cache, so if you change the dataset, you need to delete the cache
    Args:
        - args (argparse.Namespace): The arguments.
        - dataset_identifier (str): The dataset identifier.
        - return_only_vocab (bool): Whether to return only the vocabulary & aligned embeddings.
    Returns:
        - dataset (dict): The dataset.
        - vocab (Vocabulary): The vocabulary.
        - aligned_embeddings (np.array): The aligned embeddings.
    """
    # load data
    HF_datasets = list_datasets()
    try:
        local_datapath = os.path.abspath("../data/")
        assert os.path.exists(local_datapath)
    except:
        local_datapath = os.path.abspath("../../../../data/") #for testing contrib. environments
        assert os.path.exists(local_datapath)
    print(f"Data path: {local_datapath}")
    local_datasets = glob.glob(os.path.join(os.path.abspath(local_datapath), "*"), recursive=False)
    local_datasets = [path.split("/")[-1] for path in local_datasets if
                      os.path.isdir(path) and not "__pycache__" in path]
    _logger.info(
        f"Found {len(local_datasets)} available local text datasets and {len(HF_datasets)} available HuggingFace text datasets.")

    # TODO: add more datasets
    ID2HFPATH = {"wikitext": {"path": "wikitext", "name": "wikitext-2-raw-v1"},
                 "penntreebank": {"name": "ptb_text_only"}
                 }

    if dataset_identifier in HF_datasets:
        try:
            dataset = load_dataset(dataset_identifier)
        except:
            try:
                dataset = load_dataset(ID2HFPATH[dataset_identifier]["name"]) #penntreebank
            except:
                try:
                    dataset = load_dataset(ID2HFPATH[dataset_identifier]["path"], ID2HFPATH[dataset_identifier]["name"])
                except:
                    _logger.debug("Wrong path specification for HuggingFace datset.")

    else:
        _logger.debug("Dataset not found in HuggingFace datasets. Loading from local.")
        data_files = get_local_data_files(local_datapath, dataset_identifier)
        dataset = load_dataset('text', data_files=data_files)

    # read from cache
    cachedir = os.path.join(*[local_datapath, ".cache", dataset_identifier])
    if not os.path.exists(cachedir):
        os.makedirs(cachedir, exist_ok=True)
    try:
        with open(f"{cachedir}/vocab_gloveemb={args.glove_emb}.pkl", "rb") as f:
            vocab = pickle.load(f)
        preprocessed_dataset = torch.load(f"{cachedir}/preprocessed_dataset_gloveemb={args.glove_emb}.pt")
        aligned_embeddings = torch.load(f"{cachedir}/aligned_embeddings_gloveemb={args.glove_emb}.pt")
        _logger.info(f"Loaded preprocessed dataset, vocab and embeddings from cache at cache directory: {cachedir}.")
        if return_only_vocab:
            return vocab, aligned_embeddings
    except:
        # preprocess data
        _logger.info("Did not find preprocessed dataset. Preprocessing dataset and creating vocab & aligned embeddings.")
        split_dataset = create_train_val_test_splits(dataset)
        if return_only_vocab:
            vocab, aligned_embeddings = preprocess_dataset(args=args, dataset_identifier=dataset_identifier,
                                                           dataset=split_dataset, return_only_vocab=True)
            return vocab, aligned_embeddings
        preprocessed_dataset, vocab, aligned_embeddings = preprocess_dataset(args=args,
                                                                         dataset_identifier=dataset_identifier,
                                                                         dataset=split_dataset, return_only_vocab=False)

        # save to cache
        _logger.info(f"Saving preprocessed dataset and vocab to cache at cache directory: {cachedir}.")
        torch.save(preprocessed_dataset, f"{cachedir}/preprocessed_dataset_gloveemb={args.glove_emb}.pt")
        torch.save(aligned_embeddings, f"{cachedir}/aligned_embeddings_gloveemb={args.glove_emb}.pt")
        with open(f"{cachedir}/vocab_gloveemb={args.glove_emb}.pkl", "wb") as fout:
            pickle.dump(vocab, fout)

    # create batches
    _logger.info("Creating batches.")
    train_data = batchify(preprocessed_dataset['train'], args.batch_size)
    valid_data = batchify(preprocessed_dataset['valid'], args.eval_batch_size)
    test_data = batchify(preprocessed_dataset['test'], args.eval_batch_size)
    vocab_size = len(vocab)

    return vocab, vocab_size, train_data, valid_data, test_data, aligned_embeddings


if __name__ == "__main__":
    # test functionality
    args = HfArgumentParser(
        [TaskArguments, CTRNNModelArguments, RNNModelArguments, SharedModelArguments, TrainingArguments]
    ).parse_args()

    args.glove_emb = True
    dataset_identifier = 'wikitext'

    vocab, aligned_embeddings = build_text_dataset(args, dataset_identifier, return_only_vocab=True)
    vocab, vocab_size, train_data, valid_data, test_data, pretrained_emb = build_text_dataset(args, dataset_identifier)

    args.bptt = 35
    num_batches = train_data.shape[-1]
    for idx in tqdm(range(0, num_batches - 1, args.bptt), desc='Training: ',
                    leave=False):  # The last batch can't be
        data, targets = get_batch(train_data, idx, args.bptt)
        print(data.shape, targets.shape)
