import pickle
import os
import numpy as np
from tqdm import tqdm
import torch
import torchtext
from collections import OrderedDict
import re


def get_embedding_matrix(dataset_identifier):
    """Load pretrained embeddings from file and return them as a dictionary.
    Args:
        dataset_identifier: Identifier of the dataset for which the embeddings should be loaded.
    Returns:
        embeddings_matrix: Dictionary with tokens as keys and embedding vectors as values.
    """
    assert dataset_identifier == "de_wiki", "Only German pretrained embeddings are supported besides English at the moment."
    embeddings_matrix = {}
    glove_vector_path = f"../data/{dataset_identifier}/pretrained_embeddings/glove/vectors.txt"
    print(f"Loading pretrained embeddings from {glove_vector_path}...")
    glove_vector_path = os.path.abspath(glove_vector_path)
    f = open(glove_vector_path, encoding="utf8")
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_matrix[word] = coefs
    f.close()
    print(f"Found {len(embeddings_matrix)} vectors.")
    return embeddings_matrix


def get_vecs_by_tokens(vocab_from_train, glove_vectors):
    """Look up embedding vectors of tokens.
    Source: https://pytorch.org/text/stable/_modules/torchtext/vocab/vectors.html#Vectors.get_vecs_by_tokens
    Args:
        - vocab_from_train (object): a token or a list of tokens. if `vocab_from_train` is a string,
            returns a 1-D tensor of shape `self.dim`; if `vocab_from_train` is a list of strings,
            returns a 2-D tensor of shape=(len(tokens), self.dim).
        - glove_vectors (object): matrix of pretrained embeddings
    Returns:
        - vecs (Tensor): a tensor of shape=(len(tokens), self.dim), where tokens are the tokens from training that
            have a corresponding vector in the glove_vectors pretrained embeddings.
        - aligned_vocab (torchtext.vocab.Vocab): object with the tokens in the same order as the vectors in `vecs`
    """
    to_reduce = False
    if not isinstance(vocab_from_train, list):
        print("vocab_tokens is not a list but a token, returning only a single vector.")
        vocab_from_train = [vocab_from_train]
        to_reduce = True

    try:
        # keep only vocab tokens that have a corresponding vector (used for German embeddings from file)
        glove_vocab_tokens = glove_vectors.keys()
       # remove eos and unk tokens for alignment
        glove_vocab_tokens = [token for token in glove_vocab_tokens if token not in ['<eos>', '<unk>']]
        tokens_embeddings = [(token, torch.from_numpy(glove_vectors[token]).float()) for token in vocab_from_train if
                                token in glove_vocab_tokens]
    except:
        # keep only vocab tokens that have a corresponding vector (used for torchtext.Glove)
        glove_vocab_tokens = glove_vectors.stoi.keys()
        glove_vocab_tokens = [token for token in glove_vocab_tokens if token not in ['<eos>', '<unk>']]
        tokens_embeddings = [(token, glove_vectors[token]) for token in vocab_from_train if token in glove_vocab_tokens]

    tokens = [token for token, _ in tokens_embeddings]
    assert all(x in glove_vocab_tokens for x in tokens), "Some tokens are not in the pretrained embeddings vocabulary."

    # add embeddings for special tokens if they are not in the pretrained embeddings
    embeddings = [embedding for _, embedding in tokens_embeddings]
    if '<eos>' not in tokens:
        tokens = ['<eos>'] + tokens
        embeddings = [torch.ones(300)] + embeddings
    if '<unk>'not in tokens:
        tokens = ['<unk>'] + tokens
        embeddings = [torch.zeros(300)] + embeddings

    # stack the embeddings along the first dimension
    vecs = torch.stack(embeddings)

    # build a new vocab object with the tokens in the same order as the vectors in `vecs`
    special_tokens = ['<unk>', '<eos>']
    aligned_vocab = torchtext.vocab.vocab(OrderedDict([(token, 10) for token in tokens]), specials=special_tokens)
    print(f"Setting '{aligned_vocab['<unk>']}' as the default index")
    aligned_vocab.set_default_index(aligned_vocab['<unk>'])
    print(f"Found {len(aligned_vocab)} tokens from the {len(vocab_from_train)}-word "
          f"vocabulary created from the training dataset in the pretrained embeddings.")

    # assert that the alignment is correct
    print("Checking that the alignment is correct for the first 100 vectors...")
    for i, (token, embedding) in enumerate(list(zip(tokens, embeddings))[:100]):
        if token in aligned_vocab.get_itos():
            aligned_idx = aligned_vocab.get_stoi()[token]
            assert torch.all(vecs[aligned_idx] == embedding), f"Vectors for token '{token}' are not aligned!"
    print("Successfully aligned the pretrained embeddings with the vocabulary created from the training dataset.")

    if to_reduce:
        return vecs[0], aligned_vocab
    else:
        return vecs, aligned_vocab


def align_vocab_emb(vocab, dataset_identifier):
    """Align the pretrained embeddings with the vocabulary created from the training dataset.
    Args:
        vocab: torchtext.vocab.Vocab object
        dataset_identifier: str, e.g. "de_wiki"
    Returns:
        aligned_vocab: torchtext.vocab.Vocab object
        aligned_embeddings: torch.Tensor, shape=(len(aligned_vocab), 300)
    """
    if dataset_identifier == "de_wiki":
        print("Loading pretrained GloVe embeddings for German...")
        glove_pretrained_vectors = get_embedding_matrix(dataset_identifier)
    elif dataset_identifier == "en_wiki":
        print("Loading pretrained GloVe embeddings for English...")
        glove_pretrained_vectors = torchtext.vocab.GloVe(name='6B', dim=300)
    elif re.match(r"\w{2}_wiki", dataset_identifier):
        raise NotImplementedError("Pretrained embeddings for languages other than German and English are not supported yet.")
    else:
        print("Loading pretrained GloVe embeddings for English...")
        glove_pretrained_vectors = torchtext.vocab.GloVe(name='6B', dim=300)
    # align glove embeddings with vocab
    aligned_embeddings, aligned_vocab = get_vecs_by_tokens(vocab.get_itos(), glove_pretrained_vectors)

    return aligned_vocab, aligned_embeddings