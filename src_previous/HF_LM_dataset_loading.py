import time
import datasets
import torchtext
import torch
import functools
import os
import sys
from tqdm import tqdm

LOAD_FROM_HF = True

def tokenize_data(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    return {'tokens': tokens}


def numericalize_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids': ids}


def build_dataset_lang(dataset_identifier, dataset_instance=None): #, args):
    """Build dataset for language modeling task."""
    # Load data
    print("Loading corpus")
    start = time.time()

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    max_length = 35 + 1  # FIXME turn into args.bptt, added +1 here to retain 35 bptt after shifting

    if LOAD_FROM_HF:
        split_names = datasets.get_dataset_split_names(dataset_identifier, dataset_instance)
        if not split_names == ["train"]:
            raise NotImplementedError

        train_data = datasets.load_dataset(dataset_identifier, dataset_instance, split=['train'])[0]
        train_data = train_data.map(tokenize_data, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})

        # create train-valid-test splits
        # 90% train, 10% test + validation
        train_testvalid = train_data.train_test_split(test_size=0.1) #FIXME turn into parameter test_size?
        train_data = train_testvalid['train']
        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5) # or 0.25?
        valid_data = test_valid['train']
        test_data = test_valid['test']

    else:
        path = os.path.abspath(f"../data/{dataset_identifier}/")
        dataset = datasets.load_dataset("text",
                               data_files={"train": os.path.join(path, "train.txt"),
                                           "valid": os.path.join(path, "valid.txt"),
                                           "test": os.path.join(path, "test.txt")})
        train_data = dataset["train"].map(tokenize_data, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
        valid_data = dataset["valid"].map(tokenize_data, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
        test_data = dataset["test"].map(tokenize_data, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})

    print("Building vocab ...")
    min_freq = 5
    special_tokens = ['<pad>', '<unk>', '<eos>']

    vocab = torchtext.vocab.build_vocab_from_iterator(train_data['tokens'],
                                                      min_freq=min_freq,
                                                      specials=special_tokens)

    unk_index = vocab['<unk>']
    eos_index = vocab['<eos>'] #FIXME what to set here?
    pad_index = vocab['<pad>'] #NOTE gets passed to model > You can keep the <pad> vector fixed to zero by
    # setting the padding_idx argument of the nn.Embedding layer to your pad id, i.e. vocab['<pad>']
    print(unk_index, eos_index, pad_index)

    vocab.set_default_index(unk_index)

    print("Tokenizing text ...")

    train_data = train_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
    valid_data = valid_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
    test_data = test_data.map(numericalize_data, fn_kwargs={'vocab': vocab})

    train_data = train_data.with_format(type='torch', columns=['ids'])
    valid_data = valid_data.with_format(type='torch', columns=['ids'])
    test_data = test_data.with_format(type='torch', columns=['ids'])

    vocab_size = len(vocab)
    words = vocab.get_itos()
    print("Vocab size", vocab_size)

    print("Getting vectors ...")
    if not dataset_identifier == "en_wiki":
        path = f"/Users/carinakauf/repos/lang-exec-multitask-rnn/data/{dataset_identifier}/pretrained_embeddings/glove" #FIXME make rel. path
        vectors = torchtext.vocab.Vectors(name=f'{path}/glove_vectors.txt',
                                          cache=f'{dataset_identifier}_glove_vectors',
                                          unk_init=torch.Tensor.normal_)
    else:
        vectors = torchtext.vocab.Glove(name = '6B', dim = 300) #FIXME should I use these for English GloVe?
    print(vectors.vectors.shape)
    pretrained_embeddings = vectors.get_vecs_by_tokens(vocab.get_itos()) # align
    print(pretrained_embeddings.shape)

    # tot_transferred = 0
    # for v in pretrained_embeddings:
    #     if not v.equal(torch.zeros(300)): #FIXME should be check if normalized not zeros
    #         tot_transferred += 1
    #
    # print(f"{tot_transferred}/{len(vocab)} tokens were successfully transferred!")

    def collate(batch, pad_index):
        batch_ids = [i['ids'] for i in batch]
        batch_ids = torch.nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=False)
        batch = {'ids': batch_ids}
        return batch


    def get_data_target(batch):
        ids = batch['ids']
        data = ids[:-1, ...].contiguous()
        target = ids[1:, ...].contiguous().flatten()
        return data, target


    batch_size = 20 #FIXME turn into args.batch_size

    collate = functools.partial(collate, pad_index=pad_index) #FIXME CK no collate for RNN?

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   #collate_fn=collate,
                                                   shuffle=True)

    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)#, collate_fn=collate)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)#, collate_fn=collate)

    for batch in tqdm(train_dataloader, desc='evaluating...', file=sys.stdout):
        data, target = get_data_target(batch)
        print(data.shape, target.shape)

    return vocab, vocab_size, train_dataloader, valid_dataloader, test_dataloader, pretrained_embeddings


if __name__ == "__main__":
    dataset_identifier = "wikipedia"
    dataset_instance = "20220301.de"
    #dataset_identifier = "de_wiki"
    build_dataset_lang(dataset_identifier, dataset_instance)















#     def train(dataloader, model, criterion, optimizer, device):
#         model.train()
#         epoch_losses = []
#         epoch_accs = []
#
#         for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
#             ids = batch['ids'].to(device)
#             label = batch['label'].to(device)
#             prediction = model(ids)
#             loss = criterion(prediction, label)
#             accuracy = get_accuracy(prediction, label)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             epoch_losses.append(loss.item())
#             epoch_accs.append(accuracy.item())
#
#         return epoch_losses, epoch_accs
#
#     def evaluate(dataloader, model, criterion, device):
#         model.eval()
#         epoch_losses = []
#         epoch_accs = []
#
#         with torch.no_grad():
#             for batch in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
#                 ids = batch['ids'].to(device)
#                 label = batch['label'].to(device)
#                 prediction = model(ids)
#                 loss = criterion(prediction, label)
#                 accuracy = get_accuracy(prediction, label)
#                 epoch_losses.append(loss.item())
#                 epoch_accs.append(accuracy.item())
#
#         return epoch_losses, epoch_accs
#
#     # print("( %.2f )" % (time.time() - start))
#     # print("Done batchifying!")
#     #
#     # return vocab, vocab_size, train_data, val_data, test_data
#
#
# n_epochs = 10
# best_valid_loss = float('inf')
#
# train_losses = []
# train_accs = []
# valid_losses = []
# valid_accs = []
#
# for epoch in range(n_epochs):
#
#     train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
#     valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)
#
#     train_losses.extend(train_loss)
#     train_accs.extend(train_acc)
#     valid_losses.extend(valid_loss)
#     valid_accs.extend(valid_acc)
#
#     epoch_train_loss = np.mean(train_loss)
#     epoch_train_acc = np.mean(train_acc)
#     epoch_valid_loss = np.mean(valid_loss)
#     epoch_valid_acc = np.mean(valid_acc)
#
#     if epoch_valid_loss < best_valid_loss:
#         best_valid_loss = epoch_valid_loss
#         torch.save(model.state_dict(), 'cnn.pt')
#
#     print(f'epoch: {epoch + 1}')
#     print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
#     print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')
#
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(1,1,1)
# ax.plot(train_losses, label='train loss')
# ax.plot(valid_losses, label='valid loss')
# plt.legend()
# ax.set_xlabel('updates')
# ax.set_ylabel('loss');
#
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(1,1,1)
# ax.plot(train_accs, label='train accuracy')
# ax.plot(valid_accs, label='valid accuracy')
# plt.legend()
# ax.set_xlabel('updates')
# ax.set_ylabel('accuracy');
#
# model.load_state_dict(torch.load('cnn.pt'))
#
# test_loss, test_acc = evaluate(test_dataloader, model, criterion, device)
#
# epoch_test_loss = np.mean(test_loss)
# epoch_test_acc = np.mean(test_acc)

# print(f'test_loss: {epoch_test_loss:.3f}, test_acc: {epoch_test_acc:.3f}')