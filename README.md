# lang-exec-multitask

Openmind project folder: `/nese/mit/group/evlab/u/ckauf/lang-exec-multitask/`

Supported environment: `/om2/user/ckauf/anaconda39/envs/multitask_rnn310`

## Run example

Running the [CTRNN]([class CTRNN(nn.Module):](https://github.com/neurogym/ngym_usage/blob/master/yang19/models.py#L6)) network on
* the non-verbal executive function task set by [Yang et al., 2020](https://www.nature.com/articles/s41593-018-0310-2)
* language modeling for
  * English: [Wikitext corpus](https://huggingface.co/datasets/wikitext)
  * German Wikitext corpus, created via `/nese/mit/group/evlab/u/ckauf/lang-exec-multitask/data/prep_wiki_corpus_pipeline.sh`
* verbal executive function tasks delay match to sample with words for
  * English (`contrib.DelayMatchSampleWord-v0`)
  * German (`contrib.DelayMatchSampleWordGerman-v0`)

```
PYTHONPATH=/nese/mit/group/evlab/u/ckauf/lang-exec-multitask/:/nese/mit/group/evlab/u/ckauf/lang-exec-multitask/neurogym/neurogym/:/nese/mit/group/evlab/u/ckauf/lang-exec-multitask/src/ 

python3 runner.py --CTRNN \
                --tasks yang19 wikitext de_wiki contrib.DelayMatchSampleWord-v0 contrib.DelayMatchSampleWordGerman-v0 \
                --epochs 20 \
                --hidden_size 300 \
                --glove_emb \
                --seed 1 \
                --log_level DEBUG \
                --TODO train+analyze \
                --batch_size 128 \
                --eval_batch_size 128 \
                --cuda
```

## Carina's next steps

1. Finding best setup for training only the language model
* Parameters to vary: dimensionality (600/1000) > higher because you need to encode temporal dynamics, not just a word!
* Train maybe even only Adam (or AdamW)
* lr is probably not that important, use 1e-3, 1e-4
* 0.1 weight decay is high! try 0.005, 0.001, 0.0001
* train for 10 epochs
* validate more at the beginning of training (like every 1000 steps)

2. Using the best parameter setting to train the multitask model
* Use dimensionality 256 * best_lang_dimensionality (but also vary here)
* Performance on Yang task set is the success criterion (should be almost as good as the Yang model)
