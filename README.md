# lang-exec-multitask

Next steps

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
