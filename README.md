<img src="./n-grammer.png" width="400px"></img>

## N-Grammer - keras

Implementation of <a href="https://openreview.net/forum?id=GxjCYmQAody">N-Grammer</a>, augmenting Transformers with latent n-grams, in Keras

I am heavily indebted to both
- https://github.com/tensorflow/lingvo/blob/master/lingvo/jax/layers/ngrammer.py (Paper authors)
- https://github.com/lucidrains/n-grammer-pytorch/

## Install

```bash
$ pip install keras-ngrammer
````
NOTE that TensorFlow is a prerequisite

## Usage

```python
import tensorflow
from ngrammer_keras import VQNgrammer
vq_ngram = VQNgrammer(
    num_clusters = 1024,             # number of clusters
    dim_per_head = 32,               # dimension per head
    num_heads = 16,                  # number of heads
    ngram_vocab_size = 768 * 256,    # ngram vocab size
    ngram_emb_dim = 16,              # ngram embedding dimension
    decay = 0.999                    # exponential moving decay value
)
x = tf.random.uniform((1, 1024, 32 * 16))
vq_ngram(x)
```

## Learning Rates

Like product key memories, Ngrammer parameters need to have a higher learning rate (`1e-2` was recommended in the paper). The repository offers an easy way to generate the parameter groups.

## Citations

```bibtex
@inproceedings{thai2020using,
    title   = {N-grammer: Augmenting Transformers with latent n-grams},
    author  = {Anonymous},
    year    = {2021},
    url     = {https://openreview.net/forum?id=GxjCYmQAody}
}
```