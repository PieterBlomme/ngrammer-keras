import tensorflow as tf
from ngrammer_keras import VQNgrammer

def test_base():
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