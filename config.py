# This file contains configuration parameters for the GPT language model and training process.
# These parameters define the architecture and behavior of the model.
# block_size: The maximum length of input sequences (context length) that the model can handle.
# This determines the size of the positional embedding table and affects memory usage during training and inference.
block_size = 256

# n_embd: The size of the embedding dimension, which is the hidden size of the model.
# This parameter affects the capacity of the model to represent information.
# Larger values increase the model's expressive power but also increase computational cost and memory usage.
n_embd = 896

# n_head: The number of attention heads in the multi-head attention mechanism.
# This must divide n_embd evenly (n_embd % n_head == 0) to ensure proper splitting of the embedding dimension.
# More heads allow the model to attend to information from different representation subspaces simultaneously.
n_head = 14

# n_layer: The number of transformer blocks (layers) in the model.
# This determines the depth of the model, allowing for more complex feature extraction.
# Increasing this value adds more parameters and computational layers, potentially improving performance but requiring more resources.
n_layer = 16

# dropout: The dropout probability applied in various parts of the model (attention, feed-forward, etc.).
# This is a regularization technique to prevent overfitting by randomly dropping units during training.
# A value of 0.1 means 10% of units are dropped, which is a common starting point for transformer models.
dropout = 0.1