from Tensorlib import tensor, FC
from Transformerlib import Tokenizer, WordEmbeddings, PositionalEmbeddings, LayerNormalization, MultiHeadedSelfAttention, AttentionBlock, GPT
import numpy as np
import tracemalloc
import psutil
import os
import gc
import sys

def print_top_tensors(n=10):
    objs = gc.get_objects()
    tensor_objs = []
    for o in objs:
        try:
            if hasattr(o, 'matrix') and hasattr(o, '_operation'):
                tensor_objs.append(o)
        except Exception:
            continue
    tensor_objs = sorted(tensor_objs, key=lambda x: sys.getsizeof(getattr(x, 'matrix', b'')), reverse=True)
    print(f"Top {n} tensors still in memory:")
    for t in tensor_objs[:n]:
        try:
            print(f"  {t._operation} | shape: {getattr(t, 'shape', None)} | size: {getattr(t, 'matrix', None).nbytes/1024/1024:.2f} MB | id: {id(t)}")
        except Exception:
            continue

def clear_graph(t):
    # Recursively clear computation graph to break reference cycles
    visited = set()
    def _clear(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        if hasattr(node, '_children'):
            for child in list(getattr(node, '_children', [])):
                _clear(child)
            node._children = set()
        if hasattr(node, '_backward'):
            node._backward = lambda: None
    _clear(t)

def print_mem(msg=""):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / 1024 / 1024  # Resident Set Size in MB
    current, peak = tracemalloc.get_traced_memory()
    print(f"{msg} | RSS: {rss:.2f} MB | Tracemalloc current: {current/1024/1024:.2f} MB | Peak: {peak/1024/1024:.2f} MB")

# Much larger hyperparameters for stress test
vocab_size = 10000
context_length = 128
emb_dim = 512
num_heads = 8
batch_size = 64
num_epochs = 5

np.random.seed(42)
input_tokens_np = np.random.randint(0, vocab_size, size=(batch_size, context_length))
input_tokens = tensor(input_tokens_np)
target_tokens_np = np.random.randint(0, vocab_size, size=(batch_size, context_length))
target_tokens = tensor(target_tokens_np)

gpt = GPT(vocab_size, context_length, emb_dim, num_heads)

def simple_loss(logits, targets):
    # Dummy loss: sum of logits at target indices (simulate cross-entropy)
    batch, tokens, vocab = logits.shape
    idx = (np.arange(batch)[:, None], np.arange(tokens)[None, :], targets.matrix)
    selected = logits.matrix[idx]
    return tensor(selected.sum())

def sgd_step(params, lr=1e-2):
    for p in params:
        if p.grad is not None:
            p.matrix -= lr * p.grad

tracemalloc.start()
for epoch in range(num_epochs):
    print_mem(f"Epoch {epoch+1} - Start")
    logits = gpt(input_tokens)
    print_mem(f"Epoch {epoch+1} - After forward")
    loss = simple_loss(logits, target_tokens)
    print_mem(f"Epoch {epoch+1} - After loss computation")
    loss.backward()
    print_mem(f"Epoch {epoch+1} - After backward")
    sgd_step(gpt.parameters())
    print_mem(f"Epoch {epoch+1} - After optimizer step")
    for p in gpt.parameters():
        p.grad = None

    clear_graph(loss)
    del logits, loss

    print_top_tensors(10)

    top_tensor = None
    for o in gc.get_objects():
        try:
            if hasattr(o, 'matrix') and hasattr(o, '_operation') and o._operation == '@' and getattr(o, 'shape', None) == (64, 128, 10000):
                top_tensor = o
                break
        except Exception:
            continue

    if top_tensor is not None:
        print(f"Referrers to top_tensor id {id(top_tensor)}:")
        for ref in gc.get_referrers(top_tensor):
            print(f"  Type: {type(ref)}, Repr: {repr(ref)[:200]}")
tracemalloc.stop()