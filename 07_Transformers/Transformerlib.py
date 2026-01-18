import numpy as np
from Tensorlib import tensor, FC

class Tokenizer:
    def __init__(self, mode = 'char', max_vocab_size=None):
        self.mode = mode
        self.max_vocab_size = max_vocab_size

        self.vocab = set()
        self.stoi = {}
        self.itos = {}

    def train(self, dataset):
        if self.mode == 'char':
            self.vocab = sorted(set(dataset))
            self.stoi = {c:i for i, c in enumerate(self.vocab)}
            self.itos = {i:c for i, c in enumerate(self.vocab)}
        elif self.mode == 'word':
            self.vocab = sorted(set(dataset.split()))
            self.stoi = {s:i for i, s in enumerate(self.vocab)}
            self.itos = {i:s for i, s in enumerate(self.vocab)}
        else:
            print(f"{self.mode} not supported yet! We only support 'char' and 'word'!")

    def encode(self, text):
        if len(self.stoi) == 0:
            raise ValueError("First run train method on your data!")
        
        if self.mode == 'char':
            return [self.stoi[c] for c in text]
        elif self.mode == 'word':
            return [self.stoi[c] for c in text.split()]
        else:
            raise ValueError(f"{self.mode} not supported yet! We only support 'char' and 'word'!")

    def decode(self, ints):
        if len(self.itos) == 0:
            raise ValueError("First run train method on your data!")
        
        tlists = [self.itos[i] for i in ints]
        if self.mode == 'char':
            return "".join(tlists)
        elif self.mode == 'word':
            return " ".join(tlists)
        else: 
            raise ValueError(f"{self.mode} not supported yet! We only support 'char' and 'word'!")
        
class WordEmbeddings:
    def __init__(self, vocab_size=65, embedding_dim=128):
        self.table = tensor.random(shape=(vocab_size, embedding_dim), dtype=np.float32)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def __call__(self, target): #Target = tensor of shape: Batch_size x context_size. eg. 16x10 for 10 words context
                                #out_matrix = np.zeros(shape=(target.shape[0], target.shape[1], self.embedding_dim))

        '''for batch_idx, i in enumerate(target):
            for token_idx, j in enumerate(i):
                out_matrix[batch_idx][token_idx] = self.table.matrix[j]'''
        #Need to vectorise this^^
        #how ? Fancy indexing!
        # 1)
        out_matrix = self.table.matrix[target.matrix] #Fancy indexing


        def _backward(grad):
            if self.table.grad is None:
                self.table.grad = np.zeros_like(self.table.matrix, dtype=np.float32)
            np.add.at(self.table.grad, target.matrix.ravel(), grad.reshape(-1, self.embedding_dim))

        out = tensor(out_matrix, _children=(self.table, ), _operation='Embedding_Lookup')
        out._backward = _backward

        return out

class PositionalEmbeddings:
    def __init__(self, em_dim, max_tokens, stretch=10000):
        self.em_dim = em_dim
        self.stretch = stretch
        self.max_tokens = max_tokens

        positions = np.arange(0, max_tokens, dtype=int)
        div_terms = 1/(stretch ** ((np.arange(0, em_dim, 2, dtype=np.float32))/em_dim))
        angles = np.outer(positions, div_terms)

        table = np.zeros(shape=(1, max_tokens, em_dim))
        table[:, :, 0::2] = np.sin(angles)
        table[:, :, 1::2] = np.cos(angles)

        self.table = table

    def __call__(self, embeddings: tensor):
        batch, tokens, dims = embeddings.shape
        out_matrix = self.table[:, :tokens, :]

        out = tensor(out_matrix, _children=(embeddings, ), _operation='PosEmbedding')

        #We dont need a backward pass for this operation since the positional embeddings are deterministic, not learnt.

        return out
    
class MultiHeadedSelfAttention:
    def __init__(self, num_tokens, token_dims, num_heads, causal_masking=True):
        self.head_dim = token_dims//num_heads

        #We are initialising the weights with number of features as the embeddings dimension.
        self.Wk = tensor.random((token_dims, token_dims), dtype=np.float32)
        self.Wq = tensor.random((token_dims, token_dims), dtype=np.float32)
        self.Wv = tensor.random((token_dims, token_dims), dtype=np.float32)
        self.Wo = tensor.random((token_dims, token_dims), dtype=np.float32) #This is the final normalizing weight that mixes all the learnt values
        self.num_tokens = num_tokens
        self.token_dims = token_dims
        self.num_heads = num_heads
        self.causal_masking = causal_masking

    def __call__(self, in_em: tensor, returnAttention=False):

        batch, tokens, dims = in_em.shape

        #Shape = (batch, tokens, tokens)
        K = in_em @ self.Wk 
        Q = in_em @ self.Wq
        V = in_em @ self.Wv

        #Shape = (batch, tokens, number_of_heads, dims_per_head)
        Q = Q.reshape(shape=(batch, tokens, self.num_heads, self.head_dim))
        K = K.reshape(shape=(batch, tokens, self.num_heads, self.head_dim))
        V = V.reshape(shape=(batch, tokens, self.num_heads, self.head_dim))

        #Shape = (batch, number_of_heads, tokens, dims_per_head)
        Q = Q.swap_axes(1, 2)
        K = K.swap_axes(1, 2)
        V = V.swap_axes(1, 2)

        scaling_factor = np.sqrt(self.head_dim)

        scores = ((Q @ K.swap_axes(-2, -1))/scaling_factor) #Shape = (batch, number_of_heads, tokens, tokens)


        if self.causal_masking:
            mask = np.triu(np.full(shape=(1, 1, tokens, tokens), fill_value=-np.inf, dtype=np.float32), k=1)
            scores = scores + tensor(mask)

        Attention_Probs = scores.softmax()

        Attention = Attention_Probs @ V  #Shape = (batch, number_of_heads, tokens, dims_per_head)

        output = Attention.swap_axes(1, 2) #Shape = (batch, tokens, number_of_heads, dims_per_head)

        output = output.reshape(shape=(batch, tokens, dims)) #Shape = (batch, tokens, dims)

        output = output @ self.Wo #Shape = (batch, tokens, dims)


        if returnAttention:
            return Attention_Probs.matrix, output

        return output #Shape = (batch, tokens, dims)
    
    def parameters(self):
        return [self.Wk, self.Wq, self.Wv, self.Wo]
    
class FeedForward:
    def __init__(self, emb_dims, expansion_factor = 4):
        inner_dim = emb_dims * expansion_factor #Expanding the embedding dimension, Adding hidden features and more Depth.

        self.net = [
            FC(emb_dims, inner_dim),
            FC(inner_dim, emb_dims)
        ]
        #We are fixing it to two Fully Connected Layers as per the original paper.
        #The more the layers, the richer the interpretations (reasoning)
        #But let's remember that we are training on CPU.

    def __call__(self, X):

        X = self.net[1](self.net[0](X).ReLU()) #We can hardcode since we have two layers only! Note: ReLU only in between the layers, not at last.

        return X
    
    def parameters(self):
        params = []
        for fc in self.net:
            params.extend(fc.parameters())

        return params

class LayerNormalization:
    def __init__(self, emb_dim, e = 1e-6):
        self.e = e

        self.gamma = tensor.const((emb_dim,), constant=1, dtype=np.float32)
        
        self.beta = tensor.zeros((emb_dim,), dtype=np.float32)

    def __call__ (self, X: tensor):
        X = (X - X.mean(axis=-1, keepdims=True))/((X.var(axis=-1, keepdims=True)+self.e)**(1/2))
        out = (X * self.gamma) + self.beta
        return out
        #return X
    
    def parameters(self):
        return [self.gamma, self.beta]
    
class AttentionBlock:


    def __init__(self, num_tokens, emb_dims, num_heads, decoder=True, ffn_expansion=4):
        self.msa = MultiHeadedSelfAttention(num_tokens, emb_dims, num_heads, decoder)
        self.ffn = FeedForward(emb_dims, ffn_expansion)
        self.ln1 = LayerNormalization(emb_dims)
        self.ln2 = LayerNormalization(emb_dims)

    def __call__(self, x):

        x = x + self.msa(self.ln1(x)) #Pre Normalization!!, #Shape = (batch, tokens, em_dims)
        
        x = x + self.ffn(self.ln2(x)) #Shape = (batch, tokens, em_dims)

        return x #Shape = (batch, tokens, em_dims)

    def parameters(self):
        params = []
        params.extend(self.msa.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.ln1.parameters())
        params.extend(self.ln2.parameters())

        return params
    
class GPT:
    def __init__(self, vocab_size, context_length, emb_dim, num_heads, num_layers = 1, ffn_expansion=4):
        self.token_emb =WordEmbeddings(vocab_size, emb_dim)
        self.pos_emb = PositionalEmbeddings(emb_dim, context_length)
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.blocks = [AttentionBlock(
            num_tokens=context_length, 
            emb_dims=emb_dim, 
            num_heads=num_heads, 
            decoder=True,
            ffn_expansion=ffn_expansion
        ) for _ in range(num_layers)]
        self.ln_final = LayerNormalization(emb_dim)
        self.head_weight = tensor.random((emb_dim, vocab_size), dtype=np.float32)
        self.tokenizer = None
        self.num_layers = num_layers

    def __call__(self, input_tokens): #Shape = (batch, tokens)
        word_embeddings = self.token_emb(input_tokens)
        pos_embeddings = self.pos_emb(word_embeddings)
        x = word_embeddings + pos_embeddings #Shape = (batch, tokens, em_dims)

        for layer in range(self.num_layers):
            x = self.blocks[layer](x) #Shape = (batch, tokens, em_dims)

        x = self.ln_final(x) #Shape = (batch, tokens, em_dims)
        logits = x @ self.head_weight #Shape = (batch, tokens, vocab_size)
        return logits #Shape = (batch, tokens, vocab_size)
    
    def parameters(self):
        params = []
        for block in self.blocks:
            params.extend(block.parameters())
        params.append(self.token_emb.table)
        params.extend(self.ln_final.parameters())
        params.append(self.head_weight)
        return params
    
    @staticmethod
    def _get_batch(encoded_data, batch_size, context_length):
        data_len = len(encoded_data)
        starts = np.random.randint(0, data_len - context_length, size=batch_size)
        idx = np.arange(context_length)[None, :] + starts[:, None]
        x_batch = encoded_data[idx]
        y_batch = encoded_data[idx + 1]
        return x_batch, y_batch 
    
    @staticmethod
    def _cross_entropy_loss(logits:tensor, targets:np.ndarray):
        batch, tokens, dims = logits.shape

        #logits = logits - tensor(np.max(logits.matrix, axis=-1, keepdims=True))
        log_probs = logits.log_softmax()

        batch_idx = np.arange(batch, dtype=np.intp).reshape(-1,1)
        token_idx = np.arange(tokens, dtype=np.intp).reshape(1,-1)

        nll = -1 * log_probs[batch_idx, token_idx, targets]
        #print(f"nll shape: {nll.shape}")
        return nll.mean()
    
    def save_model(self, filename="Shakespearian_model.npz"):
        weights_dict = {}
        for i, param in enumerate(self.parameters()):
            weights_dict[f'param_{i}'] = param.matrix
        
        np.savez(filename, **weights_dict)
        print(f"Model saved to {filename}")

    def load_model(self, data, filename="Shakespearian_model_2002.npz", tokenization_mode='char'):
        loaded_data = np.load(filename)
        self.tokenizer = Tokenizer(tokenization_mode)
        self.tokenizer.train(data)
        for i, param in enumerate(self.parameters()):
            if param.shape == loaded_data[f'param_{i}'].shape:
                param.matrix= loaded_data[f'param_{i}']
            else:
                raise ValueError(f"Shape Mismatch at index {i}: Model expects {param.shape}, loaded {loaded_data[f'param_{i}'].shape}")

    @staticmethod
    def adam_step(params, adam_state, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            state = adam_state[i]
            if not state:
                state['m'] = np.zeros_like(p.grad)
                state['v'] = np.zeros_like(p.grad)
                state['t'] = 0
            state['t'] += 1
            state['m'] = beta1 * state['m'] + (1 - beta1) * p.grad
            state['v'] = beta2 * state['v'] + (1 - beta2) * (p.grad ** 2)
            m_hat = state['m'] / (1 - beta1 ** state['t'])
            v_hat = state['v'] / (1 - beta2 ** state['t'])
            p.matrix -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    def train(self, data, tokenization_mode = 'char', epochs=10, learning_rate=0.01, batch=16):

        #We first have to train the tokenizer
        self.tokenizer = Tokenizer(tokenization_mode)
        self.tokenizer.train(data)

        #Then we tokenize the entire dataset
        Encoded_Data = np.array(self.tokenizer.encode(data))

        #We should be keeping track of loss per epoch too.
        losses = []

        #Let us store the parameters too.
        params = self.parameters()

        epoch_gap = 0
        adam_state = [{} for _ in params]
        data_len = len(Encoded_Data)
        num_batches = (data_len - self.context_length) // batch

        for epoch in range(epochs):
            #We have to create batches of encoded data as per the specified context_length.
            #x_b, y_b = GPT._get_batch(Encoded_Data, batch, self.context_length) #Shapes: x_b: (batch * tokens), y_b: (batch * tokens)
            starts = np.arange(data_len - self.context_length)
            np.random.shuffle(starts)
            for i in range(0, len(starts), batch):

                #We are essentially batching such that the entire dataset is covered, with stride = 1, per epoch.
                batch_starts = starts[i:i+batch]
                idx = np.arange(self.context_length)[None, :] + batch_starts[:, None]
                x_b = Encoded_Data[idx]
                y_b = Encoded_Data[idx + 1]
                x_b = tensor(x_b)

                logits = self(x_b) #Shape: (batch * tokens * vocab_size), Contains the raw logits for each (class) token in vocabulary.

                loss = GPT._cross_entropy_loss(logits, y_b) #Shape: (1)

                loss.backward()

                print(f"Epoch: {epoch} | Loss: {loss}")

                losses.append((epoch, loss.matrix.item()))

                #This is the normal SGD optimization
                '''for param in params:
                    param.matrix -= learning_rate*param.grad
                    param.zero_grad()
                    param.cleanBackward()'''
                

                #Optimizing the parameters with Adam Optimizer.
                GPT.adam_step(params, adam_state, lr=learning_rate)


                #Zero Grad-ing and cleaning the computational graph.
                for param in params:
                    param.zero_grad()
                    param.cleanBackward()

                #Clearning the loss computational graph and also deleting the unused tensors
                loss.cleanBackward()
                del loss, logits, x_b, y_b

                #Both learning rate scheduler and model saver.
                if epoch_gap > 1000:
                    #learning_rate *= 0.5
                    self.save_model(f"Shakespearian_model_{epoch}.npz")
                    epoch_gap = 0
                    #print("Halved the LR!")
                epoch_gap += 1

            return losses
    
    def generate(self, start_token, max_tokens_generated, temperature=1.2):
        if isinstance(start_token, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer not trained or loaded.")
            context = self.tokenizer.encode(start_token)
        elif isinstance(start_token, (list, np.ndarray)):
            context = list(start_token)
        else:
            raise ValueError("start_token must be str, list, or np.ndarray")

        for _ in range(max_tokens_generated):   
            input_tokens = context[-self.context_length:] if len(context) > self.context_length else context #Return the last context length-ed slice
            x = tensor(np.array([input_tokens]))  #shape: (1, context_length)
            logits = self(x)  #shape: (1, context_length, vocab_size)
            last_logits = logits.matrix[0, -1]  #shape: (vocab_size,) #Picking the last row of output matrix
            probs = np.exp(last_logits / temperature) #shape: (vocab_size,)
            probs /= np.sum(probs) #shape: (vocab_size,) #Softmaxing to get probability distribution
            next_token = int(np.random.choice(len(probs), p=probs)) # Returns a random index based on probability distribution p = probs
            context.append(next_token) 

        return self.tokenizer.decode(context)