import numpy as np
from tensor import tensor
from memory_profiler import profile

class Conv2d:
    def __init__(self, in_channels=1, out_channels=1, kernel_size=2, stride=1):
        #self.kernel = tensor.random((out_channels, in_channels, kernel_size, kernel_size))
        fan_in = in_channels * kernel_size * kernel_size
        self.kernel = tensor.he_init((out_channels, in_channels, kernel_size, kernel_size), fan_in)
        self.bias = tensor.zeros((out_channels, ))
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride

    def parameters(self):
        return [self.kernel, self.bias]

    def __call__(self, X : tensor):

        batch_size = X.shape[0]

        X_col, act_h, act_w = Conv2d.im2col(X, kernel_size=self.kernel_size, stride=self.stride)
        K_col_shape = (self.out_channels, self.kernel_size*self.kernel_size*self.in_channels)
        K_col = self.kernel.reshape(K_col_shape).transpose()
        Y_col = X_col @ K_col + self.bias
        Y = Y_col.reshape((batch_size, self.out_channels, act_h, act_w))
        return Y
        
    @classmethod
    def im2col(cls, X : tensor, kernel_size=2, stride=1):

        batch_size = X.shape[0]
        channels = X.shape[1]
        image_height = X.shape[-2] #Rows
        image_width = X.shape[-1] #Columns

        #We are assuming square kernels.
        kernel_h = kernel_size
        kernel_w = kernel_size

        act_h = (((image_height - kernel_size)//stride) + 1) #height of activation
        act_w = (((image_width - kernel_size)//stride) + 1)  #width of activation

        istrides = X.matrix.strides #strides of input tensor

        intermediate_6D = np.lib.stride_tricks.as_strided(
                            X.matrix,
                            shape=(batch_size, act_h, act_w, channels, kernel_h, kernel_w),
                            strides=(istrides[0], #No of images stride bytes
                                     istrides[-2] * stride, #Activation map Vertical stride bytes
                                     istrides[-1] * stride, #Activation map Horizontal stride bytes
                                     istrides[1], #Channel stride bytes
                                     istrides[-2], #Rective field vertical stride bytes
                                     istrides[-1]) #Receptive field horizontal stride bytes
                            )
        
        out_shape = (batch_size * act_h * act_w, channels * kernel_h * kernel_w)
        out_matrix = np.reshape(intermediate_6D, shape=out_shape)


        def _backward():
            X.grad = np.zeros_like(X.matrix, dtype = np.float32) if X.grad is None else X.grad
            
            grad_6D = out.grad.reshape(batch_size, act_h, act_w, channels, kernel_h, kernel_w)
            for i in range(kernel_h):
                for j in range(kernel_w):
                    grad_slice = grad_6D[:, :, :, :, i, j]
                    
                    grad_slice_transposed = grad_slice.transpose(0, 3, 1, 2)
                    X.grad[:, :, 
                        i : i + act_h * stride : stride, 
                        j : j + act_w * stride : stride
                    ] += grad_slice_transposed

        out = tensor(out_matrix, _children=(X, ), _operation='im2col')

        out._backward = _backward

        return out, act_h, act_w


class maxpool2D:
    def __init__(self, in_channels, pool_size = 2, stride = 1):
        self.in_channels = in_channels
        self.pool_size = pool_size
        self.stride = stride


    def __call__(self, Y: tensor):

        batch_number = Y.shape[0]
        filters = Y.shape[1]
        image_height = Y.shape[-2] #Rows
        image_width = Y.shape[-1] #Columns


        #We are assuming square kernels.
        pool_h = self.pool_size
        pool_w = self.pool_size

        pooled_h = (((image_height - pool_h)//self.stride) + 1) #height of activation
        pooled_w = (((image_width - pool_w)//self.stride) + 1)  #width of activation

        istrides = Y.matrix.strides #strides of input tensor

        intermediate_6D = np.lib.stride_tricks.as_strided(
                            Y.matrix,
                            shape=(batch_number, pooled_h, pooled_w, filters, pool_h, pool_w),
                            strides=(istrides[0], #No of images stride bytes
                                     istrides[-2] * self.stride, #Activation map Vertical stride bytes
                                     istrides[-1] * self.stride, #Activation map Horizontal stride bytes
                                     istrides[1], #Channel stride bytes
                                     istrides[-2], #Rective field vertical stride bytes
                                     istrides[-1]) #Receptive field horizontal stride bytes
                            )
        
        intermediate_6D_transposed = intermediate_6D.transpose(0, 3, 1, 2, 4, 5)
        intermediate_5D = intermediate_6D_transposed.reshape(batch_number, filters, pooled_h, pooled_w, pool_h * pool_w)

        out_matrix = np.max(intermediate_5D, axis=-1)
        IndexA_for5D = np.argmax(intermediate_5D, axis=-1)

        def _backward():
            # Recover window position (i, j) from flat index in last dim
            Y.grad = np.zeros_like(Y.matrix) if Y.grad is None else Y.grad
            flat_idx = IndexA_for5D  # (B, F, pooled_h, pooled_w)
            i = flat_idx // pool_w
            j = flat_idx % pool_w

            # Build grids for batch, filter, and pooled positions
            b_grid = np.arange(batch_number).reshape(batch_number, 1, 1, 1)
            f_grid = np.arange(filters).reshape(1, filters, 1, 1)
            ph_grid = np.arange(pooled_h).reshape(1, 1, pooled_h, 1)
            pw_grid = np.arange(pooled_w).reshape(1, 1, 1, pooled_w)

            # Broadcast all to shape (B, F, pooled_h, pooled_w)
            b_idx = np.broadcast_to(b_grid, flat_idx.shape)
            f_idx = np.broadcast_to(f_grid, flat_idx.shape)
            ph = np.broadcast_to(ph_grid, flat_idx.shape)
            pw = np.broadcast_to(pw_grid, flat_idx.shape)

            # Compute actual positions in Y where max values came from
            h_idx = self.stride * ph + i
            w_idx = self.stride * pw + j

            # Accumulate gradients using 4D indexing
            np.add.at(Y.grad, (b_idx.ravel(), f_idx.ravel(), h_idx.ravel(), w_idx.ravel()), out.grad.ravel())

        out = tensor(out_matrix, _children=(Y, ), _operation="maxpool")
        out._backward = _backward

        return out
    
class FC:
    def __init__(self, in_features, out_features):
        
        self.bias = tensor.zeros((out_features, 1))
        self.weights = tensor.he_init((out_features, in_features), in_features)
        #self.weights = tensor.random((out_features, in_features))

    def parameters(self):
        return [self.weights, self.bias]

    def __call__(self, X:tensor):
        return (self.weights @ X) + self.bias
    
class CNN:
    def __init__(self, in_channels, layers, kernels_in_layers, kernels_shape, conv_strides, pool_shape, pool_strides, FCL_weights):

        self.in_channels = (in_channels, ) + kernels_in_layers
        self.FCL_weights = FCL_weights
        self.layers = layers
        #self.conv_layers = [Conv2d(in_channels[layer], kernels_in_layers[layer], kernels_shape[layer], conv_strides[layer]) for layer in range(layers)]
        self.conv_layers = [Conv2d(self.in_channels[layer], kernels_in_layers[layer], kernels_shape[layer], conv_strides[layer]) for layer in range(layers)]
        self.pool_layers = [maxpool2D(kernels_in_layers[layer], pool_shape[layer], pool_strides[layer]) for layer in range(layers)]
        self.FC_layers = [None for _ in range(layers+1)]

    def parameters(self):
        params = []
        for layer in range(self.layers):
            params.extend(self.conv_layers[layer].parameters())
            params.extend(self.FC_layers[layer].parameters())
        params.extend(self.FC_layers[layer+1].parameters())
        return params

    def __call__(self, X:tensor):
        b = X
        for layer in range(self.layers):
            a_ = self.conv_layers[layer](b)
            a = a_.ReLU()
            b = self.pool_layers[layer](a)
        
        c:tensor = b.reshape((X.shape[0], -1)).transpose()
        if self.FC_layers[0] is None:
            self.FC_layers[0] = FC(c.shape[0], self.FCL_weights[0])

        for layer in range(self.layers+1):
            if self.FC_layers[layer] is None:
                self.FC_layers[layer] = FC(self.FCL_weights[layer-1], self.FCL_weights[layer])

            c = self.FC_layers[layer](c)
            
            if layer < self.layers:
                c = c.ReLU()
        
        out = c.transpose()
        return out
    
    @classmethod
    def cross_entropy_loss(cls, ypredicted: tensor, ytrue, batch_size):
        ytrue =  tensor(ytrue) if not isinstance(ytrue, tensor) else ytrue
        cross_entropy = -1*ypredicted.log_softmax(axis=-1)
        loss = ((ytrue * cross_entropy).sum())/batch_size
        #loss = ((ytrue * cross_entropy).sum())
        return loss

    @profile
    def fit(self, X_train, y_train, epochs = 10, lr = 0.001, batch_size = 32):
        lossT = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            perm = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                idx = perm[i:i+batch_size]
                xb = tensor(X_train[idx])             
                yb = tensor(y_train[idx]) 

                y_predicted = self(xb)
                ce_loss = CNN.cross_entropy_loss(y_predicted, yb, len(idx))
                ce_loss.backward()

                for param in self.parameters():
                    if param.grad is not None:

                        #grad_clipped = np.clip(param.grad, -1.0, 1.0)
                        #param.matrix -= lr * grad_clipped
                        param.matrix -= lr * param.grad
                        param.grad = None

                epoch_loss += ce_loss.matrix.flatten()[0]
                n_batches += 1
                # Clean up
                ce_loss.cleanBackward()
                del xb, yb, y_predicted, ce_loss
            
            avg_loss = epoch_loss / n_batches    
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            lossT.append((epoch, avg_loss))

        return lossT