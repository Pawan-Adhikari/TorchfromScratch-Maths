import numpy as np
import itertools

class tensor:
    def __init__(self, fromArray=np.zeros((2,2)), _children = (), _operation = ''):
        fromArray = fromArray if isinstance(fromArray, np.ndarray) else np.array(fromArray)
        #assert len(fromArray.shape) == 2, "Only 2D Tensors or Scalar to 2D Supported!"
        self.matrix = fromArray
        #self.rows = fromArray.shape[0]
        #self.columns = fromArray.shape[1]
        self.shape = fromArray.shape
        self._prev = set(_children)
        self._operation = _operation
        self._backward = lambda grad: None
        self.grad = None


    def __repr__(self):
        return f"Tensor Values = {self.matrix}"
    
    @classmethod
    def zeros(cls, shape, dtype = np.float32):
        t = tensor()
        t.matrix = np.zeros(shape, dtype=dtype)
        t.shape = shape
        return t
    
    @classmethod
    def random(cls, shape, dtype = np.float32):
        t = tensor()
        t.matrix = (np.random.randn(*shape) * 0.02).astype(dtype=dtype) #0.02 = GPT-2 style
        t.shape = shape
        return t
    
    @classmethod
    def he_init(cls, shape, fan_in, dtype=np.float32):
        t = tensor()
        std = np.sqrt(2.0 / fan_in)
        t.matrix = (np.random.randn(*shape) * std).astype(dtype=dtype)
        t.shape = shape
        return t
    
    @classmethod
    def const(cls, shape, constant=1, dtype = np.float32):
        t = tensor()
        t.matrix = (np.full(shape, constant)).astype(dtype=dtype)
        t.shape = shape
        return t
    
    #Operations
    def __add__(self, other):
        other = self.checkOther(other)
        out_matrix = self.matrix + other.matrix

        def _backward(grad):
            if self.grad is None:
                self.grad = self.return_unbroadcasted(grad) #Derivation in the notes.
            else:
                self.grad += self.return_unbroadcasted(grad)

            if other.grad is None:
                other.grad = other.return_unbroadcasted(grad)
            else:
                other.grad += other.return_unbroadcasted(grad)

        out = tensor(out_matrix, (self, other), '+')
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        other = self.checkOther(other)
        return self + (-1 * other)
    
    
    def __rsub__(self, other):
        other = self.checkOther(other)
        return other + (-1 * other)
    

    def __mul__(self, other):
        other = self.checkOther(other)
        out_matrix = self.matrix * other.matrix
        def _backward(grad):
            if self.grad is None:
                self.grad = self.return_unbroadcasted(grad) * other.matrix #Derivation in the notes.
            else:
                self.grad += self.return_unbroadcasted(grad) * other.matrix

            if other.grad is None:
                other.grad = other.return_unbroadcasted(grad) * self.matrix
            else:
                other.grad += other.return_unbroadcasted(grad) * self.matrix

        out = tensor(out_matrix, (self, other), '*')
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        other = self.checkOther(other)
        return self*other
    
    '''
    batch multiplication might cause shape broadcasts.
    eg. (3,2,2) @ (1,2,3) = (3,2,3)
    this is similar to our element wise operations
    thus we should be handling this the same way we did for elementwise operations
    But, for now, we would be working in a controlled way (Even for CNNS)
    and wouldn't need this handling. Or maybe we do!?
    '''
    def __matmul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        assert other.shape[-2] == self.shape[-1], "Dimension Unsupported for @"
        out_matrix = self.matrix @ other.matrix
        def _backward(grad):
            if self.grad is None:
                self.grad = self.return_unbroadcasted(grad @ (other.matrix).swapaxes(-2,-1)) #Derivation in the notes.
            else:
                self.grad += self.return_unbroadcasted(grad @ (other.matrix).swapaxes(-2,-1))

            if other.grad is None:
                other.grad = self.return_unbroadcasted((self.matrix).swapaxes(-2,-1) @ grad)
            else:
                other.grad += self.return_unbroadcasted((self.matrix).swapaxes(-2,-1) @ grad)

        out = tensor(out_matrix, (self, other), '@')
        out._backward = _backward
        return out
    

    #I and thus we should learn at this point that to make our class compatible for ND tensors,
    #We need the matrix multiplication and Transpose backward to change
    #For higher dimensions, matmul = batch matmul where multiplication is done 
    #along each and every batches of 2D matrix. 
    #eg. If we have (2,3,3) shape tensor, it implies there are two batches of (3,3) matrices
    #similarly, (2,3,3,2) shape = 2x3 batches of 3x2 matrices.
    #matrix multiplication, (2,3,3) @ (2,3,2) = (2,3,2)
    def swap_axes(self, axis1, axis2):
        out_matrix = self.matrix.swapaxes(axis1, axis2)
        
        def _backward(grad):
            if self.grad is None:
                self.grad = (grad).swapaxes(axis1,axis2).copy() #Not in note, but can be derived similarly.
            else:
                self.grad += (grad).swapaxes(axis1,axis2)

        out = tensor(out_matrix, (self, ), 'T')
        out._backward = _backward

        return out

    def transpose(self):
        out_matrix = self.matrix.transpose()
        
        def _backward(grad):
            if self.grad is None:
                self.grad = (grad).transpose().copy() #Not in note, but can be derived similarly.
            else:
                self.grad += (grad).transpose()

            '''self.grad = np.zeros_like(self.matrix) if self.grad is None else self.grad
            self.grad += (grad).transpose() '''

        out = tensor(out_matrix, (self, ), 'T')
        out._backward = _backward

        return out
    
    def __rmatmul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        return other @ self
    
    def __pow__(self, N):
        assert isinstance(N, int | float), "Can only power up by scalars!"
        out_matrix = self.matrix ** N

        def _backward(grad):
            if self.grad is None:
                self.grad = N * (self.matrix ** (N-1)) * grad
            else:
                self.grad += N * (self.matrix ** (N-1)) * grad
        
        out = tensor(out_matrix, _children=(self, ), _operation="**")
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = self.checkOther(other)
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)
    
    def sum(self):
        out_matrix = np.array(([[self.matrix.sum()]]))

        def _backward(grad):
            if self.grad is None:
                self.grad = np.ones_like(self.matrix) * grad
            else:
                self.grad += np.ones_like(self.matrix) * grad

        out = tensor(out_matrix, _children=(self, ), _operation='sum()')
        out._backward = _backward
        return out

    def mean(self):
        N = np.prod(self.shape)
        out_matrix = np.array(([[self.matrix.sum()/(N)]]))

        def _backward(grad):
            if self.grad is None:
                self.grad = np.ones_like(self.matrix) * grad / N
            else:
                self.grad += np.ones_like(self.matrix) * grad / N

        out = tensor(out_matrix, _children=(self, ), _operation='mean()')
        out._backward = _backward
        return out
    
    def ReLU(self):
        out_matrix = np.maximum(0,self.matrix)

        def _backward(grad):
            if self.grad is None:
                self.grad = (self.matrix > 0).astype(self.matrix.dtype) * grad
            else:
                self.grad += (self.matrix > 0).astype(self.matrix.dtype) * grad

        out = tensor(out_matrix, (self, ), "ReLU")
        out._backward = _backward
        return out
    
    def reshape(self, shape):
        assert isinstance(shape, tuple), f"Can only reshape using shape tuples e.g. (3,3). Provided is {shape}"
        out_matrix = self.matrix.reshape(shape)

        def _backward(grad):
            if self.grad is None:
                self.grad = grad.reshape(self.shape)
            else:
                self.grad += grad.reshape(self.shape)


        out = tensor(out_matrix, (self, ), "reshape()")
        out._backward = _backward
        return out
    
    def flatten(self):
        out_matrix = self.matrix.reshape(-1,np.prod(self.shape[1:]))

        def _backward(grad):
            if self.grad is None:
                self.grad = grad.reshape(self.shape)
            else:
                self.grad += grad.reshape(self.shape)

        out = tensor(out_matrix, (self, ), "flatten()")
        out._backward = _backward
        return out
    
    #Helper Functions
    #def shape(self):
     #   return (self.rows, self.columns)

    def return_unbroadcasted(self, grad):  
        added_axis = []
        stretched_axis = []
        for index, (first_no, second_no) in enumerate(itertools.zip_longest(reversed(self.shape), reversed(grad.shape))):
            if first_no is None:
                added_axis.append(index)
            elif (first_no == 1) and (second_no > 1):
                stretched_axis.append(index)
        ndim = len(grad.shape)
        if stretched_axis:
            original_axes = tuple(ndim - 1 - i for i in stretched_axis)
            grad = np.sum(grad, axis=original_axes, keepdims=True)
        if added_axis:
            original_axes = tuple(ndim - 1 - i for i in added_axis)
            grad = np.sum(grad, axis=original_axes, keepdims=False)
        return grad

    def checkOther(self, other):
        if isinstance(other, int | float):
            other = tensor.const(self.shape, other)
        elif not isinstance(other, tensor):
            other = tensor(other)
        #assert other.shape == self.shape, "Operand Tensor sizes dont match"

        return other
    
    def zero_grad(self):
        self.grad = None
        
    def backward(self):
        self.grad = np.ones_like(self.matrix, dtype=np.float32)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for current in reversed(topo):
            current._backward(current.grad)

            '''current._prev = set()
            current._backward = lambda grad: None
            '''

    def cleanBackward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: build_topo(child)
                topo.append(v)
        build_topo(self)

        for t in reversed(topo):
            t.grad = None
            t._prev = ()
            t._backward = lambda grad: None

    def exp(self):
        out_matrix = np.exp(self.matrix)
        
        def _backward(grad):
            if self.grad is None:
                self.grad = out_matrix * grad 
            else:
                self.grad += out_matrix * grad  
        
        out = tensor(out_matrix, (self,), 'exp')
        out._backward = _backward
        return out
    
    def log(self, eps=1e-6):
        clipped = np.clip(self.matrix, eps, None)  
        out_matrix = np.log(clipped)
        
        def _backward(grad):
            if self.grad is None:
                self.grad = (grad / clipped) 
            else:
                self.grad += (grad / clipped) 
        
        out = tensor(out_matrix, (self,), 'log')
        out._backward = _backward
        return out
    
    def softmax(self, axis=-1):
        out_matrix = np.exp(self.matrix) / np.sum(np.exp(self.matrix), axis = axis, keepdims=True)

        def _backward(grad):
            if self.grad is None:
                self.grad = out_matrix*(grad - np.sum(out_matrix * grad, axis = axis, keepdims=True))
            else:
                self.grad += out_matrix*(grad - np.sum(out_matrix * grad, axis = axis, keepdims=True))

        out = tensor(out_matrix, (self, ), 'softmax')
        out._backward = _backward
        return out
    
    def log_softmax(self, axis=-1, eps = 1e-6):
        c = np.max(self.matrix, axis=axis, keepdims=True)

        out_matrix = (self.matrix-c) - np.log(np.sum(np.exp(self.matrix-c), axis=axis, keepdims=True)+eps)
        softmax = np.exp(self.matrix-c) / (np.sum(np.exp(self.matrix-c), axis = axis, keepdims=True)+eps)

        def _backward(grad):
            if self.grad is None:
                self.grad = grad - softmax * np.sum(grad, axis = axis, keepdims= True)
            else:
                self.grad += grad - softmax * np.sum(grad, axis = axis, keepdims= True)

        out = tensor(out_matrix, (self, ), 'log-softmax')
        out._backward = _backward
        return out

    def padding(self, pad_h, pad_w):
        np_padding = ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w))
        out_matrix = np.pad(self.matrix, np_padding, 'constant', constant_values=(0, ))

        def _backward(grad):
            h_end = -pad_h if pad_h > 0 else None
            w_end = -pad_w if pad_w > 0 else None
            
            if self.grad is None:
                self.grad = grad[:, :, pad_h:h_end, pad_w:w_end].copy()
            else:
                self.grad += grad[:, :, pad_h:h_end, pad_w:w_end]

        out = tensor(out_matrix, _children=(self, ), _operation='pad')
        out._backward = _backward
        return out
    
    __array_ufunc__ = None