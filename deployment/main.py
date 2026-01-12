import os
import numpy as np
import CNN
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms

os.environ['KERAS_BACKEND'] = 'jax'
# Download and load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(), 
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

X_train = np.stack([img.numpy() for img, _ in trainset]).astype(np.float32)
y_train = np.array([label for _, label in trainset])
X_test = np.stack([img.numpy() for img, _ in testset]).astype(np.float32)
y_test = np.array([label for _, label in testset])

Y_train = np.zeros((y_train.size, 10), dtype=np.float32)
Y_train[np.arange(y_train.size), y_train] = 1
Y_test = np.zeros((y_test.size, 10), dtype=np.float32)
Y_test[np.arange(y_test.size), y_test] = 1

model = CNN(
    in_channels=3,
    layers=2,
    kernels_in_layers= (5, 16, ),
    kernels_shape= (5, 5, ),
    conv_strides= (1, 1, ),
    pool_shape= (2, 2, ),
    pool_strides= (2, 2, ), 
    FCL_weights= (64, 32, 10) #3 Fully Connected Layers including one output layer. 
)

#We will be training on 1/10th of the total dataset.
loss = model.fit(X_train[:100], Y_train[:100], epochs=100, lr=0.08, batch_size=24)

#Plot Train Loss vs Epochs graph.
epochs, losses = zip(*loss)
plt.plot(epochs, losses)
plt.show()
