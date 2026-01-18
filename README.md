# MLMaths-TorchfromScratch

This repository contains code, notebooks, and resources for understanding and implementing core mathematical concepts in Deep Learning. It is basically a custom Deep Learning engine similar to pytorch, but on CPU. It is organized into modules covering traditional machine learning, perceptrons, autograd, MLPs, tensors, CNNs, transformers, and deployment tests.
Most of the operations and their upstream gradients are derived by hand, using pen and paper. I've kept almost all operations vectorised and avoided naive loops. Memory leaks and performance tweaks were also checked. Since we are constrained to CPU training and NumPy has an additional overhead, both the performance and memory efficiency takes some toll. Besides the tensor stuff, we also have the early scalar based implementations which are excellent pedagogical tools to understand internals of deep learning.


## Structure
- **00_Traditional_ML/**: Linear regression and classic ML models.
- **01_Perceptrons_Rosenblatt/**: Perceptron algorithm and class.
- **02_Scalar_Autograd/**: Micrograd autograd implementation.
- **03_Scalar_MLP/**: Scalar MLP implementation.
- **04_Tensor_Class/**: Tensor class and related notebooks.
- **05_Tensor_MLP/**: Fully connected network / Multi Layer Perceptron and housing dataset on tensor autograd.
- **06_CNN_Vectorised/**: CNN models, training, and notes. (Trained on MNIST and CFIAR-10)
- **07_Transformers/**: Transformer model, tensor library, and handwritten notes. (Trained on Shakespearen dataset)
- **99_Deployment_Tests/**: Deployment scripts and tests.


## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/Pawan-Adhikari/MLMaths-TorchfromScratch.git
   ```
3. Explore the notebooks and Python scripts in each module.


## Future Work
- Custom CUDA kernels
- Further Performance Optimisations
- BPPT, RNNs, ViTs etc.
- Although in a different realm, we can try surrogate gradient modeling and SNNs. 


## Author
[Pawan Adhikari](https://github.com/Pawan-Adhikari)
