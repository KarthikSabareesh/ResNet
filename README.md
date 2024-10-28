# ResNet
My implementation of ResNet with my custom recreation of some basic Pytorch layers.

## Why ResNet?
ResNet captures features better and prevents the effect of vanishing gradients affecting model training via skip connections. This means we get better results and faster training.
ResNet also modularises well, allowing us to create variants that are specific to certain tasks.

## What are vanishing gradients?
In deep networks, as backpropagation passes through many layers, gradients tend to get very small (or even vanish), making it difficult for the early layers to learn effectively. This leads to poor learning by our DNN models. This phenomenon is called the "vanishing gradients problem."

## What are skip connections?
Residual connections (or skip connections) allow layers to learn modifications to the identity function rather than needing to learn the full transformation. This simplifies learning and encourages layers to focus on adjustments rather than re-learning existing knowledge, making the network more efficient. They allow the output from one layer to bypass (or "skip") certain layers and add directly to the output of a deeper layer in the network. Residual connections allow gradients to be passed back through the network more directly, avoiding excessive diminishing. This keeps gradients more stable and helps the network learn even when it’s very deep. 

## Mathematical representation of Skip Connections

Input x passes through a series of convolutional layers to produce an intermediate output F(x).

This output F(x) is then added to the original input x (the residual connection), resulting in H(x)=F(x) + x.

This result H(x) is then passed to the next layer.

![image](https://github.com/user-attachments/assets/de5bd833-3baf-4074-a85a-04d18bc3d590)

## ResNet architecture


![image](https://github.com/user-attachments/assets/4f9881a3-b501-49ee-a00e-5cbbf0e50355)






