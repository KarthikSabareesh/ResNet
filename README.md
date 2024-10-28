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

## Advantages of ResNet

* Ease of Training Deep Networks: ResNet's residual connections allow models to go very deep (e.g., 50, 101, or 152 layers) without suffering from issues like vanishing gradients or high training error, which are common in traditional deep networks. This allows ResNet models to capture complex patterns in data more effectively.

* Reduced Training Error: The skip connections in ResNet reduce the problem of degradation, where the training error increases with added layers. By bypassing some layers, ResNet can learn only the necessary residuals, which improves training efficiency and reduces error rates even as the network gets deeper.

* Better Generalization: ResNet models generalize well on unseen data due to their ability to capture hierarchical features at different depths. The depth helps ResNet capture complex and abstract features, improving performance on tasks like image classification, object detection, and segmentation.

* Transfer Learning: ResNet has become a popular architecture for transfer learning because of its generalization and depth. Pre-trained ResNet models on ImageNet, for example, can be fine-tuned for a wide variety of custom applications, significantly reducing the time and data needed for new tasks.

* Modularity and Flexibility: ResNet serves as a foundation for many variations and derivative models (e.g., ResNeXt, SE-ResNet), allowing customization based on the task at hand, from lightweight versions for mobile devices to deeper versions for high-performance applications.

* Efficient Computation: Despite their depth, ResNet models are computationally efficient due to residual connections, which make it easier for gradients to propagate, thereby reducing the need for frequent weight updates in all layers.
* 
## Applications of ResNet

* Image Classification: ResNet has set benchmarks in image classification, achieving top performances on datasets like ImageNet. Its ability to learn deep hierarchical representations makes it ideal for complex classification tasks in various fields like medical imaging, security, and self-driving cars.

* Object Detection and Localization: ResNet is frequently used in object detection models, such as Faster R-CNN and YOLO, as the backbone for feature extraction. The deep layers help in detecting and localizing objects at different scales and complexities within an image.

* Image Segmentation: For semantic and instance segmentation tasks, such as in U-Net or Mask R-CNN architectures, ResNet serves as the backbone for segmenting images into meaningful parts. Applications include autonomous driving (lane detection, object segmentation), medical imaging (tumor segmentation), and satellite image analysis.

* Facial Recognition: ResNet architectures are employed in facial recognition systems, where deep feature extraction is necessary to identify unique facial features across various conditions (lighting, pose, expressions).

* Transfer Learning for Custom Applications: Pre-trained ResNet models are widely used in industry for transfer learning, where they serve as a starting point for various custom applications






