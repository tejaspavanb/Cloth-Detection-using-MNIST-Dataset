
# Cloth Detection with Fashion MNIST

This project implements a convolutional neural network (CNN) to detect and classify different types of clothing items from images in the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). The model is built using TensorFlow and Keras, and it classifies images into one of 10 clothing categories such as shirts, trousers, shoes, etc.

## Project Overview

The primary goal of this project is to create an image classification model that accurately identifies various types of clothing from grayscale 28x28 pixel images. The dataset includes 60,000 training images and 10,000 test images, each labeled with one of 10 categories.

## Dataset

The Fashion MNIST dataset consists of the following clothing categories:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Each image is a 28x28 pixel grayscale image associated with a label from one of these categories.

## Model Architecture

The model uses a Convolutional Neural Network (CNN) architecture, which includes:

- **Input layer**: 28x28 grayscale images reshaped into (28, 28, 1)
- **Convolutional layers**: Extracting spatial features from the input images
- **MaxPooling layers**: Reducing the spatial dimensions to focus on key features
- **Fully connected (Dense) layers**: Learning high-level representations for classification
- **Softmax output layer**: Outputting the probability distribution across the 10 classes

### Model Summary:
```plaintext
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320
max_pooling2d (MaxPooling2D)  (None, 13, 13, 32)        0
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
max_pooling2d_1 (MaxPooling2D) (None, 5, 5, 64)        0
conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928
flatten (Flatten)            (None, 576)               0
dense (Dense)                (None, 64)                36928
dense_1 (Dense)              (None, 10)                650
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
```

## Installation

To run this project, you'll need to install the required dependencies:

```bash
pip install tensorflow matplotlib
```

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/tejaspavanb/cloth-detection.git
   cd cloth-detection
   ```

2. Train the model by running:

   ```bash
   python train_model.py
   ```

3. The training process will output the accuracy and loss of the model. You can also visualize sample predictions:

   ```bash
   python predict_sample.py
   ```

## Results

The CNN model achieved an accuracy of **87%** on the test set. 

## Future Work

- Explore more advanced architectures like ResNet and VGG for improved accuracy.
- Implement techniques like data augmentation and regularization to avoid overfitting.
- Deploy the model using a web interface for real-time clothing detection.

## Contributing

Feel free to submit issues or pull requests if you'd like to contribute to the project!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
