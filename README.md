# MobileNetV3-Segmentation-Keras

This repository implements the semantic segmentation version of the MobileNetV3 architecture ([source](https://arxiv.org/abs/1905.02244)).

The model is implemented using Keras and TensorFlow 2.x. The MobileNetV3 backbone is based on the official model from the [keras_applications package](https://keras.io/api/applications/).

## Loading pre-trained weights from the official repository

The original implementation of the model was built using TensorFlow 1 and TF-Slim. Pre-trained weights for this model can be found [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md).

Using these weights in Keras can be done by converting the saved tensors to numpy arrays and saving them as .npy files. Those files can later be loaded as Keras weights, as long as the correct naming is used.

The script ***utils/convert_checkpoint_to_npy.py*** extracts the tensors from a checkpoint file and saves them as numpy arrays with the correct names for this Keras implementation.

Public release from XXXX-XX-XX. Contact Mission Embedded for release date.
