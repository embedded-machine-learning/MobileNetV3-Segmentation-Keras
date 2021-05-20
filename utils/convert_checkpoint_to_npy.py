#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert TensorFlow checkpoint file to numpy arrays.

Copyright 2021 Christian Doppler Laboratory for Embedded Machine Learning

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Built-in/Generic Imports
from pathlib import Path
import sys

# Libs
import numpy as np
import tensorflow as tf

__author__ = 'Bernhard Haas'
__copyright__ = 'Copyright 2021, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'Apache 2.0'
__version__ = '1.0.0'
__maintainer__ = 'Bernhard Haas'
__email__ = 'bernhardhaas55@gmail.com'
__status__ = 'Release'

def extract_weights(model_file:str, output_folder:Path):

    reader = tf.compat.v1.train.NewCheckpointReader(model_file)

    tensor_dict = reader.get_variable_to_shape_map()

    for key, _ in tensor_dict.items():
        
        filename = str(key)
        filename = filename.replace('/', '_')
        filename = filename.replace('MobilenetV3_', '')
        filename = filename.replace('BatchNorm', 'BN')
        if 'Momentum' in filename:
            continue

        # from TF to Keras naming
        filename = filename.replace('_weights', '_kernel')
        filename = filename.replace('_biases', '_bias')

        filepath = output_folder / Path(filename)
        np.save(str(filepath), reader.get_tensor(key))


if __name__ == '__main__':

    if len(sys.argv) != 3:
        raise ValueError('Encountered an invalid number of arguments. Needed arguments: model_file_path,output_folder_name')

    model_file = Path(sys.argv[1])

    output_folder = Path(sys.argv[2])
    output_folder.mkdir(exist_ok=True)

    extract_weights(sys.argv[1], output_folder)