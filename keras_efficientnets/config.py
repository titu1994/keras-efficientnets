# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.
[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

import re


class BlockArgs(object):

    def __init__(self, input_filters=None,
                 output_filters=None,
                 kernel_size=None,
                 strides=None,
                 num_repeat=None,
                 se_ratio=None,
                 expand_ratio=None,
                 identity_skip=True):

        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel_size=kernel_size
        self.strides = strides
        self.num_repeat = num_repeat
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.identity_skip = identity_skip

    def decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        self.input_filters = int(options['i'])
        self.output_filters = int(options['o'])
        self.kernel_size = int(options['k'])
        self.num_repeat = int(options['r'])
        self.identity_skip = ('noskip' not in block_string)
        self.se_ratio = float(options['se']) if 'se' in options else None
        self.expand_ratio = int(options['e'])
        self.strides = [int(options['s'][0]), int(options['s'][1])]

        return self

    def encode_block_string(self, block):
        """Encodes a block to a string.

        Encoding Schema:
        "rX_kX_sXX_eX_iX_oX{_se0.XX}{_noskip}"
         - X is replaced by a any number ranging from 0-9
         - {} encapsulates optional arguments

        To deserialize an encoded block string, use
        the class method :
        ```python
        BlockArgs.from_block_string(block_string)
        ```
        """
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]

        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)

        if block.identity_skip is False:
            args.append('noskip')

        return '_'.join(args)

    @classmethod
    def from_block_string(cls, block_string):
        """
        Encoding Schema:
        "rX_kX_sXX_eX_iX_oX{_se0.XX}{_noskip}"
         - X is replaced by a any number ranging from 0-9
         - {} encapsulates optional arguments

        To deserialize an encoded block string, use
        the class method :
        ```python
        BlockArgs.from_block_string(block_string)
        ```

        Returns:
            BlockArgs object initialized with the block
            string args.
        """
        block = cls()
        return block.decode_block_string(block_string)


# Default list of blocks for EfficientNets
def get_default_block_list():
    blocks_args = ['r1_k3_s11_e1_i32_o16_se0.25',
                 'r2_k3_s22_e6_i16_o24_se0.25',
                 'r2_k5_s22_e6_i24_o40_se0.25',
                 'r3_k3_s22_e6_i40_o80_se0.25',
                 'r3_k5_s11_e6_i80_o112_se0.25',
                 'r4_k5_s22_e6_i112_o192_se0.25',
                 'r1_k3_s11_e6_i192_o320_se0.25']
    
    DEFAULT_BLOCK_LIST = [BlockArgs.from_block_string(s)
                          for s in blocks_args]

    return DEFAULT_BLOCK_LIST
