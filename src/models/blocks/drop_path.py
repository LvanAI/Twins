# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""DropPath module."""

from mindspore import nn
from mindspore import Tensor, dtype
from mindspore.ops import operations as P
import mindspore.nn.probability.distribution as msd


class DropPath(nn.Cell):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, keep_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - keep_prob
        seed = min(seed, 0)
        self.rand = P.UniformReal(seed=seed)
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor

        return x


class DropConnect(nn.Cell):
    """
    DropConnect function.

    Args:
        keep_prob (int): Drop rate of the MBConv Block. Default: 0.

    Return:
        Tensor

    Example:
        >>> x = Tensor(((20, 16), (50, 50)), mindspore.float32)
        >>> DropConnect(0.8)(x)
    """
    def __init__(self,
                 keep_prob: float = 0.
                 ):
        super(DropConnect, self).__init__()
        self.drop_rate = keep_prob
        self.bernoulli = msd.Bernoulli(probs=0.8, dtype=dtype.int32)

    def construct(self, x: Tensor):
        if not self.training or self.drop_rate == 0.:
            return x
        return x * self.bernoulli.sample((x.shape[0],) + (1,) * (x.ndim-1))
