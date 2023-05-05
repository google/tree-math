# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pickle
from typing import Union

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np
import tree_math

from tree_math._src import test_util


ArrayLike = Union[jnp.ndarray, np.ndarray, float]


@tree_math.struct
class TestStruct:
  a: ArrayLike
  b: ArrayLike


class StructsTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='Scalars', x=TestStruct(1., 2.)),
      dict(testcase_name='Arrays', x=TestStruct(np.eye(10), np.ones([3, 4, 5])))
  )
  def testFlattenUnflatten(self, x):
    leaves, structure = jax.tree_flatten(x)
    y = jax.tree_unflatten(structure, leaves)
    np.testing.assert_allclose(x.a, y.a)
    np.testing.assert_allclose(x.b, y.b)

  @parameterized.named_parameters(
      dict(testcase_name='Addition',
           x=TestStruct(1., 2.),
           y=TestStruct(3., 4.),
           operation=lambda x, y: x + y,
           expected=TestStruct(4., 6.)),
      dict(testcase_name='Subtraction',
           x=TestStruct(1., 2.),
           y=TestStruct(3., 4.),
           operation=lambda x, y: x - y,
           expected=TestStruct(-2., -2.)),
      dict(testcase_name='Multiplication',
           x=TestStruct(1., 2.),
           y=TestStruct(3., 4.),
           operation=lambda x, y: x * y,
           expected=TestStruct(3., 8.)),
      dict(testcase_name='Division',
           x=TestStruct(1., 2.),
           y=TestStruct(3., 4.),
           operation=lambda x, y: x / y,
           expected=TestStruct(1 / 3, 2 / 4)),
  )
  def testInfixOperator(self, x, y, operation, expected):
    actual = operation(x, y)
    np.testing.assert_allclose(expected.a, actual.a)
    np.testing.assert_allclose(expected.b, actual.b)

  @parameterized.named_parameters(
      dict(testcase_name='Product',
           x=TestStruct(1., 2.),
           operation=lambda x: x.a * x.b,
           expected=TestStruct(2., 1.)),
      dict(testcase_name='SquaredNorm',
           x=TestStruct(1., 2.),
           operation=lambda x: x.a**2 + x.b**2,
           expected=TestStruct(2., 4.)),
  )
  def testGrad(self, x, operation, expected):
    actual = jax.grad(operation)(x)
    np.testing.assert_allclose(expected.a, actual.a)
    np.testing.assert_allclose(expected.b, actual.b)

  @parameterized.named_parameters(
      dict(testcase_name='Sum',
           x=TestStruct(1., 2.),
           y=TestStruct(3., 4.),
           operation=lambda x, y: 3 * x + 2 * y),
      dict(testcase_name='Product',
           x=TestStruct(1., 2.),
           y=TestStruct(3., 4.),
           operation=lambda x, y: x * y),
  )
  def testJit(self, x, y, operation):
    jitted = jax.jit(operation)(x, y)
    unjitted = operation(x, y)
    np.testing.assert_allclose(jitted.a, unjitted.a)
    np.testing.assert_allclose(jitted.b, unjitted.b)

  def testPickle(self):
    struct = TestStruct(1, 2)
    restored = pickle.loads(pickle.dumps(struct))
    self.assertTreeEqual(struct, restored, check_dtypes=True)


if __name__ == '__main__':
  absltest.main()
