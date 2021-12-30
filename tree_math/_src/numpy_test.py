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

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import tree_math as tm
from tree_math._src import test_util
import tree_math.numpy as tnp


# pylint: disable=g-complex-comprehension


class NumpyTest(test_util.TestCase):

  def test_matmul(self):
    vector1 = tm.Vector({"a": jnp.array([1, 2, 3])})
    vector2 = tm.Vector({"a": jnp.array([4, 5, 6])})
    expected = 1*4 + 2*5 + 3 * 6

    actual = tnp.dot(vector1, vector2)
    self.assertEqual(actual, expected)

    actual = tnp.matmul(vector1, vector2)
    self.assertEqual(actual, expected)

  def test_where(self):
    condition = tm.Vector({"a": jnp.array([True, False])})
    x = tm.Vector({"a": jnp.array([1, 2])})
    y = 3
    expected = tm.Vector({"a": jnp.array([1, 3])})
    actual = tnp.where(condition, x, y)
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_where_all_scalars(self):
    expected = 1
    actual = tnp.where(True, 1, 2)
    self.assertTreeEqual(actual, expected, check_dtypes=False)
    with self.assertRaisesRegex(
        TypeError, "non-tree_math.VectorMixin argument is not a scalar",
    ):
      tnp.where(True, jnp.array([1, 2]), 3)

  def test_zeros_like(self):
    x = jnp.array([1, 2])
    expected = tm.Vector({"a": jnp.zeros_like(x)})
    actual = tnp.zeros_like(tm.Vector({"a": x}))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_ones_like(self):
    x = jnp.array([1, 2])
    expected = tm.Vector({"a": jnp.ones_like(x)})
    actual = tnp.ones_like(tm.Vector({"a": x}))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_full_like(self):
    x = jnp.array([1, 2])
    expected = tm.Vector({"a": jnp.full_like(x, 3)})
    actual = tnp.full_like(tm.Vector({"a": x}), 3)
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  @parameterized.named_parameters(*(
      {"testcase_name": name, "func_name": name}
      for name in ["maximum", "minimum"]
  ))
  def test_binary_ufuncs(self, func_name):
    jnp_func = getattr(jnp, func_name)
    tree_func = getattr(tnp, func_name)
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 3, 2])
    expected = tm.Vector({"a": jnp_func(x, y)})
    actual = tree_func(tm.Vector({"a": x}), tm.Vector({"a": y}))
    self.assertTreeEqual(actual, expected, check_dtypes=True)


if __name__ == "__main__":
  absltest.main()
