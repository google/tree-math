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

import operator

from absl.testing import absltest
from absl.testing import parameterized
from jax import tree_util
import jax.numpy as jnp
import numpy as np
import tree_math as tm
from tree_math._src import test_util

# pylint: disable=g-complex-comprehension


class VectorTest(test_util.TestCase):

  def test_vector(self):
    tree = {"a": 0, "b": jnp.array([1, 2], dtype=jnp.int32)}
    vector = tm.Vector(tree)
    self.assertEqual(vector.size, 3)
    self.assertLen(vector, 3)
    self.assertEqual(vector.shape, (3,))
    self.assertEqual(vector.ndim, 1)
    self.assertEqual(vector.dtype, jnp.int32)
    self.assertEqual(repr(tm.Vector({"a": 1})),
                     "tree_math.Vector({'a': 1})")
    self.assertTreeEqual(tree_util.tree_leaves(tree),
                         tree_util.tree_leaves(vector), check_dtypes=True)
    vector2 = tree_util.tree_map(lambda x: x, vector)
    self.assertTreeEqual(vector, vector2, check_dtypes=True)

  @parameterized.named_parameters(*(
      {"testcase_name": op.__name__, "op": op}
      for op in [operator.pos, operator.neg, abs, operator.invert]
  ))
  def test_unary_math(self, op):
    tree = {"a": 1, "b": -jnp.array([2, 3])}
    expected = tm.Vector(tree_util.tree_map(op, tree))
    actual = op(tm.Vector(tree))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_arithmetic_with_scalar(self):
    vector = tm.Vector({"a": 0, "b": jnp.array([1, 2])})
    expected = tm.Vector({"a": 1, "b": jnp.array([2, 3])})
    self.assertTreeEqual(vector + 1, expected, check_dtypes=True)
    self.assertTreeEqual(1 + vector, expected, check_dtypes=True)
    with self.assertRaisesRegex(
        TypeError, "non-tree_math.VectorMixin argument is not a scalar",
    ):
      vector + jnp.ones((3,))  # pylint: disable=expression-not-assigned

  @parameterized.named_parameters(*(
      {"testcase_name": op.__name__, "op": op}
      for op in [
          operator.add,
          operator.sub,
          operator.mul,
          operator.truediv,
          operator.floordiv,
          operator.mod,
      ]
  ))
  def test_binary_arithmetic(self, op):
    rng = np.random.default_rng(0)
    tree1 = {"a": rng.standard_normal(dtype=np.float32),
             "b": rng.standard_normal((2, 3), dtype=np.float32)}
    tree2 = {"a": rng.standard_normal(dtype=np.float32),
             "b": rng.standard_normal((2, 3), dtype=np.float32)}
    expected = tm.Vector(tree_util.tree_map(op, tree1, tree2))
    actual = op(tm.Vector(tree1), tm.Vector(tree2))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_pow(self):
    expected = tm.Vector({"a": 2 ** 3})
    actual = tm.Vector({"a": 2}) ** tm.Vector({"a": 3})
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_divmod(self):
    x, y = divmod(jnp.arange(5), 2)
    expected = tm.Vector({"a": x}), tm.Vector({"a": y})
    actual = divmod(tm.Vector({"a": jnp.arange(5)}), 2)
    self.assertTreeEqual(actual, expected, check_dtypes=True)

    x, y = divmod(5, jnp.arange(5))
    expected = tm.Vector({"a": x}), tm.Vector({"a": y})
    actual = divmod(5, tm.Vector({"a": jnp.arange(5)}))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_matmul_scalars(self):
    actual = tm.Vector(1.0) @ tm.Vector(2.0)
    expected = 2.0
    self.assertAllClose(actual, expected)

  def test_matmul(self):
    rng = np.random.default_rng(0)
    tree1 = {"a": rng.standard_normal(dtype=np.float32),
             "b": rng.standard_normal((2, 3), dtype=np.float32)}
    tree2 = {"a": rng.standard_normal(dtype=np.float32),
             "b": rng.standard_normal((2, 3), dtype=np.float32)}

    expected = tree1["a"] * tree2["a"] + tree1["b"].ravel() @ tree2["b"].ravel()

    vector1 = tm.Vector(tree1)
    vector2 = tm.Vector(tree2)

    actual = vector1 @ vector2
    self.assertAllClose(actual, expected)

    actual = vector1.dot(vector2)
    self.assertAllClose(actual, expected)

    with self.assertRaisesRegex(
        TypeError,
        "matmul arguments must both be tree_math.VectorMixin objects",
    ):
      vector1 @ jnp.ones((7,))  # pylint: disable=expression-not-assigned

  # TODO(shoyer): test comparisons and bitwise ops

  def test_conj(self):
    vector = tm.Vector({"a": jnp.array([1, 1j])})
    actual = vector.conj()
    expected = tm.Vector({"a": jnp.array([1, -1j])})
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_real_imag(self):
    vector = tm.Vector({"a": jnp.array([1, 1j])})
    real_part = tm.Vector({"a": jnp.array([1.0, 0.0])})
    imag_part = tm.Vector({"a": jnp.array([0.0, 1.0])})
    self.assertTreeEqual(vector.real, real_part, check_dtypes=True)
    self.assertTreeEqual(vector.imag, imag_part, check_dtypes=True)

  def test_sum_mean_min_max(self):
    vector = tm.Vector({"a": 1, "b": jnp.array([2, 3, 4])})
    self.assertTreeEqual(vector.sum(), 10, check_dtypes=False)
    self.assertTreeEqual(vector.min(), 1, check_dtypes=False)
    self.assertTreeEqual(vector.max(), 4, check_dtypes=False)

  def test_custom_class(self):

    @tree_util.register_pytree_node_class
    class CustomVector(tm.VectorMixin):

      def __init__(self, a: int, b: float):
        self.a = a
        self.b = b

      def tree_flatten(self):
        return (self.a, self.b), None

      @classmethod
      def tree_unflatten(cls, _, args):
        return cls(*args)

    v1 = CustomVector(1, 2.0)
    v2 = v1 + 3
    self.assertTreeEqual(v2, CustomVector(4, 5.0), check_dtypes=True)

    v3 = v2 + v1
    self.assertTreeEqual(v3, CustomVector(5, 7.0), check_dtypes=True)


if __name__ == "__main__":
  absltest.main()
