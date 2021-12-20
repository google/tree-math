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
import jax.numpy as jnp
import tree_math as tm
from tree_math._src import test_util


class FuncWrappersTest(test_util.TestCase):

  def test_basic_wrap(self):
    @tm.wrap
    def f(x, y):
      return x - y

    x = {"a": 10, "b": jnp.array([20, 30])}
    y = {"a": 1, "b": jnp.array([2, 3])}
    expected = {"a": 9, "b": jnp.array([18, 27])}
    actual = f(x, y)
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_tree_out_wrap(self):
    @tm.wrap
    def f(x, y):
      return x + y, x - y

    x = {"a": 10, "b": jnp.array([20, 30])}
    y = {"a": 1, "b": jnp.array([2, 3])}
    expected = ({"a": 11, "b": jnp.array([22, 33])},
                {"a": 9, "b": jnp.array([18, 27])})
    actual = f(x, y)
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_unwrap(self):

    @tm.unwrap
    def f(x):
      return {"b": x["a"] + 1}

    actual = f(tm.Vector({"a": 1}))
    expected = tm.Vector({"b": 2})
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_unwrap_out_vectors(self):
    f = lambda *args: args

    expected = tm.Vector((1, 2))
    actual = tm.unwrap(f, out_vectors=True)(tm.Vector(1), tm.Vector(2))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

    expected = (1, 2)
    actual = tm.unwrap(f, out_vectors=False)(tm.Vector(1), tm.Vector(2))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

    expected = (tm.Vector(1), 2)
    actual = tm.unwrap(f, out_vectors=(True, False))(tm.Vector(1), tm.Vector(2))
    self.assertTreeEqual(actual, expected, check_dtypes=True)

  def test_wrap_argnums_argnames(self):

    def f(x, y):
      assert isinstance(x, tm.Vector)
      assert not isinstance(y, tm.Vector)
      return x

    tm.wrap(f, vector_argnums=0)(1, 2)
    tm.wrap(f, vector_argnums=0)(x=1, y=2)
    tm.wrap(f, vector_argnames="x")(1, 2)
    tm.wrap(f, vector_argnames="x")(x=1, y=2)

    def g(x, y):
      assert not isinstance(x, tm.Vector)
      assert isinstance(y, tm.Vector)
      return y

    tm.wrap(g, vector_argnums=1)(1, 2)
    tm.wrap(g, vector_argnums=1)(x=1, y=2)
    tm.wrap(g, vector_argnames="y")(1, 2)
    tm.wrap(g, vector_argnames="y")(x=1, y=2)


if __name__ == "__main__":
  absltest.main()
