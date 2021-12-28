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
"""Integration tests for tree_math."""

import functools

from absl.testing import absltest
import jax
from jax import lax
import tree_math as tm
from tree_math._src import test_util
import tree_math.numpy as tnp

# pylint: disable=g-complex-comprehension


class TreeMathTest(test_util.TestCase):

  def test_norm(self):

    @tm.wrap
    def norm1(x, y):
      return ((x - y) ** 2).sum() ** 0.5

    @tm.wrap
    def norm2(x, y):
      d = x - y
      return (d @ d) ** 0.5

    x = {"a": 1, "b": 1}
    y = {"a": 1 + 3, "b": 1 + 4}
    expected = 5.0
    actual = norm1(x, y)
    self.assertAllClose(actual, expected)

    actual = norm2(x, y)
    self.assertAllClose(actual, expected)

  def test_cg(self):
    # an integration test to verify non-trivial examples work
    # pylint: disable=invalid-name

    @functools.partial(tm.wrap, vector_argnames=["b", "x0"])
    def cg(A, b, x0, M=lambda x: x, maxiter=5, tol=1e-5, atol=0.0):
      """jax.scipy.sparse.linalg.cg, written with tree_math."""
      A = tm.unwrap(A)
      M = tm.unwrap(M)

      atol2 = tnp.maximum(tol**2 * (b @ b), atol**2)

      def cond_fun(value):
        x, r, gamma, p, k = value  # pylint: disable=unused-variable
        return (r @ r > atol2) & (k < maxiter)

      def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p)
        alpha = gamma / (p.conj() @ Ap)
        x_ = x + alpha * p
        r_ = r - alpha * Ap
        z_ = M(r_)
        gamma_ = r_.conj() @ z_
        beta_ = gamma_ / gamma
        p_ = z_ + beta_ * p
        return x_, r_, gamma_, p_, k + 1

      r0 = b - A(x0)
      p0 = z0 = M(r0)
      gamma0 = r0 @ z0
      initial_value = (x0, r0, gamma0, p0, 0)

      x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)
      return x_final

    A = lambda x: {"a": x["a"] + 0.5 * x["b"], "b": 0.5 * x["a"] + x["b"]}
    b = {"a": 1.0, "b": -1.0}
    x0 = {"a": 0.0, "b": 0.0}
    actual = cg(A, b, x0)

    expected = jax.device_put({"a": 2.0, "b": -2.0})
    self.assertTreeAllClose(actual, expected, check_dtypes=True)


if __name__ == "__main__":
  absltest.main()
