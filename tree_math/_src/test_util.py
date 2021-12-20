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
"""Testing utilities for JAX Tree Math."""

from absl.testing import parameterized
from jax import tree_util
import numpy as np


def _dtype(x):
  return getattr(x, 'dtype', None) or np.dtype(type(x))


class TestCase(parameterized.TestCase):
  """Internal TestCase for tree_math."""

  def assertArraysEqual(self, expected, actual, check_dtypes):
    if check_dtypes:
      self.assertEqual(_dtype(actual), _dtype(expected))
    np.testing.assert_array_equal(actual, expected)

  def assertAllClose(self, expected, actual, check_dtypes=False, **kwargs):
    if check_dtypes:
      self.assertEqual(_dtype(actual), _dtype(expected))
    np.testing.assert_allclose(expected, actual, **kwargs)

  def _assert_tree(self, method, expected, actual, check_dtypes):
    expected_leaves, expected_treedef = tree_util.tree_flatten(expected)
    actual_leaves, actual_treedef = tree_util.tree_flatten(actual)
    self.assertEqual(actual_treedef, expected_treedef)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
      method(actual_leaf, expected_leaf, check_dtypes)

  def assertTreeEqual(self, expected, actual, check_dtypes):
    self._assert_tree(self.assertArraysEqual, expected, actual, check_dtypes)

  def assertTreeAllClose(self, expected, actual, check_dtypes):
    self._assert_tree(self.assertAllClose, expected, actual, check_dtypes)
