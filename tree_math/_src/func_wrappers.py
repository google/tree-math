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
"""Function wrappers for tree_math."""

import functools

from jax import tree_util
from tree_math._src import arg_util
from tree_math._src import vector


def _infer_argnums_and_argnames(fun, argnums, argnames):
  if argnums is None and argnames is None:
    return None, None  # wrap all arguments
  return arg_util.infer_argnums_and_argnames(fun, argnums, argnames)


def _apply_argnums(wrapper, args, argnums):
  return tuple(wrapper(arg) if argnums is None or i in argnums else arg
               for i, arg in enumerate(args))


def _apply_argnames(wrapper, kwargs, argnames):
  return {k: wrapper(arg) if argnames is None or k in argnames else arg
          for k, arg in kwargs.items()}


def _maybe_get_tree(arg):
  return arg.tree if isinstance(arg, vector.Vector) else arg


def _is_vector(arg):
  return isinstance(arg, vector.Vector)


def wrap(fun, vector_argnums=None, vector_argnames=None):
  """Convert a vector -> vector function to a pytree -> pytree function."""
  vector_argnums, vector_argnames = _infer_argnums_and_argnames(
      fun, vector_argnums, vector_argnames)
  @functools.wraps(fun)
  def wrapper(*args, **kwargs):
    args = _apply_argnums(vector.Vector, args, vector_argnums)
    kwargs = _apply_argnames(vector.Vector, kwargs, vector_argnames)
    result = fun(*args, **kwargs)
    return tree_util.tree_map(_maybe_get_tree, result, is_leaf=_is_vector)
  return wrapper


def _get_tree(tree_vector):
  return tree_vector.tree


def _maybe_vector(condition, arg):
  return vector.Vector(arg) if condition else arg


def unwrap(fun, vector_argnums=None, vector_argnames=None, out_vectors=True):
  """Convert a pytree -> pytree function to a vector -> vector function."""
  vector_argnums, vector_argnames = _infer_argnums_and_argnames(
      fun, vector_argnums, vector_argnames)
  @functools.wraps(fun)
  def wrapper(*args, **kwargs):
    args = _apply_argnums(_get_tree, args, vector_argnums)
    kwargs = _apply_argnames(_get_tree, kwargs, vector_argnames)
    result = fun(*args, **kwargs)
    return tree_util.tree_map(_maybe_vector, out_vectors, result)
  return wrapper
