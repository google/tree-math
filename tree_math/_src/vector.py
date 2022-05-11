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
"""Core functionality for tree_math."""

import functools
import operator
import typing
from typing import Tuple

from jax import tree_util
import jax.numpy as jnp


def _flatten_together(*args):
  """Flatten a collection of pytrees with matching structure/shapes together."""
  all_values, all_treedefs = zip(*map(tree_util.tree_flatten, args))
  all_treedefs = typing.cast(Tuple[tree_util.PyTreeDef, ...], all_treedefs)

  if not all(treedef == all_treedefs[0] for treedef in all_treedefs[1:]):
    treedefs_str = " vs ".join(map(str, all_treedefs))
    raise ValueError(
        f"arguments have different tree structures: {treedefs_str}"
    )

  all_shapes = [list(map(jnp.shape, values)) for values in all_values]
  if not all(shapes == all_shapes[0] for shapes in all_shapes[1:]):
    shapes_str = " vs ".join(map(str, all_shapes))
    raise ValueError(f"tree leaves have different array shapes: {shapes_str}")

  return all_values, all_treedefs[0]


def _argnums_partial(f, args, static_argnums):
  def g(*args3):
    args3 = list(args3)
    for i in static_argnums:
      args3.insert(i, args[i])
    return f(*args3)
  args2 = tuple(x for i, x in enumerate(args) if i not in static_argnums)
  return g, args2


def broadcasting_map(func, *args):
  """Like tree_map, but scalar arguments are broadcast to all leaves."""
  static_argnums = [
      i for i, x in enumerate(args) if not isinstance(x, VectorMixin)
  ]
  func2, vector_args = _argnums_partial(func, args, static_argnums)
  for arg in args:
    if not isinstance(arg, VectorMixin):
      shape = jnp.shape(arg)
      if shape:
        raise TypeError(
            f"non-tree_math.VectorMixin argument is not a scalar: {arg!r}"
        )
  if not vector_args:
    return func2()  # result is a scalar
  _flatten_together(*[arg for arg in vector_args])  # check shapes
  return tree_util.tree_map(func2, *vector_args)


def _binary_method(func, name):
  """Implement a forward binary method, e.g., __add__."""
  def wrapper(self, other):
    return broadcasting_map(func, self, other)
  wrapper.__name__ = f"__{name}__"
  return wrapper


def _reflected_binary_method(func, name):
  """Implement a reflected binary method, e.g., __radd__."""
  def wrapper(self, other):
    return broadcasting_map(func, other, self)
  wrapper.__name__ = f"__r{name}__"
  return wrapper


def _numeric_methods(func, name):
  """Implement forward and reflected methods."""
  return (_binary_method(func, name), _reflected_binary_method(func, name))


def _unary_method(func, name):
  def wrapper(self):
    return tree_util.tree_map(func, self)
  wrapper.__name__ = f"__{name}__"
  return wrapper


def dot(left, right, *, precision="highest"):
  """Dot product between tree math vectors.

  Note that unlike jax.numpy.dot, tree_math.dot defaults to full (highest)
  precision. This is more useful for numerical algorithms and will be the
  default for jax.numpy in the future:
  https://github.com/google/jax/pull/7859

  Args:
    left: left argument.
    right: right argument.
    precision: precision.

  Returns:
    Resulting dot product (scalar).
  """
  if not isinstance(left, VectorMixin) or not isinstance(right, VectorMixin):
    raise TypeError(
        "matmul arguments must both be tree_math.VectorMixin objects")

  def _vector_dot(a, b):
    return jnp.dot(jnp.ravel(a), jnp.ravel(b), precision=precision)

  (left_values, right_values), _ = _flatten_together(left, right)
  parts = map(_vector_dot, left_values, right_values)
  return functools.reduce(operator.add, parts)


class VectorMixin:
  """A mixin class that adds a 1D vector-like behaviour to any custom pytree class."""

  @property
  def size(self):
    values = tree_util.tree_leaves(self)
    return sum(jnp.size(value) for value in values)

  def __len__(self):
    return self.size

  @property
  def shape(self):
    return (self.size,)

  @property
  def ndim(self):
    return 1

  @property
  def dtype(self):
    values = tree_util.tree_leaves(self)
    return jnp.result_type(*values)

  # comparison
  __lt__ = _binary_method(operator.lt, "lt")
  __le__ = _binary_method(operator.le, "le")
  __eq__ = _binary_method(operator.eq, "eq")
  __ne__ = _binary_method(operator.ne, "ne")
  __ge__ = _binary_method(operator.ge, "ge")
  __gt__ = _binary_method(operator.gt, "gt")

  # arithmetic
  __add__, __radd__ = _numeric_methods(operator.add, "add")
  __sub__, __rsub__ = _numeric_methods(operator.sub, "sub")
  __mul__, __rmul__ = _numeric_methods(operator.mul, "mul")
  __truediv__, __rtruediv__ = _numeric_methods(operator.truediv, "truediv")
  __floordiv__, __rfloordiv__ = _numeric_methods(operator.floordiv, "floordiv")
  __mod__, __rmod__ = _numeric_methods(operator.mod, "mod")
  __pow__, __rpow__ = _numeric_methods(operator.pow, "pow")
  __matmul__ = __rmatmul__ = dot

  # TODO(shoyer): implement this via divmod() on the leaves
  def __divmod__(self, other):
    return self // other, self % other

  def __rdivmod__(self, other):
    return other // self, other % self

  # bitwise
  __lshift__, __rlshift__ = _numeric_methods(operator.lshift, "lshift")
  __rshift__, __rrshift__ = _numeric_methods(operator.rshift, "rshift")
  __and__, __rand__ = _numeric_methods(operator.and_, "and")
  __xor__, __rxor__ = _numeric_methods(operator.xor, "xor")
  __or__, __ror__ = _numeric_methods(operator.or_, "or")

  # unary methods
  __neg__ = _unary_method(operator.neg, "neg")
  __pos__ = _unary_method(operator.pos, "pos")
  __abs__ = _unary_method(abs, "abs")
  __invert__ = _unary_method(operator.invert, "invert")

  # numpy methods
  conj = _unary_method(jnp.conj, "conj")
  dot = dot
  real = property(_unary_method(jnp.real, "real"))
  imag = property(_unary_method(jnp.imag, "imag"))

  def sum(self):
    parts = map(jnp.sum, tree_util.tree_leaves(self))
    return functools.reduce(operator.add, parts)

  def mean(self):
    return self.sum() / len(self)

  def min(self):
    parts = map(jnp.min, tree_util.tree_leaves(self))
    return jnp.asarray(list(parts)).min()

  def max(self):
    parts = map(jnp.max, tree_util.tree_leaves(self))
    return jnp.asarray(list(parts)).max()


@tree_util.register_pytree_node_class
class Vector(VectorMixin):
  """A wrapper for treating an arbitrary pytree as a 1D vector."""

  def __init__(self, tree):
    self._tree = tree

  @property
  def tree(self):
    return self._tree

  # TODO(shoyer): consider casting to a common dtype?

  def __repr__(self):
    return f"tree_math.Vector({self._tree!r})"

  def tree_flatten(self):
    return (self._tree,), None

  @classmethod
  def tree_unflatten(cls, _, args):
    return cls(*args)
