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
"""Helpers for constructing data classes that are JAX and tree-math enabled."""

import dataclasses
import jax
from tree_math._src.vector import VectorMixin


def struct(cls):
  """Class decorator that enables JAX function transforms as well as tree math.

  Decorating a class with `@struct` makes it a dataclass that is compatible
  with arithmetic infix operators like `+`, `-`, `*` and `/`. The decorated
  class is also a valid pytree, making it compatible with JAX function
  transformations such as `jit` and `grad`.

  Example usage:
  ```
  import jax
  import tree_math

  @tree_math.struct
  class Point:
    x: float
    y: float

  a = Point(0.0, 1.0)
  b = Point(2.0, 3.0)

  a + 3 * b  # Point(6.0, 10.0)
  jax.grad(lambda x, y: x @ y)(a, b)  # Point(2.0, 3.0)
  ```

  Args:
    cls: a class, written with the same syntax as a `dataclass`.

  Returns:
    A wrapped version of `cls` that implements dataclass, pytree and tree_math
    functionality.
  """
  @property
  def fields(self):
    return dataclasses.fields(self)

  def asdict(self):
    return {field.name: getattr(self, field.name) for field in self.fields}

  def astuple(self):
    return tuple(getattr(self, field.name) for field in self.fields)

  def tree_flatten(self):
    return self.astuple(), None

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)

  cls_as_struct = type(cls.__name__,
                       (VectorMixin, dataclasses.dataclass(cls)),
                       {'fields': fields,
                        'asdict': asdict,
                        'astuple': astuple,
                        'replace': dataclasses.replace,
                        'tree_flatten': tree_flatten,
                        'tree_unflatten': tree_unflatten,
                        '__module__': cls.__module__})
  return jax.tree_util.register_pytree_node_class(cls_as_struct)
