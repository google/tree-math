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
  @struct
  class Point:
    x: float
    y: float

  a = Point(0., 1.)
  b = Point(1., 1.)

  a + 3 * b  # Point(3., 4.)

  def norm_squared(pt):
    return pt.x**2 + pt.y**2

  jax.jit(jax.grad(norm))(b)  # Point(2., 2.)
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
                        'tree_flatten': tree_flatten,
                        'tree_unflatten': tree_unflatten})
  return jax.tree_util.register_pytree_node_class(cls_as_struct)
