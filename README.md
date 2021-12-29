# tree-math: mathematical operations for JAX pytrees

tree-math makes it easy to implement numerical algorithms that work on
[JAX pytrees](https://jax.readthedocs.io/en/latest/pytrees.html), such as
iterative methods for optimization and equation solving. It does so by providing
a wrapper class `tree_math.Vector` that defines array operations such as
infix arithmetic and dot-products on pytrees as if they were vectors.

## Why tree-math

In a library like SciPy, numerical algorithms are typically written to handle
fixed-rank arrays, e.g., [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
requires inputs of shape `(n,)`. This is convenient for implementors of
numerical methods, but not for users, because 1d arrays are typically not the
best way to keep track of state for non-trivial functions (e.g., neural networks
or PDE solvers).

tree-math provides an alternative to flattening and unflattening these more
complex data structures ("pytrees") for use in numerical algorithms. Instead,
the numerical algorithm itself can be written in way to handle arbitrary
collections of arrays stored in pytrees. This avoids unnecessary memory copies,
and gives the user more control over the memory layouts used in computation.
In practice, this can often makes a big difference for computational efficiency
as well, which is why support for flexible data structures is so prevalent
inside libraries that use JAX.

## Installation

tree-math is implemented in pure Python, and only depends upon JAX.

You can install it from PyPI: `pip install tree-math`.

## User guide

tree-math is simple to use. Just pass arbitrary pytree objects into
`tree_math.Vector` to create an a object that arithmetic as if all leaves of
the pytree were flattened and concatenated together:
```
>>> import tree_math as tm
>>> import jax.numpy as jnp
>>> v = tm.Vector({'x': 1, 'y': jnp.arange(2, 4)})
>>> v
tree_math.Vector({'x': 1, 'y': DeviceArray([2, 3], dtype=int32)})
>>> v + 1
tree_math.Vector({'x': 2, 'y': DeviceArray([3, 4], dtype=int32)})
>>> v.sum()
DeviceArray(6, dtype=int32)
```

You can also find a few functions defined on vectors in `tree_math.numpy`, which
implements a very restricted subset of `jax.numpy`. If you're interested in more
functionality, please open an issue to discuss before sending a pull request.
(In the long term, this separate module might disappear if we can support
`Vector` objects directly inside `jax.numpy`.)

Vector objects are pytrees themselves, which means the are compatible with JAX
transformations like `jit`, `vmap` and `grad`, and control flow like
`while_loop` and `cond`.

When you're done manipulating vectors, you can pull out the underlying pytrees
from the `.tree` property:
```
>>> v.tree
{'x': 1, 'y': DeviceArray([2, 3], dtype=int32)}
```

As an alternative to manipulating `Vector` objects directly, you can also use
the functional transformations `wrap` and `unwrap` (see the "Example usage"
below).

One important difference between `tree_math` and `jax.numpy` is that dot
products in `tree_math` default to full precision on all platforms, rather
than defaulting to bfloat16 precision on TPUs. This is useful for writing most
numerical algorithms, and will likely be JAX's default behavior
[in the future](https://github.com/google/jax/pull/7859).

In the near-term, we also plan to add a `Matrix` class that will make it
possible to use tree-math for numerical algorithms such as
[L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) which use matrices
to represent stacks of vectors.

## Example usage

Here is how we could write the preconditioned conjugate gradient
method. Notice how similar the implementation is to the [pseudocode from
Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method),
unlike the [implementation in JAX](https://github.com/google/jax/blob/b5aea7bc2da4fb5ef96c87a59bfd1486d8958dd7/jax/_src/scipy/sparse/linalg.py#L111-L121):

```python
import functools
from jax import lax
import tree_math as tm
import tree_math.numpy as tnp

@functools.partial(tm.wrap, vector_argnames=['b', 'x0'])
def cg(A, b, x0, M=lambda x: x, maxiter=5, tol=1e-5, atol=0.0):
  """jax.scipy.sparse.linalg.cg, written with tree_math."""
  A = tm.unwrap(A)
  M = tm.unwrap(M)

  atol2 = tnp.maximum(tol**2 * (b @ b), atol**2)

  def cond_fun(value):
    x, r, gamma, p, k = value
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
```
