# tree-math: mathematical operations for JAX pytrees

tree-math makes it easy to implement numerical algorithms that work on
[JAX pytrees](https://jax.readthedocs.io/en/latest/pytrees.html), such as
iterative methods for optimization and equation solving. It does so by providing
a wrapper class `tree_math.Vector` that defines array operations such as
infix arithmetic and dot-products on pytrees.

For example, here's how we could write the preconditioned conjugate gradient
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

For most operations, we recommend working directly with `Vector` objects or the
`wrap` and `unwrap` helper functions. You can also find a few functions defined
on vectors in `tree_math.numpy`, which implements a very restricted subset
of `jax.numpy`. (In the long term, this separate module might dissappear if we
can support `Vector` objects directly inside `jax.numpy`).

One important different between `tree_math` and `jax.numpy` is that dot
products in `tree_math` default to full precision on all platforms, rather
than defaulting to bfloat16 precision on TPUs. This is useful for writing most
numerical algorithms, and will likely be JAX's default behavior
[in the future](https://github.com/google/jax/pull/7859).

TODO(shoyer): add a full tutorial!
