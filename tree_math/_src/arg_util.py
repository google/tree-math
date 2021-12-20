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
"""Argument parsing utilities copied from jax._src.api and jax._src.api_util."""

import inspect
import operator
from typing import Any, Callable, Iterable, Mapping, Tuple, Union


# pylint: disable=g-bare-generic


def _ensure_index_tuple(x: Any) -> Tuple[int, ...]:
  """Convert x to a tuple of indices."""
  try:
    return (operator.index(x),)
  except TypeError:
    return tuple(map(operator.index, x))


def _ensure_str(x: str) -> str:
  if not isinstance(x, str):
    raise TypeError(f"argument is not a string: {x}")
  return x


def _ensure_str_tuple(x: Union[str, Iterable[str]]) -> Tuple[str, ...]:
  """Convert x to a tuple of strings."""
  if isinstance(x, str):
    return (x,)
  else:
    return tuple(map(_ensure_str, x))


_POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD


def infer_argnums_and_argnames(
    fun: Callable,
    argnums: Union[int, Iterable[int], None],
    argnames: Union[str, Iterable[str], None],
) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
  """Infer missing argnums and argnames for a function with inspect."""
  if argnums is None and argnames is None:
    argnums = ()
    argnames = ()
  elif argnums is not None and argnames is not None:
    argnums = _ensure_index_tuple(argnums)
    argnames = _ensure_str_tuple(argnames)
  else:
    try:
      signature = inspect.signature(fun)
    except ValueError:
      # In rare cases, inspect can fail, e.g., on some builtin Python functions.
      # In these cases, don't infer any parameters.
      parameters: Mapping[str, inspect.Parameter] = {}
    else:
      parameters = signature.parameters
    if argnums is None:
      assert argnames is not None
      argnames = _ensure_str_tuple(argnames)
      argnums = tuple(
          i for i, (k, param) in enumerate(parameters.items())
          if param.kind == _POSITIONAL_OR_KEYWORD and k in argnames
      )
    else:
      assert argnames is None
      argnums = _ensure_index_tuple(argnums)
      argnames = tuple(
          k for i, (k, param) in enumerate(parameters.items())
          if param.kind == _POSITIONAL_OR_KEYWORD and i in argnums
      )
  return argnums, argnames
