"""Circular buffer for storing batched tensor history — copied from atom01_train rsl_rl."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class CircularBuffer:
  """Circular buffer for storing a history of batched tensor data.

  Shape convention:
      - Internal buffer: (max_len, batch_size, *data_shape)
      - ``buffer`` property (read-out): (batch_size, max_len, *data_shape)
        with the most-recent entry last.
  """

  def __init__(self, max_len: int, batch_size: int, device: str):
    if max_len < 1:
      raise ValueError(f"Buffer size must be >= 1, got {max_len}.")
    self._batch_size = batch_size
    self._device = device
    self._ALL_INDICES = torch.arange(batch_size, device=device)

    self._max_len = torch.full((batch_size,), max_len, dtype=torch.int, device=device)
    self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
    self._pointer: int = -1
    self._buffer: torch.Tensor = None  # type: ignore

  # ------------------------------------------------------------------
  # Properties
  # ------------------------------------------------------------------

  @property
  def batch_size(self) -> int:
    return self._batch_size

  @property
  def device(self) -> str:
    return self._device

  @property
  def max_length(self) -> int:
    return int(self._max_len[0].item())

  @property
  def current_length(self) -> torch.Tensor:
    return torch.minimum(self._num_pushes, self._max_len)

  @property
  def buffer(self) -> torch.Tensor:
    """Return buffer with most-recent entry last.  Shape: (batch_size, max_len, ...)."""
    buf = self._buffer.clone()
    buf = torch.roll(buf, shifts=self.max_length - self._pointer - 1, dims=0)
    return torch.transpose(buf, dim0=0, dim1=1)

  # ------------------------------------------------------------------
  # Operations
  # ------------------------------------------------------------------

  def reset(self, batch_ids: Sequence[int] | None = None):
    if batch_ids is None:
      batch_ids = slice(None)
    self._num_pushes[batch_ids] = 0
    if self._buffer is not None:
      self._buffer[:, batch_ids] = 0.0

  def append(self, data: torch.Tensor):
    """Append (batch_size, ...) data to the buffer."""
    if data.shape[0] != self.batch_size:
      raise ValueError(
        f"Data batch size {data.shape[0]} != buffer batch size {self.batch_size}."
      )
    data = data.to(self._device)
    if self._buffer is None:
      self._pointer = -1
      self._buffer = torch.empty(
        (self.max_length, *data.shape), dtype=data.dtype, device=self._device
      )
    self._pointer = (self._pointer + 1) % self.max_length
    self._buffer[self._pointer] = data
    is_first = self._num_pushes == 0
    if torch.any(is_first):
      self._buffer[:, is_first] = data[is_first]
    self._num_pushes += 1

  def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
    if len(key) != self.batch_size:
      raise ValueError(f"Key length {len(key)} != batch size {self.batch_size}.")
    if torch.any(self._num_pushes == 0) or self._buffer is None:
      raise RuntimeError("Buffer is empty.")
    valid_keys = torch.minimum(key, self._num_pushes - 1)
    index_in_buffer = torch.remainder(self._pointer - valid_keys, self.max_length)
    return self._buffer[index_in_buffer, self._ALL_INDICES]

  def mini_batch_generator(
    self, fetch_length: int, num_mini_batches: int, num_epochs: int = 8
  ):
    """Yield mini-batches of shape (mini_batch_size, *data_shape)."""
    if torch.any(self._num_pushes == 0) or self._buffer is None:
      raise RuntimeError("Buffer is empty.")
    min_len = int(torch.min(self.current_length).item())
    if fetch_length > min_len:
      raise ValueError(f"fetch_length {fetch_length} > min current_length {min_len}.")
    epoch_batch_size = self.batch_size * fetch_length
    mini_batch_size = epoch_batch_size // num_mini_batches
    if epoch_batch_size % num_mini_batches != 0:
      raise ValueError(
        f"epoch_batch_size {epoch_batch_size} not divisible by {num_mini_batches}."
      )
    total_combinations = int(self.current_length[0].item()) * self.batch_size
    linear_indices = torch.randperm(total_combinations, device=self.device)[
      :epoch_batch_size
    ]
    indices_0 = linear_indices // self.batch_size
    indices_1 = linear_indices % self.batch_size
    for _ in range(num_epochs):
      perm = torch.randperm(epoch_batch_size, device=self.device)
      for i in range(num_mini_batches):
        start, end = i * mini_batch_size, (i + 1) * mini_batch_size
        mb_0 = indices_0[perm[start:end]]
        mb_1 = indices_1[perm[start:end]]
        yield self._buffer[mb_0, mb_1]
