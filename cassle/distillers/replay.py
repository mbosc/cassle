import argparse
from typing import Any, List, Sequence
import torch
from torch import nn
import random
import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

def not_so_default_collate(batch, device='cpu'):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out).to(device)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(elem.dtype)

            return not_so_default_collate([torch.as_tensor(b, device=device) for b in batch], device)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch, device=device)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64, device=device)
    elif isinstance(elem, int):
        return torch.tensor(batch, device=device)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: not_so_default_collate([d[key] for d in batch], device) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(not_so_default_collate(samples, device) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [not_so_default_collate(samples, device) for samples in transposed]

    raise TypeError(elem_type)

def replay_wrapper(Method=object):
    class ReplayWrapper(Method):

        def buffer_len(self):
            return min(self.buffer_seen.item(), len(self.buffer))
        
        def buffer_sample(self, batch):
            idxs, _, _ = batch
            for sample in idxs:
                if self.buffer_len() < self.buffer_size:
                    self.buffer[self.buffer_seen] = sample.item()
                else:
                    idx = random.randint(0, self.buffer_seen.item())
                    if idx < self.buffer_size:
                        self.buffer[idx] = sample.item()
                self.buffer_seen += 1

        def buffer_retrieve(self, batch_size):
            assert batch_size > 0
            sample = self.buffer[torch.randperm(self.buffer_len())[:batch_size]]
            retrieved = [self.buffer_dataset[i] for i in sample]
            return not_so_default_collate(retrieved, device=self.device)
        
        def __init__(
            self,
            replay_lamb: float,
            buffer_size: int,
            buffer_dataset: torch.utils.data.Dataset,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.register_buffer('buffer', torch.zeros(buffer_size, dtype=int))
            self.register_buffer('buffer_seen', torch.zeros(1, dtype=int))
            self.buffer_dataset = buffer_dataset
            self.buffer_size = buffer_size
            self.replay_lamb = replay_lamb
            

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("contrastive_distiller")

            parser.add_argument("--replay_lamb", type=float, default=1)
            parser.add_argument("--buffer_size", type=int, default=5000)

            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            return super().learnable_params

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            loss = out["loss"]

            if self.buffer_len():
                extra_batch = {}
                if self.online_eval:
                    extra_batch['online_eval'] = batch['online_eval']
                extra_batch[f"task{self.current_task_idx}"] = self.buffer_retrieve(self.batch_size)
                extra_out = super().training_step(extra_batch, batch_idx)
                loss += self.replay_lamb * extra_out["loss"]
            
            with torch.no_grad():
                self.buffer_sample(batch[f"task{self.current_task_idx}"])
            
            return loss

    return ReplayWrapper
