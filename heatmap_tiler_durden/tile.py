#!/usr/bin/env python3
import logging
import os
import os.path as path
import numpy as np
from coords import TILE_SIZE_PX

logger = logging.getLogger('tyler')


class Tile:
    cache_memory = {}

    def __init__(
            self,
            filepath: str,
            dtype: type,
    ):
        self._filepath = filepath
        self._dtype = dtype

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def is_empty(self) -> bool:
        return not path.isfile(self.filepath)

    @property
    def data(self):
        if not path.isfile(self.filepath):
            return np.zeros((TILE_SIZE_PX, TILE_SIZE_PX), dtype=self._dtype)
        return np.load(self.filepath)

    @data.setter
    def data(self, data):
        os.makedirs(path.dirname(self.filepath), exist_ok=True)
        empty = np.all(data == 0)
        if empty and path.isfile(self.filepath):
            os.remove(self.filepath)
        if not empty:
            np.save(self.filepath, data)
