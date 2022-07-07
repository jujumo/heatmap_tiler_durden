#!/usr/bin/env python3
import logging
import os
import os.path as path
import numpy as np
from typing import Tuple, List
from coords import nb_tiles

from tile import Tile

logger = logging.getLogger('tyler')


def get_tile_coord_list(level: int) -> List[Tuple[int, int]]:
    xx, yy = np.meshgrid(np.arange(nb_tiles(level)), np.arange(nb_tiles(level)))
    tile_coords = np.column_stack([xx.flatten(), yy.flatten()])
    tile_coords = [
        (x, y) for x, y in tile_coords
    ]
    return tile_coords


def normalizes_coord(
        tile_coord: Tuple[int, int],
        level: int):
    size = nb_tiles(level)
    return tile_coord[0] % size, tile_coord[1] % size


def get_tiles_neighborhood_33(
        tile_coord: Tuple[int, int],
        level: int
):
    tile_neighborhood_coords = [
        [
            normalizes_coord(tile_coord=(tile_coord[0] + x_shift, tile_coord[1] + y_shift),
                             level=level)
            for x_shift in [-1, 0, 1]
        ] for y_shift in [-1, 0, 1]
    ]
    return tile_neighborhood_coords


def get_tiles_lower(
        upper_tile_coord: Tuple[int, int]
):
    return [
        [
            (upper_tile_coord[0] * 2 + x_shift, upper_tile_coord[1] * 2 + y_shift)
            for x_shift in [0, 1]
        ]
        for y_shift in [0, 1]
    ]


class TileLayer:
    def __init__(
            self,
            dirpath: str,
            level: int,
            dtype: type
    ):
        self._layer_dirpath = dirpath
        self._level = level
        self._dtype = dtype
        self._tiles = {
            coord: Tile(filepath=TileLayer.tile_filepath(dirpath, coord), dtype=dtype)
            for coord in get_tile_coord_list(level)
        }

    @staticmethod
    def tile_filepath(
            root_dirpath: str,
            tile_coord: Tuple[int, int],
            file_ext: str = 'npy'
    ) -> str:
        filepath = path.join(root_dirpath, f'{tile_coord[1]:05}x{tile_coord[0]:05}.{file_ext}')
        filepath = path.abspath(filepath)
        return filepath

    @property
    def level(self):
        return self._level

    @property
    def dirpath(self):
        return self._layer_dirpath

    @property
    def dtype(self):
        return self._dtype

    @property
    def tiles(self):
        return self._tiles
