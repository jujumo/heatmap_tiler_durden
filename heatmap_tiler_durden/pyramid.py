#!/usr/bin/env python3
import logging
import os
import os.path as path
import numpy as np
from typing import Optional, Tuple, Set
from coords import get_tile_coord_list
from tqdm import tqdm

from tile_layer import TileLayer


QUAD_TREE_SHIFTS = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])


logger = logging.getLogger('tyler')


def shrinks_layer_by_aggregation(
        lower_layer: TileLayer,
        upper_layer: TileLayer
):
    """ shrink tiles by summation to the upper level """
    if lower_layer.level == 0:
        raise IndexError('cant create upper level of level 0')
    upper_level = lower_layer.level - 1
    # upper_layer = TileLayer(root_dirpath=lower_layer.root_dirpath, level=upper_level, dtype=lower_layer.dtype)
    upper_tile_coords = get_tile_coord_list(upper_level)
    # todo: speed up with is_empty test
    for upper_tile_coord in upper_tile_coords:
        # retrieve downscale tiles coords
        upper_layer_tile = upper_layer.tiles[upper_tile_coord]
        upper_tile_data = upper_layer_tile.data
        for x_shift, y_shift in QUAD_TREE_SHIFTS:
            base_tile_coord = tuple(np.array(upper_tile_coord) * 2 + np.array([x_shift, y_shift]).tolist())
            base_tile = lower_layer.tiles[base_tile_coord]
            if base_tile.is_empty:
                continue
            base_tile_data = base_tile.data
            x_slice_dest = slice(x_shift * 128, x_shift * 128 + 128)
            y_slice_dest = slice(y_shift * 128, y_shift * 128 + 128)
            upper_tile_quarter = upper_tile_data[y_slice_dest, x_slice_dest]
            upper_tile_quarter += base_tile_data[0::2, 0::2]
            upper_tile_quarter += base_tile_data[0::2, 1::2]
            upper_tile_quarter += base_tile_data[1::2, 0::2]
            upper_tile_quarter += base_tile_data[1::2, 1::2]
            if np.all(upper_tile_quarter == 0):
                continue
        upper_layer.tiles[upper_tile_coord].data = upper_tile_data


class TilePyramid:
    def __init__(
            self,
            dirpath: str,
            max_level: int,
            dtype: type
    ):
        self._dirpath = dirpath
        self.max_level = max_level
        self._pyramid_layers = {
            level: TileLayer(dirpath=path.join(dirpath, f'{level:02}'), level=level, dtype=dtype)
            for level in range(max_level)
        }

    @property
    def layers(self):
        return self._pyramid_layers

    @classmethod
    def create_from_base_layer(
            cls,
            base_layer: TileLayer,
            root_dirpath: str
    ):
        pyramid = cls(dirpath=root_dirpath,
                      max_level=base_layer.level,
                      dtype=base_layer.dtype)

        # replace base layer with given one
        pyramid._pyramid_layers[base_layer.level] = base_layer

        lower_layer = base_layer
        for level in reversed(range(base_layer.level)):
            logger.debug(f'creating layer level {level}')
            upper_layer = pyramid.layers[level]
            # upper_layer = shrinks_layer_by_aggregation(lower_layer)
            shrinks_layer_by_aggregation(lower_layer=lower_layer, upper_layer=upper_layer)
            # pyramid._pyramid_layers[level] = upper_layer
            lower_layer = upper_layer

        return pyramid

