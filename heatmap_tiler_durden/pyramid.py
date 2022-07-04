#!/usr/bin/env python3
import logging
import os
import os.path as path
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Set
from coords import ll_to_raster, raster_to_tile, tile_set_rx, keep_in_tile_rx, TILE_SIZE_PX, nb_tiles
from tqdm import tqdm

logger = logging.getLogger('tyler')

QUAD_TREE_SHIFTS = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])


class Tile:
    def __int__(self, filepath: str, data: np.ndarray):
        self._filepath = filepath
        self.data = data

    @property
    def filepath(self):
        return self._filepath

    @staticmethod
    def tile_filepath(root_dirpath: str, tile_coord: Tuple[int, int], level: int, file_ext: str = 'npy') -> str:
        filepath = path.join(root_dirpath, f'{level:02}', f'{tile_coord[1]:05}x{tile_coord[0]:05}.{file_ext}')
        filepath = path.abspath(filepath)
        return filepath

    @classmethod
    def creates(cls, filepath: str, dtype):
        data = np.zeros((TILE_SIZE_PX, TILE_SIZE_PX), dtype=dtype)
        tile = cls()
        tile.__int__(filepath, data)
        return tile

    @classmethod
    def load_from_disk(cls, filepath: str):
        if not path.isfile(filepath):
            return None
        data = np.load(filepath)
        tile = cls()
        tile.__int__(filepath, data)
        return tile

    def save_to_disk(self):
        os.makedirs(path.dirname(self._filepath), exist_ok=True)
        np.save(self._filepath, self.data)

    def dump_to_image(self, image_filepath):
        os.makedirs(path.dirname(image_filepath), exist_ok=True)
        image = Image.fromarray(np.uint8(cm.plasma(self.data / 5.0) * 255))
        image.save(image_filepath)


class TilePyramidLayer:
    def __init__(
            self,
            root_dirpath: str,
            level: int,
            dtype
    ):
        self._root_dirpath = root_dirpath
        self._level = level
        self._dtype = dtype
        self._tiles_in_cache = dict()  # (x, y) -> tile with sparsity (not every tiles are required)

    @property
    def level(self):
        return self._level

    @property
    def root_dirpath(self):
        return self._root_dirpath

    @property
    def dtype(self):
        return self._dtype

    def get_tile(self, tile_coord):
        # if not in memory yet, load or assigns the tile
        if tile_coord in self._tiles_in_cache:
            return
        filepath = Tile.tile_filepath(
            root_dirpath=self._root_dirpath, tile_coord=tile_coord, level=self._level)
        tile = Tile.load_from_disk(filepath=filepath) or Tile.creates(filepath=filepath, dtype=self._dtype)
        if tile is None:
            return None

        self._tiles_in_cache[tile_coord] = tile
        return tile

    def flush_to_disk(self):
        for tile in self._tiles_in_cache.values():
            tile.save_to_disk()
        self._tiles_in_cache.clear()

    def create_upper_layer(self):
        if self.level == 0:
            raise IndexError('cant create upper level of level 0')
        upper_level = self.level - 1
        upper_layer = TilePyramidLayer(root_dirpath=self._root_dirpath, level=upper_level, dtype=self._dtype)
        xx, yy = np.meshgrid(np.arange(nb_tiles(upper_level)), np.arange(nb_tiles(upper_level)))
        upper_tile_coords = np.column_stack([xx.flatten(), yy.flatten()])
        for upper_tile_coord in upper_tile_coords:
            # retreive downscale tiles coords
            tile_filepath = Tile.tile_filepath(root_dirpath=self._root_dirpath,
                                               tile_coord=tuple(upper_tile_coord),
                                               level=upper_level)
            upper_tile = Tile.creates(filepath=tile_filepath, dtype=self._dtype)
            for x_shift, y_shift in QUAD_TREE_SHIFTS:
                base_tile_coord = tuple(upper_tile_coord * 2 + np.array([x_shift, y_shift]).tolist())
                base_tile = self.get_tile(base_tile_coord)
                if base_tile is None:
                    continue
                x_slice_dest = slice(x_shift * 128, x_shift * 128 + 128)
                y_slice_dest = slice(y_shift * 128, y_shift * 128 + 128)
                upper_tile_quarter = upper_tile.data[y_slice_dest, x_slice_dest]
                upper_tile_quarter += base_tile.data[0::2, 0::2]
                upper_tile_quarter += base_tile.data[0::2, 1::2]
                upper_tile_quarter += base_tile.data[1::2, 0::2]
                upper_tile_quarter += base_tile.data[1::2, 1::2]

                if np.all(upper_tile_quarter == 0):
                    continue
                upper_layer._tiles_in_cache[tuple(upper_tile_coord)] = upper_tile

        return upper_layer


class TilePyramid:
    def __init__(
            self,
            root_dirpath: str,
            max_level: int,
            dtype: type
    ):
        self._root_dirpath = root_dirpath
        self.max_level = max_level
        self._pyramid_layers = {level: TilePyramidLayer(root_dirpath=root_dirpath, level=level, dtype=dtype)
                                for level in range(max_level)}

    def get_layer(self, level: int):
        return self._pyramid_layers[level]

    def flush_to_disk(self):
        for level, layer in self._pyramid_layers.items():
            logger.debug(f'saving layer {level}')
            layer.flush_to_disk()

    def dump_to_images(self, image_rootpath: str):
        self.flush_to_disk()
        for level in self._pyramid_layers:
            xx, yy = np.meshgrid(np.arange(nb_tiles(level)), np.arange(nb_tiles(level)))
            tile_coords = np.column_stack([xx.flatten(), yy.flatten()])
            for tile_coord in tqdm(tile_coords):
                tile_filepath = Tile.tile_filepath(
                    root_dirpath=self._root_dirpath, tile_coord=tile_coord, level=level)
                image_filepath = Tile.tile_filepath(
                    root_dirpath=image_rootpath, tile_coord=tile_coord, level=level, file_ext='png')

                tile = Tile.load_from_disk(filepath=tile_filepath)
                if tile is None:
                    continue
                tile.dump_to_image(image_filepath)

    @classmethod
    def create_from_base_layer(cls, base_layer: TilePyramidLayer):
        pyramid = cls(root_dirpath=base_layer.root_dirpath,
                      max_level=base_layer.level,
                      dtype=base_layer.dtype)

        pyramid._pyramid_layers[base_layer.level] = base_layer
        lower_layer = base_layer
        for level in reversed(range(pyramid.max_level-1)):
            logger.debug(f'creating layer level {level}')
            upper_layer = lower_layer.create_upper_layer()
            pyramid._pyramid_layers[level] = upper_layer
            lower_layer = upper_layer
            # self._pyramid_layers
        return pyramid


def add_spots(layer: TilePyramidLayer, coords_ll: np.ndarray):
    # load / create in memory
    coords_rx = ll_to_raster(coords_ll, layer.level)
    tiles_coords = tile_set_rx(coords_rx=coords_rx)
    for tile_coord in tqdm(tiles_coords):
        tile = layer.get_tile(tile_coord)
        coords_rx_in_tile = keep_in_tile_rx(coords_rx, tile_coord)
        coords_ti = raster_to_tile(coords_rx_in_tile, tile_coord).astype(int)
        for x, y in coords_ti.transpose():
            tile.data[y, x] += 1
