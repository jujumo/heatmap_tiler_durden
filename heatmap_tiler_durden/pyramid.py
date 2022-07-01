#!/usr/bin/env python3
import logging
import os
import os.path as path
from PIL import Image
import matplotlib.cm as cm
import numpy as np
from typing import Optional, Tuple, Set
from coords import ll_to_raster, raster_to_tile, tile_set_rx, keep_in_tile_rx, TILE_SIZE_PX, nb_tiles
from tqdm import tqdm


logger = logging.getLogger('tyler')
ext = 'jpg'


class Tile:
    def __int__(self, filepath: str, data: np.ndarray):
        self._filepath = filepath
        self.data = data

    @staticmethod
    def tile_filepath(root_dirpath: str, tile_coord: Tuple[int, int], level: int, file_ext: str) -> str:
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
        self._tiles_in_cache = dict()  # (x, y) -> tile

    def get_tile(self, tile_coord):
        # if not in memory yet, load or assigns the tile
        if tile_coord in self._tiles_in_cache:
            return
        filepath = Tile.tile_filepath(
            root_dirpath=self._root_dirpath, tile_coord=tile_coord, level=self._level, file_ext='npy')
        tile = Tile.load_from_disk(filepath=filepath) or Tile.creates(filepath=filepath, dtype=self._dtype)
        if tile is None:
            return None

        self._tiles_in_cache[tile_coord] = tile
        return tile

    def flush_to_disk(self):
        for tile in self._tiles_in_cache.values():
            tile.save_to_disk()
        self._tiles_in_cache.clear()


class TilePyramid:
    def __init__(
            self,
            root_dirpath: str,
            max_level: int,
            dtype
    ):
        self._root_dirpath = root_dirpath
        self._pyramid_layers = {level: TilePyramidLayer(root_dirpath=root_dirpath, level=level, dtype=dtype)
                                for level in range(max_level)}

    def flush_to_disk(self):
        for level, pyramid_layer in self._pyramid_layers.items():
            pyramid_layer.flush_to_disk()

    def dump_to_images(self, image_rootpath: str):
        self.flush_to_disk()
        for level in self._pyramid_layers:
            xx, yy = np.meshgrid(np.arange(nb_tiles(level)), np.arange(nb_tiles(level)))
            tile_coords = np.column_stack([xx.flatten(), yy.flatten()])
            for tile_coord in tqdm(tile_coords):
                tile_filepath = Tile.tile_filepath(
                    root_dirpath=self._root_dirpath, tile_coord=tile_coord, level=level, file_ext='npy')
                image_filepath = Tile.tile_filepath(
                    root_dirpath=image_rootpath, tile_coord=tile_coord, level=level, file_ext='png')

                tile = Tile.load_from_disk(filepath=tile_filepath)
                if tile is None:
                    continue
                tile.dump_to_image(image_filepath)

    def increments(self, coords_ll: np.ndarray):
        # load / create in memory
        for level, layer in self._pyramid_layers.items():
            coords_rx = ll_to_raster(coords_ll, level)
            tiles_coords = tile_set_rx(coords_rx=coords_rx)
            for tile_coord in tqdm(tiles_coords):
                tile = layer.get_tile(tile_coord)
                coords_rx_in_tile = keep_in_tile_rx(coords_rx, tile_coord)
                coords_ti = raster_to_tile(coords_rx_in_tile, tile_coord).astype(int)
                for x, y in coords_ti.transpose():
                    tile.data[y, x] += 1
