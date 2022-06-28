#!/usr/bin/env python3
import logging
import os.path as path
import numpy as np
from typing import Optional
from coords import ll_to_meters, meters_to_pixels, pixel_to_raster, nb_tiles, tile_bounds_rx

logger = logging.getLogger('tyler')
ext = 'jpg'


def increment_heatmap_pyramid_level_tile(
        coords_ti: np.ndarray,
        output_dirpath: str,
        tile_x: int,
        tile_y: int,
        level: int,
        force_empty_tiles: Optional[bool]
):
    tile_filepath = path.join(output_dirpath, f'{level:02}', f'{tile_x:04}x{tile_y:04}.{ext}')
    print(tile_filepath)


def create_heatmap_pyramid_level(
        coords_rx: np.ndarray,
        output_dirpath: str,
        level: int,
        force_empty_tiles: Optional[bool]
):
    logger.debug(f'level {level} has {(2 ** level) ** 2} tiles.')
    for tile_y in range(nb_tiles(level)):
        for tile_x in range(nb_tiles(level)):
            bounds_rx = tile_bounds_rx(tile_x, tile_y)
            in_tile = np.vstack([
                coords_rx > bounds_rx[:, 0:1],
                coords_rx < bounds_rx[:, 2:3],
            ])
            in_tile = np.all(in_tile, axis=0)
            coords_ti = coords_rx[:, in_tile]
            coords_ti = coords_ti - bounds_rx[:, 0:1]
            increment_heatmap_pyramid_level_tile(
                coords_ti=coords_ti,
                output_dirpath=output_dirpath,
                tile_x=tile_x, tile_y=tile_y, level=level,
                force_empty_tiles=force_empty_tiles
            )


def create_heatmap_pyramid(
        coords_ll: np.ndarray,
        output_dirpath: str,
        max_level: int,
        force_empty_tiles: Optional[bool]
):
    """ creates heatmap image file """

    coords_m = ll_to_meters(coords_ll)
    for level in range(0, max_level):
        coords_px = meters_to_pixels(coords_m, level=level)
        coords_rx = pixel_to_raster(coords_px, level=level)
        create_heatmap_pyramid_level(
            coords_rx=coords_rx,
            output_dirpath=output_dirpath,
            level=level,
            force_empty_tiles=force_empty_tiles
        )
