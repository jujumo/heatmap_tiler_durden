import os
import os.path as path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import matplotlib.cm as cm

from coords import ll_to_raster, tile_set_rx, keep_in_tile_rx, raster_to_tile, TILE_SIZE_PX
from tile import Tile
from tile_layer import TileLayer

QUAD_TREE_SHIFTS = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])


def add_spots(layer: TileLayer, coords_ll: np.ndarray):
    # load / create in memory
    coords_rx = ll_to_raster(coords_ll, layer.level)
    tiles_coords = tile_set_rx(coords_rx=coords_rx)
    for tile_coord in tqdm(tiles_coords):
        tile = layer.tiles[tile_coord]
        coords_rx_in_tile = keep_in_tile_rx(coords_rx, tile_coord)
        coords_ti = raster_to_tile(coords_rx_in_tile, tile_coord).astype(int)
        data = tile.data
        for x, y in coords_ti.transpose():
            data[y, x] += 1
        tile.data = data


def shrinks_tile_quad(
        tile: Tile,
        filepath: str,
        dtype: type,
        tile_ul, tile_ur,
        tile_dl, tile_dr
):
    upper_tile = tile.creates(filepath, dtype)
    tiles_quad = [tile_ul, tile_ur, tile_dl, tile_dr]
    for (x_shift, y_shift), base_tile in zip(QUAD_TREE_SHIFTS, tiles_quad):
        if base_tile is None:
            continue
        x_slice_dest = slice(x_shift * TILE_SIZE_PX // 2, x_shift * TILE_SIZE_PX // 2 + TILE_SIZE_PX // 2)
        y_slice_dest = slice(y_shift * TILE_SIZE_PX // 2, y_shift * TILE_SIZE_PX // 2 + TILE_SIZE_PX // 2)
        upper_tile_quarter = upper_tile.data[y_slice_dest, x_slice_dest]
        upper_tile_quarter += base_tile.data[0::2, 0::2]
        upper_tile_quarter += base_tile.data[0::2, 1::2]
        upper_tile_quarter += base_tile.data[1::2, 0::2]
        upper_tile_quarter += base_tile.data[1::2, 1::2]
    return upper_tile


# def smooth(density, max_value: float, sigma: float):
#     density = gaussian_filter(density, sigma=sigma)
#     density = np.log2(density + 1)
#     density = np.clip(density, 0, max_value)
#     density /= max_value
#     return density


# def render_tile(tile: Tile):
#     CLIP_VALUE = 2.5
#     density = tile.data
#     density = gaussian_filter(density, sigma=1.5)
#     density = np.log2(density + 1)
#     density = np.clip(density, 0, CLIP_VALUE + 1) / CLIP_VALUE
#     # plt.plot(density)
#     im = Image.fromarray(np.uint8(cm.plasma(density) * 255))
#     return im


def render_layer(
        tile_layer: TileLayer,
        dirpath: str
):
    for tile_coord, tile in tile_layer.tiles.items():
        if not tile.is_empty:
            image_filepath = path.join(dirpath, f'{tile_coord[0]:05}x{tile_coord[1]:05}.png')
            tile.dump_to_image(image_filepath)


def smooth_layer(
        input_tile_layer: TileLayer,
        output_tile_layer: TileLayer,
        sigma: float
):
    for tile_coord, data_tile in tqdm(input_tile_layer.tiles.items()):
        tile_chunk_coords = [
            [
                input_tile_layer.normalizes_coord((tile_coord[0] + x_shift, tile_coord[1] + y_shift))
                for x_shift in [-1, 0, 1]
            ] for y_shift in [-1, 0, 1]
        ]
        tile_chunk_data = np.vstack([
            np.hstack([input_tile_layer.tiles[x, y].data.astype(output_tile_layer.dtype)
                       for x, y in row])
            for row in tile_chunk_coords
        ])
        if np.all(tile_chunk_data == 0):
            continue
        tile_chunk_data = gaussian_filter(tile_chunk_data, sigma=sigma)
        tile_data = tile_chunk_data[TILE_SIZE_PX:2*TILE_SIZE_PX, TILE_SIZE_PX:2*TILE_SIZE_PX]
        output_tile_layer.tiles[tile_coord].data = tile_data
