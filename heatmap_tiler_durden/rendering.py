import os
import os.path as path
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import numpy as np
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

from tile_layer import TileLayer, get_tile_coord_list, get_tiles_neighborhood_33, get_tiles_lower
from tile import Tile, TILE_SIZE_PX

COLORMAP_BASE = cm.plasma  # Choose colormap
COLORMAP_ALPHA = COLORMAP_BASE(np.arange(COLORMAP_BASE.N))  # Get the colormap colors
COLORMAP_ALPHA[0, -1] = 0
COLORMAP_ALPHA = ListedColormap(COLORMAP_ALPHA)  # Create new colormap


def smooth_layer(
        input_tile_layer: TileLayer,
        output_tile_layer: TileLayer,
        sigma: float
):
    for tile_coord, data_tile in tqdm(input_tile_layer.tiles.items()):
        tile_chunk_coords = get_tiles_neighborhood_33(tile_coord, level=input_tile_layer.level)
        tile_chunk_data = np.vstack([
            np.hstack([input_tile_layer.tiles[x, y].data.astype(output_tile_layer.dtype)
                       for x, y in row])
            for row in tile_chunk_coords
        ])
        if np.all(tile_chunk_data == 0):
            continue
        tile_chunk_data = gaussian_filter(tile_chunk_data, sigma=sigma)
        tile_data = tile_chunk_data[TILE_SIZE_PX:2 * TILE_SIZE_PX, TILE_SIZE_PX:2 * TILE_SIZE_PX]
        output_tile_layer.tiles[tile_coord].data = tile_data


def renders_tile_to_image(
        tile: Tile,
        image_filepath
):
    os.makedirs(path.dirname(image_filepath), exist_ok=True)
    image = Image.fromarray(np.uint8(COLORMAP_ALPHA(tile.data) * 255))
    image.save(image_filepath)


def render_layer_to_images(
        tile_layer: TileLayer,
        layer_level: int,
        image_filepath_template: str
):
    for tile_coord, tile in tile_layer.tiles.items():
        if not tile.is_empty:
            image_filepath = image_filepath_template.format(x=tile_coord[0], y=tile_coord[1], z=layer_level)
            renders_tile_to_image(tile, image_filepath)


def downsample_image_layer(
        image_filepath_template: str,
        input_layer_level: int,
):
    lower_layer_level = input_layer_level
    upper_layer_level = lower_layer_level - 1
    upper_tile_coords = get_tile_coord_list(upper_layer_level)
    for upper_tile_coord in tqdm(upper_tile_coords):
        lower_tile_coords = get_tiles_lower(upper_tile_coord=upper_tile_coord)
        upper_image = Image.new('RGBA', (TILE_SIZE_PX * 2, TILE_SIZE_PX * 2))
        some_modifications = False
        for row, lower_tile_coord_row in enumerate(lower_tile_coords):
            for col, lower_tile_coord in enumerate(lower_tile_coord_row):
                lower_image_filepath = image_filepath_template.format(x=lower_tile_coord[0],
                                                                      y=lower_tile_coord[1],
                                                                      z=lower_layer_level)
                if path.isfile(lower_image_filepath):
                    lower_image = Image.open(lower_image_filepath)
                    upper_image.paste(lower_image, (col * TILE_SIZE_PX, row * TILE_SIZE_PX))
                    some_modifications = True
        if some_modifications:
            upper_image.resize((TILE_SIZE_PX, TILE_SIZE_PX), Image.BICUBIC)
            upper_image_filepath = image_filepath_template.format(x=upper_tile_coord[0],
                                                                  y=upper_tile_coord[1],
                                                                  z=upper_layer_level)
            os.makedirs(path.dirname(upper_image_filepath), exist_ok=True)
            upper_image.save(upper_image_filepath)

        # def resample_image_layer(
        #         input_image_filepath_template: str,
        #         output_image_filepath_template: str,
        # ):
        #     pass
