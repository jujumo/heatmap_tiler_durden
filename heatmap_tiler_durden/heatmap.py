import os

import numpy as np
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import matplotlib.cm as cm
from io import BytesIO
import os.path as path


def build_heatmap(coords_ti):
    density = np.zeros((256, 256), dtype=float)
    for x, y in tqdm(coords_ti.astype(int).transpose()):
        try:
            density[y, x] += 1
        except IndexError:
            pass
    CLIP_VALUE = 2.5
    density = gaussian_filter(density, sigma=1.5)
    density = np.log2(density + 1)
    density = np.clip(density, 0, CLIP_VALUE+1) / CLIP_VALUE
    # plt.plot(density)
    im = Image.fromarray(np.uint8(cm.plasma(density) * 255))
    return im


class HeatmapPyramid:
    def __init__(self, coords_ll, converter, cache_dirpath):
        self.coords_ll = coords_ll
        self.converter = converter
        self.cache_dirpath = cache_dirpath

    def get_tile(self, x, y, z):
        cache_filepath = path.abspath(path.join(self.cache_dirpath, f'{z:02}',
                                   f'{x:02}x{y:02}.png')) if self.cache_dirpath is not None else None
        tile = None
        if cache_filepath is not None and path.isfile(cache_filepath):
            tile = Image.open(cache_filepath)
        if tile is None:
            coords_ti = self.converter.ll_to_tile(self.coords_ll, x, y, z)
            tile = build_heatmap(coords_ti)

        if cache_filepath is not None and not path.isfile(cache_filepath):
            os.makedirs(path.dirname(cache_filepath), exist_ok=True)
            tile.save(cache_filepath)
            print(f'saving {cache_filepath}')

        return tile
