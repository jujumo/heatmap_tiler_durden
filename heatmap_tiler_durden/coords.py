import pandas as pd
from pyproj import Transformer
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Set

EARTH_RADIUS = 6378137.0
EARTH_CIRC_M = 2 * np.pi * EARTH_RADIUS
TILE_SIZE_PX = 256
METERS_PER_PIXEL_L0 = EARTH_CIRC_M / TILE_SIZE_PX
PYPROJ = Transformer.from_crs("EPSG:4326", "EPSG:3857")  # EPSG:900913 == EPSG:3857


# inspired from
# - https://www.maptiler.com/google-maps-coordinates-tile-bounds-projection/#5/150.51/68.42


def load_coords(filepath: str):
    """ Loads csv file with coordinates, and returns a numpy array with col-wise coordinates. """
    df = pd.read_csv(filepath, sep=',')
    coords_ll = df[['longitude', 'latitude']].to_numpy()
    coords_ll = coords_ll.transpose()
    return coords_ll


def ll_to_meters(coords_ll):
    # Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:3857 (aka EPSG:900913)
    coords_m = PYPROJ.transform(coords_ll[1], coords_ll[0])
    coords_m = np.array(coords_m)
    return coords_m


def meters_to_pixels(coords_m, level):
    # Converts EPSG:3857 to pyramid pixel coordinates in given zoom level
    coords_px = (coords_m + (EARTH_CIRC_M / 2.0)) / meters_per_pixel(level)
    return coords_px


def pixels_to_meters(coords_px, level):
    # Converts pixel coordinates in given zoom level to EPSG:3857
    coords_m = (coords_px * meters_per_pixel(level)) - (EARTH_CIRC_M / 2)
    return coords_m


def pixel_to_raster(coords_px, level):
    # Move the origin of pixel coordinates to top-left corner (like image)
    coords_rx = np.empty_like(coords_px)
    coords_rx[0] = coords_px[0]
    coords_rx[1] = map_size(level) - coords_px[1]
    return coords_rx


def raster_to_tile(coords_rx, tile_coord: Tuple[int, int]):
    # Move the origin of pixel coordinates to top-left corner (like image)
    # filter out points outside the tile
    bounds_rx = tile_bounds_rx(tile_coord)
    coords_ti = coords_rx - bounds_rx[:, 0:1]
    return coords_ti


def keep_in_tile_rx(coords_rx, tile_coord: Tuple[int, int]):
    bounds_rx = tile_bounds_rx(tile_coord)
    in_tile = np.vstack([
        coords_rx > bounds_rx[:, 0:1],
        coords_rx < bounds_rx[:, 2:3],
    ])
    in_tile = np.all(in_tile, axis=0)
    return coords_rx[:, in_tile]


def ll_to_raster(coords_ll, level):
    coords_m = ll_to_meters(coords_ll)
    coords_px = meters_to_pixels(coords_m, level=level)
    coords_rx = pixel_to_raster(coords_px, level=level)
    return coords_rx


def ll_to_tile(coords_ll, tile_coord: Tuple[int, int], level):
    coords_m = ll_to_meters(coords_ll)
    coords_px = meters_to_pixels(coords_m, level=level)
    coords_rx = pixel_to_raster(coords_px, level=level)
    coords_ti = raster_to_tile(coords_rx, tile_coord)
    return coords_ti


def nb_tiles(level):
    return 2 ** level


def map_size(level):
    return TILE_SIZE_PX << level


def tile_bounds_rx(tile_coords: Tuple[int, int]):
    # Returns bounds of the given tile in EPSG:900913 coordinates
    coords_tile = np.array([[tile_coords[0], tile_coords[1]]]).transpose()
    square = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ]).transpose()

    coord_bounds_px = (square + coords_tile) * TILE_SIZE_PX
    return coord_bounds_px


def tile_set_rx(coords_rx: np.ndarray) -> Set[Tuple[int, int]]:
    # returns coordinates of tiles involved
    tile_coords = {(x, y) for x, y in np.floor(coords_rx / TILE_SIZE_PX).astype(int).transpose()}
    return tile_coords


def meters_per_pixel(level):
    # Resolution (meters/pixel) for given zoom level (measured at Equator)
    return METERS_PER_PIXEL_L0 / nb_tiles(level)
