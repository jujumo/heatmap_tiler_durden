#!/usr/bin/env python3
import argparse
import logging
import os.path as path
from typing import Optional
from pyramid import TilePyramid, TilePyramidLayer, add_spots
from coords import load_coords, map_size
import numpy as np

logger = logging.getLogger('tyler')


class VerbosityParsor(argparse.Action):
    """ accept debug, info, ... or theirs corresponding integer value formatted as string."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:  # in case it represents an int, directly get it
            values = int(values)
        except ValueError:  # else ask logging to sort it out
            assert isinstance(values, str)
            values = logging.getLevelName(values.upper())
        setattr(namespace, self.dest, values)


def main():
    try:
        parser = argparse.ArgumentParser(description='Generates the tile pyramid.')
        parser_verbosity = parser.add_mutually_exclusive_group()
        parser_verbosity.add_argument(
            '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO, action=VerbosityParsor,
            help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
        parser_verbosity.add_argument(
            '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
        parser.add_argument('-i', '--input', type=str, required=True,
                            help='input csv file')
        parser.add_argument('-o', '--output', type=str, required=True,
                            help='output directory')
        parser.add_argument('-l', '--levels', type=int, default=8,
                            help='max pyramid level [8]')

        args = parser.parse_args()
        # normalize input path
        args.input = path.abspath(args.input)
        args.output = path.abspath(args.output)
        # logger stdout
        logger.setLevel(args.verbose)
        # config

        logger.debug('config:\n' + '\n'.join(f'\t\t{k}={v}' for k, v in vars(args).items()))

        data_path = path.join(args.output, 'data')
        coords_ll = load_coords(args.input)

        logger.info(f'adding tracks to heatmaps.')

        base_layer = TilePyramidLayer(root_dirpath=data_path , level=args.levels, dtype=np.uint8)
        add_spots(base_layer, coords_ll)
        logger.info(f'saving to disk.')
        base_layer.flush_to_disk()
        # upper_layer = base_layer.create_upper_layer()
        pyramid = TilePyramid.create_from_base_layer(base_layer)
        pyramid.flush_to_disk()
        # logger.info(f'exporting to images.')

        # data_pyramid.dump_to_images(image_rootpath=path.join(args.output, 'image'))

    except Exception as e:
        logger.critical(e)
        if args.verbose <= logging.DEBUG:
            raise


if __name__ == '__main__':
    logger_formatter = logging.Formatter('%(name)s::%(levelname)-8s: %(message)s')
    logger_stream = logging.StreamHandler()
    logger_stream.setFormatter(logger_formatter)
    logger.addHandler(logger_stream)
    main()
