#!/usr/bin/env python3
import argparse
import logging
import os.path as path
from tile_layer import TileLayer
from coords import load_coords
import numpy as np
from heatmap import add_spots

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
        parser = argparse.ArgumentParser(description='Update heat-map tiles from csv file.')
        parser.add_argument(
            '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO, action=VerbosityParsor,
            help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
        parser.add_argument(
            '-i', '--input', type=str, required=True,
            help='input csv file')
        parser.add_argument(
            '-o', '--output', type=str, required=True,
            help='output directory')
        parser.add_argument(
            '-l', '--nb_levels', type=int, default=9,
            help='max pyramid level [9]')

        args = parser.parse_args()
        # normalize input path
        args.input = path.abspath(args.input)
        args.output = path.abspath(args.output)
        logger.setLevel(args.verbose)
        # config
        logger.debug('config:\n' + '\n'.join(f'\t\t{k}={v}' for k, v in vars(args).items()))

        logger.info(f'loading csv.')
        coords_ll = load_coords(args.input)
        logger.info(f'adding tracks to heatmaps.')
        base_level = args.nb_levels - 1
        base_layer = TileLayer(dirpath=args.output, level=base_level, dtype=np.uint8)
        add_spots(base_layer, coords_ll)
        logger.info(f'done.')

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
