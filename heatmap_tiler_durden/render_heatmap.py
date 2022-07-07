#!/usr/bin/env python3
import argparse
import logging
import os.path as path
from tile_layer import TileLayer
from rendering import render_layer_to_images, downsample_image_layer, smooth_layer

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
        parser.add_argument(
            '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO, action=VerbosityParsor,
            help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
        parser.add_argument('-i', '--input', type=str, required=True,
                            help='input tile layer')
        parser.add_argument('-o', '--output', type=str, required=True,
                            help='output image directory')
        parser.add_argument('-c', '--cache', type=str, required=True,
                            help='cache directory (to store the pyramid)')
        parser.add_argument('-l', '--nb_levels', type=int, required=True,
                            help='input tile layer')

        args = parser.parse_args()
        args.input = path.abspath(args.input)
        args.output = path.abspath(args.output)
        logger.setLevel(args.verbose)
        logger.debug('config:\n' + '\n'.join(f'\t\t{k}={v}' for k, v in vars(args).items()))

        image_filepath_template = path.join(args.output, '{z}/{x:05}x{y:05}.png')
        base_level = args.nb_levels-1
        data_base_layer = TileLayer(dirpath=args.input, level=base_level, dtype=int)
        logger.info(f'smoothing data.')
        proc_base_layer = TileLayer(dirpath=args.cache + f'/smooth/{base_level}', level=base_level, dtype=float)
        smooth_layer(data_base_layer, proc_base_layer, sigma=3.)
        render_layer_to_images(tile_layer=proc_base_layer,
                               layer_level=base_level,
                               image_filepath_template=image_filepath_template)
        for level in reversed(range(1, base_level+1)):
            logger.info(f'resampling {level}')
            downsample_image_layer(image_filepath_template=image_filepath_template,
                                   input_layer_level=level)

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
