#!/usr/bin/env python3
import argparse
import logging
import os.path as path
from typing import Optional
from pyramid import create_heatmap_pyramid
from coords import load_coords


logger = logging.getLogger('tyler')


class VerbosityParsor(argparse.Action):
    """ accept debug, info, ... or theirs corresponding integer value formatted as string."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:  # in case it represent an int, directly get it
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
        parser.add_argument('-o', '--output', type=str,
                            help='output directory')
        parser.add_argument('-l', '--levels', type=int, default=8,
                            help='max pyramid level [8]')
        parser.add_argument('-e', '--empty', nargs='?', choices=['skip', 'keep'], default='skip', const='keep',
                            help='max pyramid level [8]')

        args = parser.parse_args()
        # normalize input path
        args.input = path.abspath(args.input)
        args.output = path.abspath(args.output)
        # logger stdout
        logger.setLevel(args.verbose)
        # config
        logger.debug('config:\n' + '\n'.join(f'\t\t{k}={v}' for k, v in vars(args).items()))

        coords_ll = load_coords(args.input)
        create_heatmap_pyramid(
            coords_ll=coords_ll,
            output_dirpath=args.output,
            max_level=args.levels,
            force_empty_tiles=(args.empty == 'keep')
        )

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