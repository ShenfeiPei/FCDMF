import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('FCDMF2', parent_package, top_path)
    config.add_subpackage('FCDMF')
    #  config.add_subpackage('FCDMF_valarray')
    config.add_subpackage('Public')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
