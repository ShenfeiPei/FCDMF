import os
import sys
import numpy
from Cython.Build import cythonize
from FCDMF_pack.Public import cg

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('FCDMF_valarray', parent_package, top_path)
    config.add_extension('FCDMF_',
                         sources=['FCDMF_.pyx'],
                         include_dirs=[numpy.get_include(), cg.cppfuns_path],
                         language="c++",

                         extra_compile_args=cg.ext_comp_args,
                         extra_link_args=cg.ext_link_args,

                         library_dirs=cg.library_dirs,
                         libraries=cg.library_dirs,

                         #  define_macros=cg.define_macros,
                         )

    config.ext_modules = cythonize(config.ext_modules, compiler_directives={'language_level': 3})

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
