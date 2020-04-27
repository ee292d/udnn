import os
import re
import sys
import platform
import subprocess
import multiprocessing
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    @staticmethod
    def is_windows():
        tag = platform.system().lower()
        return tag == "windows"

    @staticmethod
    def is_mendel():
        return "mendel" in platform.platform()

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        # test env variable to determine whether to build in debug
        if os.environ.get("DEBUG") is not None:
            cfg = 'Debug'
        else:
            cfg = 'Release'
        build_args = ['--config', cfg]
        env = os.environ.copy()

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        if self.is_windows():
            cmake_args += ["-G", "Unix Makefiles"]
            # make sure clang is in the PATH
            clang_path = shutil.which("clang")
            assert clang_path is not None, \
                "Unable to find clang. Currently only clang is supported "\
                "on Windows"
            cxx_path = shutil.which("clang++")
            rc_path = shutil.which("llvm-rc")
            cmake_args += ["-DCMAKE_C_COMPILER:PATH=" + clang_path]
            cmake_args += ["-DCMAKE_CXX_COMPILER:PATH=" + cxx_path]
            cmake_args += ["-DCMAKE_RC_COMPILER:PATH=" + rc_path]
            assert "MAKE_PROGRAM" in env,\
                "Currently only unix make is supported on windows"
            make_program = env["MAKE_PROGRAM"]
            cmake_args += [R"-DCMAKE_MAKE_PROGRAM=" + make_program]

        cpu_count = max(1, multiprocessing.cpu_count() // 2)
        if self.is_mendel():
            cpu_count = 1
        build_args += ['--', '-j{0}'.format(cpu_count)]

        python_path = sys.executable
        cmake_args += ['-DPYTHON_EXECUTABLE:FILEPATH=' + python_path]

        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)

        subprocess.check_call(
            ['cmake', '--build', '.', "--target", "_udnn"] + build_args,
            cwd=self.build_temp)


setup(
    name='udnn',
    packages=[
        "udnn"
    ],
    version='0.0.1',
    author='Keyi Zhang',
    author_email='keyi@cs.stanford.edu',
    description='SIMD DNN Library',
    ext_modules=[CMakeExtension('_udnn')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False
)
