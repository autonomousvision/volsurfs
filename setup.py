import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from distutils.command.install_headers import install_headers as install_headers_orig

import shutil

if shutil.which("nvcc") is None:
    raise RuntimeError("CUDA compiler (nvcc) is not found! Ensure CUDA is installed.")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            #   '-GNinja'
        ]

        # cfg = 'Debug' if self.debug else 'Release'
        # build_args = ['--config', cfg]
        build_args = []

        if platform.system() == "Windows":
            cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            # cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ["--", "-j4"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # print ("build temp is ", self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


with open("README.md", "r") as fh:
    long_description = fh.read()


# install headers while retaining the structure of the tree folder
# https://stackoverflow.com/a/50114715
class install_headers(install_headers_orig):
    def run(self):
        headers = self.distribution.headers or []
        for header in headers:
            dst = os.path.join(self.install_dir, os.path.dirname(header))
            print("----------------copying in ", dst)
            self.mkpath(dst)
            (out, _) = self.copy_file(header, dst)
            self.outfiles.append(out)


setup(
    name="volsurfs",
    version="1.0.0",
    author="Stefano Esposito",
    author_email="stefano.esposito@uni-tuebingen.de",
    description="volsurfs",
    long_description=long_description,
    ext_modules=[CMakeExtension("volsurfs")],
    cmdclass={
        "build_ext": CMakeBuild,
        "install_headers": install_headers,
    },
    setup_requires=["setuptools", "pybind11[global]", "torch>=2.1.0"],
    install_requires=[
        "hjson",
        "ipython",
        "wandb",
        "piq==0.8.0",
        "scikit-image==0.21.0",
        "pymeshlab==2023.12.post2",
        "xatlas==0.0.9",
        "trimesh==4.6.0",
        "gdown",
    ],
    zip_safe=False,
)
