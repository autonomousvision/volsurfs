import os
import sys
import subprocess
import urllib.request
import tarfile
import shutil

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

_src_path = os.path.abspath(os.path.dirname(__file__))


def get_pybind11_include():
    PYBIND11_WEB_URL = "https://github.com/pybind/pybind11/archive/refs/tags/v2.11.0.tar.gz"
    TMP_PYBIND11_FILE = "tmp_pybind11.tar.gz"
    PYBIND11_DIRNAME = "pybind11-2.11.0"

    target_dir = os.path.join(_src_path, "cpp/third_party", PYBIND11_DIRNAME)
    if os.path.exists(target_dir):
        return target_dir
    else:
        print("Couldn't find pybind11 locally, downloading...")
        req = urllib.request.Request(
            PYBIND11_WEB_URL,
            data=None,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        ext_dir = os.path.join(_src_path, "cpp/third_party")
        os.makedirs(ext_dir, exist_ok=True)

        pybind11_archive_path = os.path.join(ext_dir, TMP_PYBIND11_FILE)
        with urllib.request.urlopen(req) as resp, open(pybind11_archive_path, "wb") as file:
            file.write(resp.read())

        with tarfile.open(pybind11_archive_path) as tar:
            tar.extractall(path=ext_dir)

        os.remove(pybind11_archive_path)
        return target_dir
            
                        
def get_eigen_include():
    EIGEN_WEB_URL = (
        "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2"
    )
    TMP_EIGEN_FILE = "tmp_eigen.tar.bz2"
    EIGEN3_DIRNAME = "eigen-3.3.7"

    target_dir = os.path.join(_src_path, "cpp/third_party", EIGEN3_DIRNAME)
    if os.path.exists(target_dir):
        return target_dir
    else:
        print("Couldn't find Eigen locally, downloading...")
        req = urllib.request.Request(
            EIGEN_WEB_URL,
            data=None,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        ext_dir = os.path.join(_src_path, "cpp/third_party")
        os.makedirs(ext_dir, exist_ok=True)

        eigen_archive_path = os.path.join(ext_dir, TMP_EIGEN_FILE)
        with urllib.request.urlopen(req) as resp, open(eigen_archive_path, "wb") as file:
            file.write(resp.read())

        with tarfile.open(eigen_archive_path) as tar:
            tar.extractall(path=ext_dir)

        os.remove(eigen_archive_path)
        return target_dir


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="cpp"):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_dir = self.build_temp
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        eigen_dir = get_eigen_include()
        pybind11_dir = get_pybind11_include()

        os.makedirs(build_dir, exist_ok=True)

        if shutil.which("ninja") is None:
            raise RuntimeError("Ninja is not installed or not in PATH. Please install ninja-build.")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DEIGEN3_INCLUDE_DIR={eigen_dir}",
            f"-DPYBIND11_DIR={pybind11_dir}",
        ]
        build_args = ["--", "-j4"]

        # Use Ninja generator
        subprocess.check_call(["cmake", ext.sourcedir, "-G", "Ninja"] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_dir)


setup(
    name="volsurfs",
    version="1.0.0",
    description="CUDA-accelerated volumetric rendering with PyTorch",
    author="Stefano Esposito",
    author_email="your.email@example.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["volsurfs", "volsurfs.*"]),
    ext_modules=[CMakeExtension("volsurfs._volsurfs")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "torch>=2.1",
        "numpy",
        "hjson",
        "wandb",
        "piq==0.8.0",
        "scikit-image==0.21.0",
        "pymeshlab==2023.12.post2",
        "xatlas==0.0.9",
        "trimesh==4.6.0",
        "gdown"
    ],
    python_requires=">=3.8",
    zip_safe=False,
)