from __future__ import annotations

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


ext_modules = [
    Pybind11Extension(
        "bondviz._native",
        ["src/bondviz/_native.cpp"],
        cxx_std=17,
    )
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

