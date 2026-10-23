"""Build the eight-qubit CustomDevice density-matrix and statevector kernels."""
from pathlib import Path
import os
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

os.chdir(Path(__file__).resolve().parent)
options = ["/O2", "/utf-8"] if sys.platform == "win32" else ["-O3"]
setup(
    name="qml-hybrid-custom-device", version="0.1.0",
    ext_modules=[Pybind11Extension(
        f"engine8q_{kind}", sources=[f"engine_{kind}.cpp"],
        define_macros=[("NUM_QUBITS", "8"), ("MODULE_NAME", f"engine8q_{kind}")],
        extra_compile_args=options, cxx_std=17,
    ) for kind in ("dm", "sv")],
    cmdclass={"build_ext": build_ext},
)
