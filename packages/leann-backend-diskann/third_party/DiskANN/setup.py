# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win-amd64": "x64"
}


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        extdir = ext_fullpath.parent.resolve()
        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DVERSION_INFO={self.distribution.get_version()}"  # commented out, we want this set in the CMake file
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        # cmake_args += [f"-DVERSION_INFO={self.distribution.get_version()}"]  # type: ignore[attr-defined]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja  # noqa: F401

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        # this next line is problematic. we tell it to use the ext.sourcedir but, when
        # using `python -m build`, we actually have a copy of everything made and pushed
        # into a venv isolation area
        if os.environ.get("USE_CONDA", "") == '1' and os.environ.get("CONDA_PREFIX", "") != "":
            subprocess.run(
                ["cmake", "-DPYBIND=True", "-DCMAKE_PREFIX_PATH=" + os.environ.get("CONDA_PREFIX", ""), 
                "-DProtobuf_DIR=" + os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib/cmake/protobuf"), 
                ext.sourcedir] + cmake_args, cwd=build_temp, check=True
            )
        else:
            subprocess.run(
                ["cmake", "-DPYBIND=True", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
            )
        
        subprocess.run(
            ["cmake", "--build", "."] + build_args, cwd=build_temp, check=True
        )


class InstallCMakeLibs(install_lib):
    def run(self):
        """
        Windows only copy from the x64/Release directory and place them in the package
        """

        self.announce("Moving library files", level=3)

        self.skip_build = True

        # we only need to move the windows build output
        windows_build_output_dir = Path('.') / 'x64' / 'Release'

        if windows_build_output_dir.exists():
            libs = [
                os.path.join(windows_build_output_dir, _lib) for _lib in
                os.listdir(windows_build_output_dir) if
                os.path.isfile(os.path.join(windows_build_output_dir, _lib)) and
                os.path.splitext(_lib)[1] in [".dll", '.lib', '.pyd', '.exp']
            ]

            for lib in libs:
                shutil.move(
                    lib,
                    os.path.join(self.build_dir, 'diskannpy', os.path.basename(lib))
                )

        super().run()


setup(
    ext_modules=[CMakeExtension("diskannpy._diskannpy", ".")],
    cmdclass={
        "build_ext": CMakeBuild,
        'install_lib': InstallCMakeLibs
    },
    zip_safe=False,
    package_dir={"diskannpy": "python/src"},
    exclude_package_data={"diskannpy": ["diskann_bindings.cpp"]}
)
