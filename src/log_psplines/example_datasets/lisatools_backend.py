"""Register lisatools CPU/GPU backends with gpubackendtools if missing."""

from __future__ import annotations


def ensure_lisatools_backends() -> None:
    try:
        import numpy as np
        from gpubackendtools.globals import Globals, add_backends
        from gpubackendtools.gpubackendtools import (
            BackendMethods,
            BackendNotInstalled,
            CpuBackend,
            Cuda11xBackend,
            Cuda12xBackend,
        )
    except Exception:
        return

    manager = Globals().backends_manager
    if "lisatools_cpu" in manager.backend_list:
        return

    class LisatoolsCpuBackend(CpuBackend):
        _name = "lisatools_cpu"
        _backend_name = "lisatools_backend_cpu"

        @staticmethod
        def cpu_methods_loader() -> BackendMethods:
            return BackendMethods(xp=np)

        def __init__(self):
            super().__init__()
            try:
                from lisatools_backend_cpu import pycppdetector as cpp
            except ModuleNotFoundError as exc:
                raise BackendNotInstalled(
                    "The 'lisatools_cpu' backend is not installed."
                ) from exc
            self.OrbitsWrap = getattr(cpp, "OrbitsWrapCPU", None)
            self.Orbits = getattr(cpp, "OrbitsCPU", None)

    class LisatoolsCuda11xBackend(Cuda11xBackend):
        _name = "lisatools_cuda11x"
        _backend_name = "lisatools_backend_cuda11x"

        @staticmethod
        def cuda11x_module_loader() -> BackendMethods:
            import cupy

            return BackendMethods(xp=cupy)

    class LisatoolsCuda12xBackend(Cuda12xBackend):
        _name = "lisatools_cuda12x"
        _backend_name = "lisatools_backend_cuda12x"

        @staticmethod
        def cuda12x_module_loader() -> BackendMethods:
            import cupy

            return BackendMethods(xp=cupy)

    manager.add_backends(
        {
            "lisatools_cpu": LisatoolsCpuBackend,
            "lisatools_cuda11x": LisatoolsCuda11xBackend,
            "lisatools_cuda12x": LisatoolsCuda12xBackend,
        }
    )
