import torch


class rocmBLASModule:
    def __getattr__(self, name):
        assert name == "allow_f8", "Unknown attribute" + name
        return torch._C._get_rocm_rocblas_allow_f8()

    def __setattr__(self, name, value):
        assert name == "allow_f8", "Unknown attribute" + name
        return torch._C._set_rocm_rocblas_allow_f8(value)


class rocmMIOpenModule:
    def __getattr__(self, name):
        assert name == "allow_f8", "Unknown attribute" + name
        return torch._C._get_rocm_miopen_allow_f8()

    def __setattr__(self, name, value):
        assert name == "allow_f8", "Unknown attribute" + name
        return torch._C._set_rocm_miopen_allow_f8(value)


rocblas = rocmBLASModule()
miopen = rocmMIOpenModule()


