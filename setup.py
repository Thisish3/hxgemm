import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="hxgemm_ext",
    ext_modules=[CUDAExtension(
        "hxgemm_ext",
        ["csrc/torch_bindings.cpp", "csrc/dispatch.cu"] + sorted(glob.glob("csrc/kernels/*.cu")),
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-gencode=arch=compute_86,code=sm_86", "-lineinfo"]},
    )],
    cmdclass={"build_ext": BuildExtension},
)
