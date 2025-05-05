# set(LIBNVTOOLSEXT ${CUDA_nvToolsExt_LIBRARY})
# message(STATUS "LIBNVTOOLSEXT": ${LIBNVTOOLSEXT})

cmake_minimum_required(VERSION 3.18)

project(volsurfs VERSION 1.0 LANGUAGES CXX CUDA)

### ──────────────── Build Settings ────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(CUDA)

### ──────────────── Eigen (user-supplied or system) ────────────────

message(STATUS "Using provided Eigen include dir: ${EIGEN3_INCLUDE_DIR}")

### ──────────────── pybind11 (system first, vendored fallback) ────────────────

message(STATUS "Using vendored pybind11: ${PYBIND11_DIR}")
add_subdirectory("${PYBIND11_DIR}" EXCLUDE_FROM_ALL)

### ──────────────── PyTorch ────────────────
execute_process(
  COMMAND python -c "import torch, os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
if (TorchPath STREQUAL "")
  message(FATAL_ERROR "PyTorch not found. Please install it.")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TorchPath}")
find_package(Torch REQUIRED)

execute_process(
  COMMAND python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
  OUTPUT_VARIABLE TORCH_USES_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

### ──────────────── CUDA Arch Detection ────────────────
option(VOLSURFS_CUDA_ARCH_FALLBACK "Use default CUDA archs if detection fails" ON)

function(auto_detect_cuda_arch out_var)
  file(WRITE ${PROJECT_BINARY_DIR}/detect_arch.cu "
#include <cuda_runtime.h>
#include <cstdio>
int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return -1;
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
      printf(\"%d%d;\", prop.major, prop.minor);
  }
  return 0;
}")
  try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${PROJECT_BINARY_DIR}/detect_arch.cu
    RUN_OUTPUT_VARIABLE archs)
  if (run_result EQUAL 0)
    list(REMOVE_DUPLICATES archs)
    set(${out_var} ${archs} PARENT_SCOPE)
  elseif (VOLSURFS_CUDA_ARCH_FALLBACK)
    message(WARNING "CUDA architecture detection failed. Using fallback.")
    set(${out_var} "75;86" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "Failed to detect CUDA architectures and fallback is disabled.")
  endif()
endfunction()

auto_detect_cuda_arch(CMAKE_CUDA_ARCHITECTURES)
message(STATUS "Detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

### ──────────────── CUDA Library ────────────────
file(GLOB_RECURSE VOLSURFS_CUDA_SOURCES
  "${CMAKE_SOURCE_DIR}/src/*.cu"
)

add_library(volsurfs_cu SHARED ${VOLSURFS_CUDA_SOURCES})
target_compile_options(volsurfs_cu PUBLIC
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr --disable-warnings>
)
target_include_directories(volsurfs_cu PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/kernels
  ${EIGEN3_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(volsurfs_cu PRIVATE ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})

### ──────────────── Pybind11 Module ────────────────
pybind11_add_module(volsurfs src/PyBridge.cxx)
target_include_directories(volsurfs PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/kernels
  ${EIGEN3_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)
target_compile_definitions(volsurfs PRIVATE
  $<$<BOOL:${TORCH_USES_CXX11_ABI}>:_GLIBCXX_USE_CXX11_ABI=1>
  $<$<NOT:$<BOOL:${TORCH_USES_CXX11_ABI}>>:_GLIBCXX_USE_CXX11_ABI=0>
)
target_link_libraries(volsurfs PRIVATE volsurfs_cu)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TorchPath}/lib")
if (NOT TORCH_PYTHON_LIBRARY)
  message(FATAL_ERROR "torch_python not found.")
endif()
target_link_libraries(volsurfs PRIVATE ${TORCH_PYTHON_LIBRARY})
