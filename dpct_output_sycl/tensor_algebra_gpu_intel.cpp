/** Tensor Algebra Library for INTel GPU: INT-TAL (SYCL based).
    AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com, liakhdi@ornl.gov
    REVISION: 2020/07/21

    Copyright (C) 2014-2020 Dmitry I. Lyakh (Liakh)
    Copyright (C) 2014-2020 Oak Ridge National Laboratory (UT-Battelle)

    This file is part of ExaTensor.

    ExaTensor is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ExaTensor is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with ExaTensor. If not, see <http://www.gnu.org/licenses/>.
    ------------------------------------------------------------------------
    OPTIONS:
    # -D CUDA_ARCH=350: target GPU compute capability (default is 130);
    # -D NO_GPU: disables GPU usage;
    # -D NO_BLAS: disables cuBLAS calls, they will be replaced by in-house
routines (slower); # -D USE_CUTT: enables an optimized tensor transpose via the
cuTT library; # -D DEBUG_GPU: collection of debugging information will be
activated; NOTES: # Minimal required compute capability is 1.1 (1.3 for double
precision). # cuBLAS.v2 is required when BLAS is enabled. # Non-blocking tensor
algebra functions carry an additional output argument <sycl_task> (task handle).
    # Non-blocking tensor algebra functions carry an additional input argument
<coherence_ctrl> which controls the tensor data consistency synchronization
accross different devices after the tensor operation has completed successfully.
    FOR DEVELOPERS ONLY:
    # Currently used device resources:
    - Global memory pointer (any device);
    - Argument buffer entry handle (any device);
    - Multi-index entry * (Host pinned memory, entry length = MAX_TENSOR_RANK);
    - GPU constant-memory entry handle (Nvidia GPU);
    - CUDA stream handle (Nvidia GPU);
    - SYCL queue handle (Nvidia GPU).
    # A life cycle of a C object (for example, tensBlck_t):
    a) Allocate memory for the object itself, if needed: Suffix _alloc or
_create (includes cleaning); b) Clean (initialize to null) an allocated (empty)
object: Suffix _clean (normally included in _create); c) Construct (define or
redefine) an existing object (resources will be acquired/released): Suffix
_construct; d) Destruct a defined object (resources will be released, the object
will be reset to clean): Suffix _destruct; e) Free the memory occupied by an
object: Suffix _free or _destroy (may include _destruct, if needed). Thus, as a
rule, the device resource acquisition/release occurs solely in _construct and
_destruct functions. # A state of a C object: a) Undefined: After the memory
allocation (either dynamic or static); b) Defined-empty (clean): After cleaning
or destruction (dynamic object creation produces a clean object); c) Defined to
a value (value-defined): After construction; d) Dead: After memory deallocation
(if it was allocated dynamically). # Resource acquisition/release:
    - Tensor block constructor/destructor acquires/releases global memory
resources, including both pointers and buffer entries, as well as multi-index
bank entries (pinned Host memory).
    - CUDA task constructor/destructor acquires/releases CUDA resources (stream,
events).
    - Tensor operation scheduling functions acquire GPU global memory resources,
    GPU constant memory resources, Host pinned multi-index entries.
    - CUDA task completion/error query functions release GPU global memory
resources, GPU constant memory resources, Host pinned multi-index entries.
    - Coherence control is only applied to successfully finished CUDA tasks.
    # Functions which construct tensor blocks or perform asynchronous operations
on them allocate resources (global/constant memory, etc). In case the
corresponding resource allocator returns TRY_LATER or DEVICE_UNABLE (or an
error), the corresponding function must clean the partially created tensor block
or the CUDA task before returning: The corresponding object will be kept in its
initial state if no SUCCESS. # Some CUDA kernels operating on two or more
arguments assume no aliases for GPU pointers (__restrict__). Check each specific
operation to see whether it is ok for the two tensor arguments to refer to the
same tensor body. TO BE FIXED: # In tensor operation scheduling functions, if a
scheduling error occurs after the data transfer or CUDA kernel has been
scheduled, the CUDA task finalization must not begin until the partially
scheduled CUDA task has completed on GPU. Insert cudaStreamSynchronize in the
finalization procedure. # Invoke cudaDeviceCanAccessPeer() in tensor operations
to check whether two devices of the same kind can access each other's memory. #
Account for implicit data transfers to/from peer GPUs in their statistics. #
User-provided Alpha factors for gpu_tensor_block_contract() and
    gpu_tensor_block_add() reside on Host, thus requiring a slab in GPU
    memory (either global or constant) as a temporary for BLAS references.
**/

#include "device_algebra.h"
#include "mem_manager.h"
#include "talsh_complex.h"
#include "tensor_algebra.h"
#include <CL/sycl.hpp>
#include "sycl_device.hpp"

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <cmath>

#include <chrono>

template <typename T, int DIM>
using local_accessor =
    cl::sycl::accessor<T, DIM, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>;
template <typename T, int DIM>
using constant_accessor =
    cl::sycl::accessor<T, DIM, cl::sycl::access::mode::read,
                       cl::sycl::access::target::constant>;

template <typename T, int DIM> using globalBuffer = cl::sycl::buffer<T, DIM>;

template <typename T>
using atomic_ref = cl::sycl::ONEAPI::atomic_ref< T, cl::sycl::ONEAPI::memory_order::relaxed,
						 cl::sycl::ONEAPI::memory_scope::device,
						 cl::sycl::access::address_space::global_space>;

#ifndef NO_GPU
// PARAMETERS:
#define GPU_DEBUG_DUMP_SIZE 128 // size of the GPU debug dump (int array)
#endif                          /*NO_GPU*/
//----------------------------------------------------------------------
// FUNCTION PROTOTYPES:
// LOCAL (PRIVATE):
static int prmn_convert(int n, const int *o2n, int *n2o);
static int non_trivial_prmn(int n, const int *prm);
#ifndef NO_GPU
static int sycl_queue_get(int gpu_num, int *sycl_queue_handle);
static int sycl_queue_release(int gpu_num, int sycl_queue_handle);
static cl::sycl::queue **sycl_stream_ptr(int gpu_num, int sycl_stream_handle);
static int sycl_event_get(int gpu_num, int *sycl_event_handle);
static int sycl_event_release(int gpu_num, int sycl_event_handle);
static cl::sycl::event *sycl_event_ptr(int gpu_num, int sycl_event_handle);
static void limit_sycl_workgroups2d(int max_blocks, int *bx, int *by);
static int tens_op_best_gpu(const tensBlck_t *tens0 = nullptr,
                            const tensBlck_t *tens1 = nullptr,
                            const tensBlck_t *tens2 = nullptr);
static int sycl_task_set_arg(cudaTask_t *sycl_task, unsigned int arg_num,
                             tensBlck_t *tens_p);
static int sycl_task_set_prefactor(cudaTask_t *sycl_task,
                                   talshComplex4 prefactor);
static int sycl_task_set_prefactor(cudaTask_t *sycl_task,
                                   talshComplex8 prefactor);
static int sycl_task_record(cudaTask_t *sycl_task, unsigned int coh_ctrl,
                            unsigned int err_code = 0);
static int sycl_task_finalize(cudaTask_t *sycl_task);
// SYCL KERNELS:
template <typename T>
void gpu_array_norm2__(size_t tsize, const T *__restrict__ arr,
                       volatile double *bnorm2, cl::sycl::nd_item<2>& item,
                       uint8_t *local_ptr, int *norm2_wr_lock);
template <typename T>
void gpu_array_init__(size_t tsize, T *arr, T val, cl::sycl::nd_item<1> &item);
template <typename T>
void gpu_scalar_multiply__(const T *left_arg, const T *right_arg, T *dest_arg,
                           T alpha, int left_conj = 0, int right_conj = 0);
template <typename T>
void gpu_array_scale__(size_t tsize, T *arr, T alpha, sycl::nd_item<1> &item);
template <typename T>
void gpu_array_add__(size_t tsize, T *__restrict__ arr0,
                     const T *__restrict__ arr1, T alpha,
                     cl::sycl::nd_item<1> &item, int left_conj = 0);
template <typename T>
void gpu_array_add__(size_t tsize, T *__restrict__ arr0,
                     const T *__restrict__ arr1, const T *__restrict__ scalar,
                     T alpha, cl::sycl::nd_item<1> &item, int left_conj = 0);
template <typename T>
void gpu_array_dot_product__(size_t tsize, const T *arr1, const T *arr2,
                             volatile T *dprod, T alpha,
                             cl::sycl::nd_item<1> &item, uint8_t *local_ptr,
                             int *dot_product_wr_lock, int left_conj = 0,
                             int right_conj = 0);
template <typename T>
void gpu_array_product__(size_t tsize1, const T *arr1, size_t tsize2,
                         const T *arr2, T *arr0, T alpha, cl::sycl::nd_item<2>& item,
                         T *lbuf, T *rbuf, talshComplex4 *lbuf,
                         talshComplex4 *rbuf, talshComplex8 *lbuf,
                         talshComplex8 *rbuf, int left_conj = 0,
                         int right_conj = 0);
template <typename T>
void gpu_tensor_block_add_dlf__(
    int dmo, int drc, int dim_num, int const_args_pos,
    const T *__restrict__ tens_in, T *__restrict__ tens_out,
    sycl::nd_item<3> item, dpct::accessor<int, dpct::device, 2> const_args_dims,
    dpct::accessor<int, dpct::device, 2> const_args_prmn, int *gpu_error_count,
    T *buf0, float *val, size_t *base_in, size_t *base_out, size_t *ftb,
    size_t *gtb, int *htb, int *stb, int *dim_in, int *dim_out, int *o2n,
    int *n2o, int *pri, int *tmp0, int *err_code, int *minor, int *minor_in,
    int *minor_out, int *s1_ind, int *s1_ond, int *s1_step, int *s1_dim,
    int *s2_ind, int *s2_ond, int *s2_step, int *s2_dim, int *ns1, int *ns2,
    size_t *vol, size_t *vol_ext);
template <typename T>
void gpu_tensor_block_copy_dlf__(
    int dmo, int drc, int dim_num, int const_args_pos,
    const T *__restrict__ tens_in, T *__restrict__ tens_out,
    cl::sycl::nd_item<1> &item, constant_accessor<int, 2> &const_args_dims,
    constant_accessor<int, 2> &const_args_prmn, int *gpu_error_count, T *buf0,
    float *val, size_t *base_in, size_t *base_out, size_t *ftb, size_t *gtb,
    int *htb, int *stb, int *dim_in, int *dim_out, int *o2n, int *n2o, int *pri,
    int *tmp0, int *err_code, int *minor, int *minor_in, int *minor_out,
    int *s1_ind, int *s1_ond, int *s1_step, int *s1_dim, int *s2_ind,
    int *s2_ond, int *s2_step, int *s2_dim, int *ns1, int *ns2, size_t *vol,
    size_t *vol_ext);
template <typename T>
void gpu_tensor_block_copy_cmplx_split_in_dlf__(
    int dmo, int drc, int dim_num, int const_args_pos,
    const T *__restrict__ tens_in, T *__restrict__ tens_out,
    cl::sycl::nd_item<1> &item,
    dpct::accessor<int, dpct::device, 2> const_args_dims,
    dpct::accessor<int, dpct::device, 2> const_args_prmn, int *gpu_error_count,
    T *buf0, float *val, size_t *base_in, size_t *base_out, size_t *ftb,
    size_t *gtb, int *htb, int *stb, int *dim_in, int *dim_out, int *o2n,
    int *n2o, int *pri, int *tmp0, int *err_code, int *minor, int *minor_in,
    int *minor_out, int *s1_ind, int *s1_ond, int *s1_step, int *s1_dim,
    int *s2_ind, int *s2_ond, int *s2_step, int *s2_dim, int *ns1, int *ns2,
    size_t *vol, size_t *vol_ext);
template <typename T>
void gpu_tensor_block_copy_cmplx_split_out_dlf__(
    int dmo, int drc, int dim_num, int const_args_pos,
    const T *__restrict__ tens_in, T *__restrict__ tens_out,
    sycl::nd_item<3> item, dpct::accessor<int, dpct::device, 2> const_args_dims,
    dpct::accessor<int, dpct::device, 2> const_args_prmn, int *gpu_error_count,
    T *buf0, float *val, size_t *base_in, size_t *base_out, size_t *ftb,
    size_t *gtb, int *htb, int *stb, int *dim_in, int *dim_out, int *o2n,
    int *n2o, int *pri, int *tmp0, int *err_code, int *minor, int *minor_in,
    int *minor_out, int *s1_ind, int *s1_ond, int *s1_step, int *s1_dim,
    int *s2_ind, int *s2_ond, int *s2_step, int *s2_dim, int *ns1, int *ns2,
    size_t *vol, size_t *vol_ext);
template <typename T>
void gpu_tensor_block_copy_scatter_dlf__(
    int dmo, int drc, int dim_num, int const_args_pos,
    const T *__restrict__ tens_in, T *__restrict__ tens_out,
    cl::sycl::nd_item<1> &item, constant_accessor<int, 2> &const_args_dims,
    constant_accessor<int, 2> &const_args_prmn, int *gpu_error_count, int *n2o,
    size_t *vol, size_t *base_in, size_t *base_out);
template <typename T>
void gpu_matrix_multiply_tn__(size_t ll, size_t lr, size_t lc, const T *arg1,
                              const T *arg2, T *arg0, T alpha,
                              cl::sycl::nd_item<2> &item, int *gpu_error_count,
                              local_accessor<T, 2> &buf1,
                              local_accessor<T, 2> &buf2);
template <typename T>
void gpu_matrix_multiply_nt__(size_t ll, size_t lr, size_t lc, const T *arg1,
                              const T *arg2, T *arg0, T alpha);
template <typename T>
void gpu_matrix_multiply_nn__(size_t ll, size_t lr, size_t lc, const T *arg1,
                              const T *arg2, T *arg0, T alpha);
template <typename T>
void gpu_matrix_multiply_tt__(size_t ll, size_t lr, size_t lc, const T *arg1,
                              const T *arg2, T *arg0, T alpha);
#endif /*NO_GPU*/
//---------------------------------------------------------------------------------------------------------------------------
// PARAMETERS:
static int VERBOSE = 1; // verbosity for error messages
static int DEBUG = 0;   // debugging mode
#ifndef NO_GPU
// GLOBAL DATA:
// GPU control on the current MPI process:
static int gpu_up[MAX_GPUS_PER_NODE] = {GPU_OFF}; // GPU_OFF(0): GPU is disabled; GPU_MINE(1): GPU is enabled; GPU_MINE_ONEMKL(2): GPU is BLAS enabled
static talsh_stats_t gpu_stats[MAX_GPUS_PER_NODE]; // runtime statistics for all GPUs present on the node

#ifndef NO_BLAS
// Infrastructure for CUBLAS:
static sycl::queue *cublas_handle[MAX_GPUS_PER_NODE]; // each GPU present on a node obtains its own cuBLAS context handle
#endif /*NO_BLAS*/

// Slabs for the GPU asynchronous resources:
//  SYCL queue handles:
static cl::sycl::queue SYCLQueueBank[MAX_GPUS_PER_NODE][MAX_SYCL_TASKS]; // pre-allocated SYCL queues (for each SYCL device)
static int SYCLQueueFreeHandle[MAX_GPUS_PER_NODE][MAX_SYCL_TASKS]; // free CUDA stream handles
static int SYCLQueueFFE[MAX_GPUS_PER_NODE]; // number of free handles left in SYCLQueueFreeHandle

//  SYCL event handles:
static cl::sycl::event SYCLEventBank[MAX_GPUS_PER_NODE][MAX_SYCL_EVENTS]; // pre-allocated SYCL queue handles (for each SYCL device)
static int SYCLEventFreeHandle[MAX_GPUS_PER_NODE][MAX_SYCL_EVENTS]; // free SYCL queue handles
static int SYCLEventFFE[MAX_GPUS_PER_NODE]; // number of free handles left in SYCLEventFreeHandle
// Mapped slab of tensor operation prefactors for GPU usage:
static slab_t prefactors;        // mapped slab of prefactors
static void *gpu_prefs_base_ptr; // mapped device pointer of the slab base

// Slab of GPU constant memory arguments for each GPU (managed by
// "mem_manager.cpp"):
globalBuffer<int, 2> const_args_dims(cl::sycl::range<2>(
    MAX_GPU_ARGS, MAX_TENSOR_RANK)); // storage for device constant memory
                                     // arguments: dimension extents
globalBuffer<int, 2> const_args_prmn(cl::sycl::range<2>(
    MAX_GPU_ARGS, MAX_TENSOR_RANK)); // storage for device constant memory
                                     // arguments: permutation

// dpct::device_memory<int, 2> const_args_dims(MAX_GPU_ARGS, MAX_TENSOR_RANK);
// // storage for device constant memory arguments: dimension extents
// dpct::device_memory<int, 2> const_args_prmn(MAX_GPU_ARGS, MAX_TENSOR_RANK);
// // storage for device constant memory arguments: permutation

// GPU error control and debugging for each GPU:
dpct::device_memory<int, 0> gpu_error_count(0); // total number of CUDA errors registered on device till the current moment
dpct::device_memory<int, 1> gpu_debug_dump(GPU_DEBUG_DUMP_SIZE); // debug dump
// Global SYCL queue recording policy:
static int PRINT_TIMING = 1; // non-zero value enables time printing statements
// Infrastructure for function <gpu_tensor_block_copy_dlf> (blocking and
// non-blocking):
static int TRANS_SHMEM = EFF_TRN_ON; // switch between shared-memory tensor
                                     // transpose and scatter tensor transpose
// Infrastructure for <gpu_tensor_block_contract_dlf> (non-blocking):
#ifndef NO_BLAS
static int DISABLE_BLAS = 0; // non-zero value will disable cuBLAS usage (if it had been cuBLAS compiled/linked)
#else
static int DISABLE_BLAS = 1; // non-zero value will disable cuBLAS usage (if it had been cuBLAS compiled/linked)
#endif                                          /*NO_BLAS*/

static cudaTask_t *LastTask[MAX_GPUS_PER_NODE]; // last CUDA task successfully scheduled on each GPU
static float sgemm_alpha_plus = 1.0f;   // default alpha constant for SGEMM
static float sgemm_alpha_minus = -1.0f; // default alpha constant for SGEMM
static float sgemm_beta_one = 1.0f;     // default beta constant SGEMM
static float sgemm_beta_zero = 0.0f;    // zero beta constant SGEMM
static double dgemm_alpha_plus = 1.0;   // default alpha constant for DGEMM
static double dgemm_alpha_minus = -1.0; // default alpha constant for DGEMM
static double dgemm_beta_one = 1.0;     // default beta constant DGEMM
static double dgemm_beta_zero = 0.0;    // zero beta constant DGEMM
static std::complex<float> cgemm_alpha_plus(1.0f, 0.0f); // default alpha constant CGEMM
static std::complex<float> cgemm_alpha_minus(-1.0f, 0.0f); // default alpha constant CGEMM
static std::complex<float> cgemm_beta_one(1.0f, 0.0f); // default beta constant CGEMM
static std::complex<float> cgemm_beta_zero(0.0f, 0.0f); // zero beta constant CGEMM
static std::complex<double> zgemm_alpha_plus(1.0, 0.0); // default alpha constant ZGEMM
static std::complex<double> zgemm_alpha_minus(-1.0, 0.0); // default alpha constant ZGEMM
static std::complex<double> zgemm_beta_one(1.0, 0.0); // default beta constant ZGEMM
static std::complex<double> zgemm_beta_zero(0.0, 0.0); // zero beta constant ZGEMM
// Infrastructure for kernels <gpu_array_norm2__>:
dpct::device_memory<int, 0> norm2_wr_lock(0); // write lock shared by all <gpu_array_norm2__> running on GPU
// Infrastructure for kernels <gpu_array_dot_product__>:
dpct::device_memory<int, 0> dot_product_wr_lock(0); // write lock shared by all <gpu_array_dot_product__> running on GPU
#endif  /*NO_GPU*/
//--------------------------------------------------------------------------------------------------------------
#ifndef NO_GPU
// CUDA KERNELS:
// SUM OF THE SQUARES OF ABSOLUTE VALUES OF ALL ARRAY ELEMENTS:
// REAL:
template <typename T>
void gpu_array_norm2__(size_t tsize, const T *__restrict__ arr,
                       volatile double *bnorm2, cl::sycl::nd_item<1> &item,
                       uint8_t *local_ptr, int *norm2_wr_lock)
/** Computes the squared 2-norm of array arr(0:tsize-1)
    INPUT:
    # tsize - size of the array;
    # arr(0:tsize-1) - array;
    # bnorm2 - must be zero on entrance (resides on device as well);
    OUTPUT:
    # bnorm2 - squared 2-norm of the array (resides on device as well);
**/
{
  size_t i, n;
  double _thread_norm2;
  auto thread_norms2 = (double *)
      local_ptr; // size = blockDim.x*sizeof(double) Bytes per thread block

  n = item.get_global_range(0);
  _thread_norm2 = 0.0;
  for (i = item.get_global_id(0) i < tsize; i += n)
    _thread_norm2 += arr[i] * arr[i];
  thread_norms2[item.get_local_id(0)] = _thread_norm2;
  item.barrier(cl::sycl::access::fence_space::local_space);
  if (item.get_local_id(0) == 0) { // global reduction among thread blocks
    _thread_norm2 = thread_norms2[0];
    for (i = 1; i < item.get_local_range(0); i++)
      _thread_norm2 += thread_norms2[i];
    i = 1;

    auto atm = atomic_ref<int>(norm2_wr_lock);

    while (i == 1) {
      i = atm.fetch_max(1);
    } // waiting for the lock to unlock, then lock

    cl::sycl::ONEAPI::atomic_fence(cl::sycl::ONEAPI::memory_order::relaxed,
                                   cl::sycl::ONEAPI::memory_scope::work_group);
    *bnorm2 += _thread_norm2;

    cl::sycl::ONEAPI::atomic_fence(cl::sycl::ONEAPI::memory_order::relaxed,
                                   cl::sycl::ONEAPI::memory_scope::work_group);
    i = atm.exchange(0); // unlock
  }
  item.barrier(cl::sycl::access::fence_space::local_space);
  return;
}
// COMPLEX4:
template <>
void gpu_array_norm2__<talshComplex4>(size_t tsize,
                                      const talshComplex4 *__restrict__ arr,
                                      volatile double *bnorm2,
                                      cl::sycl::nd_item<1>& item, uint8_t *local_ptr,
                                      int *norm2_wr_lock)
/** Computes the squared 2-norm of array arr(0:tsize-1)
    INPUT:
    # tsize - size of the array;
    # arr(0:tsize-1) - array;
    # bnorm2 - must be zero on entrance (resides on device as well);
    OUTPUT:
    # bnorm2 - squared 2-norm of the array (resides on device as well);
**/
{
  size_t i, n;
  double _thread_norm2;
  auto thread_norms2 = (double *)
      local_ptr; // size = blockDim.x*sizeof(double) Bytes per thread block

  n = item.get_global_range(0);
  _thread_norm2 = 0.0;
  for (i = item.get_global_id(0) i < tsize; i += n)
    _thread_norm2 += talshComplex4Asq(arr[i]);
  thread_norms2[item.get_local_id(0)] = _thread_norm2;
  item.barrier(cl::sycl::access::fence_space::local_space);
  if (item.get_local_id(0) ==
      0) { // global reduction among thread blocks (one thread per block)
    _thread_norm2 = thread_norms2[0];
    for (i = 1; i < item.get_local_range().get(0); i++)
      _thread_norm2 += thread_norms2[i];
    i = 1;
    auto atm = atomic_ref<int>(norm2_wr_lock);
    while (i == 1) {
      i = atm.fetch_max(1);
    } // waiting for the lock to unlock, then lock

    cl::sycl::ONEAPI::atomic_fence(cl::sycl::ONEAPI::memory_order::relaxed, cl::sycl::ONEAPI::memory_scope::work_group);
    *bnorm2 += _thread_norm2;
    cl::sycl::ONEAPI::atomic_fence(cl::sycl::ONEAPI::memory_order::relaxed, cl::sycl::ONEAPI::memory_scope::work_group);
    i = atm.exchange(0) // unlock
  }
  item.barrier(cl::sycl::access::fence_space::local_space);
  return;
}
// COMPLEX8:
template <>
void gpu_array_norm2__<talshComplex8>(size_t tsize,
                                      const talshComplex8 *__restrict__ arr,
                                      volatile double *bnorm2,
                                      cl::sycl::nd_item<1>& item, uint8_t *local_ptr,
                                      int *norm2_wr_lock)
/** Computes the squared 2-norm of array arr(0:tsize-1)
    INPUT:
    # tsize - size of the array;
    # arr(0:tsize-1) - array;
    # bnorm2 - must be zero on entrance (resides on device as well);
    OUTPUT:
    # bnorm2 - squared 2-norm of the array (resides on device as well);
**/
{
  size_t i, n;
  double _thread_norm2;
  auto thread_norms2 = (double *)
      local_ptr; // size = blockDim.x*sizeof(double) Bytes per thread block

  n = item.get_global_range(0);
  _thread_norm2 = 0.0;
  for (i = item.get_global_id(0) i < tsize; i += n)
    _thread_norm2 += talshComplex8Asq(arr[i]);
  thread_norms2[item.get_local_id(0)] = _thread_norm2;
  item.barrier(cl::sycl::access::fence_space::local_space);
  if (item.get_local_id(0) ==
      0) { // global reduction among thread blocks (one thread per block)
    _thread_norm2 = thread_norms2[0];
    for (i = 1; i < item.get_local_range().get(0); i++)
      _thread_norm2 += thread_norms2[i];
    i = 1;
    auto atm = atomic_ref<int>(norm2_wr_lock);
    while (i == 1) {
      i = atm.fetch_max(1);
    } // waiting for the lock to unlock, then lock

    cl::sycl::ONEAPI::atomic_fence(cl::sycl::ONEAPI::memory_order::relaxed, cl::sycl::ONEAPI::memory_scope::work_group);
    *bnorm2 += _thread_norm2;
    cl::sycl::ONEAPI::atomic_fence(cl::sycl::ONEAPI::memory_order::relaxed, cl::sycl::ONEAPI::memory_scope::work_group);
    i = atm.exchange(0) // unlock
  }
  item.barrier(cl::sycl::access::fence_space::local_space);
  return;
}
//------------------------------------------------------------
// ARRAY INITIALIZATION:
template <typename T>
void gpu_array_init__(size_t tsize, T *arr, T val, cl::sycl::nd_item<1> &item)
/** arr(0:tsize-1)=val **/
{
  size_t _ti = item.get_global_id(0);
  size_t _gd = item.get_global_range(0);
  for (size_t l = _ti; l < tsize; l += _gd)
    arr[l] = val;
  return;
}
//---------------------------------------------------------------------------------------------------
// SCALAR MULTIPLICATION:
// REAL:
template <typename T>
void gpu_scalar_multiply__(const T *left_arg, const T *right_arg, T *dest_arg,
                           T alpha, int left_conj, int right_conj)
/** Scalar += Scalar * Scalar * Alpha **/
{
  *dest_arg += (*left_arg) * (*right_arg) * alpha;
  return;
}
// COMPLEX4:
template <>
void gpu_scalar_multiply__<talshComplex4>(const talshComplex4 *left_arg,
                                          const talshComplex4 *right_arg,
                                          talshComplex4 *dest_arg,
                                          talshComplex4 alpha, int left_conj,
                                          int right_conj)
/** Scalar += Scalar * Scalar * Alpha **/
{
  if (left_conj != 0) {
    if (right_conj != 0) {
      *dest_arg = talshComplex4Add(
          *dest_arg,
          talshComplex4Mul(talshComplex4Mul(talshComplex4Conjg(*left_arg),
                                            talshComplex4Conjg(*right_arg)),
                           alpha));
    } else {
      *dest_arg = talshComplex4Add(
          *dest_arg,
          talshComplex4Mul(
              talshComplex4Mul(talshComplex4Conjg(*left_arg), *right_arg),
              alpha));
    }
  } else {
    if (right_conj != 0) {
      *dest_arg = talshComplex4Add(
          *dest_arg,
          talshComplex4Mul(
              talshComplex4Mul(*left_arg, talshComplex4Conjg(*right_arg)),
              alpha));
    } else {
      *dest_arg = talshComplex4Add(
          *dest_arg,
          talshComplex4Mul(talshComplex4Mul(*left_arg, *right_arg), alpha));
    }
  }
  return;
}
// COMPLEX8:
template <>
void gpu_scalar_multiply__<talshComplex8>(const talshComplex8 *left_arg,
                                          const talshComplex8 *right_arg,
                                          talshComplex8 *dest_arg,
                                          talshComplex8 alpha, int left_conj,
                                          int right_conj)
/** Scalar += Scalar * Scalar * Alpha **/
{
  if (left_conj != 0) {
    if (right_conj != 0) {
      *dest_arg = talshComplex8Add(*dest_arg, talshComplex8Mul(talshComplex8Mul(talshComplex8Conjg(*left_arg), talshComplex8Conjg(*right_arg)), alpha));
    } else {
      *dest_arg = talshComplex8Add(*dest_arg, talshComplex8Mul(talshComplex8Mul(talshComplex8Conjg(*left_arg), *right_arg), alpha));
    }
  } else {
    if (right_conj != 0) {
      *dest_arg = talshComplex8Add(*dest_arg, talshComplex8Mul(talshComplex8Mul(*left_arg, talshComplex8Conjg(*right_arg)), alpha));
    } else {
      *dest_arg = talshComplex8Add(*dest_arg, talshComplex8Mul(talshComplex8Mul(*left_arg, *right_arg), alpha));
    }
  }
  return;
}
//---------------------------------------------------------------
// ARRAY RESCALING:
// REAL:
template <typename T>
void gpu_array_scale__(size_t tsize, T *arr, T alpha,
                       cl::sycl::nd_item<1> &item)
/** arr(0:tsize-1)*=alpha **/
{
  size_t _ti = item.get_global_id(0);
  size_t _gd = item.get_global_range(0);
  for (size_t l = _ti; l < tsize; l += _gd)
    arr[l] *= alpha;
  return;
}
// COMPLEX4:
template <>
void gpu_array_scale__<talshComplex4>(size_t tsize, talshComplex4 *arr,
                                      talshComplex4 alpha,
                                      cl::sycl::nd_item<1> &item)
/** arr(0:tsize-1)*=alpha **/
{
  size_t _ti = item.get_global_id(0);
  size_t _gd = item.get_global_range(0);
  for (size_t l = _ti; l < tsize; l += _gd)
    arr[l] = talshComplex4Mul(arr[l], alpha);
  return;
}
// COMPLEX8:
template <>
void gpu_array_scale__<talshComplex8>(size_t tsize, talshComplex8 *arr,
                                      talshComplex8 alpha,
                                      cl::sycl::nd_item<1> &item)
/** arr(0:tsize-1)*=alpha **/
{
  size_t _ti = item.get_global_id(0);
  size_t _gd = item.get_global_range(0);
  for (size_t l = _ti; l < tsize; l += _gd)
    arr[l] = talshComplex8Mul(arr[l], alpha);
  return;
}
//-----------------------------------------------------------------------------------------------------------------------
// ARRAY ADDITION:
// REAL:
template <typename T>
void gpu_array_add__(size_t tsize, T *__restrict__ arr0,
                     const T *__restrict__ arr1, T alpha,
                     cl::sycl::nd_item<1> &item, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*alpha **/
{
  size_t _ti = item.get_global_id(0);
  size_t _gd = item.get_global_range(0);
  for (size_t l = _ti; l < tsize; l += _gd)
    arr0[l] += (arr1[l] * alpha);
  return;
}
// COMPLEX4:
template <>
void gpu_array_add__<talshComplex4>(size_t tsize,
                                    talshComplex4 *__restrict__ arr0,
                                    const talshComplex4 *__restrict__ arr1,
                                    talshComplex4 alpha,
                                    cl::sycl::nd_item<1> &item, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*alpha **/
{
  size_t _ti = item.get_global_id(0);
  size_t _gd = item.get_global_range(0);
  if (left_conj != 0) {
    for (size_t l = _ti; l < tsize; l += _gd)
      arr0[l] = talshComplex4Add(
          arr0[l], talshComplex4Mul(talshComplex4Conjg(arr1[l]), alpha));
  } else {
    for (size_t l = _ti; l < tsize; l += _gd)
      arr0[l] = talshComplex4Add(arr0[l], talshComplex4Mul(arr1[l], alpha));
  }
  return;
}
// COMPLEX8:
template <>
void gpu_array_add__<talshComplex8>(size_t tsize,
                                    talshComplex8 *__restrict__ arr0,
                                    const talshComplex8 *__restrict__ arr1,
                                    talshComplex8 alpha,
                                    cl::sycl::nd_item<1> &item, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*alpha **/
{
  size_t _ti = item.get_global_id(0);
  size_t _gd = item.get_global_range(0);
  if (left_conj != 0) {
    for (size_t l = _ti; l < tsize; l += _gd)
      arr0[l] = talshComplex8Add(arr0[l], talshComplex8Mul(talshComplex8Conjg(arr1[l]), alpha));
  } else {
    for (size_t l = _ti; l < tsize; l += _gd)
      arr0[l] = talshComplex8Add(arr0[l], talshComplex8Mul(arr1[l], alpha));
  }
  return;
}
//------------------------------------------------------------------------------------------------------------------------------
// ARRAY ADDITION AND SCALING:
// REAL:
template <typename T>
void gpu_array_add__(size_t tsize, T *__restrict__ arr0,
                     const T *__restrict__ arr1, const T *__restrict__ scalar,
                     T alpha, cl::sycl::nd_item<1> &item, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*scalar*alpha **/
{
  size_t _ti = item.get_global_id(0);
  size_t _gd = item.get_global_range(0);
  T pref = (*scalar) * alpha;
  for (size_t l = _ti; l < tsize; l += _gd)
    arr0[l] += (arr1[l] * pref);
  return;
}
// COMPLEX4:
template <>
void gpu_array_add__<talshComplex4>(size_t tsize,
                                    talshComplex4 *__restrict__ arr0,
                                    const talshComplex4 *__restrict__ arr1,
                                    const talshComplex4 *__restrict__ scalar,
                                    talshComplex4 alpha,
                                    cl::sycl::nd_item<1> &item, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*scalar*alpha **/
{
  size_t _ti = item.get_global_id(0);
  size_t _gd = item.get_global_range(0);
  talshComplex4 pref = talshComplex4Mul(*scalar, alpha);
  if (left_conj != 0) {
    for (size_t l = _ti; l < tsize; l += _gd)
      arr0[l] = talshComplex4Add(arr0[l], talshComplex4Mul(talshComplex4Conjg(arr1[l]), pref));
  } else {
    for (size_t l = _ti; l < tsize; l += _gd)
      arr0[l] = talshComplex4Add(arr0[l], talshComplex4Mul(arr1[l], pref));
  }
  return;
}
// COMPLEX8:
template <>
void gpu_array_add__<talshComplex8>(size_t tsize,
                                    talshComplex8 *__restrict__ arr0,
                                    const talshComplex8 *__restrict__ arr1,
                                    const talshComplex8 *__restrict__ scalar,
                                    talshComplex8 alpha,
                                    cl::sycl::nd_item<1> &item, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*scalar*alpha **/
{
  size_t _ti = item.get_global_id(0);
  size_t _gd = item.get_global_range(0);
  talshComplex8 pref = talshComplex8Mul(*scalar, alpha);
  if (left_conj != 0) {
    for (size_t l = _ti; l < tsize; l += _gd)
      arr0[l] = talshComplex8Add(arr0[l], talshComplex8Mul(talshComplex8Conjg(arr1[l]), pref));
  } else {
    for (size_t l = _ti; l < tsize; l += _gd)
      arr0[l] = talshComplex8Add(arr0[l], talshComplex8Mul(arr1[l], pref));
  }
  return;
}
//-------------------------------------------------------------------------------------------------------
// ARRAY DOT-PRODUCT:
// REAL:
template <typename T>
void gpu_array_dot_product__(size_t tsize, const T *arr1, const T *arr2,
                             volatile T *dprod, T alpha,
                             cl::sycl::nd_item<1> &item, uint8_t *local_ptr,
                             int *dot_product_wr_lock, int left_conj,
                             int right_conj)
/** Scalar (GPU) += arr1(0:tsize-1) * arr2(0:tsize-1) * alpha **/
{
  auto sh_buf = (char *)local_ptr; // size = blockDim.x * sizeof(T) Bytes per thread block
  T *dprs;
  T dpr;
  size_t l;
  unsigned int j, s;
  int i;
  size_t threadIdx_x = item.get_local_id(0);

  dprs = (T *)(&sh_buf[0]); // dynamic shared memory buffer
  dpr = static_cast<T>(0.0);
  for (l = item.get_global_id(0); l < tsize; l += item.get_global_range(0))
    dpr += arr1[l] * arr2[l];
  dprs[threadIdx_x] = dpr * alpha;
  item.barrier(cl::sycl::access::fence_space::local_space);
  s = item.get_local_range(0);
  while (s > 1) {
    j = (s + 1U) >> 1;
    if (threadIdx_x + j < s)
      dprs[threadIdx_x] += dprs[threadIdx_x + j];
    item.barrier(cl::sycl::access::fence_space::local_space);
    s = j;
  }
  if (threadIdx_x == 0) {
    i = 1;
    auto atm = atomic_ref<int>(dot_product_wr_lock);
    while (i != 0) {
      i = atm.fetch_max(1);
      if (i == 0)
        *dprod += dprs[0];
    }
    cl::sycl::ONEAPI::atomic_fence(cl::sycl::ONEAPI::memory_order::relaxed,
                                   cl::sycl::ONEAPI::memory_scope::work_group);
    i = atm.exchange(0); // unlock
  }
  item.barrier(cl::sycl::access::fence_space::local_space);
  return;
}
// COMPLEX4:
template <>
void gpu_array_dot_product__<talshComplex4>(
    size_t tsize, const talshComplex4 *arr1, const talshComplex4 *arr2,
    volatile talshComplex4 *dprod, talshComplex4 alpha,
    cl::sycl::nd_item<1> &item, uint8_t *local_ptr, int *dot_product_wr_lock,
    int left_conj, int right_conj)
/** Scalar (GPU) += arr1(0:tsize-1) * arr2(0:tsize-1) * alpha **/
{
  auto sh_buf = (char *)local_ptr; // size = blockDim.x * sizeof(T) Bytes per thread block
  talshComplex4 *dprs;
  talshComplex4 dpr;
  size_t l;
  unsigned int j, s;
  int i;
  size_t threadIdx_x = item.get_local_id(0);
  size_t globalIdx_x = item.get_global_id(0);
  size_t globalDim_x = item.get_global_range(0);

  dprs = (talshComplex4 *)(&sh_buf[0]); // dynamic shared memory buffer
  dpr = talshComplex4Set(0.0f, 0.0f);
  if (left_conj != 0) {
    if (right_conj != 0) {
      for (l = globalIdx_x; l < tsize;
           l += globalDim_x) {
        dpr = talshComplex4Add(dpr, talshComplex4Mul(talshComplex4Conjg(arr1[l]), talshComplex4Conjg(arr2[l])));
      }
    } else {
      for (l = globalIdx_x; l < tsize;
           l += globalDim_x) {
        dpr = talshComplex4Add(dpr, talshComplex4Mul(talshComplex4Conjg(arr1[l]), arr2[l]));
      }
    }
  } else {
    if (right_conj != 0) {
      for (l = globalIdx_x; l < tsize;
           l += globalDim_x) {
        dpr = talshComplex4Add(dpr, talshComplex4Mul(arr1[l], talshComplex4Conjg(arr2[l])));
      }
    } else {
      for (l = globalIdx_x; l < tsize;
           l += globalDim_x) {
        dpr = talshComplex4Add(dpr, talshComplex4Mul(arr1[l], arr2[l]));
      }
    }
  }
  dprs[threadIdx_x] = talshComplex4Mul(dpr, alpha);
  item.barrier(cl::sycl::access::fence_space::local_space);
  s = item.get_local_range(0);
  while (s > 1) {
    j = (s + 1U) >> 1;
    if (threadIdx_x + j < s)
      dprs[threadIdx_x] = talshComplex4Add(dprs[threadIdx_x], dprs[threadIdx_x + j]);
    item.barrier(cl::sycl::access::fence_space::local_space);
    s = j;
  }
  if (threadIdx_x == 0) {
    i = 1;
    auto atm = atomic_ref<int>(dot_product_wr_lock);
    while (i != 0) {
      i = atm.fetch_max(1);
      if (i == 0) {
        dprod->x += talshComplex4Real(dprs[0]);
        dprod->y += talshComplex4Imag(dprs[0]);
      }
    }
    cl::sycl::ONEAPI::atomic_fence(cl::sycl::ONEAPI::memory_order::relaxed,
                                   cl::sycl::ONEAPI::memory_scope::work_group);
    i = atm.exchange(0); // unlock
  }
  item.barrier(cl::sycl::access::fence_space::local_space);
  return;
}
// COMPLEX8:
template <>
void gpu_array_dot_product__<talshComplex8>(
    size_t tsize, const talshComplex8 *arr1, const talshComplex8 *arr2,
    volatile talshComplex8 *dprod, talshComplex8 alpha,
    cl::sycl::nd_item<3> &item, uint8_t *local_ptr, int *dot_product_wr_lock,
    int left_conj, int right_conj)
/** Scalar (GPU) += arr1(0:tsize-1) * arr2(0:tsize-1) * alpha **/
{
  auto sh_buf = (char *)local_ptr; // size = blockDim.x * sizeof(T) Bytes per thread block
  talshComplex8 *dprs;
  talshComplex8 dpr;
  size_t l;
  unsigned int j, s;
  int i;
  size_t threadIdx_x = item.get_local_id(0);
  size_t globalIdx_x = item.get_global_id(0);
  size_t globalDim_x = item.get_global_range(0);

  dprs = (talshComplex8 *)(&sh_buf[0]); // dynamic shared memory buffer
  dpr = talshComplex8Set(0.0, 0.0);
  if (left_conj != 0) {
    if (right_conj != 0) {
      for (l = globalIdx_x; l < tsize;
           l += globalDim_x) {
        dpr = talshComplex8Add(dpr, talshComplex8Mul(talshComplex8Conjg(arr1[l]), talshComplex8Conjg(arr2[l])));
      }
    } else {
      for (l = globalIdx_x; l < tsize;
           l += globalDim_x) {
        dpr = talshComplex8Add(dpr, talshComplex8Mul(talshComplex8Conjg(arr1[l]), arr2[l]));
      }
    }
  } else {
    if (right_conj != 0) {
      for (l = globalIdx_x; l < tsize;
           l += globalDim_x) {
        dpr = talshComplex8Add(dpr, talshComplex8Mul(arr1[l], talshComplex8Conjg(arr2[l])));
      }
    } else {
      for (l = globalIdx_x; l < tsize;
           l += globalDim_x) {
        dpr = talshComplex8Add(dpr, talshComplex8Mul(arr1[l], arr2[l]));
      }
    }
  }
  dprs[threadIdx_x] = talshComplex8Mul(dpr, alpha);
  item.barrier(cl::sycl::access::fence_space::local_space);
  s = item.get_local_range(0);
  while (s > 1) {
    j = (s + 1U) >> 1;
    if (threadIdx_x + j < s)
      dprs[threadIdx_x] = talshComplex8Add(dprs[threadIdx_x], dprs[threadIdx_x + j]);
    item.barrier(cl::sycl::access::fence_space::local_space);
    s = j;
  }
  if (threadIdx_x == 0) {
    i = 1;
    auto atm = atomic_ref<int>(dot_product_wr_lock);
    while (i != 0) {
      i = atm.fetch_max(1);
      if (i == 0) {
        dprod->x += talshComplex8Real(dprs[0]);
        dprod->y += talshComplex8Imag(dprs[0]);
      }
    }
    cl::sycl::ONEAPI::atomic_fence(cl::sycl::ONEAPI::memory_order::relaxed,
                                   cl::sycl::ONEAPI::memory_scope::work_group);
    i = atm.exchange(0); // unlock
  }
  item.barrier(cl::sycl::access::fence_space::local_space);
  return;
}
//---------------------------------------------------------------------------------------------------------
// ARRAY DIRECT PRODUCT:
// REAL:
template <typename T>
void gpu_array_product__(size_t tsize1, const T *arr1, size_t tsize2,
                         const T *arr2, T *arr0, T alpha,
                         cl::sycl::nd_item<2> &item, T *lbuf, T *rbuf,
                         talshComplex4 *lbuf, talshComplex4 *rbuf,
                         talshComplex8 *lbuf, talshComplex8 *rbuf,
                         int left_conj, int right_conj)
/** arr0[0:tsize2-1][0:tsize1-1]+=arr1[0:tsize1-1]*arr2[0:tsize2-1]*alpha **/
{
  size_t _ib, _in, _jb, _jn, _tx, _jc;

  _tx = (size_t)item.get_local_id(1);
  for (_jb = item.get_group(0) * THRDS_ARRAY_PRODUCT; _jb < tsize2;
       _jb += item.get_group_range(0) * THRDS_ARRAY_PRODUCT) {
    if (_jb + THRDS_ARRAY_PRODUCT > tsize2) {
      _jn = tsize2 - _jb;
    } else {
      _jn = THRDS_ARRAY_PRODUCT;
    }
    if (_tx < _jn)
      rbuf[_tx] = arr2[_jb + _tx] * alpha;
    for (_ib = item.get_group(1) * THRDS_ARRAY_PRODUCT; _ib < tsize1;
         _ib += item.get_group_range(1) * THRDS_ARRAY_PRODUCT) {
      if (_ib + THRDS_ARRAY_PRODUCT > tsize1) {
        _in = tsize1 - _ib;
      } else {
        _in = THRDS_ARRAY_PRODUCT;
      }
      if (_tx < _in)
        lbuf[_tx] = arr1[_ib + _tx];
      item.barrier(cl::sycl::access::fence_space::local_space);
      for (_jc = 0; _jc < _jn; _jc++) {
        if (_tx < _in)
          arr0[(_jb + _jc) * tsize1 + _ib + _tx] += lbuf[_tx] * rbuf[_jc];
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
    }
  }
  return;
}
// COMPLEX4:
template <>
void gpu_array_product__<talshComplex4>(
    size_t tsize1, const talshComplex4 *arr1, size_t tsize2,
    const talshComplex4 *arr2, talshComplex4 *arr0, talshComplex4 alpha,
    cl::sycl::nd_item<2> &item, T *lbuf, T *rbuf, talshComplex4 *lbuf,
    talshComplex4 *rbuf, talshComplex8 *lbuf, talshComplex8 *rbuf,
    int left_conj, int right_conj)
/** arr0[0:tsize2-1][0:tsize1-1]+=arr1[0:tsize1-1]*arr2[0:tsize2-1]*alpha **/
{
  size_t _ib, _in, _jb, _jn, _tx, _jc, _ja;

  _tx = (size_t)item.get_local_id(1);
  for (_jb = item.get_group(0) * THRDS_ARRAY_PRODUCT; _jb < tsize2;
       _jb += item.get_group_range(0) * THRDS_ARRAY_PRODUCT) {
    if (_jb + THRDS_ARRAY_PRODUCT > tsize2) {
      _jn = tsize2 - _jb;
    } else {
      _jn = THRDS_ARRAY_PRODUCT;
    }
    if (right_conj != 0) {
      if (_tx < _jn)
        rbuf[_tx] =
            talshComplex4Mul(talshComplex4Conjg(arr2[_jb + _tx]), alpha);
    } else {
      if (_tx < _jn)
        rbuf[_tx] = talshComplex4Mul(arr2[_jb + _tx], alpha);
    }
    for (_ib = item.get_group(1) * THRDS_ARRAY_PRODUCT; _ib < tsize1;
         _ib += item.get_group_range(1) * THRDS_ARRAY_PRODUCT) {
      if (_ib + THRDS_ARRAY_PRODUCT > tsize1) {
        _in = tsize1 - _ib;
      } else {
        _in = THRDS_ARRAY_PRODUCT;
      }
      if (left_conj != 0) {
        if (_tx < _in)
          lbuf[_tx] = talshComplex4Conjg(arr1[_ib + _tx]);
      } else {
        if (_tx < _in)
          lbuf[_tx] = arr1[_ib + _tx];
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
      for (_jc = 0; _jc < _jn; _jc++) {
        if (_tx < _in) {
          _ja = (_jb + _jc) * tsize1 + (_ib + _tx);
          arr0[_ja] = talshComplex4Add(arr0[_ja],
                                       talshComplex4Mul(lbuf[_tx], rbuf[_jc]));
        }
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
    }
  }
  return;
}
// COMPLEX8:
template <>
void gpu_array_product__<talshComplex8>(
    size_t tsize1, const talshComplex8 *arr1, size_t tsize2,
    const talshComplex8 *arr2, talshComplex8 *arr0, talshComplex8 alpha,
    cl::sycl::nd_item<2> &item, T *lbuf, T *rbuf, talshComplex4 *lbuf,
    talshComplex4 *rbuf, talshComplex8 *lbuf, talshComplex8 *rbuf,
    int left_conj, int right_conj)
/** arr0[0:tsize2-1][0:tsize1-1]+=arr1[0:tsize1-1]*arr2[0:tsize2-1]*alpha **/
{
  size_t _ib, _in, _jb, _jn, _tx, _jc, _ja;

  _tx = (size_t)item.get_local_id(1);
  for (_jb = item.get_group(0) * THRDS_ARRAY_PRODUCT; _jb < tsize2;
       _jb += item.get_group_range(0) * THRDS_ARRAY_PRODUCT) {
    if (_jb + THRDS_ARRAY_PRODUCT > tsize2) {
      _jn = tsize2 - _jb;
    } else {
      _jn = THRDS_ARRAY_PRODUCT;
    }
    if (right_conj != 0) {
      if (_tx < _jn)
        rbuf[_tx] =
            talshComplex8Mul(talshComplex8Conjg(arr2[_jb + _tx]), alpha);
    } else {
      if (_tx < _jn)
        rbuf[_tx] = talshComplex8Mul(arr2[_jb + _tx], alpha);
    }
    for (_ib = item.get_group(1) * THRDS_ARRAY_PRODUCT; _ib < tsize1;
         _ib += item.get_group_range(1) * THRDS_ARRAY_PRODUCT) {
      if (_ib + THRDS_ARRAY_PRODUCT > tsize1) {
        _in = tsize1 - _ib;
      } else {
        _in = THRDS_ARRAY_PRODUCT;
      }
      if (left_conj != 0) {
        if (_tx < _in)
          lbuf[_tx] = talshComplex8Conjg(arr1[_ib + _tx]);
      } else {
        if (_tx < _in)
          lbuf[_tx] = arr1[_ib + _tx];
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
      for (_jc = 0; _jc < _jn; _jc++) {
        if (_tx < _in) {
          _ja = (_jb + _jc) * tsize1 + (_ib + _tx);
          arr0[_ja] = talshComplex8Add(arr0[_ja],
                                       talshComplex8Mul(lbuf[_tx], rbuf[_jc]));
        }
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
    }
  }
  return;
}
//---------------------------------------------------------------------------------------------------
// TENSOR TRANSPOSE-ADD (shared-memory version):
template <typename T>
void gpu_tensor_block_add_dlf__(
    int dmo, int drc, int dim_num, int const_args_pos,
    const T *__restrict__ tens_in, T *__restrict__ tens_out,
    cl::sycl::nd_item<3>& item,
    dpct::accessor<int, dpct::device, 2>& const_args_dims,
    dpct::accessor<int, dpct::device, 2>& const_args_prmn,
    int *gpu_error_count,
    T *buf0, float *val, size_t *base_in, size_t *base_out, size_t *ftb,
    size_t *gtb, int *htb, int *stb, int *dim_in, int *dim_out, int *o2n,
    int *n2o, int *pri, int *tmp0, int *err_code, int *minor, int *minor_in,
    int *minor_out, int *s1_ind, int *s1_ond, int *s1_step, int *s1_dim,
    int *s2_ind, int *s2_ond, int *s2_step, int *s2_dim, int *ns1, int *ns2,
    size_t *vol, size_t *vol_ext)
// void gpu_tensor_block_add_dlf__(int dmo, int drc, int dim_num,
//                                 int const_args_pos,
//                                 const T *__restrict__ tens_in,
//                                 T *__restrict__ tens_out)
/**
   Shared-memory version of tensor transpose-add: tens_out+=TRN(tens_in):
   INPUT:
   # dmo - dimension extents order (0: normal, as it is in <const_args>; not 0:
permuted dimension order will be imposed); # drc - index permutation direction
(0: normal, as it is in <const_args>; not 0: inversed permutation will be used);
   # dim_num - tensor block rank;
   # const_args_pos - entry in the __constant__ memory bank where tensor block
dimension extents (const_args_dims) and index permutation (const_args_prmn) are
stored; # tens_in[0:] - input tensor; OUTPUT: # tens_out[0:] - output
(transposed) tensor in which accumulation is performed; NOTES: # Minimal CUDA
execution configuration is <<<1,warpSize>>> # Number of threads per block must
be multiple of the warpSize!
**/
{
    local_accessor<T, 1> buf0_acc(TENS_TRANSP_BUF_SIZE, cgh);
    local_accessor<float, 1> val_acc(1, cgh);
    local_accessor<size_t, 1> base_in_acc(MAX_TENSOR_RANK, cgh);
    local_accessor<size_t, 1> base_out_acc(MAX_TENSOR_RANK, cgh);
    local_accessor<size_t, 1> ftb_acc(TENS_TRANSP_TAB_SIZE, cgh);
    local_accessor<size_t, 1> gtb_acc(TENS_TRANSP_TAB_SIZE, cgh);
    local_accessor<int, 1> htb_acc(TENS_TRANSP_TAB_SIZE, cgh);
    local_accessor<int, 1> stb_acc(TENS_TRANSP_TAB_SIZE, cgh);
    local_accessor<size_t, 1> ftb_acc(TENS_TRANSP_TAB_SIZE, cgh);
    local_accessor<int, 1> dim_in_acc(MAX_TENSOR_RANK, cgh);
    local_accessor<int, 1> dim_out_acc(MAX_TENSOR_RANK, cgh);
    local_accessor<int, 1> o2n_acc(MAX_TENSOR_RANK, cgh);
    local_accessor<int, 1> pri_acc(MAX_TENSOR_RANK, cgh);
    local_accessor<int, 1> tmp0_acc(MAX_TENSOR_RANK, cgh);
    local_accessor<int, 1> err_code_acc(1, cgh);
    local_accessor<int, 1> minor_acc(1, cgh);
    local_accessor<int, 1> minor_in_acc(1, cgh);
    local_accessor<int, 1> minor_out_acc(1, cgh);
    local_accessor<size_t, 1> vol_acc(1, cgh);
    local_accessor<size_t, 1> vol_ext_acc(1, cgh);

    T *buf0 = buf0_acc.get_pointer();
    float *val = val_acc.get_pointer();
    size_t *base_in = base_in_acc.get_pointer();
    size_t *base_out = base_out_acc.get_pointer();
    size_t *ftb = ftb_acc.get_pointer();
    size_t *gtb = gtb_acc.get_pointer();
    int *htb = htb_acc.get_pointer();
    int *stb = stb_acc.get_pointer();
    int *dim_in = dim_in_acc.get_pointer();
    int *dim_out = dim_out_acc.get_pointer();
    int *o2n = o2n_acc.get_pointer();
    int *n2o = n2o_acc.get_pointer();
    int *pri = pri_acc.get_pointer();
    int *tmp0 = tmp0_acc.get_pointer();
    int *err_code = err_code_acc.get_pointer();
    int *minor = minor_acc.get_pointer();
    int *minor_in = minor_in_acc.get_pointer();
    int *minor_out = minor_out_acc.get_pointer();
    size_t *vol = vol_acc.get_pointer();
    size_t *vol_ext = vol_ext_acc.get_pointer();

    size_t _vol, _addr_in, _addr_out, _addr, _work_piece;
    int i, j, k, l, m, n, _vol_minor, _vol_in, _vol_out, _s1, _s2;

    /*
      SHARED MEMORY USE (bytes) =
      + TENS_TRANSP_BUF_SIZE*sizeof(T)
      + MAX_TENSOR_RANK*(8+8+4+4+4+4+4+4)
      + TENS_TRANSP_TAB_SIZE*(8+8+4+4)
      + 4*15 + 8*2
      MIN REGISTER USE (bytes) per thread =
      + 4*4 + 4*11 + 8*5 = 100
    */

    size_t threadIdx_x = item.get_local_id(0);
    size_t blockDim_x = item.get_local_range(0);

    // Determine the minor index set (only the master thread in each thread
    // block):
    if (threadIdx_x == 0) {
      *err_code = 0;
      if (dim_num >= 0 && dim_num <= MAX_TENSOR_RANK &&
          blockDim_x >= warpSize && blockDim_x % warpSize == 0) {
        *s1_ind = dim_num + 1;
        *s2_ind = dim_num - 1;
        _vol = 1;
        for (i = 0; i < dim_num; i++) {
          _vol *= const_args_dims[const_args_pos][i];
          if (const_args_prmn[const_args_pos][i] != i + 1)
            *s1_ind = 0;
        }
        *vol = _vol;        // total volume (number of tensor elements)
        if (*s1_ind == 0) { // non-trivial permutation
          // Set input/output permutations and dimension extents:
          if (drc == 0) { // normal index permutation
            for (i = 0; i < dim_num; i++)
              o2n[i] = const_args_prmn[const_args_pos][i] - 1;
            for (i = 0; i < dim_num; i++)
              n2o[o2n[i]] = i;
          } else { // inversed index permutation
            for (i = 0; i < dim_num; i++)
              n2o[i] = const_args_prmn[const_args_pos][i] - 1;
            for (i = 0; i < dim_num; i++)
              o2n[n2o[i]] = i;
          }
          if (dmo == 0) { // normal dimension order
            for (i = 0; i < dim_num; i++)
              dim_in[i] = const_args_dims[const_args_pos][i];
            for (i = 0; i < dim_num; i++)
              dim_out[o2n[i]] = dim_in[i];
          } else { // inversed dimension order
            for (i = 0; i < dim_num; i++)
              dim_out[i] = const_args_dims[const_args_pos][i];
            for (i = 0; i < dim_num; i++)
              dim_in[n2o[i]] = dim_out[i];
          }
          *s1_step = dim_in[(*s1_ind)];
          *s2_step = dim_in[(*s2_ind)];
          if (_vol > TENS_TRANSP_BUF_SIZE) { // tensor block does not fit into
                                             // the shared memory buffer
            // Determine the input/output minor index sets and the combined
            // minor index set:
            l = (int)(cl::sycl::sqrt((float)TENS_TRANSP_BUF_SIZE));
            *minor_in = 0;
            _vol_in = 1;
            for (i = 0; i < dim_num; i++) {
              j = _vol_in * dim_in[i];
              if (j > l)
                break;
              (*minor_in)++;
              _vol_in = j;
            }
            *minor_out = 0;
            _vol_out = 1;
            for (i = 0; i < dim_num; i++) {
              j = _vol_out * dim_out[i];
              if (j > l)
                break;
              (*minor_out)++;
              _vol_out = j;
            }
            *minor = *minor_in;
            _vol_minor = _vol_in;
            for (i = 0; i < *minor_out; i++) {
              if (n2o[i] >= *minor_in) {
                (*minor)++;
                _vol_minor *= dim_out[i];
              }
            }
            m = 1;
            _s1 = 0;
            _s2 = 0;
            while (_vol_minor < TENS_TRANSP_BUF_SIZE && m != 0) {
              m = 0;
              if (_s1 == 0) {
                for (i = *minor_in; i < dim_num; i++) {
                  if (o2n[i] < *minor_out) {
                    (*minor_in)++;
                    _vol_in *= dim_in[i];
                  } else {
                    break;
                  }
                }
              }
              if (_s2 == 0) {
                for (i = *minor_out; i < dim_num; i++) {
                  if (n2o[i] < *minor_in) {
                    (*minor_out)++;
                    _vol_out *= dim_out[i];
                  } else {
                    break;
                  }
                }
              }
              j = dim_in[(*minor_in)];
              l = dim_out[(*minor_out)];
              if (*minor_in == n2o[(*minor_out)] &&
                  _s1 + _s2 == 0) { // same candidate index to both the input
                                    // and output index sets
                if (j > 1 && TENS_TRANSP_BUF_SIZE < _vol_minor * 2)
                  break;
                if (_vol_minor * j > TENS_TRANSP_BUF_SIZE) {
                  *s1_ind = *minor_in;
                  *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                  _s1++;
                  _s2++;
                }
                (*minor_in)++;
                _vol_in *= j;
                (*minor_out)++;
                _vol_out *= j;
                (*minor)++;
                _vol_minor *= j;
                m++;
              } else { // the input and output index sets consider two different
                       // candidates
                if (_vol_minor * j * l <= TENS_TRANSP_BUF_SIZE &&
                    _s1 + _s2 == 0) { // accept both, no splitting
                  (*minor_in)++;
                  _vol_in *= j;
                  (*minor_out)++;
                  _vol_out *= l;
                  *minor += 2;
                  _vol_minor *= (j * l);
                  m++;
                } else { // try to accept either one of the two OR both with
                         // splitting
                  if (j == 1 || l == 1) {
                    if (j == 1 && _s1 == 0) {
                      (*minor_in)++;
                      (*minor)++;
                      m++;
                    }
                    if (l == 1 && _s2 == 0) {
                      (*minor_out)++;
                      (*minor)++;
                      m++;
                    }
                  } else {
                    if (_vol_minor * j <= TENS_TRANSP_BUF_SIZE &&
                        _vol_minor * l > TENS_TRANSP_BUF_SIZE &&
                        _vol_out >= warpSize &&
                        _s1 == 0) { // accept the input index, no splitting
                      (*minor_in)++;
                      _vol_in *= j;
                      (*minor)++;
                      _vol_minor *= j;
                      m++;
                    } else if (_vol_minor * j > TENS_TRANSP_BUF_SIZE &&
                               _vol_minor * l <= TENS_TRANSP_BUF_SIZE &&
                               _vol_in >= warpSize &&
                               _s2 ==
                                   0) { // accept the output index, no splitting
                      (*minor_out)++;
                      _vol_out *= l;
                      (*minor)++;
                      _vol_minor *= l;
                      m++;
                    } else { // splitting is unavoidable (both OR one OR none)
                      if (TENS_TRANSP_BUF_SIZE >= _vol_minor * 2) {
                        if (j >= 4 && l >= 4) { // dimension extents are large
                                                // enough to be split
                          if (_vol_minor * 4 >
                              TENS_TRANSP_BUF_SIZE) { // impossible to split
                                                      // both indices
                            if (_vol_in <= _vol_out &&
                                _s1 == 0) { // split the input candidate index
                              *s1_ind = *minor_in;
                              *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                              (*minor_in)++;
                              _vol_in *= j;
                              (*minor)++;
                              _vol_minor *= j;
                              _s1++;
                              m++;
                            } else { // split the output candidate index
                              if (_s2 == 0) {
                                *s1_ind = n2o[(*minor_out)];
                                *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                                (*minor_out)++;
                                _vol_out *= l;
                                (*minor)++;
                                _vol_minor *= l;
                                _s2++;
                                m++;
                              }
                            }
                          } else { // possible to split both indices
                            i = (int)cl::sycl::sqrt(
                                ((float)TENS_TRANSP_BUF_SIZE) /
                                (float)_vol_minor);
                            if (i < 2)
                              i = 2; // uniform splitting
                            *s1_step = i;
                            *s2_step = i;
                            *val = (float)_vol_out / (float)_vol_in;
                            if (*val <
                                1.0f) { // scale the initial uniform splitting
                                        // to reflect the disbalance between
                                        // _vol_in and _vol_out
                              if (*val * (float)i < 1.0f)
                                *val = 1.0f / (float)i;
                              if (*val * (float)l < (float)i)
                                *val = (float)i / (float)l;
                            } else {
                              if (*val * (float)i > (float)j)
                                *val = (float)j / (float)i;
                              if (*val > float(i))
                                *val = (float)i;
                            }
                            *s1_step = (int)(((float)i) * *val);
                            *s2_step = (int)(((float)i) / val);
                            if (*s1_step >= 2 &&
                                _s1 == 0) { //&& s1_step <= dim_in[minor_in]
                              *s1_ind = *minor_in;
                              (*minor_in)++;
                              _vol_in *= j;
                              (*minor)++;
                              _vol_minor *= j;
                              _s1++;
                              m++;
                            } else {
                              *s1_step = dim_in[(*s1_ind)];
                            }
                            if (*s2_step >= 2 &&
                                _s2 == 0) { //&& s2_step <= dim_out[minor_out]
                              *s2_ind = n2o[(*minor_out)];
                              (*minor_out)++;
                              _vol_out *= l;
                              (*minor)++;
                              _vol_minor *= l;
                              _s2++;
                              m++;
                            } else {
                              *s2_step = dim_in[(*s2_ind)];
                            }
                          }
                        } else if (j >= 4 && l < 4 &&
                                   _s1 ==
                                       0) { // split the input candidate index
                          *s1_ind = *minor_in;
                          *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                          (*minor_in)++;
                          _vol_in *= j;
                          (*minor)++;
                          _vol_minor *= j;
                          _s1++;
                          m++;
                        } else if (j < 4 && l >= 4 &&
                                   _s2 ==
                                       0) { // split the output candidate index
                          *s1_ind = n2o[(*minor_out)];
                          *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                          (*minor_out)++;
                          _vol_out *= l;
                          (*minor)++;
                          _vol_minor *= l;
                          _s2++;
                          m++;
                        } else { // both candidate indices have too small extent
                                 // to be split: try to add one of them fully
                          if (_vol_minor * j <= TENS_TRANSP_BUF_SIZE &&
                              _s1 == 0) {
                            (*minor_in)++;
                            _vol_in *= j;
                            (*minor)++;
                            _vol_minor *= j;
                            m++;
                          } else if (_vol_minor * l <= TENS_TRANSP_BUF_SIZE &&
                                     _s2 == 0) {
                            (*minor_out)++;
                            _vol_out *= l;
                            (*minor)++;
                            _vol_minor *= l;
                            m++;
                          }
                        }
                      } else { // unable to add more indices in the minor set
                        break;
                      }
                    }
                  }
                }
              }
            }
            if (*s1_ind == dim_num - 1 && *s2_ind == dim_num - 1) {
              *s2_ind = 0;
              *s2_step = dim_in[0];
            }      // s1_ind was set while s2_ind was not
          } else { // tensor block fits into the shared memory buffer from the
                   // beginning
            *minor = dim_num;
            *minor_in = dim_num;
            *minor_out = dim_num;
            _vol_minor = _vol;
            _vol_in = _vol;
            _vol_out = _vol;
          }
          // Share the tensor transpose configuration with other threads in each
          // block:
          *vol_ext = _vol / _vol_minor;
          *s1_dim = dim_in[(*s1_ind)];
          *s2_dim = dim_in[(*s2_ind)];
          // Set indexing bases (OUT:{out,in_c,ext_in}_new;
          // IN:{in,out_c,ext_in}_old):
          //  OUTPUT indexing (dim_out[], base_out[]: prioritized new
          //  numeration):
          for (i = 0; i < dim_num; i++) {
            tmp0[i] = dim_out[i];
          } // save output dimension extents (new numeration)
          j = 0;
          for (i = 0; i < *minor_out; i++) {
            pri[j++] = i;
          } // output minor index set (new numeration))
          for (i = 0; i < dim_num; i++) {
            if (o2n[i] >= *minor_out)
              pri[j++] = o2n[i];
          } //{compl.input minor + external} index set (new numeration)
          j = 1;
          for (i = 0; i < dim_num; i++) {
            dim_out[i] = j;
            j *= tmp0[i];
          } // output bases (new numeration)
          for (i = 0; i < dim_num; i++) {
            base_out[i] = dim_out[pri[i]];
          } // output bases (prioritized new numeration)
          for (i = 0; i < dim_num; i++) {
            dim_out[i] = tmp0[pri[i]];
          } // output extents (prioritized new numeration)
          for (i = 0; i < dim_num; i++) {
            if (n2o[pri[i]] == *s1_ind) {
              *s1_ond = i;
            } else if (n2o[pri[i]] == *s2_ind) {
              *s2_ond = i;
            }
          } // split indices (prioritized new numeration)
          //  INPUT indexing (dim_in[], base_in[]: prioritized old numeration):
          for (i = 0; i < dim_num; i++) {
            tmp0[i] = dim_in[i];
          } // save input dimension extents (old numeration)
          j = 0;
          for (i = 0; i < *minor_in; i++) {
            pri[j++] = i;
          } // input minor index set (old numeration)
          for (i = 0; i < *minor_out; i++) {
            if (n2o[i] >= *minor_in)
              pri[j++] = n2o[i];
          } // compl.output minor idex set (old numeration)
          for (i = j; i < dim_num; i++) {
            pri[i] = n2o[pri[i]];
          } // external index set (just convert new numbers to old ones for
            // consistency)
          j = 1;
          for (i = 0; i < dim_num; i++) {
            dim_in[i] = j;
            j *= tmp0[i];
          } // input bases (old numeration)
          for (i = 0; i < dim_num; i++) {
            base_in[i] = dim_in[pri[i]];
          } // input bases (prioritized old numeration)
          for (i = 0; i < dim_num; i++) {
            dim_in[i] = tmp0[pri[i]];
          } // input extents (prioritized old numeration)
          for (i = 0; i < dim_num; i++) {
            if (pri[i] == *s1_ind) {
              _s1 = i;
            } else if (pri[i] == *s2_ind) {
              _s2 = i;
            }
          } // split indices (prioritized old numeration)
          *s1_ind = _s1;
          *s2_ind = _s2;
          *ns1 =
              1 +
              (*s1_dim - 1) /
                  *s1_step; // number of segments from the 1st split minor index
          *ns2 =
              1 +
              (*s2_dim - 1) /
                  *s2_step; // number of segments from the 2nd split minor index
          //  Index position correspondence for the minor index set (pri-new -->
          //  pri-old):
          j = 0;
          for (i = 0; i < *minor_out; i++) {
            if (n2o[i] < *minor_in) {
              pri[i] = n2o[i];
            } else {
              pri[i] = (*minor_in + j);
              j++;
            }
          }
          j = 0;
          for (i = 0; i < *minor_in; i++) {
            if (o2n[i] < *minor_out) {
              pri[o2n[i]] = i;
            } else {
              pri[*minor_out + j] = i;
              j++;
            }
          }
          // Check tensor transpose configuration parameters:
          if (*minor <= 0 || *minor_in <= 0 || *minor_out <= 0 || _vol <= 0 ||
              _vol_minor <= 0)
            *err_code += 5000; // trap
          if (*s1_ind >= dim_num || *s2_ind >= dim_num || *s1_ond >= dim_num ||
              *s2_ond >= dim_num || *s1_ind == *s2_ind || *s1_ond == *s2_ond ||
              *s1_step <= 0 || *s2_step <= 0)
            *err_code += 1000; // trap
          if ((*s1_step != dim_in[(*s1_ind)] && *s1_ind != *minor_in - 1 &&
               *s1_ond != *minor_out - 1) ||
              (*s2_step != dim_in[(*s2_ind)] && *s2_ind != *minor_in - 1 &&
               *s2_ond != *minor_out - 1))
            *err_code += 500; // trap
          if ((_vol_minor * *s1_step * *s2_step) / (*s1_dim * *s2_dim) >
              TENS_TRANSP_BUF_SIZE)
            *err_code += 100; // trap
        }                     // endif: non-trivial permutation
      } else {
        *err_code = 1 + 2 * blockDim_x % warpSize;
      }
  } // endif: Master thread.
  item.barrier(cl::sycl::access::fence_space::local_space);

  // Proceed:
  if (*err_code == 0) {
    if (*s1_ind > dim_num) { // tag of a trivial permutation
                             // Direct copy:
      _vol = *vol;
      j = item.get_global_range(0);
      i = item.get_global_id(0);
      _addr_in = _vol - _vol % j;
      for (_addr = 0; _addr < _addr_in; _addr += j) {
        _addr_out = _addr + i;
        tens_out[_addr_out] += tens_in[_addr_out];
      }
      _addr_out = _addr_in + i;
      if (_addr_out < _vol)
        tens_out[_addr_out] += tens_in[_addr_out];
    } else {                      // non-trivial permutation
      l = threadIdx_x / warpSize; // l: warp number
      // Distribute work accross SYCL work-groups (external multi-index + splitting):
      for (_work_piece = item.get_group(0);
           _work_piece < *vol_ext * *ns1 * *ns2;
           _work_piece +=
           item.get_group_range(0)) { //(ns1*ns2*vol_ext) is the total number
                                      // of independent tasks
        _addr = _work_piece;
        _addr /= *vol_ext;
        _vol = _work_piece - _addr * *vol_ext;
        _s2 = (int)(_addr / *ns1);
        _s1 = (int)(_addr - _s2 * *ns1); //{_addr_ext,_s1,_s2} --> tensor
                                         //subblock (CUDA block)
        //  Modify dimension extents due to possible dimension splitting:
        if (threadIdx_x == 0) {
          if (_s1 + 1 == *ns1) { // last segment of the 1st split index
            j = *s1_dim - _s1 * *s1_step;
            dim_in[(*s1_ind)] = j;
            dim_out[(*s1_ond)] = j;
          } else { // internal segment of the 1st split index
            dim_in[(*s1_ind)] = *s1_step;
            dim_out[(*s1_ond)] = *s1_step;
          }
          if (_s2 + 1 == *ns2) { // last segment of the 2nd split index
            j = *s2_dim - _s2 * *s2_step;
            dim_in[(*s2_ind)] = j;
            dim_out[(*s2_ond)] = j;
          } else { // internal segment of the 2nd split index
            dim_in[(*s2_ind)] = *s2_step;
            dim_out[(*s2_ond)] = *s2_step;
          }
          j = 1;
          for (i = 0; i < *minor; i++) {
            tmp0[i] = j;
            j *= dim_in[i];
          } // minor buffer bases (pri-old)
          for (i = 0; i < *minor; i++)
            n2o[i] = tmp0[pri[i]]; // look up table to accelerate further
                                   // accesses to tmp0[]
        }
        item.barrier(cl::sycl::access::fence_space::local_space);
        //  Mount input/output volumes and bases:
        _vol_in = dim_in[0];
        for (i = 1; i < *minor_in; i++) {
          _vol_in *= dim_in[i];
        }
        _vol_out = dim_out[0];
        for (i = 1; i < *minor_out; i++) {
          _vol_out *= dim_out[i];
        }
        _vol_minor = _vol_out;
        for (i = *minor_out; i < *minor; i++) {
          _vol_minor *= dim_out[i];
        }
        _addr_in = (_s1 * *s1_step) * base_in[(*s1_ind)] +
                   (_s2 * *s2_step) * base_in[(*s2_ind)];
        _addr_out = _vol;
        for (i = *minor; i < dim_num; i++) {
          _addr = _vol / dim_in[i];
          _addr_in += (_vol - _addr * dim_in[i]) * base_in[i];
          _vol = _addr;
        }
        _vol = _addr_out;
        _addr_out = (_s1 * *s1_step) * base_out[(*s1_ond)] +
                    (_s2 * *s2_step) * base_out[(*s2_ond)];
        for (i = *minor; i < dim_num; i++) {
          _addr = _vol / dim_out[i];
          _addr_out += (_vol - _addr * dim_out[i]) * base_out[i];
          _vol = _addr;
        }
        if (_vol_out > TENS_TRANSP_TAB_SIZE ||
            _vol_minor > _vol_in * TENS_TRANSP_TAB_SIZE ||
            _vol_minor > _vol_out * TENS_TRANSP_TAB_SIZE) {
          //  Algorithm 0 (slower):
          //   Read the minor volume into the buffer from the input tensor
          //   block:
          _vol_minor /= _vol_in;              // vol_in_c
          _s1 = 1 + (_vol_in - 1) / warpSize; // number of warps (lines) which
                                              // fully cover the input volume
          _s2 = blockDim_x / warpSize; // number of whole warps in a thread
                                       // block (each warp treats one line)
          for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
            m = j / _s1;
            _addr = _addr_in;
            n = m; // n: Input column number (in_c)
            for (i = *minor_in; i < *minor; i++) {
              k = m / dim_in[i];
              _addr += (m - k * dim_in[i]) * base_in[i];
              m = k;
            }
            //    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset in
            //    the input volume
            m = threadIdx_x +
                (j - n * _s1 - l) * warpSize; // elemental offset in the input
                                              // volume (alternative)
            if (m < _vol_in) {
              buf0[n * _vol_in + m] = tens_in[_addr + m];
            }
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
          //   Write the minor volume from the buffer into the output tensor
          //   block:
          _vol_minor = (_vol_minor * _vol_in) / _vol_out; // vol_out_c
          _s1 = 1 + (_vol_out - 1) / warpSize; // number of warps (lines) which
                                               // fully cover the output volume
          for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
            n = j / _s1;
            _addr = _addr_out;
            _vol = n;
            _vol_in = 0; //_vol: Output column number (out_c)
            //    for(i=minor_out;i<minor;i++){m=n%dim_out[i]; n/=dim_out[i];
            //    _addr+=m*base_out[i]; _vol_in+=m*tmp0[pri[i]];}
            for (i = *minor_out; i < *minor; i++) {
              k = n / dim_out[i];
              m = n - k * dim_out[i];
              n = k;
              _addr += m * base_out[i];
              _vol_in += m * n2o[i];
            }
            //    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset in
            //    the output volume
            m = threadIdx_x + (j - (int)_vol * _s1 - l) *
                                  warpSize; // elemental offset in the output
                                            // volume (alternative)
            if (m < _vol_out) {
              _addr += m;
              //     for(i=0;i<minor_out;i++){_vol_in+=(m%dim_out[i])*tmp0[pri[i]];
              //     m/=dim_out[i];}
              for (i = 0; i < *minor_out; i++) {
                k = m / dim_out[i];
                _vol_in += (m - k * dim_out[i]) * n2o[i];
                m = k;
              }
              tens_out[_addr] += buf0[_vol_in];
            }
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
        } else {
          //  Algorithm 1 (presumably faster):
          //   Create per-block look-up tables:
          m = _vol_minor / _vol_in; // vol_in_c
          for (j = threadIdx_x; j < m;
               j += blockDim_x) { // column number (input)
            _addr = 0;
            _s1 = j;
            //    for(i=minor_in;i<minor;i++){_addr+=(_s1%dim_in[i])*base_in[i];
            //    _s1/=dim_in[i];}
            for (i = *minor_in; i < *minor; i++) {
              _s2 = _s1 / dim_in[i];
              _addr += (_s1 - _s2 * dim_in[i]) * base_in[i];
              _s1 = _s2;
            }
            ftb[j] = _addr;
          }
          m = _vol_minor / _vol_out; // vol_out_c
          for (j = threadIdx_x; j < m;
               j += blockDim_x) { // column number (output)
            _addr = 0;
            _s1 = j;
            //    for(i=minor_out;i<minor;i++){_addr+=(_s1%dim_out[i])*base_out[i];
            //    _s1/=dim_out[i];}
            for (i = *minor_out; i < *minor; i++) {
              _s2 = _s1 / dim_out[i];
              _addr += (_s1 - _s2 * dim_out[i]) * base_out[i];
              _s1 = _s2;
            }
            gtb[j] = _addr;
          }
          for (j = threadIdx_x; j < m;
               j += blockDim_x) { // column number (output)
            n = 0;
            _s1 = j;
            //    for(i=minor_out;i<minor;i++){n+=(_s1%dim_out[i])*n2o[i];
            //    _s1/=dim_out[i];}
            for (i = *minor_out; i < *minor; i++) {
              _s2 = _s1 / dim_out[i];
              n += (_s1 - _s2 * dim_out[i]) * n2o[i];
              _s1 = _s2;
            }
            htb[j] = n;
          }
          for (j = threadIdx_x; j < _vol_out; j += blockDim_x) {
            n = 0;
            _s1 = j;
            //    for(i=0;i<minor_out;i++){n+=(_s1%dim_out[i])*n2o[i];
            //    _s1/=dim_out[i];}
            for (i = 0; i < *minor_out; i++) {
              _s2 = _s1 / dim_out[i];
              n += (_s1 - _s2 * dim_out[i]) * n2o[i];
              _s1 = _s2;
            }
            stb[j] = n;
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
          //   Read the minor volume into the buffer from the input tensor
          //   block:
          _vol_minor /= _vol_in;              // vol_in_c
          _s1 = 1 + (_vol_in - 1) / warpSize; // number of warps (lines) which
                                              // fully cover the input volume
          _s2 = blockDim_x / warpSize; // number of whole warps in a thread
                                       // block (each warp treats one line)
          for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
            m = j / _s1;
            n = threadIdx_x +
                (j - m * _s1 - l) * warpSize; // m: Input column number (in_c);
                                              // n: Offset in the column
            if (n < _vol_in) {
              _addr = _addr_in + ftb[m] + n;
              buf0[m * _vol_in + n] = tens_in[_addr];
            }
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
          //   Write the minor volume from the buffer into the output tensor
          //   block:
          _vol_minor = (_vol_minor * _vol_in) / _vol_out; // vol_out_c
          _s1 = 1 + (_vol_out - 1) / warpSize; // number of warps (lines) which
                                               // fully cover the output volume
          for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
            m = j / _s1;
            n = threadIdx_x +
                (j - m * _s1 - l) * warpSize; // m: Output column number
                                              // (out_c); n: Offset in the column
            if (n < _vol_out) {
              _addr = _addr_out + gtb[m] + n;
              _vol_in = htb[m] + stb[n];
              tens_out[_addr] += buf0[_vol_in];
            }
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
        }
      } // enddo _work_piece: independent work distribution among thread blocks
    }
  }
  // Record errors if occured (for each block):
  if (threadIdx_x == 0) {
    if (*err_code != 0)
      i = atomic_ref<int>(gpu_error_count).fetch_add(1);
  }
  return;
}
//----------------------------------------------------------------------------------------------------
// TENSOR TRANSPOSE (shared-memory version):
template <typename T>
void gpu_tensor_block_copy_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
				 const T *__restrict__ tens_in, T *__restrict__ tens_out,
				 cl::sycl::nd_item<1>& item,
				 constant_accessor<int, 2>& const_args_dims, constant_accessor<int, 2>& const_args_prmn,
				 int *gpu_error_count,
				 T *buf0, float *val, size_t *base_in, size_t *base_out, size_t *ftb,
				 size_t *gtb, int *htb, int *stb, int *dim_in, int *dim_out, int *o2n,
				 int *n2o, int *pri, int *tmp0, int *err_code, int *minor, int *minor_in,
				 int *minor_out, int *s1_ind, int *s1_ond, int *s1_step, int *s1_dim,
				 int *s2_ind, int *s2_ond, int *s2_step, int *s2_dim, int *ns1, int *ns2,
				 size_t *vol, size_t *vol_ext)
/**
   Shared-memory version of tensor transpose: tens_out=TRN(tens_in):
   INPUT:
   # dmo - dimension extents order (0: normal, as it is in <const_args>; not 0:
   permuted dimension order will be imposed); # drc - index permutation direction
   (0: normal, as it is in <const_args>; not 0: inversed permutation will be used);
   # dim_num - tensor block rank;
   # const_args_pos - entry in the __constant__ memory bank where tensor block
   dimension extents (const_args_dims) and index permutation (const_args_prmn) are
   stored; # tens_in[0:] - input tensor; OUTPUT: # tens_out[0:] - output
   (transposed) tensor; NOTES: # Minimal SYCL execution configuration is
   <<<1,warpSize>>> # Number of threads per block must be multiple of the warpSize!
**/
{
  size_t _vol, _addr_in, _addr_out, _addr, _work_piece;
  int i, j, k, l, m, n, _vol_minor, _vol_in, _vol_out, _s1, _s2;
  /*
    SHARED MEMORY USE (bytes) =
    + TENS_TRANSP_BUF_SIZE*sizeof(T)
    + MAX_TENSOR_RANK*(8+8+4+4+4+4+4+4)
    + TENS_TRANSP_TAB_SIZE*(8+8+4+4)
    + 4*15 + 8*2
    MIN REGISTER USE (bytes) per thread =
    + 4*4 + 4*11 + 8*5 = 100
  */

  size_t threadIdx_x = item.get_local_id(0);
  size_t blockDim_x = item.get_local_range(0);
  size_t gridDim_x = item.get_group_range(0);
  size_t blockIdx_x = item.get_group(0);

  // Determine the minor index set (only the master thread in each thread
  // block):
  if (threadIdx_x == 0) {
    *err_code = 0;
    if (dim_num >= 0 && dim_num <= MAX_TENSOR_RANK && blockDim_x >= warpSize &&
        blockDim_x % warpSize == 0) {
      *s1_ind = dim_num + 1;
      *s2_ind = dim_num - 1;
      _vol = 1;
      for (i = 0; i < dim_num; i++) {
        _vol *= const_args_dims[const_args_pos][i];
        if (const_args_prmn[const_args_pos][i] != i + 1)
          *s1_ind = 0;
      };
      *vol = _vol;        // total volume (number of tensor elements)
      if (*s1_ind == 0) { // non-trivial permutation
        // Set input/output permutations and dimension extents:
        if (drc == 0) { // normal index permutation
          for (i = 0; i < dim_num; i++)
            o2n[i] = const_args_prmn[const_args_pos][i] - 1;
          for (i = 0; i < dim_num; i++)
            n2o[o2n[i]] = i;
        } else { // inversed index permutation
          for (i = 0; i < dim_num; i++)
            n2o[i] = const_args_prmn[const_args_pos][i] - 1;
          for (i = 0; i < dim_num; i++)
            o2n[n2o[i]] = i;
        }
        if (dmo == 0) { // normal dimension order
          for (i = 0; i < dim_num; i++)
            dim_in[i] = const_args_dims[const_args_pos][i];
          for (i = 0; i < dim_num; i++)
            dim_out[o2n[i]] = dim_in[i];
        } else { // inversed dimension order
          for (i = 0; i < dim_num; i++)
            dim_out[i] = const_args_dims[const_args_pos][i];
          for (i = 0; i < dim_num; i++)
            dim_in[n2o[i]] = dim_out[i];
        }
        *s1_step = dim_in[(*s1_ind)];
        *s2_step = dim_in[(*s2_ind)];
        if (_vol > TENS_TRANSP_BUF_SIZE) { // tensor block does not fit into the
                                           // shared memory buffer
          // Determine the input/output minor index sets and the combined minor
          // index set:
          l = (int)(cl::sycl::sqrt((float)TENS_TRANSP_BUF_SIZE));
          *minor_in = 0;
          _vol_in = 1;
          for (i = 0; i < dim_num; i++) {
            j = _vol_in * dim_in[i];
            if (j > l)
              break;
            (*minor_in)++;
            _vol_in = j;
          }
          *minor_out = 0;
          _vol_out = 1;
          for (i = 0; i < dim_num; i++) {
            j = _vol_out * dim_out[i];
            if (j > l)
              break;
            (*minor_out)++;
            _vol_out = j;
          }
          *minor = *minor_in;
          _vol_minor = _vol_in;
          for (i = 0; i < *minor_out; i++) {
            if (n2o[i] >= *minor_in) {
              (*minor)++;
              _vol_minor *= dim_out[i];
            }
          }
          m = 1;
          _s1 = 0;
          _s2 = 0;
          while (_vol_minor < TENS_TRANSP_BUF_SIZE && m != 0) {
            m = 0;
            if (_s1 == 0) {
              for (i = *minor_in; i < dim_num; i++) {
                if (o2n[i] < *minor_out) {
                  (*minor_in)++;
                  _vol_in *= dim_in[i];
                } else {
                  break;
                }
              }
            }
            if (_s2 == 0) {
              for (i = *minor_out; i < dim_num; i++) {
                if (n2o[i] < *minor_in) {
                  (*minor_out)++;
                  _vol_out *= dim_out[i];
                } else {
                  break;
                }
              }
            }
            j = dim_in[(*minor_in)];
            l = dim_out[(*minor_out)];
            if (*minor_in == n2o[(*minor_out)] &&
                _s1 + _s2 == 0) { // same candidate index to both the
              // input and output index sets
              if (j > 1 && TENS_TRANSP_BUF_SIZE < _vol_minor * 2)
                break;
              if (_vol_minor * j > TENS_TRANSP_BUF_SIZE) {
                *s1_ind = *minor_in;
                *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                _s1++;
                _s2++;
              }
              (*minor_in)++;
              _vol_in *= j;
              (*minor_out)++;
              _vol_out *= j;
              (*minor)++;
              _vol_minor *= j;
              m++;
            } else { // the input and output index sets consider two different
                     // candidates
              if (_vol_minor * j * l <= TENS_TRANSP_BUF_SIZE &&
                  _s1 + _s2 == 0) { // accept both, no splitting
                (*minor_in)++;
                _vol_in *= j;
                (*minor_out)++;
                _vol_out *= l;
                *minor += 2;
                _vol_minor *= (j * l);
                m++;
              } else { // try to accept either one of the two OR both with
                       // splitting
                if (j == 1 || l == 1) {
                  if (j == 1 && _s1 == 0) {
                    (*minor_in)++;
                    (*minor)++;
                    m++;
                  }
                  if (l == 1 && _s2 == 0) {
                    (*minor_out)++;
                    (*minor)++;
                    m++;
                  }
                } else {
                  if (_vol_minor * j <= TENS_TRANSP_BUF_SIZE &&
                      _vol_minor * l > TENS_TRANSP_BUF_SIZE &&
                      _vol_out >= warpSize &&
                      _s1 == 0) { // accept the input index, no splitting
                    (*minor_in)++;
                    _vol_in *= j;
                    (*minor)++;
                    _vol_minor *= j;
                    m++;
                  } else if (_vol_minor * j > TENS_TRANSP_BUF_SIZE &&
                             _vol_minor * l <= TENS_TRANSP_BUF_SIZE &&
                             _vol_in >= warpSize &&
                             _s2 ==
                                 0) { // accept the output index, no splitting
                    (*minor_out)++;
                    _vol_out *= l;
                    (*minor)++;
                    _vol_minor *= l;
                    m++;
                  } else { // splitting is unavoidable (both OR one OR none)
                    if (TENS_TRANSP_BUF_SIZE >= _vol_minor * 2) {
                      if (j >= 4 && l >= 4) { // dimension extents are large
                                              // enough to be split
                        if (_vol_minor * 4 >
                            TENS_TRANSP_BUF_SIZE) { // impossible to split both
                                                    // indices
                          if (_vol_in <= _vol_out &&
                              _s1 == 0) { // split the input candidate index
                            *s1_ind = *minor_in;
                            *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                            (*minor_in)++;
                            _vol_in *= j;
                            (*minor)++;
                            _vol_minor *= j;
                            _s1++;
                            m++;
                          } else { // split the output candidate index
                            if (_s2 == 0) {
                              *s1_ind = n2o[(*minor_out)];
                              *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                              (*minor_out)++;
                              _vol_out *= l;
                              (*minor)++;
                              _vol_minor *= l;
                              _s2++;
                              m++;
                            }
                          }
                        } else { // possible to split both indices
                          i = (int)cl::sycl::sqrt(
                              ((float)TENS_TRANSP_BUF_SIZE) /
                              (float)_vol_minor);
                          if (i < 2)
                            i = 2; // uniform splitting
                          *s1_step = i;
                          *s2_step = i;
                          *val = (float)_vol_out / (float)_vol_in;
                          if (*val <
                              1.0f) { // scale the initial uniform splitting to
                            // reflect the disbalance between _vol_in and
                            // _vol_out
                            if (*val * (float)i < 1.0f)
                              *val = 1.0f / (float)i;
                            if (*val * (float)l < (float)i)
                              *val = (float)i / (float)l;
                          } else {
                            if (*val * (float)i > (float)j)
                              *val = (float)j / (float)i;
                            if (*val > float(i))
                              *val = (float)i;
                          }
                          *s1_step = (int)(((float)i) * *val);
                          *s2_step = (int)(((float)i) / *val);
                          if (*s1_step >= 2 &&
                              _s1 == 0) { //&& s1_step <= dim_in[minor_in]
                            *s1_ind = *minor_in;
                            (*minor_in)++;
                            _vol_in *= j;
                            (*minor)++;
                            _vol_minor *= j;
                            _s1++;
                            m++;
                          } else {
                            *s1_step = dim_in[(*s1_ind)];
                          }
                          if (*s2_step >= 2 &&
                              _s2 == 0) { //&& s2_step <= dim_out[minor_out]
                            *s2_ind = n2o[(*minor_out)];
                            (*minor_out)++;
                            _vol_out *= l;
                            (*minor)++;
                            _vol_minor *= l;
                            _s2++;
                            m++;
                          } else {
                            *s2_step = dim_in[(*s2_ind)];
                          }
                        }
                      } else if (j >= 4 && l < 4 &&
                                 _s1 == 0) { // split the input candidate index
                        *s1_ind = *minor_in;
                        *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                        (*minor_in)++;
                        _vol_in *= j;
                        (*minor)++;
                        _vol_minor *= j;
                        _s1++;
                        m++;
                      } else if (j < 4 && l >= 4 &&
                                 _s2 == 0) { // split the output candidate index
                        *s1_ind = n2o[(*minor_out)];
                        *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                        (*minor_out)++;
                        _vol_out *= l;
                        (*minor)++;
                        _vol_minor *= l;
                        _s2++;
                        m++;
                      } else { // both candidate indices have too small extent
                               // to be split: try to add one of them fully
                        if (_vol_minor * j <= TENS_TRANSP_BUF_SIZE &&
                            _s1 == 0) {
                          (*minor_in)++;
                          _vol_in *= j;
                          (*minor)++;
                          _vol_minor *= j;
                          m++;
                        } else if (_vol_minor * l <= TENS_TRANSP_BUF_SIZE &&
                                   _s2 == 0) {
                          (*minor_out)++;
                          _vol_out *= l;
                          (*minor)++;
                          _vol_minor *= l;
                          m++;
                        }
                      }
                    } else { // unable to add more indices in the minor set
                      break;
                    }
                  }
                }
              }
            }
          }
          if (*s1_ind == dim_num - 1 && *s2_ind == dim_num - 1) {
            *s2_ind = 0;
            *s2_step = dim_in[0];
          }      // s1_ind was set while s2_ind was not
        } else { // tensor block fits into the shared memory buffer from the
                 // beginning
          *minor = dim_num;
          *minor_in = dim_num;
          *minor_out = dim_num;
          _vol_minor = _vol;
          _vol_in = _vol;
          _vol_out = _vol;
        }
        // Share the tensor transpose configuration with other threads in each
        // block:
        *vol_ext = _vol / _vol_minor;
        *s1_dim = dim_in[(*s1_ind)];
        *s2_dim = dim_in[(*s2_ind)];
        // Set indexing bases (OUT:{out,in_c,ext_in}_new;
        // IN:{in,out_c,ext_in}_old):
        //  OUTPUT indexing (dim_out[], base_out[]: prioritized new numeration):
        for (i = 0; i < dim_num; i++) {
          tmp0[i] = dim_out[i];
        } // save output dimension extents (new numeration)
        j = 0;
        for (i = 0; i < *minor_out; i++) {
          pri[j++] = i;
        } // output minor index set (new numeration))
        for (i = 0; i < dim_num; i++) {
          if (o2n[i] >= *minor_out)
            pri[j++] = o2n[i];
        } //{compl.input minor + external} index set (new numeration)
        j = 1;
        for (i = 0; i < dim_num; i++) {
          dim_out[i] = j;
          j *= tmp0[i];
        } // output bases (new numeration)
        for (i = 0; i < dim_num; i++) {
          base_out[i] = dim_out[pri[i]];
        } // output bases (prioritized new numeration)
        for (i = 0; i < dim_num; i++) {
          dim_out[i] = tmp0[pri[i]];
        } // output extents (prioritized new numeration)
        for (i = 0; i < dim_num; i++) {
          if (n2o[pri[i]] == *s1_ind) {
            *s1_ond = i;
          } else if (n2o[pri[i]] == *s2_ind) {
            *s2_ond = i;
          }
        } // split indices (prioritized new numeration)
        //  INPUT indexing (dim_in[], base_in[]: prioritized old numeration):
        for (i = 0; i < dim_num; i++) {
          tmp0[i] = dim_in[i];
        } // save input dimension extents (old numeration)
        j = 0;
        for (i = 0; i < *minor_in; i++) {
          pri[j++] = i;
        } // input minor index set (old numeration)
        for (i = 0; i < *minor_out; i++) {
          if (n2o[i] >= *minor_in)
            pri[j++] = n2o[i];
        } // compl.output minor idex set (old numeration)
        for (i = j; i < dim_num; i++) {
          pri[i] = n2o[pri[i]];
        } // external index set (just convert new numbers to old ones for
          // consistency)
        j = 1;
        for (i = 0; i < dim_num; i++) {
          dim_in[i] = j;
          j *= tmp0[i];
        } // input bases (old numeration)
        for (i = 0; i < dim_num; i++) {
          base_in[i] = dim_in[pri[i]];
        } // input bases (prioritized old numeration)
        for (i = 0; i < dim_num; i++) {
          dim_in[i] = tmp0[pri[i]];
        } // input extents (prioritized old numeration)
        for (i = 0; i < dim_num; i++) {
          if (pri[i] == *s1_ind) {
            _s1 = i;
          } else if (pri[i] == *s2_ind) {
            _s2 = i;
          }
        } // split indices (prioritized old numeration)
        *s1_ind = _s1;
        *s2_ind = _s2;
        *ns1 =
            1 +
            (*s1_dim - 1) /
                *s1_step; // number of segments from the 1st split minor index
        *ns2 =
            1 +
            (*s2_dim - 1) /
                *s2_step; // number of segments from the 2nd split minor index
        //  Index position correspondence for the minor index set (pri-new -->
        //  pri-old):
        j = 0;
        for (i = 0; i < *minor_out; i++) {
          if (n2o[i] < *minor_in) {
            pri[i] = n2o[i];
          } else {
            pri[i] = (*minor_in + j);
            j++;
          }
        }
        j = 0;
        for (i = 0; i < *minor_in; i++) {
          if (o2n[i] < *minor_out) {
            pri[o2n[i]] = i;
          } else {
            pri[*minor_out + j] = i;
            j++;
          }
        }
        // Check tensor transpose configuration parameters:
        if (*minor <= 0 || *minor_in <= 0 || *minor_out <= 0 || _vol <= 0 ||
            _vol_minor <= 0)
          *err_code += 5000; // trap
        if (*s1_ind >= dim_num || *s2_ind >= dim_num || *s1_ond >= dim_num ||
            *s2_ond >= dim_num || *s1_ind == *s2_ind || *s1_ond == *s2_ond ||
            *s1_step <= 0 || *s2_step <= 0)
          *err_code += 1000; // trap
        if ((*s1_step != dim_in[(*s1_ind)] && *s1_ind != *minor_in - 1 &&
             *s1_ond != *minor_out - 1) ||
            (*s2_step != dim_in[(*s2_ind)] && *s2_ind != *minor_in - 1 &&
             *s2_ond != *minor_out - 1))
          *err_code += 500; // trap
        if ((_vol_minor * *s1_step * *s2_step) / (*s1_dim * *s2_dim) >
            TENS_TRANSP_BUF_SIZE)
          *err_code += 100; // trap
      }                     // endif: non-trivial permutation
    } else {
      *err_code = 1 + 2 * blockDim_x % warpSize;
    }
  } // endif: Master thread.
  item.barrier(cl::sycl::access::fence_space::local_space);

  // Proceed:
  if (*err_code == 0) {
    if (*s1_ind > dim_num) { // tag of a trivial permutation
                             // Direct copy:
      _vol = *vol;
      j = gridDim_x * blockDim_x;
      i = item.get_global_id(0);
      _addr_in = _vol - _vol % j;
      for (_addr = 0; _addr < _addr_in; _addr += j) {
        _addr_out = _addr + i;
        tens_out[_addr_out] = tens_in[_addr_out];
      }
      _addr_out = _addr_in + i;
      if (_addr_out < _vol)
        tens_out[_addr_out] = tens_in[_addr_out];
    } else {                      // non-trivial permutation
      l = threadIdx_x / warpSize; // l: warp number
      // Distribute work accross CUDA blocks (external multi-index + splitting):
      for (_work_piece = blockIdx_x; _work_piece < *vol_ext * *ns1 * *ns2;
           _work_piece += gridDim_x) { //(ns1*ns2*vol_ext) is the total
        // number of independent tasks
        _addr = _work_piece;
        _addr /= *vol_ext;
        _vol = _work_piece - _addr * *vol_ext;
        _s2 = (int)(_addr / *ns1);
        _s1 = (int)(_addr - _s2 * *ns1); //{_addr_ext,_s1,_s2} --> tensor
                                         //subblock (CUDA block)
        //  Modify dimension extents due to possible dimension splitting:
        if (threadIdx_x == 0) {
          if (_s1 + 1 == *ns1) { // last segment of the 1st split index
            j = *s1_dim - _s1 * *s1_step;
            dim_in[(*s1_ind)] = j;
            dim_out[(*s1_ond)] = j;
          } else { // internal segment of the 1st split index
            dim_in[(*s1_ind)] = *s1_step;
            dim_out[(*s1_ond)] = *s1_step;
          }
          if (_s2 + 1 == *ns2) { // last segment of the 2nd split index
            j = *s2_dim - _s2 * *s2_step;
            dim_in[(*s2_ind)] = j;
            dim_out[(*s2_ond)] = j;
          } else { // internal segment of the 2nd split index
            dim_in[(*s2_ind)] = *s2_step;
            dim_out[(*s2_ond)] = *s2_step;
          }
          j = 1;
          for (i = 0; i < *minor; i++) {
            tmp0[i] = j;
            j *= dim_in[i];
          } // minor buffer bases (pri-old)
          for (i = 0; i < *minor; i++)
            n2o[i] = tmp0[pri[i]]; // look up table to accelerate further
                                   // accesses to tmp0[]
        }
        item.barrier(cl::sycl::access::fence_space::local_space);
        //  Mount input/output volumes and bases:
        _vol_in = dim_in[0];
        for (i = 1; i < *minor_in; i++) {
          _vol_in *= dim_in[i];
        }
        _vol_out = dim_out[0];
        for (i = 1; i < *minor_out; i++) {
          _vol_out *= dim_out[i];
        }
        _vol_minor = _vol_out;
        for (i = *minor_out; i < *minor; i++) {
          _vol_minor *= dim_out[i];
        }
        _addr_in = (_s1 * *s1_step) * base_in[(*s1_ind)] +
                   (_s2 * *s2_step) * base_in[(*s2_ind)];
        _addr_out = _vol;
        for (i = *minor; i < dim_num; i++) {
          _addr = _vol / dim_in[i];
          _addr_in += (_vol - _addr * dim_in[i]) * base_in[i];
          _vol = _addr;
        }
        _vol = _addr_out;
        _addr_out = (_s1 * *s1_step) * base_out[(*s1_ond)] +
                    (_s2 * *s2_step) * base_out[(*s2_ond)];
        for (i = *minor; i < dim_num; i++) {
          _addr = _vol / dim_out[i];
          _addr_out += (_vol - _addr * dim_out[i]) * base_out[i];
          _vol = _addr;
        }
        if (_vol_out > TENS_TRANSP_TAB_SIZE ||
            _vol_minor > _vol_in * TENS_TRANSP_TAB_SIZE ||
            _vol_minor > _vol_out * TENS_TRANSP_TAB_SIZE) {
          //  Algorithm 0 (slower):
          //   Read the minor volume into the buffer from the input tensor
          //   block:
          _vol_minor /= _vol_in;              // vol_in_c
          _s1 = 1 + (_vol_in - 1) / warpSize; // number of warps (lines) which
                                              // fully cover the input volume
          _s2 = blockDim_x / warpSize; // number of whole warps in a thread
                                       // block (each warp treats one line)
          for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
            m = j / _s1;
            _addr = _addr_in;
            n = m; // n: Input column number (in_c)
            for (i = *minor_in; i < *minor; i++) {
              k = m / dim_in[i];
              _addr += (m - k * dim_in[i]) * base_in[i];
              m = k;
            }
            //    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset in
            //    the input volume
            m = threadIdx_x +
                (j - n * _s1 - l) * warpSize; // elemental offset in the input
                                              // volume (alternative)
            if (m < _vol_in) {
              buf0[n * _vol_in + m] = tens_in[_addr + m];
            }
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
          //   Write the minor volume from the buffer into the output tensor
          //   block:
          _vol_minor = (_vol_minor * _vol_in) / _vol_out; // vol_out_c
          _s1 = 1 + (_vol_out - 1) / warpSize; // number of warps (lines) which
                                               // fully cover the output volume
          for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
            n = j / _s1;
            _addr = _addr_out;
            _vol = n;
            _vol_in = 0; //_vol: Output column number (out_c)
            //    for(i=minor_out;i<minor;i++){m=n%dim_out[i]; n/=dim_out[i];
            //    _addr+=m*base_out[i]; _vol_in+=m*tmp0[pri[i]];}
            for (i = *minor_out; i < *minor; i++) {
              k = n / dim_out[i];
              m = n - k * dim_out[i];
              n = k;
              _addr += m * base_out[i];
              _vol_in += m * n2o[i];
            }
            //    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset in
            //    the output volume
            m = threadIdx_x + (j - (int)_vol * _s1 - l) *
                                  warpSize; // elemental offset in the output
                                            // volume (alternative)
            if (m < _vol_out) {
              _addr += m;
              //     for(i=0;i<minor_out;i++){_vol_in+=(m%dim_out[i])*tmp0[pri[i]];
              //     m/=dim_out[i];}
              for (i = 0; i < *minor_out; i++) {
                k = m / dim_out[i];
                _vol_in += (m - k * dim_out[i]) * n2o[i];
                m = k;
              }
              tens_out[_addr] = buf0[_vol_in];
            }
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
        } else {
          //  Algorithm 1 (presumably faster):
          //   Create per-block look-up tables:
          m = _vol_minor / _vol_in; // vol_in_c
          for (j = threadIdx_x; j < m;
               j += blockDim_x) { // column number (input)
            _addr = 0;
            _s1 = j;
            //    for(i=minor_in;i<minor;i++){_addr+=(_s1%dim_in[i])*base_in[i];
            //    _s1/=dim_in[i];}
            for (i = *minor_in; i < *minor; i++) {
              _s2 = _s1 / dim_in[i];
              _addr += (_s1 - _s2 * dim_in[i]) * base_in[i];
              _s1 = _s2;
            }
            ftb[j] = _addr;
          }
          m = _vol_minor / _vol_out; // vol_out_c
          for (j = threadIdx_x; j < m;
               j += blockDim_x) { // column number (output)
            _addr = 0;
            _s1 = j;
            //    for(i=minor_out;i<minor;i++){_addr+=(_s1%dim_out[i])*base_out[i];
            //    _s1/=dim_out[i];}
            for (i = *minor_out; i < *minor; i++) {
              _s2 = _s1 / dim_out[i];
              _addr += (_s1 - _s2 * dim_out[i]) * base_out[i];
              _s1 = _s2;
            }
            gtb[j] = _addr;
          }
          for (j = threadIdx_x; j < m;
               j += blockDim_x) { // column number (output)
            n = 0;
            _s1 = j;
            //    for(i=minor_out;i<minor;i++){n+=(_s1%dim_out[i])*n2o[i];
            //    _s1/=dim_out[i];}
            for (i = *minor_out; i < *minor; i++) {
              _s2 = _s1 / dim_out[i];
              n += (_s1 - _s2 * dim_out[i]) * n2o[i];
              _s1 = _s2;
            }
            htb[j] = n;
          }
          for (j = threadIdx_x; j < _vol_out; j += blockDim_x) {
            n = 0;
            _s1 = j;
            //    for(i=0;i<minor_out;i++){n+=(_s1%dim_out[i])*n2o[i];
            //    _s1/=dim_out[i];}
            for (i = 0; i < *minor_out; i++) {
              _s2 = _s1 / dim_out[i];
              n += (_s1 - _s2 * dim_out[i]) * n2o[i];
              _s1 = _s2;
            }
            stb[j] = n;
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
          //   Read the minor volume into the buffer from the input tensor
          //   block:
          _vol_minor /= _vol_in;              // vol_in_c
          _s1 = 1 + (_vol_in - 1) / warpSize; // number of warps (lines) which
                                              // fully cover the input volume
          _s2 = blockDim_x / warpSize; // number of whole warps in a thread
                                       // block (each warp treats one line)
          for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
            m = j / _s1;
            n = threadIdx_x +
                (j - m * _s1 - l) * warpSize; // m: Input column number (in_c);
                                              // n: Offset in the column
            if (n < _vol_in) {
              _addr = _addr_in + ftb[m] + n;
              buf0[m * _vol_in + n] = tens_in[_addr];
            }
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
          //   Write the minor volume from the buffer into the output tensor
          //   block:
          _vol_minor = (_vol_minor * _vol_in) / _vol_out; // vol_out_c
          _s1 = 1 + (_vol_out - 1) / warpSize; // number of warps (lines) which
                                               // fully cover the output volume
          for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
            m = j / _s1;
            n = threadIdx_x +
                (j - m * _s1 - l) * warpSize; // m: Output column number
            // (out_c); n: Offset in the column
            if (n < _vol_out) {
              _addr = _addr_out + gtb[m] + n;
              _vol_in = htb[m] + stb[n];
              tens_out[_addr] = buf0[_vol_in];
            }
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
        }
      } // enddo _work_piece: independent work distribution among thread blocks
    }
  }
  // Record errors if occured (for each block):
  if (threadIdx_x == 0) {
    if (*err_code != 0)
      i = atomic_ref<int>(gpu_error_count).fetch_add(1);
  }
  return;
}
//-------------------------------------------------------------------------------------------------------------------
// TENSOR TRANSPOSE (shared-memory version):
template <typename T>
void gpu_tensor_block_copy_cmplx_split_in_dlf__(
    int dmo, int drc, int dim_num, int const_args_pos,
    const T *__restrict__ tens_in, T *__restrict__ tens_out,
    cl::sycl::nd_item<1>& item,
    dpct::accessor<int, dpct::device, 2>& const_args_dims,
    dpct::accessor<int, dpct::device, 2>& const_args_prmn, int *gpu_error_count,
    T *buf0, float *val, size_t *base_in, size_t *base_out, size_t *ftb,
    size_t *gtb, int *htb, int *stb, int *dim_in, int *dim_out, int *o2n,
    int *n2o, int *pri, int *tmp0, int *err_code, int *minor, int *minor_in,
    int *minor_out, int *s1_ind, int *s1_ond, int *s1_step, int *s1_dim,
    int *s2_ind, int *s2_ond, int *s2_step, int *s2_dim, int *ns1, int *ns2,
    size_t *vol, size_t *vol_ext)
/**
   Shared-memory version of tensor transpose: tens_out=TRN(tens_in): Complex
arguments only: INPUT: # dmo - dimension extents order (0: normal, as it is in
<const_args>; not 0: permuted dimension order will be imposed); # drc - index
permutation direction (0: normal, as it is in <const_args>; not 0: inversed
permutation will be used); # dim_num - tensor block rank; # const_args_pos -
entry in the __constant__ memory bank where tensor block dimension extents
(const_args_dims) and index permutation (const_args_prmn) are stored; #
tens_in[0:] - complex input tensor in split representation; OUTPUT: #
tens_out[0:] - complex output (transposed) tensor in normal representation;
   NOTES:
   # Minimal CUDA execution configuration is <<<1,warpSize>>>
   # Number of threads per block must be multiple of the warpSize!
**/
{
    size_t threadIdx_x = item.get_local_id(0);
    size_t blockDim_x = item.get_local_range(0);
    size_t gridDim_x = item.get_group_range(0);
    size_t blockIdx_x = item.get_group(0);

    size_t _vol, _addr_in, _addr_out, _addr, _work_piece;
    int i, j, k, l, m, n, _vol_minor, _vol_in, _vol_out, _s1, _s2;
    /*
      SHARED MEMORY USE (bytes) =
      + TENS_TRANSP_BUF_SIZE*sizeof(T)
      + MAX_TENSOR_RANK*(8+8+4+4+4+4+4+4)
      + TENS_TRANSP_TAB_SIZE*(8+8+4+4)
      + 4*15 + 8*2
      MIN REGISTER USE (bytes) per thread =
      + 4*4 + 4*11 + 8*5 = 100
    */

    static_assert(ComplexType<T>::valid, "Non-complex types are not allowed!");
    typename ComplexType<T>::RealType *tens_in_real =
        (typename ComplexType<T>::RealType *)tens_in;
    // Determine the minor index set (only the master thread in each thread
    // block):
    if (threadIdx_x == 0) {
      *err_code = 0;
      if (dim_num >= 0 && dim_num <= MAX_TENSOR_RANK &&
          blockDim_x >= warpSize && blockDim_x % warpSize == 0) {
        *s1_ind = dim_num + 1;
        *s2_ind = dim_num - 1;
        _vol = 1;
        for (i = 0; i < dim_num; i++) {
          _vol *= const_args_dims[const_args_pos][i];
          if (const_args_prmn[const_args_pos][i] != i + 1)
            *s1_ind = 0;
        };
        *vol = _vol;        // total volume (number of tensor elements)
        if (*s1_ind == 0) { // non-trivial permutation
          // Set input/output permutations and dimension extents:
          if (drc == 0) { // normal index permutation
            for (i = 0; i < dim_num; i++)
              o2n[i] = const_args_prmn[const_args_pos][i] - 1;
            for (i = 0; i < dim_num; i++)
              n2o[o2n[i]] = i;
          } else { // inversed index permutation
            for (i = 0; i < dim_num; i++)
              n2o[i] = const_args_prmn[const_args_pos][i] - 1;
            for (i = 0; i < dim_num; i++)
              o2n[n2o[i]] = i;
          }
          if (dmo == 0) { // normal dimension order
            for (i = 0; i < dim_num; i++)
              dim_in[i] = const_args_dims[const_args_pos][i];
            for (i = 0; i < dim_num; i++)
              dim_out[o2n[i]] = dim_in[i];
          } else { // inversed dimension order
            for (i = 0; i < dim_num; i++)
              dim_out[i] = const_args_dims[const_args_pos][i];
            for (i = 0; i < dim_num; i++)
              dim_in[n2o[i]] = dim_out[i];
          }
          *s1_step = dim_in[(*s1_ind)];
          *s2_step = dim_in[(*s2_ind)];
          if (_vol > TENS_TRANSP_BUF_SIZE) { // tensor block does not fit into
                                             // the shared memory buffer
            // Determine the input/output minor index sets and the combined
            // minor index set:
            l = (int)(cl::sycl::sqrt((float)TENS_TRANSP_BUF_SIZE));
            *minor_in = 0;
            _vol_in = 1;
            for (i = 0; i < dim_num; i++) {
              j = _vol_in * dim_in[i];
              if (j > l)
                break;
              (*minor_in)++;
              _vol_in = j;
            }
            *minor_out = 0;
            _vol_out = 1;
            for (i = 0; i < dim_num; i++) {
              j = _vol_out * dim_out[i];
              if (j > l)
                break;
              (*minor_out)++;
              _vol_out = j;
            }
            *minor = *minor_in;
            _vol_minor = _vol_in;
            for (i = 0; i < *minor_out; i++) {
              if (n2o[i] >= *minor_in) {
                (*minor)++;
                _vol_minor *= dim_out[i];
              }
            }
            m = 1;
            _s1 = 0;
            _s2 = 0;
            while (_vol_minor < TENS_TRANSP_BUF_SIZE && m != 0) {
              m = 0;
              if (_s1 == 0) {
                for (i = *minor_in; i < dim_num; i++) {
                  if (o2n[i] < *minor_out) {
                    (*minor_in)++;
                    _vol_in *= dim_in[i];
                  } else {
                    break;
                  }
                }
              }
              if (_s2 == 0) {
                for (i = *minor_out; i < dim_num; i++) {
                  if (n2o[i] < *minor_in) {
                    (*minor_out)++;
                    _vol_out *= dim_out[i];
                  } else {
                    break;
                  }
                }
              }
              j = dim_in[(*minor_in)];
              l = dim_out[(*minor_out)];
              if (*minor_in == n2o[(*minor_out)] &&
                  _s1 + _s2 == 0) { // same candidate index to both the input
                                    // and output index sets
                if (j > 1 && TENS_TRANSP_BUF_SIZE < _vol_minor * 2)
                  break;
                if (_vol_minor * j > TENS_TRANSP_BUF_SIZE) {
                  *s1_ind = *minor_in;
                  *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                  _s1++;
                  _s2++;
                }
                (*minor_in)++;
                _vol_in *= j;
                (*minor_out)++;
                _vol_out *= j;
                (*minor)++;
                _vol_minor *= j;
                m++;
              } else { // the input and output index sets consider two different
                       // candidates
                if (_vol_minor * j * l <= TENS_TRANSP_BUF_SIZE &&
                    _s1 + _s2 == 0) { // accept both, no splitting
                  (*minor_in)++;
                  _vol_in *= j;
                  (*minor_out)++;
                  _vol_out *= l;
                  *minor += 2;
                  _vol_minor *= (j * l);
                  m++;
                } else { // try to accept either one of the two OR both with
                         // splitting
                  if (j == 1 || l == 1) {
                    if (j == 1 && _s1 == 0) {
                      (*minor_in)++;
                      (*minor)++;
                      m++;
                    }
                    if (l == 1 && _s2 == 0) {
                      (*minor_out)++;
                      (*minor)++;
                      m++;
                    }
                  } else {
                    if (_vol_minor * j <= TENS_TRANSP_BUF_SIZE &&
                        _vol_minor * l > TENS_TRANSP_BUF_SIZE &&
                        _vol_out >= warpSize &&
                        _s1 == 0) { // accept the input index, no splitting
                      (*minor_in)++;
                      _vol_in *= j;
                      (*minor)++;
                      _vol_minor *= j;
                      m++;
                    } else if (_vol_minor * j > TENS_TRANSP_BUF_SIZE &&
                               _vol_minor * l <= TENS_TRANSP_BUF_SIZE &&
                               _vol_in >= warpSize &&
                               _s2 ==
                                   0) { // accept the output index, no splitting
                      (*minor_out)++;
                      _vol_out *= l;
                      (*minor)++;
                      _vol_minor *= l;
                      m++;
                    } else { // splitting is unavoidable (both OR one OR none)
                      if (TENS_TRANSP_BUF_SIZE >= _vol_minor * 2) {
                        if (j >= 4 && l >= 4) { // dimension extents are large
                                                // enough to be split
                          if (_vol_minor * 4 >
                              TENS_TRANSP_BUF_SIZE) { // impossible to split
                                                      // both indices
                            if (_vol_in <= _vol_out &&
                                _s1 == 0) { // split the input candidate index
                              *s1_ind = *minor_in;
                              *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                              (*minor_in)++;
                              _vol_in *= j;
                              (*minor)++;
                              _vol_minor *= j;
                              _s1++;
                              m++;
                            } else { // split the output candidate index
                              if (_s2 == 0) {
                                *s1_ind = n2o[(*minor_out)];
                                *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                                (*minor_out)++;
                                _vol_out *= l;
                                (*minor)++;
                                _vol_minor *= l;
                                _s2++;
                                m++;
                              }
                            }
                          } else { // possible to split both indices
                            i = (int)cl::sycl::sqrt(
                                ((float)TENS_TRANSP_BUF_SIZE) /
                                (float)_vol_minor);
                            if (i < 2)
                              i = 2; // uniform splitting
                            *s1_step = i;
                            *s2_step = i;
                            *val = (float)_vol_out / (float)_vol_in;
                            if (*val < 1.0f) { // scale the initial uniform
                                               // splitting to
                              // reflect the disbalance between _vol_in and
                              // vol_out
                              if (*val * (float)i < 1.0f)
                                *val = 1.0f / (float)i;
                              if (*val * (float)l < (float)i)
                                *val = (float)i / (float)l;
                            } else {
                              if (*val * (float)i > (float)j)
                                *val = (float)j / (float)i;
                              if (*val > float(i))
                                *val = (float)i;
                            }
                            *s1_step = (int)(((float)i) * *val);
                            *s2_step = (int)(((float)i) / *val);
                            if (*s1_step >= 2 &&
                                _s1 == 0) { //&& s1_step <= dim_in[minor_in]
                              *s1_ind = *minor_in;
                              (*minor_in)++;
                              _vol_in *= j;
                              (*minor)++;
                              _vol_minor *= j;
                              _s1++;
                              m++;
                            } else {
                              *s1_step = dim_in[(*s1_ind)];
                            }
                            if (*s2_step >= 2 &&
                                _s2 == 0) { //&& s2_step <= dim_out[minor_out]
                              *s2_ind = n2o[(*minor_out)];
                              (*minor_out)++;
                              _vol_out *= l;
                              (*minor)++;
                              _vol_minor *= l;
                              _s2++;
                              m++;
                            } else {
                              *s2_step = dim_in[(*s2_ind)];
                            }
                          }
                        } else if (j >= 4 && l < 4 &&
                                   _s1 ==
                                       0) { // split the input candidate index
                          *s1_ind = *minor_in;
                          *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                          (*minor_in)++;
                          _vol_in *= j;
                          (*minor)++;
                          _vol_minor *= j;
                          _s1++;
                          m++;
                        } else if (j < 4 && l >= 4 &&
                                   _s2 ==
                                       0) { // split the output candidate index
                          *s1_ind = n2o[(*minor_out)];
                          *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                          (*minor_out)++;
                          _vol_out *= l;
                          (*minor)++;
                          _vol_minor *= l;
                          _s2++;
                          m++;
                        } else { // both candidate indices have too small extent
                                 // to be split: try to add one of them fully
                          if (_vol_minor * j <= TENS_TRANSP_BUF_SIZE &&
                              _s1 == 0) {
                            (*minor_in)++;
                            _vol_in *= j;
                            (*minor)++;
                            _vol_minor *= j;
                            m++;
                          } else if (_vol_minor * l <= TENS_TRANSP_BUF_SIZE &&
                                     _s2 == 0) {
                            (*minor_out)++;
                            _vol_out *= l;
                            (*minor)++;
                            _vol_minor *= l;
                            m++;
                          }
                        }
                      } else { // unable to add more indices in the minor set
                        break;
                      }
                    }
                  }
                }
              }
            }
            if (*s1_ind == dim_num - 1 && *s2_ind == dim_num - 1) {
              *s2_ind = 0;
              *s2_step = dim_in[0];
            }      // s1_ind was set while s2_ind was not
          } else { // tensor block fits into the shared memory buffer from the
                   // beginning
            *minor = dim_num;
            *minor_in = dim_num;
            *minor_out = dim_num;
            _vol_minor = _vol;
            _vol_in = _vol;
            _vol_out = _vol;
          }
          // Share the tensor transpose configuration with other threads in each
          // block:
          *vol_ext = _vol / _vol_minor;
          *s1_dim = dim_in[(*s1_ind)];
          *s2_dim = dim_in[(*s2_ind)];
          // Set indexing bases (OUT:{out,in_c,ext_in}_new;
          // IN:{in,out_c,ext_in}_old):
          //  OUTPUT indexing (dim_out[], base_out[]: prioritized new
          //  numeration):
          for (i = 0; i < dim_num; i++) {
            tmp0[i] = dim_out[i];
          } // save output dimension extents (new numeration)
          j = 0;
          for (i = 0; i < *minor_out; i++) {
            pri[j++] = i;
          } // output minor index set (new numeration))
          for (i = 0; i < dim_num; i++) {
            if (o2n[i] >= *minor_out)
              pri[j++] = o2n[i];
          } //{compl.input minor + external} index set (new numeration)
          j = 1;
          for (i = 0; i < dim_num; i++) {
            dim_out[i] = j;
            j *= tmp0[i];
          } // output bases (new numeration)
          for (i = 0; i < dim_num; i++) {
            base_out[i] = dim_out[pri[i]];
          } // output bases (prioritized new numeration)
          for (i = 0; i < dim_num; i++) {
            dim_out[i] = tmp0[pri[i]];
          } // output extents (prioritized new numeration)
          for (i = 0; i < dim_num; i++) {
            if (n2o[pri[i]] == *s1_ind) {
              *s1_ond = i;
            } else if (n2o[pri[i]] == *s2_ind) {
              *s2_ond = i;
            }
          } // split indices (prioritized new numeration)
          //  INPUT indexing (dim_in[], base_in[]: prioritized old numeration):
          for (i = 0; i < dim_num; i++) {
            tmp0[i] = dim_in[i];
          } // save input dimension extents (old numeration)
          j = 0;
          for (i = 0; i < *minor_in; i++) {
            pri[j++] = i;
          } // input minor index set (old numeration)
          for (i = 0; i < *minor_out; i++) {
            if (n2o[i] >= *minor_in)
              pri[j++] = n2o[i];
          } // compl.output minor idex set (old numeration)
          for (i = j; i < dim_num; i++) {
            pri[i] = n2o[pri[i]];
          } // external index set (just convert new numbers to old ones for
            // consistency)
          j = 1;
          for (i = 0; i < dim_num; i++) {
            dim_in[i] = j;
            j *= tmp0[i];
          } // input bases (old numeration)
          for (i = 0; i < dim_num; i++) {
            base_in[i] = dim_in[pri[i]];
          } // input bases (prioritized old numeration)
          for (i = 0; i < dim_num; i++) {
            dim_in[i] = tmp0[pri[i]];
          } // input extents (prioritized old numeration)
          for (i = 0; i < dim_num; i++) {
            if (pri[i] == *s1_ind) {
              _s1 = i;
            } else if (pri[i] == *s2_ind) {
              _s2 = i;
            }
          } // split indices (prioritized old numeration)
          *s1_ind = _s1;
          *s2_ind = _s2;
          *ns1 =
              1 +
              (*s1_dim - 1) /
                  *s1_step; // number of segments from the 1st split minor index
          *ns2 =
              1 +
              (*s2_dim - 1) /
                  *s2_step; // number of segments from the 2nd split minor index
          //  Index position correspondence for the minor index set (pri-new -->
          //  pri-old):
          j = 0;
          for (i = 0; i < *minor_out; i++) {
            if (n2o[i] < *minor_in) {
              pri[i] = n2o[i];
            } else {
              pri[i] = (*minor_in + j);
              j++;
            }
          }
          j = 0;
          for (i = 0; i < *minor_in; i++) {
            if (o2n[i] < *minor_out) {
              pri[o2n[i]] = i;
            } else {
              pri[*minor_out + j] = i;
              j++;
            }
          }
          // Check tensor transpose configuration parameters:
          if (*minor <= 0 || *minor_in <= 0 || *minor_out <= 0 || _vol <= 0 ||
              _vol_minor <= 0)
            *err_code += 5000; // trap
          if (*s1_ind >= dim_num || *s2_ind >= dim_num || *s1_ond >= dim_num ||
              *s2_ond >= dim_num || *s1_ind == *s2_ind || *s1_ond == *s2_ond ||
              *s1_step <= 0 || *s2_step <= 0)
            *err_code += 1000; // trap
          if ((*s1_step != dim_in[(*s1_ind)] && *s1_ind != *minor_in - 1 &&
               *s1_ond != *minor_out - 1) ||
              (*s2_step != dim_in[(*s2_ind)] && *s2_ind != *minor_in - 1 &&
               *s2_ond != *minor_out - 1))
            *err_code += 500; // trap
          if ((_vol_minor * *s1_step * *s2_step) / (*s1_dim * *s2_dim) >
              TENS_TRANSP_BUF_SIZE)
            *err_code += 100; // trap
        }                     // endif: non-trivial permutation
      } else {
        *err_code = 1 + 2 * blockDim_x % warpSize;
      }
    } // endif: Master thread.
    item.barrier(cl::sycl::access::fence_space::local_space);

    // Proceed:
    if (*err_code == 0) {
      if (*s1_ind > dim_num) { // tag of a trivial permutation
                               // Direct copy:
        _vol = *vol;
        j = item.get_global_range(0);
        i = item.get_global_id(0);
        _addr_in = _vol - _vol % j;
        for (_addr = 0; _addr < _addr_in; _addr += j) {
          _addr_out = _addr + i;
          auto real_part = tens_in_real[_addr_out];
          auto imag_part = tens_in_real[_addr_out + _vol];
          tens_out[_addr_out] = T{real_part, imag_part};
        }
        _addr_out = _addr_in + i;
        if (_addr_out < _vol) {
          auto real_part = tens_in_real[_addr_out];
          auto imag_part = tens_in_real[_addr_out + _vol];
          tens_out[_addr_out] = T{real_part, imag_part};
        }
      } else {                      // non-trivial permutation
        l = threadIdx_x / warpSize; // l: warp number
        // Distribute work accross CUDA blocks (external multi-index +
        // splitting):
        for (_work_piece = blockIdx_x; _work_piece < *vol_ext * *ns1 * *ns2;
             _work_piece += gridDim_x) { //(ns1*ns2*vol_ext)
          // is the total number of independent tasks
          _addr = _work_piece;
          _addr /= *vol_ext;
          _vol = _work_piece - _addr * *vol_ext;
          _s2 = (int)(_addr / *ns1);
          _s1 = (int)(_addr - _s2 * *ns1); //{_addr_ext,_s1,_s2} --> tensor
                                           //subblock (CUDA block)
          //  Modify dimension extents due to possible dimension splitting:
          if (threadIdx_x == 0) {
            if (_s1 + 1 == *ns1) { // last segment of the 1st split index
              j = *s1_dim - _s1 * *s1_step;
              dim_in[(*s1_ind)] = j;
              dim_out[(*s1_ond)] = j;
            } else { // internal segment of the 1st split index
              dim_in[(*s1_ind)] = *s1_step;
              dim_out[(*s1_ond)] = *s1_step;
            }
            if (_s2 + 1 == *ns2) { // last segment of the 2nd split index
              j = *s2_dim - _s2 * *s2_step;
              dim_in[(*s2_ind)] = j;
              dim_out[(*s2_ond)] = j;
            } else { // internal segment of the 2nd split index
              dim_in[(*s2_ind)] = *s2_step;
              dim_out[(*s2_ond)] = *s2_step;
            }
            j = 1;
            for (i = 0; i < *minor; i++) {
              tmp0[i] = j;
              j *= dim_in[i];
            } // minor buffer bases (pri-old)
            for (i = 0; i < *minor; i++)
              n2o[i] = tmp0[pri[i]]; // look up table to accelerate further
                                     // accesses to tmp0[]
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
          //  Mount input/output volumes and bases:
          _vol_in = dim_in[0];
          for (i = 1; i < *minor_in; i++) {
            _vol_in *= dim_in[i];
          }
          _vol_out = dim_out[0];
          for (i = 1; i < *minor_out; i++) {
            _vol_out *= dim_out[i];
          }
          _vol_minor = _vol_out;
          for (i = *minor_out; i < *minor; i++) {
            _vol_minor *= dim_out[i];
          }
          _addr_in = (_s1 * *s1_step) * base_in[(*s1_ind)] +
                     (_s2 * *s2_step) * base_in[(*s2_ind)];
          _addr_out = _vol;
          for (i = *minor; i < dim_num; i++) {
            _addr = _vol / dim_in[i];
            _addr_in += (_vol - _addr * dim_in[i]) * base_in[i];
            _vol = _addr;
          }
          _vol = _addr_out;
          _addr_out = (_s1 * *s1_step) * base_out[(*s1_ond)] +
                      (_s2 * *s2_step) * base_out[(*s2_ond)];
          for (i = *minor; i < dim_num; i++) {
            _addr = _vol / dim_out[i];
            _addr_out += (_vol - _addr * dim_out[i]) * base_out[i];
            _vol = _addr;
          }
          if (_vol_out > TENS_TRANSP_TAB_SIZE ||
              _vol_minor > _vol_in * TENS_TRANSP_TAB_SIZE ||
              _vol_minor > _vol_out * TENS_TRANSP_TAB_SIZE) {
            //  Algorithm 0 (slower):
            //   Read the minor volume into the buffer from the input tensor
            //   block:
            _vol_minor /= _vol_in; // vol_in_c
            _s1 = 1 +
                  (_vol_in - 1) / warpSize; // number of warps (lines)
                                            // whichfully cover the input volume
            _s2 = blockDim_x / warpSize;    // number of whole warps in a thread
                                            // block (each warp treats one line)
            for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
              m = j / _s1;
              _addr = _addr_in;
              n = m; // n: Input column number (in_c)
              for (i = *minor_in; i < *minor; i++) {
                k = m / dim_in[i];
                _addr += (m - k * dim_in[i]) * base_in[i];
                m = k;
              }
              //    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset
              //    in the input volume
              m = threadIdx_x +
                  (j - n * _s1 - l) * warpSize; // elemental offset in the input
                                                // volume (alternative)
              if (m < _vol_in) {
                auto real_part = tens_in_real[_addr + m];
                auto imag_part = tens_in_real[_addr + m + _vol];
                buf0[n * _vol_in + m] = T{real_part, imag_part};
              }
            }
            item.barrier(cl::sycl::access::fence_space::local_space);
            //   Write the minor volume from the buffer into the output tensor
            //   block:
            _vol_minor = (_vol_minor * _vol_in) / _vol_out; // vol_out_c
            _s1 =
                1 + (_vol_out - 1) / warpSize; // number of warps (lines) which
                                               // fully cover the output volume
            for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
              n = j / _s1;
              _addr = _addr_out;
              _vol = n;
              _vol_in = 0; //_vol: Output column number (out_c)
              //    for(i=minor_out;i<minor;i++){m=n%dim_out[i]; n/=dim_out[i];
              //    _addr+=m*base_out[i]; _vol_in+=m*tmp0[pri[i]];}
              for (i = *minor_out; i < *minor; i++) {
                k = n / dim_out[i];
                m = n - k * dim_out[i];
                n = k;
                _addr += m * base_out[i];
                _vol_in += m * n2o[i];
              }
              //    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset
              //    in the output volume
              m = threadIdx_x + (j - (int)_vol * _s1 - l) *
                                    warpSize; // elemental offset in the output
                                              // volume (alternative)
              if (m < _vol_out) {
                _addr += m;
                //     for(i=0;i<minor_out;i++){_vol_in+=(m%dim_out[i])*tmp0[pri[i]];
                //     m/=dim_out[i];}
                for (i = 0; i < *minor_out; i++) {
                  k = m / dim_out[i];
                  _vol_in += (m - k * dim_out[i]) * n2o[i];
                  m = k;
                }
                tens_out[_addr] = buf0[_vol_in];
              }
            }
            item.barrier(cl::sycl::access::fence_space::local_space);
          } else {
            //  Algorithm 1 (presumably faster):
            //   Create per-block look-up tables:
            m = _vol_minor / _vol_in; // vol_in_c
            for (j = threadIdx_x; j < m;
                 j += blockDim_x) { // column number (input)
              _addr = 0;
              _s1 = j;
              //    for(i=minor_in;i<minor;i++){_addr+=(_s1%dim_in[i])*base_in[i];
              //    _s1/=dim_in[i];}
              for (i = *minor_in; i < *minor; i++) {
                _s2 = _s1 / dim_in[i];
                _addr += (_s1 - _s2 * dim_in[i]) * base_in[i];
                _s1 = _s2;
              }
              ftb[j] = _addr;
            }
            m = _vol_minor / _vol_out; // vol_out_c
            for (j = threadIdx_x; j < m;
                 j += blockDim_x) { // column number (output)
              _addr = 0;
              _s1 = j;
              //    for(i=minor_out;i<minor;i++){_addr+=(_s1%dim_out[i])*base_out[i];
              //    _s1/=dim_out[i];}
              for (i = *minor_out; i < *minor; i++) {
                _s2 = _s1 / dim_out[i];
                _addr += (_s1 - _s2 * dim_out[i]) * base_out[i];
                _s1 = _s2;
              }
              gtb[j] = _addr;
            }
            for (j = threadIdx_x; j < m;
                 j += blockDim_x) { // column number (output)
              n = 0;
              _s1 = j;
              //    for(i=minor_out;i<minor;i++){n+=(_s1%dim_out[i])*n2o[i];
              //    _s1/=dim_out[i];}
              for (i = *minor_out; i < *minor; i++) {
                _s2 = _s1 / dim_out[i];
                n += (_s1 - _s2 * dim_out[i]) * n2o[i];
                _s1 = _s2;
              }
              htb[j] = n;
            }
            for (j = threadIdx_x; j < _vol_out; j += blockDim_x) {
              n = 0;
              _s1 = j;
              //    for(i=0;i<minor_out;i++){n+=(_s1%dim_out[i])*n2o[i];
              //    _s1/=dim_out[i];}
              for (i = 0; i < *minor_out; i++) {
                _s2 = _s1 / dim_out[i];
                n += (_s1 - _s2 * dim_out[i]) * n2o[i];
                _s1 = _s2;
              }
              stb[j] = n;
            }
            item.barrier(cl::sycl::access::fence_space::local_space);
            //   Read the minor volume into the buffer from the input tensor
            //   block:
            _vol_minor /= _vol_in;              // vol_in_c
            _s1 = 1 + (_vol_in - 1) / warpSize; // number of warps (lines) which
                                                // fully cover the input volume
            _s2 = blockDim_x / warpSize; // number of whole warps in a thread
                                         // block (each warp treats one line)
            for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
              m = j / _s1;
              n = threadIdx_x + (j - m * _s1 - l) *
                                    warpSize; // m: Input column number (in_c);
                                              // n: Offset in the column
              if (n < _vol_in) {
                _addr = _addr_in + ftb[m] + n;
                auto real_part = tens_in_real[_addr];
                auto imag_part = tens_in_real[_addr + _vol];
                buf0[m * _vol_in + n] = T{real_part, imag_part};
              }
            }
            item.barrier(cl::sycl::access::fence_space::local_space);
            //   Write the minor volume from the buffer into the output tensor
            //   block:
            _vol_minor = (_vol_minor * _vol_in) / _vol_out; // vol_out_c
            _s1 =
                1 + (_vol_out - 1) / warpSize; // number of warps (lines) which
                                               // fully cover the output volume
            for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
              m = j / _s1;
              n = threadIdx_x +
                  (j - m * _s1 - l) * warpSize; // m: Output column number
              // (out_c); n: Offset in the column
              if (n < _vol_out) {
                _addr = _addr_out + gtb[m] + n;
                _vol_in = htb[m] + stb[n];
                tens_out[_addr] = buf0[_vol_in];
              }
            }
            item.barrier(cl::sycl::access::fence_space::local_space);
          }
        } // enddo _work_piece: independent work distribution among thread
          // blocks
      }
  }
  // Record errors if occured (for each block):
  if (threadIdx_x == 0) {
    if (*err_code != 0)
      i = atomic_ref<int>(gpu_error_count).fetch_add(1);
  }
  return;
}
//--------------------------------------------------------------------------------------------------------------------
// TENSOR TRANSPOSE (shared-memory version):
template <typename T>
void gpu_tensor_block_copy_cmplx_split_out_dlf__(
    int dmo, int drc, int dim_num, int const_args_pos,
    const T *__restrict__ tens_in, T *__restrict__ tens_out,
    sycl::nd_item<3> item,
    dpct::accessor<int, dpct::device, 2> const_args_dims,
    dpct::accessor<int, dpct::device, 2> const_args_prmn, int *gpu_error_count,
    T *buf0, float *val, size_t *base_in, size_t *base_out, size_t *ftb,
    size_t *gtb, int *htb, int *stb, int *dim_in, int *dim_out, int *o2n,
    int *n2o, int *pri, int *tmp0, int *err_code, int *minor, int *minor_in,
    int *minor_out, int *s1_ind, int *s1_ond, int *s1_step, int *s1_dim,
    int *s2_ind, int *s2_ond, int *s2_step, int *s2_dim, int *ns1, int *ns2,
    size_t *vol, size_t *vol_ext)
/**
Shared-memory version of tensor transpose: tens_out=TRN(tens_in): Complex
arguments only: INPUT: # dmo - dimension extents order (0: normal, as it is in
<const_args>; not 0: permuted dimension order will be imposed); # drc - index
permutation direction (0: normal, as it is in <const_args>; not 0: inversed
permutation will be used); # dim_num - tensor block rank; # const_args_pos -
entry in the __constant__ memory bank where tensor block dimension extents
(const_args_dims) and index permutation (const_args_prmn) are stored; #
tens_in[0:] - complex input tensor in normal representation; OUTPUT: #
tens_out[0:] - complex output (transposed) tensor in split representation;
NOTES:
# Minimal CUDA execution configuration is <<<1,warpSize>>>
# Number of threads per block must be multiple of the warpSize!
**/
{
    size_t threadIdx_x = item.get_local_id(2);
    size_t blockDim_x = item.get_local_range(2);
    size_t gridDim_x = item.get_group_range(2);
    size_t blockIdx_x = item.get_group(2);

    size_t _vol, _addr_in, _addr_out, _addr, _work_piece;
    int i, j, k, l, m, n, _vol_minor, _vol_in, _vol_out, _s1, _s2;
    /*
      SHARED MEMORY USE (bytes) =
      + TENS_TRANSP_BUF_SIZE*sizeof(T)
      + MAX_TENSOR_RANK*(8+8+4+4+4+4+4+4)
      + TENS_TRANSP_TAB_SIZE*(8+8+4+4)
      + 4*15 + 8*2
      MIN REGISTER USE (bytes) per thread =
      + 4*4 + 4*11 + 8*5 = 100
    */

    static_assert(ComplexType<T>::valid, "Non-complex types are not allowed!");
    typename ComplexType<T>::RealType *tens_out_real =
        (typename ComplexType<T>::RealType *)tens_out;
    // Determine the minor index set (only the master thread in each thread
    // block):
    if (threadIdx_x == 0) {
      *err_code = 0;
      if (dim_num >= 0 && dim_num <= MAX_TENSOR_RANK &&
          blockDim_x >= warpSize && blockDim_x % warpSize == 0) {
        *s1_ind = dim_num + 1;
        *s2_ind = dim_num - 1;
        _vol = 1;
        for (i = 0; i < dim_num; i++) {
          _vol *= const_args_dims[const_args_pos][i];
          if (const_args_prmn[const_args_pos][i] != i + 1)
            *s1_ind = 0;
        };
        *vol = _vol; // total volume
        // (number of tensor elements)
        if (*s1_ind == 0) { // non-trivial permutation
          // Set input/output permutations and dimension extents:
          if (drc == 0) { // normal index permutation
            for (i = 0; i < dim_num; i++)
              o2n[i] = const_args_prmn[const_args_pos][i] - 1;
            for (i = 0; i < dim_num; i++)
              n2o[o2n[i]] = i;
          } else { // inversed index permutation
            for (i = 0; i < dim_num; i++)
              n2o[i] = const_args_prmn[const_args_pos][i] - 1;
            for (i = 0; i < dim_num; i++)
              o2n[n2o[i]] = i;
          }
          if (dmo == 0) { // normal dimension order
            for (i = 0; i < dim_num; i++)
              dim_in[i] = const_args_dims[const_args_pos][i];
            for (i = 0; i < dim_num; i++)
              dim_out[o2n[i]] = dim_in[i];
          } else { // inversed dimension order
            for (i = 0; i < dim_num; i++)
              dim_out[i] = const_args_dims[const_args_pos][i];
            for (i = 0; i < dim_num; i++)
              dim_in[n2o[i]] = dim_out[i];
          }
          *s1_step = dim_in[(*s1_ind)];
          *s2_step = dim_in[(*s2_ind)];
          if (_vol > TENS_TRANSP_BUF_SIZE) { // tensor block does not fit into
                                             // the shared memory buffer
            // Determine the input/output minor index sets and the combined
            // minor index set:
            l = (int)(cl::sycl::sqrt((float)TENS_TRANSP_BUF_SIZE));
            *minor_in = 0;
            _vol_in = 1;
            for (i = 0; i < dim_num; i++) {
              j = _vol_in * dim_in[i];
              if (j > l)
                break;
              (*minor_in)++;
              _vol_in = j;
            }
            *minor_out = 0;
            _vol_out = 1;
            for (i = 0; i < dim_num; i++) {
              j = _vol_out * dim_out[i];
              if (j > l)
                break;
              (*minor_out)++;
              _vol_out = j;
            }
            *minor = *minor_in;
            _vol_minor = _vol_in;
            for (i = 0; i < *minor_out; i++) {
              if (n2o[i] >= *minor_in) {
                (*minor)++;
                _vol_minor *= dim_out[i];
              }
            }
            m = 1;
            _s1 = 0;
            _s2 = 0;
            while (_vol_minor < TENS_TRANSP_BUF_SIZE && m != 0) {
              m = 0;
              if (_s1 == 0) {
                for (i = *minor_in; i < dim_num; i++) {
                  if (o2n[i] < *minor_out) {
                    (*minor_in)++;
                    _vol_in *= dim_in[i];
                  } else {
                    break;
                  }
                }
              }
              if (_s2 == 0) {
                for (i = *minor_out; i < dim_num; i++) {
                  if (n2o[i] < *minor_in) {
                    (*minor_out)++;
                    _vol_out *= dim_out[i];
                  } else {
                    break;
                  }
                }
              }
              j = dim_in[(*minor_in)];
              l = dim_out[(*minor_out)];
              if (*minor_in == n2o[(*minor_out)] &&
                  _s1 + _s2 == 0) { // same candidate index to both the input
                                    // and output index sets
                if (j > 1 && TENS_TRANSP_BUF_SIZE < _vol_minor * 2)
                  break;
                if (_vol_minor * j > TENS_TRANSP_BUF_SIZE) {
                  *s1_ind = *minor_in;
                  *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                  _s1++;
                  _s2++;
                }
                (*minor_in)++;
                _vol_in *= j;
                (*minor_out)++;
                _vol_out *= j;
                (*minor)++;
                _vol_minor *= j;
                m++;
              } else { // the input and output index sets consider two different
                       // candidates
                if (_vol_minor * j * l <= TENS_TRANSP_BUF_SIZE &&
                    _s1 + _s2 == 0) { // accept both, no splitting
                  (*minor_in)++;
                  _vol_in *= j;
                  (*minor_out)++;
                  _vol_out *= l;
                  *minor += 2;
                  _vol_minor *= (j * l);
                  m++;
                } else { // try to accept either one of the two OR both with
                         // splitting
                  if (j == 1 || l == 1) {
                    if (j == 1 && _s1 == 0) {
                      (*minor_in)++;
                      (*minor)++;
                      m++;
                    }
                    if (l == 1 && _s2 == 0) {
                      (*minor_out)++;
                      (*minor)++;
                      m++;
                    }
                  } else {
                    if (_vol_minor * j <= TENS_TRANSP_BUF_SIZE &&
                        _vol_minor * l > TENS_TRANSP_BUF_SIZE &&
                        _vol_out >= warpSize &&
                        _s1 == 0) { // accept the input index, no splitting
                      (*minor_in)++;
                      _vol_in *= j;
                      (*minor)++;
                      _vol_minor *= j;
                      m++;
                    } else if (_vol_minor * j > TENS_TRANSP_BUF_SIZE &&
                               _vol_minor * l <= TENS_TRANSP_BUF_SIZE &&
                               _vol_in >= warpSize &&
                               _s2 ==
                                   0) { // accept the output index, no splitting
                      (*minor_out)++;
                      _vol_out *= l;
                      (*minor)++;
                      _vol_minor *= l;
                      m++;
                    } else { // splitting is unavoidable (both OR one OR none)
                      if (TENS_TRANSP_BUF_SIZE >= _vol_minor * 2) {
                        if (j >= 4 && l >= 4) { // dimension extents are large
                                                // enough to be split
                          if (_vol_minor * 4 >
                              TENS_TRANSP_BUF_SIZE) { // impossible to split
                                                      // both indices
                            if (_vol_in <= _vol_out &&
                                _s1 == 0) { // split the input candidate index
                              *s1_ind = *minor_in;
                              *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                              (*minor_in)++;
                              _vol_in *= j;
                              (*minor)++;
                              _vol_minor *= j;
                              _s1++;
                              m++;
                            } else { // split the output candidate index
                              if (_s2 == 0) {
                                *s1_ind = n2o[(*minor_out)];
                                *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                                (*minor_out)++;
                                _vol_out *= l;
                                (*minor)++;
                                _vol_minor *= l;
                                _s2++;
                                m++;
                              }
                            }
                          } else { // possible to split both indices
                            i = (int)cl::sycl::sqrt(
                                ((float)TENS_TRANSP_BUF_SIZE) /
                                (float)_vol_minor);
                            if (i < 2)
                              i = 2; // uniform splitting
                            *s1_step = i;
                            *s2_step = i;
                            *val = (float)_vol_out / (float)_vol_in;
                            if (*val < 1.0f) { // scale the initial uniform
                                               // splitting to
                              // reflect the disbalance between _vol_in and
                              // _vol_out
                              if (*val * (float)i < 1.0f)
                                *val = 1.0f / (float)i;
                              if (*val * (float)l < (float)i)
                                *val = (float)i / (float)l;
                            } else {
                              if (*val * (float)i > (float)j)
                                *val = (float)j / (float)i;
                              if (*val > float(i))
                                *val = (float)i;
                            }
                            *s1_step = (int)(((float)i) * *val);
                            *s2_step = (int)(((float)i) / *val);
                            if (*s1_step >= 2 &&
                                _s1 == 0) { //&& s1_step <= dim_in[minor_in]
                              *s1_ind = *minor_in;
                              (*minor_in)++;
                              _vol_in *= j;
                              (*minor)++;
                              _vol_minor *= j;
                              _s1++;
                              m++;
                            } else {
                              *s1_step = dim_in[(*s1_ind)];
                            }
                            if (*s2_step >= 2 &&
                                _s2 == 0) { //&& s2_step <= dim_out[minor_out]
                              *s2_ind = n2o[(*minor_out)];
                              (*minor_out)++;
                              _vol_out *= l;
                              (*minor)++;
                              _vol_minor *= l;
                              _s2++;
                              m++;
                            } else {
                              *s2_step = dim_in[(*s2_ind)];
                            }
                          }
                        } else if (j >= 4 && l < 4 &&
                                   _s1 ==
                                       0) { // split the input candidate index
                          *s1_ind = *minor_in;
                          *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                          (*minor_in)++;
                          _vol_in *= j;
                          (*minor)++;
                          _vol_minor *= j;
                          _s1++;
                          m++;
                        } else if (j < 4 && l >= 4 &&
                                   _s2 ==
                                       0) { // split the output candidate index
                          *s1_ind = n2o[(*minor_out)];
                          *s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                          (*minor_out)++;
                          _vol_out *= l;
                          (*minor)++;
                          _vol_minor *= l;
                          _s2++;
                          m++;
                        } else { // both candidate indices have too small extent
                                 // to be split: try to add one of them fully
                          if (_vol_minor * j <= TENS_TRANSP_BUF_SIZE &&
                              _s1 == 0) {
                            (*minor_in)++;
                            _vol_in *= j;
                            (*minor)++;
                            _vol_minor *= j;
                            m++;
                          } else if (_vol_minor * l <= TENS_TRANSP_BUF_SIZE &&
                                     _s2 == 0) {
                            (*minor_out)++;
                            _vol_out *= l;
                            (*minor)++;
                            _vol_minor *= l;
                            m++;
                          }
                        }
                      } else { // unable to add more indices in the minor set
                        break;
                      }
                    }
                  }
                }
              }
            }
            if (*s1_ind == dim_num - 1 && *s2_ind == dim_num - 1) {
              *s2_ind = 0;
              *s2_step = dim_in[0];
            }      // s1_ind was set while s2_ind was not
          } else { // tensor block fits into the shared memory buffer from the
                   // beginning
            *minor = dim_num;
            *minor_in = dim_num;
            *minor_out = dim_num;
            _vol_minor = _vol;
            _vol_in = _vol;
            _vol_out = _vol;
          }
          // Share the tensor transpose configuration with other threads in each
          // block:
          *vol_ext = _vol / _vol_minor;
          *s1_dim = dim_in[(*s1_ind)];
          *s2_dim = dim_in[(*s2_ind)];
          // Set indexing bases (OUT:{out,in_c,ext_in}_new;
          // IN:{in,out_c,ext_in}_old):
          //  OUTPUT indexing (dim_out[], base_out[]: prioritized new
          //  numeration):
          for (i = 0; i < dim_num; i++) {
            tmp0[i] = dim_out[i];
          } // save output dimension extents (new numeration)
          j = 0;
          for (i = 0; i < *minor_out; i++) {
            pri[j++] = i;
          } // output minor index set (new numeration))
          for (i = 0; i < dim_num; i++) {
            if (o2n[i] >= *minor_out)
              pri[j++] = o2n[i];
          } //{compl.input minor + external} index set (new numeration)
          j = 1;
          for (i = 0; i < dim_num; i++) {
            dim_out[i] = j;
            j *= tmp0[i];
          } // output bases (new numeration)
          for (i = 0; i < dim_num; i++) {
            base_out[i] = dim_out[pri[i]];
          } // output bases (prioritized new numeration)
          for (i = 0; i < dim_num; i++) {
            dim_out[i] = tmp0[pri[i]];
          } // output extents (prioritized new numeration)
          for (i = 0; i < dim_num; i++) {
            if (n2o[pri[i]] == *s1_ind) {
              *s1_ond = i;
            } else if (n2o[pri[i]] == *s2_ind) {
              *s2_ond = i;
            }
          } // split indices (prioritized new numeration)
          //  INPUT indexing (dim_in[], base_in[]: prioritized old numeration):
          for (i = 0; i < dim_num; i++) {
            tmp0[i] = dim_in[i];
          } // save input dimension extents (old numeration)
          j = 0;
          for (i = 0; i < *minor_in; i++) {
            pri[j++] = i;
          } // input minor index set (old numeration)
          for (i = 0; i < *minor_out; i++) {
            if (n2o[i] >= *minor_in)
              pri[j++] = n2o[i];
          } // compl.output minor idex set (old numeration)
          for (i = j; i < dim_num; i++) {
            pri[i] = n2o[pri[i]];
          } // external index set (just convert new numbers to old ones for
            // consistency)
          j = 1;
          for (i = 0; i < dim_num; i++) {
            dim_in[i] = j;
            j *= tmp0[i];
          } // input bases (old numeration)
          for (i = 0; i < dim_num; i++) {
            base_in[i] = dim_in[pri[i]];
          } // input bases (prioritized old numeration)
          for (i = 0; i < dim_num; i++) {
            dim_in[i] = tmp0[pri[i]];
          } // input extents (prioritized old numeration)
          for (i = 0; i < dim_num; i++) {
            if (pri[i] == *s1_ind) {
              _s1 = i;
            } else if (pri[i] == *s2_ind) {
              _s2 = i;
            }
          } // split indices (prioritized old numeration)
          *s1_ind = _s1;
          *s2_ind = _s2;
          *ns1 =
              1 +
              (*s1_dim - 1) /
                  *s1_step; // number of segments from the 1st split minor index
          *ns2 =
              1 +
              (*s2_dim - 1) /
                  *s2_step; // number of segments from the 2nd split minor index
          //  Index position correspondence for the minor index set (pri-new -->
          //  pri-old):
          j = 0;
          for (i = 0; i < *minor_out; i++) {
            if (n2o[i] < *minor_in) {
              pri[i] = n2o[i];
            } else {
              pri[i] = (*minor_in + j);
              j++;
            }
          }
          j = 0;
          for (i = 0; i < *minor_in; i++) {
            if (o2n[i] < *minor_out) {
              pri[o2n[i]] = i;
            } else {
              pri[*minor_out + j] = i;
              j++;
            }
          }
          // Check tensor transpose configuration parameters:
          if (*minor <= 0 || *minor_in <= 0 || *minor_out <= 0 || _vol <= 0 ||
              _vol_minor <= 0)
            *err_code += 5000; // trap
          if (*s1_ind >= dim_num || *s2_ind >= dim_num || *s1_ond >= dim_num ||
              *s2_ond >= dim_num || *s1_ind == *s2_ind || *s1_ond == *s2_ond ||
              *s1_step <= 0 || *s2_step <= 0)
            *err_code += 1000; // trap
          if ((*s1_step != dim_in[(*s1_ind)] && *s1_ind != *minor_in - 1 &&
               *s1_ond != *minor_out - 1) ||
              (*s2_step != dim_in[(*s2_ind)] && *s2_ind != *minor_in - 1 &&
               *s2_ond != *minor_out - 1))
            *err_code += 500; // trap
          if ((_vol_minor * *s1_step * *s2_step) / (*s1_dim * *s2_dim) >
              TENS_TRANSP_BUF_SIZE)
            *err_code += 100; // trap
        }                     // endif: non-trivial permutation
      } else {
        *err_code = 1 + 2 * blockDim_x % warpSize;
      }
    } // endif: Master thread.
    item.barrier(cl::sycl::access::fence_space::local_space);

    // Proceed:
    if (*err_code == 0) {
      if (*s1_ind > dim_num) { // tag of a trivial permutation
                               // Direct copy:
        _vol = *vol;
        j = item.get_global_range(2);
        i = item.get_global_id(2);
        _addr_in = _vol - _vol % j;
        for (_addr = 0; _addr < _addr_in; _addr += j) {
          _addr_out = _addr + i;
          auto cmplx_val = tens_in[_addr_out];
          tens_out_real[_addr_out] = talshComplexReal(cmplx_val);
          tens_out_real[_addr_out + _vol] = talshComplexImag(cmplx_val);
        }
        _addr_out = _addr_in + i;
        if (_addr_out < _vol) {
          auto cmplx_val = tens_in[_addr_out];
          tens_out_real[_addr_out] = talshComplexReal(cmplx_val);
          tens_out_real[_addr_out + _vol] = talshComplexImag(cmplx_val);
        }
      } else {                      // non-trivial permutation
        l = threadIdx_x / warpSize; // l: warp number
        // Distribute work accross CUDA blocks (external multi-index +
        // splitting):
        for (_work_piece = blockIdx_x; _work_piece < *vol_ext * *ns1 * *ns2;
             _work_piece += gridDim_x) { //(ns1*ns2*vol_ext) is the total number
                                         //of independent tasks
          _addr = _work_piece;
          _addr /= *vol_ext;
          _vol = _work_piece - _addr * *vol_ext;
          _s2 = (int)(_addr / *ns1);
          _s1 = (int)(_addr - _s2 * *ns1); //{_addr_ext,_s1,_s2} --> tensor
                                           // subblock (CUDA block)
          //  Modify dimension extents due to possible dimension splitting:
          if (threadIdx_x == 0) {
            if (_s1 + 1 == *ns1) { // last segment of the 1st split index
              j = *s1_dim - _s1 * *s1_step;
              dim_in[(*s1_ind)] = j;
              dim_out[(*s1_ond)] = j;
            } else { // internal segment of the 1st split index
              dim_in[(*s1_ind)] = *s1_step;
              dim_out[(*s1_ond)] = *s1_step;
            }
            if (_s2 + 1 == *ns2) { // last segment of the 2nd split index
              j = *s2_dim - _s2 * *s2_step;
              dim_in[(*s2_ind)] = j;
              dim_out[(*s2_ond)] = j;
            } else { // internal segment of the 2nd split index
              dim_in[(*s2_ind)] = *s2_step;
              dim_out[(*s2_ond)] = *s2_step;
            }
            j = 1;
            for (i = 0; i < *minor; i++) {
              tmp0[i] = j;
              j *= dim_in[i];
            } // minor buffer bases (pri-old)
            for (i = 0; i < *minor; i++)
              n2o[i] = tmp0[pri[i]]; // look up table to accelerate further
                                     // accesses to tmp0[]
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
          //  Mount input/output volumes and bases:
          _vol_in = dim_in[0];
          for (i = 1; i < *minor_in; i++) {
            _vol_in *= dim_in[i];
          }
          _vol_out = dim_out[0];
          for (i = 1; i < *minor_out; i++) {
            _vol_out *= dim_out[i];
          }
          _vol_minor = _vol_out;
          for (i = *minor_out; i < *minor; i++) {
            _vol_minor *= dim_out[i];
          }
          _addr_in = (_s1 * *s1_step) * base_in[(*s1_ind)] +
                     (_s2 * *s2_step) * base_in[(*s2_ind)];
          _addr_out = _vol;
          for (i = *minor; i < dim_num; i++) {
            _addr = _vol / dim_in[i];
            _addr_in += (_vol - _addr * dim_in[i]) * base_in[i];
            _vol = _addr;
          }
          _vol = _addr_out;
          _addr_out = (_s1 * *s1_step) * base_out[(*s1_ond)] +
                      (_s2 * *s2_step) * base_out[(*s2_ond)];
          for (i = *minor; i < dim_num; i++) {
            _addr = _vol / dim_out[i];
            _addr_out += (_vol - _addr * dim_out[i]) * base_out[i];
            _vol = _addr;
          }
          if (_vol_out > TENS_TRANSP_TAB_SIZE ||
              _vol_minor > _vol_in * TENS_TRANSP_TAB_SIZE ||
              _vol_minor > _vol_out * TENS_TRANSP_TAB_SIZE) {
            //  Algorithm 0 (slower):
            //   Read the minor volume into the buffer from the input tensor
            //   block:
            _vol_minor /= _vol_in;              // vol_in_c
            _s1 = 1 + (_vol_in - 1) / warpSize; // number of warps (lines) which
                                                // fully cover the input volume
            _s2 = blockDim_x / warpSize; // number of whole warps in a thread
                                         // block (each warp treats one line)
            for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
              m = j / _s1;
              _addr = _addr_in;
              n = m; // n: Input column number (in_c)
              for (i = *minor_in; i < *minor; i++) {
                k = m / dim_in[i];
                _addr += (m - k * dim_in[i]) * base_in[i];
                m = k;
              }
              //    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset
              //    in the input volume
              m = threadIdx_x +
                  (j - n * _s1 - l) * warpSize; // elemental offset in the input
                                                // volume (alternative)
              if (m < _vol_in)
                buf0[n * _vol_in + m] = tens_in[_addr + m];
            }
            item.barrier(cl::sycl::access::fence_space::local_space);
            //   Write the minor volume from the buffer into the output tensor
            //   block:
            _vol_minor = (_vol_minor * _vol_in) / _vol_out; // vol_out_c
            _s1 =
                1 + (_vol_out - 1) / warpSize; // number of warps (lines) which
                                               // fully cover the output volume
            for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
              n = j / _s1;
              _addr = _addr_out;
              _vol = n;
              _vol_in = 0; //_vol: Output column number (out_c)
              //    for(i=minor_out;i<minor;i++){m=n%dim_out[i]; n/=dim_out[i];
              //    _addr+=m*base_out[i]; _vol_in+=m*tmp0[pri[i]];}
              for (i = *minor_out; i < *minor; i++) {
                k = n / dim_out[i];
                m = n - k * dim_out[i];
                n = k;
                _addr += m * base_out[i];
                _vol_in += m * n2o[i];
              }
              //    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset
              //    in the output volume
              m = threadIdx_x + (j - (int)_vol * _s1 - l) *
                                    warpSize; // elemental offset in the output
                                              // volume (alternative)
              if (m < _vol_out) {
                _addr += m;
                //     for(i=0;i<minor_out;i++){_vol_in+=(m%dim_out[i])*tmp0[pri[i]];
                //     m/=dim_out[i];}
                for (i = 0; i < *minor_out; i++) {
                  k = m / dim_out[i];
                  _vol_in += (m - k * dim_out[i]) * n2o[i];
                  m = k;
                }
                auto cmplx_val = buf0[_vol_in];
                tens_out_real[_addr] = talshComplexReal(cmplx_val);
                tens_out_real[_addr + _vol] = talshComplexImag(cmplx_val);
              }
            }
            item.barrier(cl::sycl::access::fence_space::local_space);
          } else {
            //  Algorithm 1 (presumably faster):
            //   Create per-block look-up tables:
            m = _vol_minor / _vol_in; // vol_in_c
            for (j = threadIdx_x; j < m;
                 j += blockDim_x) { // column number (input)
              _addr = 0;
              _s1 = j;
              //    for(i=minor_in;i<minor;i++){_addr+=(_s1%dim_in[i])*base_in[i];
              //    _s1/=dim_in[i];}
              for (i = *minor_in; i < *minor; i++) {
                _s2 = _s1 / dim_in[i];
                _addr += (_s1 - _s2 * dim_in[i]) * base_in[i];
                _s1 = _s2;
              }
              ftb[j] = _addr;
            }
            m = _vol_minor / _vol_out; // vol_out_c
            for (j = threadIdx_x; j < m;
                 j += blockDim_x) { // column number (output)
              _addr = 0;
              _s1 = j;
              //    for(i=minor_out;i<minor;i++){_addr+=(_s1%dim_out[i])*base_out[i];
              //    _s1/=dim_out[i];}
              for (i = *minor_out; i < *minor; i++) {
                _s2 = _s1 / dim_out[i];
                _addr += (_s1 - _s2 * dim_out[i]) * base_out[i];
                _s1 = _s2;
              }
              gtb[j] = _addr;
            }
            for (j = threadIdx_x; j < m;
                 j += blockDim_x) { // column number (output)
              n = 0;
              _s1 = j;
              //    for(i=minor_out;i<minor;i++){n+=(_s1%dim_out[i])*n2o[i];
              //    _s1/=dim_out[i];}
              for (i = *minor_out; i < *minor; i++) {
                _s2 = _s1 / dim_out[i];
                n += (_s1 - _s2 * dim_out[i]) * n2o[i];
                _s1 = _s2;
              }
              htb[j] = n;
            }
            for (j = threadIdx_x; j < _vol_out; j += blockDim_x) {
              n = 0;
              _s1 = j;
              //    for(i=0;i<minor_out;i++){n+=(_s1%dim_out[i])*n2o[i];
              //    _s1/=dim_out[i];}
              for (i = 0; i < *minor_out; i++) {
                _s2 = _s1 / dim_out[i];
                n += (_s1 - _s2 * dim_out[i]) * n2o[i];
                _s1 = _s2;
              }
              stb[j] = n;
            }
            item.barrier(cl::sycl::access::fence_space::local_space);
            //   Read the minor volume into the buffer from the input tensor
            //   block:
            _vol_minor /= _vol_in;              // vol_in_c
            _s1 = 1 + (_vol_in - 1) / warpSize; // number of warps (lines) which
                                                // fully cover the input volume
            _s2 = blockDim_x / warpSize; // number of whole warps in a thread
                                         // block (each warp treats one line)
            for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
              m = j / _s1;
              n = threadIdx_x + (j - m * _s1 - l) *
                                    warpSize; // m: Input column number (in_c);
                                              // n: Offset in the column
              if (n < _vol_in) {
                _addr = _addr_in + ftb[m] + n;
                buf0[m * _vol_in + n] = tens_in[_addr];
              }
            }
            item.barrier(cl::sycl::access::fence_space::local_space);
            //   Write the minor volume from the buffer into the output tensor
            //   block:
            _vol_minor = (_vol_minor * _vol_in) / _vol_out; // vol_out_c
            _s1 =
                1 + (_vol_out - 1) / warpSize; // number of warps (lines) which
                                               // fully cover the output volume
            for (j = l; j < _s1 * _vol_minor; j += _s2) { // j: Line number
              m = j / _s1;
              n = threadIdx_x +
                  (j - m * _s1 - l) *
                      warpSize; // m: Output column number
                                // (out_c); n: Offset in the column
              if (n < _vol_out) {
                _addr = _addr_out + gtb[m] + n;
                _vol_in = htb[m] + stb[n];
                auto cmplx_val = buf0[_vol_in];
                tens_out_real[_addr] = talshComplexReal(cmplx_val);
                tens_out_real[_addr + _vol] = talshComplexImag(cmplx_val);
              }
            }
            item.barrier(cl::sycl::access::fence_space::local_space);
          }
        } // enddo _work_piece: independent work distribution among thread
          // blocks
      }
  }
  // Record errors if occured (for each block):
  if (threadIdx_x == 0) {
    if (*err_code != 0)
      i = atomic_ref<int>(gpu_error_count).fetch_add(1);
  }
  return;
}
//------------------------------------------------------------------------------------------------------------
// TENSOR TRANSPOSE (naive scatter version):
template <typename T>
void gpu_tensor_block_copy_scatter_dlf__(
    int dmo, int drc, int dim_num, int const_args_pos,
    const T *__restrict__ tens_in, T *__restrict__ tens_out,
    cl::sycl::nd_item<1>& item,
    constant_accessor<int, 2>& const_args_dims,
    constant_accessor<int, 2>& const_args_prmn, int *gpu_error_count,
    int *n2o, size_t *vol, size_t *base_in, size_t *base_out)
/**
   Scattering version of tensor transpose: tens_out=TRN(tens_in):
   INPUT:
   # dmo - dimension extents order (0: normal, as it is in <const_args>; not 0:
permuted dimension order will be imposed); # drc - index permutation direction
(0: normal, as it is in <const_args>; not 0: inversed permutation will be used);
   # dim_num - tensor block rank;
   # const_args_pos - entry in the __constant__ memory bank where tensor block
dimension extents (const_args_dims) and index permutation (const_args_prmn) are
stored; # tens_in[0:] - input tensor; OUTPUT: # tens_out[0:] - output
(transposed) tensor;
**/
{
    size_t threadIdx_x = item.get_local_id(0);
    size_t blockDim_x = item.get_local_range(0);
    size_t gridDim_x = item.get_group_range(0);
    size_t blockIdx_x = item.get_group(0);
    size_t globalIdx_x = item.get_global_id(0);
    size_t globalDim_x = item.get_global_range(0);

    int i, j, k;
    size_t _vol, _addr_in, _addr_out, _si;

    if (dim_num == 0) {
      if (blockIdx_x == 0 && threadIdx_x == 0)
        tens_out[0] = tens_in[0];
    } else if (dim_num == 1) {
      _vol = const_args_dims[const_args_pos][0];
      j = globalIdx_x;
      for (_addr_in = j; _addr_in < _vol; _addr_in += globalDim_x) {
        tens_out[_addr_in] = tens_in[_addr_in];
      }
    } else if (dim_num > 1) {

      if (threadIdx_x == 0) {
        k = 0;
        for (i = 0; i < dim_num; i++) {
          j = const_args_prmn[const_args_pos][i] - 1;
          n2o[j] = i;
          if (j != i)
            k = 1;
        }
        if (k == 0) {       // trivial permutation
          n2o[0] = dim_num; // trivial permutation flag
          _vol = 1;
          for (i = 0; i < dim_num; i++) {
            _vol *= const_args_dims[const_args_pos][i];
          }
          *vol = _vol;
        } else {          // non-trivial permutation
          if (dmo == 0) { // normal dimension order
            _vol = 1;
            for (i = 0; i < dim_num; i++) {
              base_in[i] = _vol;
              _vol *= const_args_dims[const_args_pos][i];
            }
            *vol = _vol;
            if (drc == 0) { // normal index permutation
              _vol = 1;
              for (i = 0; i < dim_num; i++) {
                k = n2o[i];
                base_out[k] = _vol;
                _vol *= const_args_dims[const_args_pos][k];
              }
            } else { // inversed index permutation
              _vol = 1;
              for (i = 0; i < dim_num; i++) {
                k = const_args_prmn[const_args_pos][i] - 1;
                base_out[k] = _vol;
                _vol *= const_args_dims[const_args_pos][k];
              }
            }
          } else {          // inversed dimension order
            if (drc == 0) { // normal index permutation
              _vol = 1;
              for (i = 0; i < dim_num; i++) {
                k = const_args_prmn[const_args_pos][i] - 1;
                base_in[i] = _vol;
                _vol *= const_args_dims[const_args_pos][k];
              };
              *vol = _vol;
              _vol = 1;
              for (i = 0; i < dim_num; i++) {
                k = n2o[i];
                base_out[k] = _vol;
                _vol *= const_args_dims[const_args_pos][i];
              }
            } else { // inversed index permutation
              _vol = 1;
              for (i = 0; i < dim_num; i++) {
                k = n2o[i];
                base_in[i] = _vol;
                _vol *= const_args_dims[const_args_pos][k];
              };
              *vol = _vol;
              _vol = 1;
              for (i = 0; i < dim_num; i++) {
                k = const_args_prmn[const_args_pos][i] - 1;
                base_out[k] = _vol;
                _vol *= const_args_dims[const_args_pos][i];
              }
            }
          }
        }
      }
      item.barrier(cl::sycl::access::fence_space::local_space);

      _vol = *vol;
      if (n2o[0] >= dim_num) { // trivial permutation
        k = globalDim_x;
        j = globalIdx_x;
        for (_addr_in = j; _addr_in < _vol; _addr_in += k) {
          tens_out[_addr_in] = tens_in[_addr_in];
        }
      } else { // non-trivial permutation
        j = globalIdx_x;
        for (_addr_in = j; _addr_in < _vol; _addr_in += globalDim_x) {
          _addr_out = 0;
          _si = _addr_in;
          for (i = dim_num - 1; i >= 0; i--) {
            _addr_out += (_si / base_in[i]) * base_out[i];
            _si %= base_in[i];
          }
          tens_out[_addr_out] = tens_in[_addr_in];
        }
      }
    } else { // dim_num < 0
      if (threadIdx_x == 0)
        i = atomic_ref<int>(gpu_error_count)
                .fetch_add(1); // record an error (for each thread block)
  }
  return;
}
//--------------------------------------------------------------------------------------------------------------------------
// MATRIX MULTIPLICATION (slow):
template <typename T>
void gpu_matrix_multiply_tn__(size_t ll, size_t lr, size_t lc, const T *arg1,
                              const T *arg2, T *arg0, T alpha,
                              cl::sycl::nd_item<2>& item, int *gpu_error_count,
			      local_accessor<T, 2>& buf1, local_accessor<T, 2>& buf2)
/** arg0(0:ll-1,0:lr-1)+=arg1(0:lc-1,0:ll-1)*arg2(0:lc-1,0:lr-1)*alpha
    NOTES:
    # Thread block dimensions (.x and .y) must be equal to
MAT_MULT_TILE_DIM(X,Y), respectively.
**/
{
  size_t k, _col, _row, _col_base, _row_base;
  int i, j, l, m;
  T _val;

  size_t threadIdx_x = item.get_local_id(1);
  size_t threadIdx_y = item.get_local_id(0);
  size_t blockDim_x = item.get_local_range(1);
  size_t blockDim_y = item.get_local_range(0);
  size_t blockIdx_x = item.get_group(1);
  size_t blockIdx_y = item.get_group(0);

  if (lc > 0 && ll > 0 && lr > 0 && blockDim_x == MAT_MULT_TILE_DIMX && blockDim_y == MAT_MULT_TILE_DIMY) {
    _val = static_cast<T>(0.0);
    j = threadIdx_y;
    i = threadIdx_x;
    _col_base = blockIdx_y * MAT_MULT_TILE_DIMY;

    while (_col_base < lr) {
      _row_base = blockIdx_x * MAT_MULT_TILE_DIMX;
      while (_row_base < ll) {
        for (k = 0; k < lc; k += MAT_MULT_TILE_DIMX) {
          _col = _col_base + j;
          _row = _row_base + j;
          // Load two blocks into shared memory:
          if (k + MAT_MULT_TILE_DIMX > lc) {
            m = lc - k;
          } else {
            m = MAT_MULT_TILE_DIMX;
          }
          if (i < m) { //(k+i)<lc
            for (l = 0; l < MAT_MULT_TILE_DIMX; l += MAT_MULT_TILE_DIMY) {
              if (_row < ll) {
                buf1[l + j][i] = arg1[_row * lc + (k + i)] * alpha;
              } // Load a block of the 1st argument into the shared memory
              _row += MAT_MULT_TILE_DIMY;
            }
            if (_col < lr) {
              buf2[j][i] = arg2[_col * lc + (k + i)];
            } // Load a block of the 2nd argument into the shared memory
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
          // Multiply the two blocks:
          _row = _row_base + i;
          if (_col < lr) {
            if (_row < ll) {
              _col = _col * ll + _row;
              for (l = 0; l < m; l++) {
                _val += buf1[i][l] * buf2[j][l];
              }
              arg0[_col] += _val;
              _val = static_cast<T>(0.0);
            }
          }
          item.barrier(cl::sycl::access::fence_space::local_space);
        }
        _row_base += item.get_group_range(1) * MAT_MULT_TILE_DIMX;
      }
      _col_base += item.get_group_range(0) * MAT_MULT_TILE_DIMY;
    }
  } else {
    if (threadIdx_x == 0 && threadIdx_y == 0)
      i = atomic_ref<int>(gpu_error_count).fetch_add(1); // record an error (for each thread block)
  }
  return;
}

// GPU DEBUG FUNCTIONS:
int gpu_get_error_count()
/** Returns the total number of SYCL errors occured on current GPU.
    A negative return status means an error occurred. **/
  try {
    int i;
    talsh::get_queue().memcpy((void *)&i, gpu_error_count.get_ptr(), sizeof(gpu_error_count)).wait();
    if (err == 0) {
      return i;
    } else {
      return -1;
    }
  } catch (cl::sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
	      << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

int gpu_get_debug_dump(int *dump)
/** Returns the debug dump (int array) from current GPU.
    A positive return status is the length of the debug dump.
    A negative return status means an error occurred. **/
  try {
    talsh::get_queue().memcpy((void *)dump, gpu_debug_dump.get_ptr(), sizeof(int) * GPU_DEBUG_DUMP_SIZE).wait();
    if (err == 0) {
      return GPU_DEBUG_DUMP_SIZE;
    } else {
      return -1;
    }
  } catch (cl::sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
	      << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }
#endif /*NO_GPU*/

// AUXILIARY FUNCTIONS:
static int prmn_convert(int n, const int *o2n, int *n2o)
/** Converts an O2N permutation into N2O (length = n). Both permutations
    are sign-free and the numeration starts from 1. **/
{
  int i, j;
  if (n >= 0) {
    for (i = 0; i < n; i++) {
      j = o2n[i] - 1;
      if (j >= 0 && j < n) {
        n2o[j] = i + 1;
      } else {
        return 1;
      }
    }
  } else {
    return 2;
  }
  return 0;
}

static int non_trivial_prmn(int n, const int *prm)
/** Returns NOPE if the permutation prm[0:n-1] is trivial, YEP otherwise.
    The permutation is sign-free and the numeration starts from 1. No error
   check. **/
{
  int i, f = NOPE;
  for (i = 0; i < n; i++) {
    if (prm[i] != i + 1) {
      f = YEP;
      break;
    }
  }
  return f;
}

#ifndef NO_GPU
static int sycl_queue_get(int gpu_num, int *sycl_queue_handle)
/** For GPU#gpu_num, returns a usable CUDA stream handle <sycl_queue_handle>.
    Non-zero return status means an error, except the return status TRY_LATER
   means no free resources are currently available (not an error). **/
{
  *sycl_queue_handle = -1;
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_is_mine(gpu_num) > GPU_OFF) {
      if (SYCLQueueFFE[gpu_num] > 0) { // number of free handles left on GPU#gpu_num
        *sycl_queue_handle = SYCLQueueFreeHandle[gpu_num][--SYCLQueueFFE[gpu_num]];
        if (*sycl_queue_handle < 0 || *sycl_queue_handle >= MAX_SYCL_TASKS) {
          *sycl_queue_handle = -1;
          return 3; // invalid handle: corruption
        }
      } else {
        return TRY_LATER; // all handles are currently busy
      }
    } else {
      return 2;
    }
  } else {
    return 1;
  }
  return 0;
}

static int sycl_queue_release(int gpu_num, int sycl_queue_handle)
/** For GPU#gpu_num, releases a CUDA stream handle <sycl_queue_handle>.
    Non-zero return status means an error. **/
{
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_is_mine(gpu_num) > GPU_OFF) {
      if (sycl_queue_handle >= 0 && sycl_queue_handle < MAX_SYCL_TASKS) {
        if (SYCLQueueFFE[gpu_num] < 0 || SYCLQueueFFE[gpu_num] > MAX_SYCL_TASKS)
          return 5; // corrupted
        if (SYCLQueueFFE[gpu_num] < MAX_SYCL_TASKS) {
          SYCLQueueFreeHandle[gpu_num][SYCLQueueFFE[gpu_num]++] = sycl_queue_handle;
        } else {
          return 4; // an attempt to release a non-existing handle
        }
      } else {
        return 3;
      }
    } else {
      return 2;
    }
  } else {
    return 1;
  }
  return 0;
}

static cl::sycl::queue **sycl_queue_ptr(int gpu_num, int sycl_queue_handle) {
  /** Returns a pointer to a valid CUDA stream handle. **/
  if (gpu_num < 0 || gpu_num >= MAX_GPUS_PER_NODE)
    return nullptr;
  if (sycl_queue_handle < 0 || sycl_queue_handle >= MAX_SYCL_TASKS)
    return nullptr;
  if (gpu_is_mine(gpu_num) > GPU_OFF)
    return &(SYCLQueueBank[gpu_num][sycl_queue_handle]);
  return nullptr;
}

static int sycl_event_get(int gpu_num, int *sycl_event_handle)
/** For GPU#gpu_num, returns a usable SYCL queue handle <sycl_event_handle>.
    Non-zero return status means an error, except the return status TRY_LATER
   means no free resources are currently available (not an error). **/
{
  *sycl_event_handle = -1;
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_is_mine(gpu_num) > GPU_OFF) {
      if (CUDAEventFFE[gpu_num] > 0) { // number of free handles left on GPU#gpu_num
        *sycl_event_handle = CUDAEventFreeHandle[gpu_num][--CUDAEventFFE[gpu_num]];
        if (*sycl_event_handle < 0 || *sycl_event_handle >= MAX_SYCL_EVENTS) {
          *sycl_event_handle = -1;
          return 3; // invalid handle: corruption
        }
      } else {
        return TRY_LATER; // all handles are currently busy
      }
    } else {
      return 2;
    }
  } else {
    return 1;
  }
  return 0;
}

static int sycl_event_release(int gpu_num, int sycl_event_handle)
/** For GPU#gpu_num, releases a SYCL queue handle <sycl_event_handle>.
    Non-zero return status means an error. **/
{
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_is_mine(gpu_num) > GPU_OFF) {
      if (sycl_event_handle >= 0 && sycl_event_handle < MAX_SYCL_EVENTS) {
        if (CUDAEventFFE[gpu_num] < 0 || CUDAEventFFE[gpu_num] > MAX_SYCL_EVENTS)
          return 5; // corrupted
        if (CUDAEventFFE[gpu_num] < MAX_SYCL_EVENTS) {
          CUDAEventFreeHandle[gpu_num][CUDAEventFFE[gpu_num]++] = sycl_event_handle;
        } else {
          return 4; // an attempt to release a non-existing handle
        }
      } else {
        return 3;
      }
    } else {
      return 2;
    }
  } else {
    return 1;
  }
  return 0;
}

static cl::sycl::event *sycl_event_ptr(int gpu_num, int sycl_event_handle) {
  /** Returns a pointer to a valid SYCL event handle. **/
  if (gpu_num < 0 || gpu_num >= MAX_GPUS_PER_NODE)
    return nullptr;
  if (sycl_event_handle < 0 || sycl_event_handle >= MAX_SYCL_EVENTS)
    return nullptr;
  if (gpu_is_mine(gpu_num) > GPU_OFF)
    return &(SYCLEventBank[gpu_num][sycl_event_handle]);
  return nullptr;
}

// limit_sycl_workgroups2d
static void limit_sycl_workgroups2d(int max_blocks, int *bx, int *by)
/** Limits the SYCL local-size (or work-group size) in a 2d grid to <max_blocks>.
    No argument validity check! **/
{
  if (max_blocks > 1) {
    double rdc = ((double)max_blocks) / (((double)(*bx)) * ((double)(*by)));
    if (rdc < 1.0) {
      rdc = sqrt(rdc);
      if (*bx > *by) {
        *by = (int)(rdc * ((double)(*by)));
        if (*by < 1) {
          *by = 1;
          *bx = max_blocks;
          return;
        }
        *bx = (int)(rdc * ((double)(*bx)));
      } else {
        *bx = (int)(rdc * ((double)(*bx)));
        if (*bx < 1) {
          *bx = 1;
          *by = max_blocks;
          return;
        }
        *by = (int)(rdc * ((double)(*by)));
      }
      if ((*bx) * (*by) > max_blocks) {
        if (*bx > *by) {
          (*bx)--;
        } else {
          (*by)--;
        }
      }
    }
  } else {
    *bx = 1;
    *by = 1;
  }
  return;
}

static int tens_op_best_gpu(const tensBlck_t *tens0, const tensBlck_t *tens1, const tensBlck_t *tens2)
/** Returns the optimal GPU for a given set of tensor arguments (from the data
   locality point of view). A negative return status means an error. All
   arguments are optional. **/
{
  int gpu, dev_kind, gpu0, gpu1, gpu2, s0, s1, s2;

  gpu = -1;
  if (tens0 != nullptr) {
    if (tens0->src_rsc == nullptr)
      return -1;
    gpu0 = decode_device_id((tens0->src_rsc)->dev_id, &dev_kind);
    if (dev_kind != DEV_INTEL_GPU)
      gpu0 = -1;
    if (tens1 != nullptr) {
      if (tens1->src_rsc == nullptr)
        return -1;
      gpu1 = decode_device_id((tens1->src_rsc)->dev_id, &dev_kind);
      if (dev_kind != DEV_INTEL_GPU)
        gpu1 = -1;
      if (gpu1 >= 0 && gpu1 == gpu0) {
        gpu = gpu1;
      } else {
        if (tens2 != nullptr) {
          if (tens2->src_rsc == nullptr)
            return -1;
          gpu2 = decode_device_id((tens2->src_rsc)->dev_id, &dev_kind);
          if (dev_kind != DEV_INTEL_GPU)
            gpu2 = -1;
          if (gpu2 >= 0 && (gpu2 == gpu1 || gpu2 == gpu0)) {
            gpu = gpu2;
          } else {
            s0 = 0;
            s1 = 0;
            s2 = 0;
            if (gpu0 >= 0)
              s0 = gpu_stats[gpu0].tasks_submitted - (gpu_stats[gpu0].tasks_completed +
                    gpu_stats[gpu0].tasks_deferred + gpu_stats[gpu0].tasks_failed);
            if (gpu1 >= 0)
              s1 = gpu_stats[gpu1].tasks_submitted - (gpu_stats[gpu1].tasks_completed +
                    gpu_stats[gpu1].tasks_deferred + gpu_stats[gpu1].tasks_failed);
            if (gpu2 >= 0)
              s2 = gpu_stats[gpu2].tasks_submitted - (gpu_stats[gpu2].tasks_completed +
                    gpu_stats[gpu2].tasks_deferred + gpu_stats[gpu2].tasks_failed);
            if (gpu0 >= 0 && (gpu1 < 0 || s0 <= s1) && (gpu2 < 0 || s0 <= s2)) {
              gpu = gpu0;
            } else if (gpu1 >= 0 && (gpu0 < 0 || s1 <= s0) && (gpu2 < 0 || s1 <= s2)) {
              gpu = gpu1;
            } else if (gpu2 >= 0 && (gpu1 < 0 || s2 <= s1) && (gpu0 < 0 || s2 <= s0)) {
              gpu = gpu2;
            }
          }
        } else {
          s0 = 0;
          s1 = 0;
          if (gpu0 >= 0)
            s0 = gpu_stats[gpu0].tasks_submitted - (gpu_stats[gpu0].tasks_completed +
                 gpu_stats[gpu0].tasks_deferred + gpu_stats[gpu0].tasks_failed);
          if (gpu1 >= 0)
            s1 = gpu_stats[gpu1].tasks_submitted - (gpu_stats[gpu1].tasks_completed +
                 gpu_stats[gpu1].tasks_deferred + gpu_stats[gpu1].tasks_failed);
          if (gpu0 >= 0 && (gpu1 < 0 || s0 <= s1)) {
            gpu = gpu0;
          } else if (gpu1 >= 0 && (gpu0 < 0 || s1 <= s0)) {
            gpu = gpu1;
          }
        }
      }
    } else {
      gpu = gpu0;
    }
  }
  if (gpu < 0 || gpu >= MAX_GPUS_PER_NODE)
    gpu = gpu_busy_least();
  if (gpu_is_mine(gpu) <= GPU_OFF)
    gpu = -1; // for safety
  return gpu;
}

// INT-TAL INITIALIZATION/SHUTDOWN (internal use only):
int init_gpus(int gpu_beg, int gpu_end)
/** Initializes all GPU contexts for the current MPI process. Returned
    positive value is the number of initialized GPUs. A negative return
    status means an error occured. Each enabled GPU from the range
    [gpu_beg:gpu_end] will obtain its own sycl queueu as well. The first GPU
    from the given range will be left active at the end. If <gpu_beg> >
    <gpu_end>, no GPU will be initialized. **/
  try {
    int i, j, n, errc;
    void *base_ptr;

    n = 0;
    for (i = 0; i < MAX_GPUS_PER_NODE; i++)
      gpu_up[i] = GPU_OFF; // initial GPU status

    if (gpu_beg >= 0 && gpu_end >= gpu_beg) {

      cl::sycl::gpu_selector device_selector;
      cl::sycl::platform platform(device_selector);
      auto const &gpu_devices = platform.get_devices();
      for (int k = 0; k < gpu_devices.size(); k++) {
	if (gpu_devices[k].is_gpu())
	  i++;
      }
      if (gpu_end >= MAX_GPUS_PER_NODE || gpu_end >= i)
	return -2;

      // Initialize a mapped bank for tensor operation prefactors for GPU usage:
      errc = slab_clean(&prefactors);
      if (errc != 0)
	return -3;
      errc = slab_construct(&prefactors, sizeof(talshComplex8), (size_t)(MAX_GPUS_PER_NODE * MAX_SYCL_TASKS), sizeof(talshComplex8), 1U);
      if (errc != 0)
	return -4;
      errc = slab_get_base_ptr(&prefactors, &base_ptr);
      if (errc != 0)
	return -5;
      *(&gpu_prefs_base_ptr) = base_ptr; // cudaHostGetDevicePointer(&gpu_prefs_base_ptr,base_ptr,0);

      // Initialize each GPU device:
      for (i = gpu_end; i >= gpu_beg; i--) {
	talsh::select_device(i);
	gpu_up[i] = GPU_MINE;
	if (gpu_up[i] > GPU_OFF) {
	  // SHMEM width: Note SYCL can't set shared memory bank width
#ifndef NO_BLAS
	  gpu_up[i]=GPU_MINE_ONEMKL;
#endif
	}

	// SYCL queue bank:
	if (gpu_up[i] > GPU_OFF) {
	  for (j = 0; j < MAX_SYCL_TASKS; j++)
	    SYCLQueueFreeHandle[i][j] = j;
	  SYCLQueueFFE[i] = MAX_SYCL_TASKS;
	  for (j = 0; j < MAX_SYCL_TASKS; j++) {
	    SYCLQueueBank[i][j] = talsh::get_current_device().create_queue();
	  }
	}
	// SYCL event bank:
	if (gpu_up[i] > GPU_OFF) {
	  for (j = 0; j < MAX_SYCL_EVENTS; j++)
	    SYCLEventFreeHandle[i][j] = j;
	  SYCLEventFFE[i] = MAX_SYCL_EVENTS;
	  for (j = 0; j < MAX_SYCL_EVENTS; j++) {
	    SYCLEventBank[i][j] = talsh::get_current_device().create_event();
	  }
	}
	// Last task:
	LastTask[i] = nullptr;
	// Clear GPU statistics:
	gpu_stats[i].tasks_submitted = 0;
	gpu_stats[i].tasks_completed = 0;
	gpu_stats[i].tasks_deferred = 0;
	gpu_stats[i].tasks_failed = 0;
	gpu_stats[i].flops = 0.0;
	gpu_stats[i].traffic_in = 0.0;
	gpu_stats[i].traffic_out = 0.0;
	gpu_stats[i].time_active = 0.0;
	gpu_stats[i].time_start = std::chrono::steady_clock::now();
	// Accept GPU as ready (active):
	if (gpu_up[i] > GPU_OFF)
	  n++;
      }
      // Peer memory access (UVA based):
#ifdef UNIFIED_ADDRESSING
      for (i = gpu_end; i >= gpu_beg; i--) {
	if (gpu_up[i] > GPU_OFF) {
	  if (gpu_devices[i].get_info<cl::sycl::info::device::host_unified_memory>()) {
	    for (j = gpu_end; j >= gpu_beg; j--) {
	      if (j != i && gpu_up[j] > GPU_OFF) {
		if (!gpu_devices[j].get_info<cl::sycl::info::device::host_unified_memory>()) {
		  if (VERBOSE)
		    printf("\n#MSG(tensor_algebra_gpu_intel): GPU peer no access: %d->%d\n", i, j);
		}
	      }
	    }
	  } else {
	    gpu_up[i] = GPU_OFF;
	    n--;
	  }
	}
      }
#endif
    }

    return n; // number of initialized GPU's
  } catch (cl::sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
	      << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

int free_gpus(int gpu_beg, int gpu_end)
/** Destroys all GPU/CUBLAS contexts on all GPU devices belonging to the MPI
    process. A positive value returned is the number of failed GPUs; a
    negative one is an error.
    If <gpu_beg> > <gpu_end>, nothing wil be done. **/
  try {
    int i, j, n, failure;
    int err;

    failure = 0;
    n = 0;
    if (gpu_beg >= 0 && gpu_end >= gpu_beg) {
      cl::sycl::gpu_selector device_selector;
      cl::sycl::platform platform(device_selector);
      auto const &gpu_devices = platform.get_devices();
      for (int k = 0; k < gpu_devices.size(); k++) {
	if (gpu_devices[k].is_gpu())
	  i++;
      }
      if (gpu_end >= MAX_GPUS_PER_NODE || gpu_end >= i)
	return -2;
      // Free the mapped bank of tensor operation prefactors:
      i = slab_destruct(&prefactors);
      if (i != 0)
	failure++;
      gpu_prefs_base_ptr = nullptr;
      // Free GPU devices:
      for (i = gpu_beg; i <= gpu_end; i++) {
	if (gpu_up[i] > GPU_OFF) {
	  n++;
	  talsh::set_device(i);
	  if (err == 0) {
#ifndef NO_BLAS
	    if (gpu_up[i] >= GPU_MINE_ONEMKL) {
	      err_onemkl = (cublas_handle[i] = nullptr, 0);
	      if (err_onemkl == 0)
		gpu_up[i] = GPU_MINE;
            }
#endif
            // SYCL queue bank:
	    if (gpu_up[i] > GPU_OFF) {
	      for (j = 0; j < MAX_SYCL_TASKS; j++)
		SYCLQueueFreeHandle[i][j] = j;
	      SYCLQueueFFE[i] = MAX_SYCL_TASKS;
	      for (j = 0; j < MAX_SYCL_TASKS; j++) {
		talsh::get_current_device().destroy_queue(SYCLQueueBank[i][j]);
	      }
	    }
	    // SYCL event bank:
	    if (gpu_up[i] > GPU_OFF) {
	      for (j = 0; j < MAX_SYCL_EVENTS; j++)
		SYCLEventFreeHandle[i][j] = j;
	      SYCLEventFFE[i] = MAX_SYCL_EVENTS;
	      for (j = 0; j < MAX_SYCL_EVENTS; j++) {
		talsh::get_current_device().destroy_event(SYCLEventBank[i][j]);
	      }
	    }
	    // Last task:
	    LastTask[i] = nullptr;
	    n--;
	    talsh::get_current_device().reset();
          }
          gpu_up[i] = GPU_OFF; // GPU is taken out of use regardless of its status!
        }
      }
    }
    if (failure && VERBOSE)
      printf("#WARNING(tensor_algebra_gpu_intel:free_gpus): Resource deallocation was not fully successful!");
    return n;
  } catch (cl::sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
	      << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

int gpu_get_device_count(int *dev_count) {
  /** Returns the total number of Intel GPUs found on the node. **/
  talsh::get_device_count(dev_count);
  return 0;
}

int gpu_is_mine(int gpu_num)
/** Positive return: GPU is mine; 0: GPU is not mine; -1: invalid <gpu_num>. **/
{
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    return gpu_up[gpu_num];
  } else {
    return -1;
  }
}

int gpu_busy_least()
/** Returns the ID of the least busy GPU (non-negative) or -1 (no GPU found). **/
{
  int i, j, m, n;
  m = -1;
  n = -1;
  for (i = 0; i < MAX_GPUS_PER_NODE; i++) {
    if (gpu_up[i] > GPU_OFF) {
      j = gpu_stats[i].tasks_submitted -
          (gpu_stats[i].tasks_completed + gpu_stats[i].tasks_deferred +
           gpu_stats[i].tasks_failed);
      if (m >= 0) {
        if (j < m) {
          m = j;
          n = i;
        };
      } else {
        m = j;
        n = i;
      }
    }
  }
  return n;
}

int gpu_in_focus(int gpu_num)
/** If <gpu_num> is not passed here, returns the id of the current GPU in focus.
    If <gpu_num> is passed here, returns YEP if it is currently in focus, NOPE otherwise.
    In case of error, returns NVTAL_FAILURE (negative integer). **/
{
  int n;
  n = talsh::dev_mgr::instance().current_device_id();
  if (gpu_num >= 0) {
    if (n == gpu_num) {
      return YEP;
    } else {
      return NOPE;
    }
  }
  if (n < 0 || n >= MAX_GPUS_PER_NODE)
    return NVTAL_FAILURE; // GPU id must not exceed the TALSH limit per node
  return n;
}

int gpu_activate(int gpu_num)
/** If GPU is enabled (mine), does cudaSetDevice; returns non-zero otherwise (error). **/
{
  int cur_gpu;
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_up[gpu_num] > GPU_OFF) {
      cur_gpu = gpu_in_focus();
      if (cur_gpu != gpu_num) {
	talsh::set_device(gpu_num);
      }
    } else {
      return 2; // GPU is not mine
    }
  } else {
    return 1; // invalid <gpu_num>
  }
  return 0;
}

size_t gpu_device_memory_size(int gpu_num)
/** Returns the total memory (bytes) for a given GPU device. **/
{
  size_t bytes = 0;
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_up[gpu_num] > GPU_OFF)
      bytes = talsh::get_device(gpu_num).get_info<cl::sycl::info::device::global_mem_size>();
  }
  return bytes;
}

double gpu_get_flops(int gpu_num)
/** Returns the current flop count executed by GPU #gpu_num,
    or by all avaialble GPU devices if gpu_num = -1. **/
{
  int i, b, f;
  double total_flops;

  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    b = gpu_num;
    f = gpu_num; // select a specific GPU
  } else if (gpu_num == -1) {
    b = 0;
    f = MAX_GPUS_PER_NODE - 1; // select all GPUs
  } else {
    return -1.0; // invalid GPU number
  }
  total_flops = 0.0;
  for (i = b; i <= f; i++) {
    if (gpu_is_mine(i) != GPU_OFF)
      total_flops += gpu_stats[i].flops;
  }
  return total_flops;
}

// INT-TAL INTERNAL CONTROL:
int gpu_set_shmem_width(int width) {
  /** Sets the GPU shared memory bank width:
      <width> = R4: 4 bytes;
      <width> = R8: 8 bytes. **/
}

int gpu_enable_fast_math(int gpu_num) {
  /** Enables fast math on GPU. **/
}

int gpu_disable_fast_math(int gpu_num) {
  /** Disables fast math on GPU. **/
}

int gpu_query_fast_math(int gpu_num) {
  /** Queries the status of fast math on given GPU. **/
}

void gpu_set_transpose_algorithm(int alg) {
  /** Activates either the scatter or the shared-memory based tensor transpose
     algorithm. Invalid <alg> values will activate the basic shared-memory
     algorithm (default). **/
  if (alg == EFF_TRN_OFF) {
    TRANS_SHMEM = EFF_TRN_OFF;
  } else {
    TRANS_SHMEM = EFF_TRN_ON;
  } // any other value will result in the default setting
  return;
}

void gpu_set_matmult_algorithm(int alg) {
/** Activates either cuBLAS (fast) or my own (slow) BLAS CUDA kernels. **/
#ifndef NO_BLAS
  if (alg == BLAS_ON) {
    DISABLE_BLAS = BLAS_ON;
  } else {
    DISABLE_BLAS = BLAS_OFF;
  };
#endif
  return;
}

int gpu_print_stats(int gpu_num)
/** Prints GPU statistics for GPU#<gpu_num>. If <gpu_num>=-1,
    prints GPU statistics for all active GPUs.
    A negative return status means invalid <gpu_num>. **/
{
  int i, b, f;
  double total_flops, total_traffic_in, total_traffic_out;

  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    b = gpu_num;
    f = gpu_num; // select a specific GPU
  } else if (gpu_num == -1) {
    b = 0;
    f = MAX_GPUS_PER_NODE - 1; // select all GPUs
  } else {
    return -1; // invalid GPU number
  }
  total_flops = 0.0;
  total_traffic_in = 0.0;
  total_traffic_out = 0.0;
  for (i = b; i <= f; i++) {
    if (gpu_is_mine(i) != GPU_OFF) {
      std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - gpu_stats[i].time_stats;
      gpu_stats[i].time_active = elapsed_seconds.count();
      total_flops += gpu_stats[i].flops;
      total_traffic_in += gpu_stats[i].traffic_in;
      total_traffic_out += gpu_stats[i].traffic_out;
      printf("\n#MSG(TAL-SH::INT-TAL): Statistics on GPU #%d:\n", i);
      printf(" Number of tasks submitted: %llu\n", gpu_stats[i].tasks_submitted);
      printf(" Number of tasks completed: %llu\n", gpu_stats[i].tasks_completed);
      printf(" Number of tasks deferred : %llu\n", gpu_stats[i].tasks_deferred);
      printf(" Number of tasks failed   : %llu\n", gpu_stats[i].tasks_failed);
      printf(" Number of Flops processed: %G\n", gpu_stats[i].flops);
      printf(" Number of Bytes to GPU   : %G\n", gpu_stats[i].traffic_in);
      printf(" Number of Bytes from GPU : %G\n", gpu_stats[i].traffic_out);
      printf(" Time active (sec)        : %f\n", gpu_stats[i].time_active);
      printf("#END_MSG\n");
      //  }else{
      //   printf("\n#MSG(TAL-SH::INT-TAL): Statistics on GPU #%d: GPU is
      //   OFF\n",i);
    }
  }
  if (gpu_num == -1) {
    printf("\n#MSG(TAL-SH::INT-TAL): Statistics across all GPU devices:\n");
    printf(" Number of Flops processed   : %G\n", total_flops);
    printf(" Number of Bytes to GPUs     : %G\n", total_traffic_in);
    printf(" Number of Bytes from GPUs   : %G\n", total_traffic_out);
    if (total_traffic_in + total_traffic_out > 0.0) {
      printf(" Average arithmetic intensity: %G\n", total_flops / (total_traffic_in + total_traffic_out));
    } else {
      printf(" Average arithmetic intensity: %G\n", 0.0);
    }
    printf("#END_MSG\n");
  }
  return 0;
}
#endif /*NO_GPU*/

// TENSOR BLOCK API:
int tensBlck_create(tensBlck_t **ctens)
/** Creates an empty instance of tensBlck_t and initializes it to null (on
   Host). **/
{
  *ctens = (tensBlck_t *)malloc(sizeof(tensBlck_t));
  if (*ctens == nullptr)
    return TRY_LATER;
  return tensBlck_clean(*ctens);
}

int tensBlck_clean(tensBlck_t *ctens)
/** Cleans an undefined tensBlck_t object. **/
{
  if (ctens == nullptr)
    return -1;
  ctens->data_kind = NO_TYPE;
  ctens->src_rsc = nullptr; // source memory resource (where the tensor body is before the operation)
  ctens->dst_rsc = nullptr; // destination memory resource (where the tensor body will be after the operation)
  ctens->tmp_rsc = nullptr; // temporary memory resource (where the tensor body can be during the operation)
  return tensShape_clean(&(ctens->shape));
}

int tensBlck_destroy(tensBlck_t *ctens)
/** Destroys a defined instance of tensBlck_t (either nullified or
   shape-defined). A return status NOT_CLEAN indicates an unsuccessful resource
   release, which
    can be considered as a tolerable error (the object will still be destroyed).
   **/
{
  int n, errc;

  errc = 0;
  n = 0;
  if (ctens == nullptr)
    return -1;
  errc = tensBlck_destruct(ctens);
  if (errc)
    n = NOT_CLEAN;
  if (ctens->tmp_rsc != nullptr) {
    errc = tensDevRsc_destroy(ctens->tmp_rsc);
    if (errc)
      n = NOT_CLEAN;
  }
  if (ctens->dst_rsc != nullptr && ctens->dst_rsc != ctens->src_rsc) {
    errc = tensDevRsc_destroy(ctens->dst_rsc);
    if (errc)
      n = NOT_CLEAN;
  }
  if (ctens->src_rsc != nullptr) {
    errc = tensDevRsc_destroy(ctens->src_rsc);
    if (errc)
      n = NOT_CLEAN;
  }
  ctens->src_rsc = nullptr;
  ctens->dst_rsc = nullptr;
  ctens->tmp_rsc = nullptr;
  free(ctens);
  return n;
}

int tensBlck_construct(
    tensBlck_t *ctens, // pointer to defined tensor block (either nullified or defined to a value)
    int pinned, // YEP: tensor shape multi-indices will be pinned (for GPU),
                // NOPE: regular malloc (not pinned)
    int trank,       // tensor rank
    const int *dims, // tensor dimension extents (when trank > 0)
    const int *divs, // tensor dimension dividers (when trank > 0, optional)
    const int *grps) // tensor dimension groups (when trank > 0, optional)
/** Constructs (defines/redefines) a tensor block without attaching its body
   (only the shape). If the tensor block is to be used on Nvidia GPUs or other
   asynchronous devices, argument <pinned> must be set to YEP (NOPE will not use
   pinned memory). A return status NOT_CLEAN indicates an unsuccessful resource
   release, which,
    can be considered as a tolerable error (the object will still be
   constructed). **/
{
  int n, errc;

  n = 0;
  if (ctens == nullptr)
    return -1;
  if (trank < 0 || trank > MAX_TENSOR_RANK)
    return -2; // invalid tensor rank
  if (trank > 0 && dims == nullptr)
    return -3; // dimension extents must be present for rank>0 tensors
  errc = tensBlck_destruct(ctens);
  if (errc != 0) {
    if (errc == NOT_CLEAN) {
      n = errc;
    } else {
      return 1;
    }
  }
  errc = tensShape_construct(&(ctens->shape), pinned, trank, dims, divs, grps);
  if (errc != 0) {
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      return errc;
    } else {
      return 2;
    }
  }
  return n; // either 0 or NOT_CLEAN
}

int tensBlck_attach_body(
    tensBlck_t *ctens, // pointer to a shape-defined (constructed) tensor block
    int data_kind,     // data kind (R4,R8,C4,C8)
    int dev_id, // flat device id where the body resides (or should reside):
                // Defaults to Host
    void *body_ptr, // pointer to the tensor body (global memory of device
                    // <dev_id>)
    int buf_entry) // argument buffer entry handle corresponding to the
                   // <body_ptr> (optional)
/** Attaches a body to a shape-defined tensor block (with an empty body). If
   both <body_ptr> and <buf_entry> are absent, a resource will be allocated on
   device <dev_id> in the device argument buffer (if available). If <buf_entry>
   is absent, a defined <body_ptr> points to an external memory (either pinned
   or not). If both <body_ptr> and <buf_entry> are defined, the external memory
   is assumed to be within that argument buffer entry. In all cases, the memory
   resource will be associated with the .src_rsc component of tensBlck_t. It is
   forbidden to attempt allocating/attaching a memory resource when an existing
   memory resource is still in use (this will result in an error). A return
   status of TRY_LATER or DEVICE_UNABLE indicates the current or permanent
   shortage in the necessary resources and is not an error. **/
{
  int errc, dks;
  size_t vol, body_size;

  if (ctens == nullptr)
    return -1;
  errc = tens_valid_data_kind(data_kind, &dks);
  if (errc != YEP || data_kind == NO_TYPE)
    return -2;
  if (ctens->shape.num_dim < 0 || ctens->shape.num_dim > MAX_TENSOR_RANK)
    return -3; // tensor block must be shape-defined
  if (body_ptr == nullptr && buf_entry >= 0)
    return -4; // a defined argument buffer entry must be supplied with the
               // corresponding pointer
  if (dev_id < 0) {
    dev_id = encode_device_id(DEV_HOST, 0);
    if (dev_id < 0 || dev_id >= DEV_MAX)
      return -5;
  } // dev_id defaults to Host
  if (ctens->src_rsc == nullptr) {
    errc = tensDevRsc_create(&(ctens->src_rsc));
    if (errc != 0 || ctens->src_rsc == nullptr)
      return 1;
  } else {
    if (tensDevRsc_is_empty(ctens->src_rsc) == NOPE)
      return 2; // source resource is not empty (release it first)
  }
  vol = tensShape_volume(
      &(ctens->shape));            // tensor body volume (number of elements)
  body_size = vol * ((size_t)dks); // tensor body size in bytes
  if (body_ptr == nullptr) {       // allocate memory in the argument buffer
    errc = tensDevRsc_allocate_mem(ctens->src_rsc, dev_id, body_size, YEP);
    if (errc != 0) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        return errc;
      } else {
        return 3;
      }
    }
  } else { // associate memory
    errc = tensDevRsc_attach_mem(ctens->src_rsc, dev_id, body_ptr, buf_entry);
    if (errc != 0) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        return errc;
      } else {
        return 4;
      }
    }
  }
  ctens->data_kind = data_kind;
  return 0;
}

int tensBlck_destruct(tensBlck_t *ctens, int release_body, int which_body)
/** Destructs a defined tensor block (releases all resources and initializes the
   tensor block to null). If <release_body> == YEP/NOPE, the global memory
   resources will be released/kept. Argument <which_body> can further regulate
   which tensor body to be released/kept (SOURCE, DESTINATION, TEMPORARY,
   EVERYTHING). A return status NOT_CLEAN indicates an unsuccessful resource
   release that may be considered as a tolerable error since the tensor block
   will be nullified anyway. Although device resources are
    released the resource objects themselves are not (they are only destroyed in
   _destroy method). **/
{
  int n, errc;

  n = 0;
  if (ctens == nullptr)
    return -1;
  if (ctens->shape.num_dim >= 0) { // shape-defined tensor block
    if (ctens->shape.num_dim > MAX_TENSOR_RANK)
      return -2;
    // Release the TEMPORARY resource:
    if (ctens->tmp_rsc != nullptr &&
        ((release_body == YEP &&
          (which_body == EVERYTHING || which_body == TEMPORARY)) ||
         (release_body == NOPE &&
          (which_body != EVERYTHING && which_body != TEMPORARY)))) {
      errc = tensDevRsc_release_all(ctens->tmp_rsc);
      if (errc != 0)
        n = NOT_CLEAN; // Note: Resource object is not destroyed!
    }
    ctens->tmp_rsc = nullptr;
    // Release the DESTINATION resource (only if different from the SOURCE
    // resource):
    if (ctens->dst_rsc != nullptr &&
        ((release_body == YEP &&
          (which_body == EVERYTHING || which_body == DESTINATION)) ||
         (release_body == NOPE &&
          (which_body != EVERYTHING && which_body != DESTINATION)))) {
      if (ctens->dst_rsc != ctens->src_rsc) {
        errc = tensDevRsc_release_all(ctens->dst_rsc);
        if (errc != 0)
          n = NOT_CLEAN; // Note: Resource object is not destroyed!
      } else {
        ctens->dst_rsc = nullptr; // destination resource simply pointed to the
                                  // source resource
      }
    }
    ctens->dst_rsc = nullptr;
    // Release the SOURCE resource:
    if (ctens->src_rsc != nullptr &&
        ((release_body == YEP &&
          (which_body == EVERYTHING || which_body == SOURCE)) ||
         (release_body == NOPE &&
          (which_body != EVERYTHING && which_body != SOURCE)))) {
      errc = tensDevRsc_release_all(ctens->src_rsc);
      if (errc != 0)
        n = NOT_CLEAN; // Note: Resource object is not destroyed!
    }
    ctens->src_rsc = nullptr;
    if (tens_valid_data_kind(ctens->data_kind) != YEP)
      n = NOT_CLEAN;
  }
  ctens->data_kind = NO_TYPE;
  errc = tensShape_destruct(&(ctens->shape));
  if (errc) {
    if (errc == NOT_CLEAN) {
      n = NOT_CLEAN;
    } else {
      return 1;
    }
  }
  return n;
}

int tensBlck_src_dev_id(const tensBlck_t *ctens, int *dev_kind)
/** Returns the device id on which the source data (tensor body) resides.
    If <dev_kind> is provided (!=nullptr), the device id will be kind-specific,
    belonging to the device kind <dev_kind>. Otherwise, it will be the flat id.
    A return status DEV_NULL indicates no current source data. A return
    status DEV_MAX indicates a failure (error). **/
{
  int dev_id;

  dev_id = DEV_NULL;
  if (dev_kind != nullptr)
    *dev_kind = DEV_NULL;
  if (ctens == nullptr)
    return DEV_MAX;
  if (ctens->src_rsc != nullptr) {
    if (dev_kind == nullptr) {
      dev_id = ((*ctens).src_rsc)->dev_id;
    } else {
      dev_id = decode_device_id(((*ctens).src_rsc)->dev_id, dev_kind);
    }
  }
  return dev_id;
}

int tensBlck_present(const tensBlck_t *ctens, int dev_id, int dev_kind)
/** Returns YEP/NOPE if the tensor body is present/absent on the device
   specified by a device id <dev_id> and a device kind <dev_kind>. When <dev_id>
   is present, the presence of <dev_kind> determines whether <dev_id> is a flat
   or kind-specific. When <dev_id> is absent but <dev_kind> is present, the
   presence will be checked against the specified device kind. If both <dev_id>
   and <dev_kind> are absent, any presence will be checked (on any device). A
   return status NVTAL_FAILURE indicates invalid arguments. **/
{
  int src_dev, dst_dev, devn, devk;

  if (ctens == nullptr)
    return NVTAL_FAILURE;
  if (ctens->src_rsc != nullptr) {
    src_dev = ctens->src_rsc->dev_id;
  } else {
    src_dev = DEV_NULL;
  }
  if (ctens->dst_rsc != nullptr) {
    dst_dev = ctens->dst_rsc->dev_id;
  } else {
    dst_dev = DEV_NULL;
  }
  if (dev_kind == DEV_NULL) {
    if (dev_id == DEV_NULL) {
      if (src_dev >= 0 || dst_dev >= 0)
        return YEP;
    } else {
      if (dev_id < 0 || dev_id >= DEV_MAX)
        return NVTAL_FAILURE;
      if (src_dev == dev_id || dst_dev == dev_id)
        return YEP;
    }
  } else {
    if (valid_device_kind(dev_kind) != YEP)
      return NVTAL_FAILURE;
    if (dev_id == DEV_NULL) {
      devn = decode_device_id(src_dev, &devk);
      if (devn >= 0 && devk == dev_kind)
        return YEP;
      devn = decode_device_id(dst_dev, &devk);
      if (devn >= 0 && devk == dev_kind)
        return YEP;
    } else {
      devn = encode_device_id(dev_id, dev_kind);
      if (devn >= DEV_MAX)
        return NVTAL_FAILURE;
      if (src_dev == devn || dst_dev == devn)
        return YEP;
    }
  }
  return NOPE;
}

size_t tensBlck_volume(const tensBlck_t *ctens)
/** Returns the volume of a tensor block (number of elements)
    or zero in cases of an empty tensor block or an error. **/
{
  if (ctens == nullptr)
    return 0;
  size_t tvol = tensShape_volume(&(ctens->shape));
  return tvol;
}

void tensBlck_print(const tensBlck_t *ctens)
/** Print info on a given tensor block. **/
{
  if (ctens != nullptr) {
    printf("\n#MESSAGE: Printing tensor block info:\n");
    printf(" Tensor block address   : %p\n", ctens);
    printf(" Tensor block data kind : %d\n", ctens->data_kind);
    printf(" Tensor block rank      : %d\n", ctens->shape.num_dim);
    if (ctens->shape.num_dim >= 0 && ctens->shape.num_dim <= MAX_TENSOR_RANK) {
      printf(" Tensor block dimensions:");
      for (int i = 0; i < (ctens->shape.num_dim); i++)
        printf(" %d", ctens->shape.dims[i]);
      printf("\n Tensor block source resource: %p:\n", ctens->src_rsc);
      if (ctens->src_rsc != nullptr) {
        printf("  Device ID     : %d\n", ctens->src_rsc->dev_id);
        printf("  Memory address: %p\n", ctens->src_rsc->gmem_p);
        printf("  Buffer entry  : %d\n", ctens->src_rsc->buf_entry);
        printf("  External mem  : %d\n", ctens->src_rsc->mem_attached);
      }
      printf(" Tensor block destination resource: %p:\n", ctens->dst_rsc);
      if (ctens->dst_rsc != nullptr) {
        printf("  Device ID     : %d\n", ctens->dst_rsc->dev_id);
        printf("  Memory address: %p\n", ctens->dst_rsc->gmem_p);
        printf("  Buffer entry  : %d\n", ctens->dst_rsc->buf_entry);
        printf("  External mem  : %d\n", ctens->dst_rsc->mem_attached);
      }
      printf(" Tensor block temporary resource: %p:\n", ctens->tmp_rsc);
      if (ctens->tmp_rsc != nullptr) {
        printf("  Device ID     : %d\n", ctens->tmp_rsc->dev_id);
        printf("  Memory address: %p\n", ctens->tmp_rsc->gmem_p);
        printf("  Buffer entry  : %d\n", ctens->tmp_rsc->buf_entry);
        printf("  External mem  : %d\n", ctens->tmp_rsc->mem_attached);
      }
    }
    printf("#END OF MESSAGE\n");
  } else {
    printf("\n#WARNING(tensor_algebra_gpu_intel:tensBlck_print): nullptr "
           "pointer!\n");
  }
  return;
}

int tensBlck_init_host(tensBlck_t *ctens, double init_val)
/** Initializes a tensor block on Host. **/
{
  int i, dev_kind;
  size_t vol;
  float fval;
  float *fp;
  double *dp;
  if (ctens == nullptr)
    return -1;
  if (ctens->shape.num_dim < 0 || ctens->src_rsc == nullptr)
    return -2;
  if (ctens->src_rsc->gmem_p == nullptr)
    return -3;
  if (tens_valid_data_kind(ctens->data_kind) != YEP ||
      ctens->data_kind == NO_TYPE)
    return -4;
  i = decode_device_id(ctens->src_rsc->dev_id, &dev_kind);
  if (dev_kind != DEV_HOST || i != 0)
    return 1;
  vol = tensBlck_volume(ctens);
  if (vol == 0)
    return -5;
  switch (ctens->data_kind) {
  case R4:
    fval = (float)init_val;
    fp = (float *)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol, fp, fval) schedule(guided)
    for (size_t l = 0; l < vol; l++)
      fp[l] = fval;
    break;
  case R8:
    dp = (double *)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol, dp, init_val) schedule(guided)
    for (size_t l = 0; l < vol; l++)
      dp[l] = init_val;
    break;
  default:
    return 2;
  }
  return 0;
}

double tensBlck_norm2_host(const tensBlck_t *ctens)
/** Computes the squared 2-norm of the tensor block on Host. **/
{
  int i, dev_kind;
  size_t vol;
  double nrm2;
  float *fp;
  double *dp;
  if (ctens == nullptr)
    return -1.;
  if (ctens->shape.num_dim < 0 || ctens->src_rsc == nullptr)
    return -2.;
  if (ctens->src_rsc->gmem_p == nullptr)
    return -3.;
  if (tens_valid_data_kind(ctens->data_kind) != YEP ||
      ctens->data_kind == NO_TYPE)
    return -4.;
  i = decode_device_id(ctens->src_rsc->dev_id, &dev_kind);
  if (dev_kind != DEV_HOST || i != 0)
    return -5.;
  vol = tensBlck_volume(ctens);
  if (vol == 0)
    return -6.;
  nrm2 = 0.0;
  switch (ctens->data_kind) {
  case R4:
    fp = (float *)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol, fp) schedule(guided) reduction(+ : nrm2)
    for (size_t l = 0; l < vol; l++)
      nrm2 += (double)(fp[l] * fp[l]);
    break;
  case R8:
    dp = (double *)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol, dp) schedule(guided) reduction(+ : nrm2)
    for (size_t l = 0; l < vol; l++)
      nrm2 += dp[l] * dp[l];
    break;
  default:
    return -7.;
  }
  return nrm2;
}

#ifndef NO_GPU
// SYCL TASK API:
int sycl_task_create(cudaTask_t **sycl_task)
/** Creates an empty instance of cudaTask_t. An unsuccessful attempt
    to allocate memory for the CUDA task returns status TRY_LATER. **/
{
  int errc = 0;
  // if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_intel:sycl_task_create): New
  // CUDA task: sizeof(cudaTask_t) = %d",sizeof(cudaTask_t)); //debug
  *sycl_task = (cudaTask_t *)malloc(sizeof(cudaTask_t));
  if (*sycl_task == nullptr)
    return TRY_LATER;
  errc = sycl_task_clean(*sycl_task);
  errc = 0;
  return errc;
}

int sycl_task_clean(cudaTask_t *sycl_task)
/** Cleans (initializes to null) a freshly allocated CUDA task. **/
{
  if (sycl_task == nullptr)
    return -1;
  sycl_task->task_error = -1;
  sycl_task->gpu_id = -1;
  sycl_task->num_args = 0;
  sycl_task->queue_hl = -1;
  sycl_task->event_hl = -1;
  // sycl_task->event_start_hl = -1;
  // sycl_task->event_comput_hl = -1;
  // sycl_task->event_output_hl = -1;
  // sycl_task->event_finish_hl = -1;
  for (int i = 0; i < MAX_TENSOR_OPERANDS; ++i) {
    sycl_task->tens_args[i].tens_p = nullptr;
    sycl_task->tens_args[i].prmn_p = nullptr;
    sycl_task->tens_args[i].const_mem_entry = -1;
  }
  sycl_task->pref_ptr = nullptr;
  return 0;
}

int sycl_task_construct(cudaTask_t *sycl_task, int gpu_id)
/** Constructs a CUDA task ready for recording on GPU#gpu_id (acquires
   resources). If <gpu_id> is not passed here (negative), the currently active
   GPU will be used. Returns TRY_LATER or DEVICE_UNABLE in case of temporary or
   permanent
    shortage of GPU resources, respectively (CUDA task is left clean). **/
{
  int i, errc;

  errc = 0;
  if (sycl_task == nullptr)
    return -1;
  if (sycl_task->task_error >= 0 || sycl_task->gpu_id >= 0 ||
      sycl_task->num_args > 0)
    return 1; // CUDA task is not clean: Destruct/clean it first
  i = sycl_task_clean(sycl_task); // just in case
  if (gpu_id < 0)
    gpu_id = gpu_in_focus();
  if (gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE)
    return 2; // gpu_id is out of range
  if (gpu_is_mine(gpu_id) > GPU_OFF) {
    errc = sycl_queue_get(gpu_id, &(sycl_task->queue_hl));
    if (errc != 0) {
      sycl_task->queue_hl = -1;
      if (errc != TRY_LATER && errc != DEVICE_UNABLE)
        errc = 3;
    } else {
      errc = sycl_event_get(gpu_id, &(sycl_task->event_hl));
      if (errc != 0) {
        sycl_task->event_hl = -1;
        if (errc != TRY_LATER && errc != DEVICE_UNABLE)
          errc = 4;
      }
    }
    if (errc == 0) {
      sycl_task->task_error = -1;
      sycl_task->gpu_id = gpu_id;
    } else {
      i = sycl_event_release(gpu_id, sycl_task->event_hl);
      sycl_task->event_hl = -1;
      i = sycl_queue_release(gpu_id, sycl_task->queue_hl);
      sycl_task->queue_hl = -1;
      i = sycl_task_clean(sycl_task);
    }
  } else {
    return DEVICE_UNABLE;
  }
  return errc;
}

int sycl_task_destruct(cudaTask_t *sycl_task)
/** Destructs a defined completed CUDA task or does nothing. If the CUDA task
    is defined but not completed, a return status TRY_LATER is returned.
    If any of the resources used by the CUDA task cannot be released cleanly,
    a return status NOT_CLEAN is returned. Nevertheless, the CUDA task will be
    clean at the end. **/
{
  int n, errc;

  if (sycl_task == nullptr)
    return -1;
  errc = sycl_task_completed(sycl_task); // SYCL task is finalized there (if completed or failed)
  if (errc == SYCL_TASK_EMPTY)
    return 0;
  n = 0; // number of unsuccessful resource releases
  if (errc == SYCL_TASK_COMPLETED || errc == SYCL_TASK_ERROR) {
    if (sycl_task->gpu_id < 0 || sycl_task->gpu_id >= MAX_GPUS_PER_NODE)
      return -2; // GPU id is out of allowed range
    if (sycl_task == LastTask[sycl_task->gpu_id])
      LastTask[sycl_task->gpu_id] = nullptr; // clear task dependency
    // Release SYCL resources:
    errc = sycl_queue_release(sycl_task->gpu_id, sycl_task->queue_hl);
    sycl_task->queue_hl = -1;
    if (errc != 0)
      n++;
    errc = sycl_event_release(sycl_task->gpu_id, sycl_task->event_hl);
    sycl_task->event_hl = -1;
    if (errc != 0)
      n++;
    // Release prefactor entry, if needed:
    if (sycl_task->pref_ptr != nullptr) {
      errc = slab_entry_release(&prefactors, sycl_task->pref_ptr);
      if (errc != 0)
        n++;
    }
    // Clean the SYCL task:
    errc = sycl_task_clean(sycl_task);
  } else {
    return TRY_LATER; // SYCL task is still in progress
  }
  if (n != 0)
    n = NOT_CLEAN;
  return n;
}

int sycl_task_destroy(cudaTask_t *sycl_task)
/** Destroys an instance of cudaTask_t if the CUDA task has completed or empty.
    If the CUDA task is still in progress, a return status TRY_LATER is
   returned. If any of the CUDA task resources could not be released cleanly, a
   return status NOT_CLEAN will be returned but the CUDA task will still be
   destroyed. **/
{
  int n, errc;

  n = 0;
  if (sycl_task == nullptr)
    return -1;
  errc = sycl_task_completed(sycl_task); // SYCL task is finalized there (if completed or failed)
  if (errc == SYCL_TASK_COMPLETED || errc == SYCL_TASK_ERROR) {
    errc = sycl_task_destruct(sycl_task);
    if (errc != 0)
      n = NOT_CLEAN;
  } else {
    if (errc != SYCL_TASK_EMPTY)
      return TRY_LATER; // SYCL task is still in progress
  }
  free(sycl_task);
  return n;
}

int sycl_task_gpu_id(const cudaTask_t *sycl_task)
/** Returns the GPU id associated with a SYCL task. A negative
    return value means a null or empty task was passed here. **/
{
  if (sycl_task == nullptr)
    return -2;
  if (sycl_task->gpu_id >= 0 && sycl_task->gpu_id < MAX_GPUS_PER_NODE)
    return sycl_task->gpu_id;
  return -1;
}

int sycl_task_status(cudaTask_t *sycl_task)
/** Checks the status of a SYCL task. Possible status values are listed in
    tensor_algebra.h and tensor_algebra.inc (keep them consistent!). Both
    SYCL_TASK_COMPLETED (no errors) and SYCL_TASK_ERROR (error occurred)
    suggest a completion of the SYCL task. An unsuccessful attempt to find
    out the status of the SYCL task results in a return status NVTAL_FAILURE.
**/
  try {
    int task_stat, cur_gpu, errc;
    cl::sycl::event *evnt_p;
    cl::sycl::event::info::event_command_status err;

    if (sycl_task == nullptr)
      return SYCL_TASK_EMPTY; // nullptr task pointer is treated as an empty task here
    if (sycl_task->task_error < 0 && sycl_task->gpu_id < 0)
      return SYCL_TASK_EMPTY; // empty SYCL task
    if (sycl_task->task_error >= 0 && sycl_task->gpu_id < 0)
      return NVTAL_FAILURE; // completed task without an assigned GPU
    if (sycl_task->task_error == 0)
      return SYCL_TASK_COMPLETED; // SYCL task had completed successfully
    if (sycl_task->task_error > 0)
      return SYCL_TASK_ERROR; // SYCL task error had been registered

    cur_gpu = gpu_in_focus();
    if (cur_gpu < 0 || cur_gpu >= MAX_GPUS_PER_NODE)
      return NVTAL_FAILURE; // get current GPU
    errc = gpu_activate(sycl_task->gpu_id);
    if (errc != 0)
      return NVTAL_FAILURE; // could not activate the SYCL task GPU

    evnt_p = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_hl);
    if (evnt_p == nullptr)
      return NVTAL_FAILURE;
    err = evnt_p->get_info<cl::sycl::info::event::command_execution_status>();
    if (err == cl::sycl::info::event_command_status::complete) {
      sycl_task->task_error = 0;
      errc = sycl_task_finalize(sycl_task); // release unneeded memory resources occupied by the task arguments
      if (errc == 0) {
	sycl_task->task_error = 0;
	task_stat = SYCL_TASK_COMPLETED; // SYCL task completed, memory released cleanly
      } else {
        if (VERBOSE)
	  printf("#ERROR(INT-TAL:sycl_task_status): sycl_task_finalize error %d\n", errc);
	sycl_task->task_error = 127;
	task_stat = SYCL_TASK_ERROR; // SYCL task completed, memory could not be released cleanly
      }
      gpu_stats[sycl_task->gpu_id].tasks_completed++;
    } else if (err == cl::sycl::info::event_command_status::running) {
      task_stat = SYCL_TASK_INPUT_THERE; // computation started, input data is on device (can be reused later)
    }
    else if (err == cl::sycl::info::event_command_status::submitted) {
      task_stat = SYCL_TASK_SCHEDULED; // task has not started yet
    }

    errc = gpu_activate(cur_gpu);
    return task_stat;
  } catch (cl::sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
	      << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

int sycl_task_completed(cudaTask_t *sycl_task)
/** Returns SYCL_TASK_COMPLETED or SYCL_TASK_ERROR if an existing SYCL task
    <sycl_task> has completed successfully or due to a scheduling/execution
    failure, respectively. Note that having had successfully checked the CUDA
    task for completion before will immediately suggest completion later
    (without further querying)! Other possible outputs: SYCL_TASK_EMPTY,
    SYCL_TASK_SCHEDULED. An inability to check the completion status of the
    SYCL task results in return status NVTAL_FAILURE. **/
{
  int cur_gpu, ret_stat, errc;
  cl::sycl::queue **strm_p;
  cl::sycl::event::info::event_command_status err;

  if (sycl_task == nullptr)
    return SYCL_TASK_EMPTY; // null SYCL task is treated as empty
  if (sycl_task->gpu_id < 0)
    return SYCL_TASK_EMPTY;
  if (sycl_task->task_error == 0)
    return SYCL_TASK_COMPLETED; // successful completion had occurred
  if (sycl_task->task_error > 0)
    return SYCL_TASK_ERROR; // completion due to an error had occurred
  cur_gpu = gpu_in_focus();
  if (cur_gpu < 0 || cur_gpu >= MAX_GPUS_PER_NODE)
    return NVTAL_FAILURE;
  errc = gpu_activate(sycl_task->gpu_id);
  if (errc != 0)
    return NVTAL_FAILURE;

  evnt_p = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_hl);
  if (evnt_p == nullptr)
    return NVTAL_FAILURE;
  err = evnt_p->get_info<cl::sycl::info::event::command_execution_status>();
  if (err == cl::sycl::info::event_command_status::complete) {
    ret_stat = SYCL_TASK_COMPLETED;
    if (sycl_task->task_error < 0) {
      sycl_task->task_error = 0;
      gpu_stats[sycl_task->gpu_id].tasks_completed++;
    }
  } else if (err == cl::sycl::info::event_command_status::running ||
             err == cl::sycl::info::event_command_status::
                        submitted) { // task is still in progress
    ret_stat = SYCL_TASK_SCHEDULED;
  } else {
    ret_stat = SYCL_TASK_EMPTY;
  }

  // strm_p = sycl_stream_ptr(sycl_task->gpu_id, sycl_task->queue_hl);
  // if (strm_p == nullptr)
  //   return NVTAL_FAILURE;
  // err = 0; // todo here
  // if (err != 0 && err != 400) { // task is still in progress
  //   ret_stat = SYCL_TASK_SCHEDULED;
  // } else { // task completed successfully or has never been scheduled
  //   if (err == 400) { // stream does not exist
  //     ret_stat = SYCL_TASK_EMPTY;
  //   } else {
  //     ret_stat = SYCL_TASK_COMPLETED;
  //     if (sycl_task->task_error < 0) {
  // 	sycl_task->task_error = 0;
  // 	gpu_stats[sycl_task->gpu_id].tasks_completed++;
  //     }
  //   }
  // }

  if (ret_stat == SYCL_TASK_COMPLETED) {
    errc = sycl_task_finalize(sycl_task);
    if (errc != 0) {
      if (VERBOSE)
	printf("#ERROR(INT-TAL:sycl_task_completed): sycl_task_finalize error %d\n", errc);
      sycl_task->task_error = 127; // resources could not be released properly
    }
  }
  errc = gpu_activate(cur_gpu);
  return ret_stat;
}

int sycl_task_wait(syclTask_t *sycl_task)
/** Waits upon completion of a SYCL task: Returns the output of
   sycl_task_completed(..). Possible returns are SYCL_TASK_COMPLETED,
   SYCL_TASK_ERROR, SYCL_TASK_SCHEDULED, SYCL_TASK_EMPTY.
    In case the completion of a SYCL task cannot be determined, a return status
   NVTAL_FAILURE is returned. **/
{
  int i, j;

  i = SYCL_TASK_SCHEDULED;
  j = 1;
  while (j > 0) {
    i = sycl_task_completed(sycl_task);
    if (i != SYCL_TASK_SCHEDULED)
      j--;
  }
  return i;
}

int sycl_tasks_wait(unsigned int num_tasks, cudaTask_t **sycl_tasks,
                    int *task_stats)
/** Waits upon completion of a series of SYCL tasks. Returns zero on success,
   non-zero on error. On success, <task_stats> will contain the completion
   status for each task. Note that
    <sycl_tasks> points to an array of SYCL task pointers. **/
{
  int i, j, n;

  if (num_tasks > 0) {
    if (sycl_tasks != nullptr && task_stats != nullptr) {
      for (i = 0; i < num_tasks; i++) {
        task_stats[i] = SYCL_TASK_SCHEDULED;
      }
      n = num_tasks;
      while (n > 0) {
        for (i = 0; i < num_tasks; i++) {
          if (task_stats[i] == SYCL_TASK_SCHEDULED) {
            if (sycl_tasks[i] != nullptr) {
              j = sycl_task_completed(sycl_tasks[i]);
              task_stats[i] = j;
              if (j != SYCL_TASK_SCHEDULED)
                n--;
            } else {
              return 1;
            }
          }
        }
      }
    } else {
      return 2;
    }
  }
  return 0;
}

int sycl_task_error_code(const cudaTask_t *sycl_task)
/** Returns the current .task_error member variable. **/
{
  return sycl_task->task_error;
}

int sycl_task_dev_rsc_copy(const cudaTask_t *sycl_task, unsigned int arg_num,
                           char which, talsh_dev_rsc_t *dev_rsc)
/** Clones the device resource object from a tensor argument of a SYCL task into
   <dev_rsc>: <which> selects bewteen 's':source, 't':temporary, 'd':destination (resource). **/
{
  int errc;
  tensBlck_t *ctens;

  if (sycl_task == nullptr)
    return -1;
  if (dev_rsc == nullptr)
    return -2;
  if (arg_num >= sycl_task->num_args)
    return 1;
  ctens = sycl_task->tens_args[arg_num].tens_p;
  if (ctens) {
    switch (which) {
    case 's':
      errc = tensDevRsc_clone(ctens->src_rsc, dev_rsc);
      break;
    case 't':
      errc = tensDevRsc_clone(ctens->tmp_rsc, dev_rsc);
      break;
    case 'd':
      errc = tensDevRsc_clone(ctens->dst_rsc, dev_rsc);
      break;
    default:
      errc = 2;
    }
  } else {
    errc = 3;
  }
  return errc;
}

int sycl_task_dev_rsc_move(cudaTask_t *sycl_task, unsigned int arg_num,
                           char which, talsh_dev_rsc_t *dev_rsc)
/** Moves the device resource object from a tensor argument of a SYCL task into
   <dev_rsc>: <which> selects bewteen 's':source, 't':temporary, 'd':destination (resource). **/
{
  int errc;
  tensBlck_t *ctens;

  if (sycl_task == nullptr)
    return -1;
  if (dev_rsc == nullptr)
    return -2;
  if (arg_num >= sycl_task->num_args)
    return 1;
  ctens = sycl_task->tens_args[arg_num].tens_p;
  if (ctens) {
    switch (which) {
    case 's':
      errc = tensDevRsc_clone(ctens->src_rsc, dev_rsc);
      if (errc == 0) {
        free(ctens->src_rsc);
        ctens->src_rsc = nullptr;
      }
      break;
    case 't':
      errc = tensDevRsc_clone(ctens->tmp_rsc, dev_rsc);
      if (errc == 0) {
        free(ctens->tmp_rsc);
        ctens->tmp_rsc = nullptr;
      }
      break;
    case 'd':
      errc = tensDevRsc_clone(ctens->dst_rsc, dev_rsc);
      if (errc == 0) {
        free(ctens->dst_rsc);
        ctens->dst_rsc = nullptr;
      }
      break;
    default:
      errc = 2;
    }
  } else {
    errc = 3;
  }
  return errc;
}

int sycl_task_arg_has_resource(cudaTask_t *sycl_task, unsigned int arg_num,
                               char which, int *ierr)
/** Queries the existence of a SYCL task resource for tensor argument <arg_num>.
    <which> selects bewteen 's':source, 't':temporary, 'd':destination
   (resource). **/
{
  int ans;
  tensBlck_t *ctens;

  ans = NOPE;
  *ierr = 0;
  if (sycl_task == nullptr) {
    *ierr = -1;
    return ans;
  }
  if (arg_num >= sycl_task->num_args) {
    *ierr = 1;
    return ans;
  }
  ctens = sycl_task->tens_args[arg_num].tens_p;
  if (ctens == nullptr) {
    *ierr = 2;
    return ans;
  }
  switch (which) {
  case 's':
    if (ctens->src_rsc != nullptr)
      ans = YEP;
    break;
  case 't':
    if (ctens->tmp_rsc != nullptr)
      ans = YEP;
    break;
  case 'd':
    if (ctens->dst_rsc != nullptr)
      ans = YEP;
    break;
  default:
    *ierr = 3;
  }
  return ans;
}

int sycl_task_arg_destroy(cudaTask_t *sycl_task, int arg_num) // internal use only
/** Destroys a specific <tensBlck_t> argument in a SYCL task. If <arg_num> is
   not specified (negative), all arguments of the SYCL task will be destroyed. **/
{
  int i, errc;

  errc = 0;
  if (sycl_task == nullptr)
    return -1;
  if (arg_num >= sycl_task->num_args)
    return 1;
  if (arg_num < 0) { // destroy all tensor arguments
    while (sycl_task->num_args > 0) {
      i = tensBlck_destroy(sycl_task->tens_args[sycl_task->num_args - 1].tens_p);
      if ((i == 0 || i == NOT_CLEAN) && errc == 0) {
        errc = i;
      } else {
        errc = 2;
      }
      sycl_task->tens_args[--(sycl_task->num_args)].tens_p = nullptr;
    }
  } else { // destroy a specific tensor argument
    i = tensBlck_destroy(sycl_task->tens_args[arg_num].tens_p);
    if ((i == 0 || i == NOT_CLEAN) && errc == 0) {
      errc = i;
    } else {
      errc = 3;
    }
    sycl_task->tens_args[arg_num].tens_p = nullptr;
  }
  return errc;
}

float sycl_task_time(const cudaTask_t *sycl_task, float *in_copy,
                     float *out_copy, float *comp, float *mmul)
/** Returns the time (in seconds) the SYCL task took to complete. Also,
    <in_copy> is the input copying time, <out_copy> is the output copying
    time, <comp> is the computing time, and <mmul> is the matrix
    multiplication time in seconds. A negative return value means an error
    occurred. **/
{
  int cur_gpu, errc;
  float time_ms;
  cl::sycl::event *evnt_p;

  if (sycl_task != nullptr) {
    if (sycl_task->task_error < 0)
      return -10.0f; // unfinished or empty task
    if (sycl_task->gpu_id < 0 || sycl_task->gpu_id >= MAX_GPUS_PER_NODE)
      return -9.0f;
    cur_gpu = gpu_in_focus();
    if (cur_gpu < 0 || cur_gpu >= MAX_GPUS_PER_NODE)
      return -8.0f;
    errc = gpu_activate(sycl_task->gpu_id);
    if (errc != 0)
      return -7.0f;

    evnt_p = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_hl);
    if (evnt_p == nullptr)
      return -6.0f;

    *in_copy = -1.0f;
    *comp=-1.0f;
    *out_copy=-1.0f;
    *mmul=-1.0f;

    auto start_time = evnt_p.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
    auto end_time = evnt_p.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    auto execution_time = ((end_time - start_time) / 1.0e+09f); // ns to s
    errc = gpu_activate(cur_gpu);
    return time_ms;
  } else {
    return -13.666f; // null task
  }
}

float sycl_task_time_(const syclTask_t *sycl_task, float *in_copy,
                      float *out_copy, float *comp, float *mmul) {
  return sycl_task_time(sycl_task, in_copy, out_copy, comp, mmul);
}

void sycl_task_print(const syclTask_t *sycl_task)
/** Prints SYCL task info. **/
{
  if (sycl_task != nullptr) {
    printf("\n#MESSAGE: Printing SYCL task info:\n");
    printf(" SYCL task status             : %d\n", sycl_task->task_error);
    printf(" SYCL task GPU id             : %d\n", sycl_task->gpu_id);
    printf(" SYCL task stream handle      : %d\n", sycl_task->queue_hl);
    printf(" SYCL task event handle       : %d\n", sycl_task->event_hl);
    printf(" SYCL task coherence_var      : %u\n", sycl_task->coherence);
    printf(" SYCL task num_args           : %u\n", sycl_task->num_args);
    if (sycl_task->num_args <= MAX_TENSOR_OPERANDS) {
      for (int i = 0; i < sycl_task->num_args; ++i) {
        printf("  Tensor argument #%d address: %p\n", i,
               sycl_task->tens_args[i].tens_p);
        tensBlck_print(sycl_task->tens_args[i].tens_p);
      }
    } else {
      printf(" ERROR: Invalid number of arguments!!!\n");
    }
    printf("#END OF MESSAGE\n");
  } else {
    printf("\n#WARNING(tensor_algebra_gpu_intel:sycl_task_print): nullptr "
           "pointer!\n");
  }
  return;
}

static int sycl_task_set_arg(syclTask_t *sycl_task, unsigned int arg_num, tensBlck_t *tens_p)
/** Sets a specific tensor argument in a SYCL task. The tensor argument is
   associated with the provided tensor block and the required temporary
   multi-index entries are acquired. If the multi-index resources cannot be
   acquired at this time, TRY_LATER is returned. **/
{
  int cae, errc, i;
  unsigned int n;

  if (sycl_task == nullptr)
    return -1;
  if (sycl_task->task_error >= 0 || sycl_task->gpu_id < 0 ||
      sycl_task->gpu_id >= MAX_GPUS_PER_NODE)
    return -2; // finished or empty SYCL task
  if (arg_num >= MAX_TENSOR_OPERANDS)
    return -3; //[0..MAX_TENSOR_OPERANDS-1]
  if (tens_p == nullptr)
    return -4;
  if (gpu_is_mine(sycl_task->gpu_id) > GPU_OFF) {
    // Associate the tensor block:
    sycl_task->tens_args[arg_num].tens_p = tens_p; // no checks, just do it
    // Acquire a multi-index entry in pinned Host memory:
    errc = mi_entry_get(&(sycl_task->tens_args[arg_num].prmn_p));
    if (errc) {
      sycl_task->tens_args[arg_num].prmn_p = nullptr;
      sycl_task->tens_args[arg_num].tens_p = nullptr;
      return TRY_LATER;
    }
    // Acquire a paired multi-index entry in GPU constant memory:
    errc = const_args_entry_get(sycl_task->gpu_id, &cae);
    if (errc == 0) {
      sycl_task->tens_args[arg_num].const_mem_entry = cae;
    } else {
      sycl_task->tens_args[arg_num].prmn_p = nullptr;
      sycl_task->tens_args[arg_num].tens_p = nullptr;
      return TRY_LATER;
    }
    // Update number of arguments:
    sycl_task->num_args = MAX(
        sycl_task->num_args,
        arg_num +
            1); // it is user's responsibility to set all preceding arguments
  } else {
    return 1;
  }
  return 0;
}

static int sycl_task_set_prefactor(syclTask_t *sycl_task, talshComplex4 prefactor)
/** Sets a complex prefactor for the tensor operation in a SYCL task (single precision). **/
{
  int errc;
  void *pref_p;

  if (sycl_task == nullptr)
    return -1;
  if (sycl_task->task_error >= 0 || sycl_task->gpu_id < 0 ||
      sycl_task->gpu_id >= MAX_GPUS_PER_NODE)
    return -2; // finished or empty SYCL task
  errc = slab_entry_get(&prefactors, &pref_p);
  if (errc != 0)
    return -3;
  sycl_task->pref_ptr = pref_p;
  *((talshComplex4 *)(sycl_task->pref_ptr)) = prefactor;
  return 0;
}

static int sycl_task_set_prefactor(syclTask_t *sycl_task, talshComplex8 prefactor)
/** Sets a complex prefactor for the tensor operation in a SYCL task (double precision). **/
{
  int errc;
  void *pref_p;

  if (sycl_task == nullptr)
    return -1;
  if (sycl_task->task_error >= 0 || sycl_task->gpu_id < 0 ||
      sycl_task->gpu_id >= MAX_GPUS_PER_NODE)
    return -2; // finished or empty SYCL task
  errc = slab_entry_get(&prefactors, &pref_p);
  if (errc != 0)
    return -3;
  sycl_task->pref_ptr = pref_p;
  *((talshComplex8 *)(sycl_task->pref_ptr)) = prefactor;
  return 0;
}

static int sycl_task_record(syclTask_t *sycl_task, unsigned int coh_ctrl, unsigned int err_code)
/** Records a scheduled SYCL task. A successfully scheduled SYCL task has
   <err_code>=0, otherwise a positive <err_code> indicates a task scheduling
   failure. In the latter case, the SYCL task will be finalized here. Special
   error code NVTAL_DEFERRED is a non-critical task scheduling failure, not
   considered as an error.  **/
{
  int i, errc;

  if (sycl_task == nullptr)
    return -1;
  if (sycl_task->task_error >= 0)
    return -2; // SYCL task is not clean
  if (sycl_task->gpu_id < 0 || sycl_task->gpu_id >= MAX_GPUS_PER_NODE)
    return -3; // GPU ID is out of range or SYCL task is not clean
  if (sycl_task->num_args == 0 || sycl_task->num_args > MAX_TENSOR_OPERANDS)
    return -4; // no operands associated with the task
  for (i = 0; i < sycl_task->num_args; i++) {
    if (sycl_task->tens_args[i].tens_p == nullptr)
      return -5;
  }                    // all tensor arguments must be set
  if (err_code == 0) { // successfully scheduled SYCL task
    if (gpu_is_mine(sycl_task->gpu_id) > GPU_OFF) {
      sycl_task->task_error = -1;
      sycl_task->coherence = coh_ctrl;
    } else {
      sycl_task->task_error = 13;
      sycl_task->coherence = coh_ctrl; // GPU is not mine
      errc = sycl_task_finalize(sycl_task);
      gpu_stats[sycl_task->gpu_id].tasks_failed++;
    }
  } else { // SYCL task that failed scheduling
    sycl_task->task_error = err_code;
    sycl_task->coherence = coh_ctrl;
    errc = sycl_task_finalize(sycl_task);
    if (err_code == NVTAL_DEFERRED) {
      gpu_stats[sycl_task->gpu_id].tasks_deferred++;
    } else {
      gpu_stats[sycl_task->gpu_id].tasks_failed++;
    }
  }
  return 0;
}

static int sycl_task_finalize(syclTask_t *sycl_task) // do not call this function in tensor operations
/** Releases unneeded (temporary and other) memory resources right after a SYCL
   task has completed or failed. In case the resources cannot be released
   cleanly, it returns NOT_CLEAN just as a warning, but the SYCL task is
   finalized anyway. It also applies the coherence control protocol (for
   successfully completed tasks only). Note that the SYCL task is not destructed
   here, namely SYCL stream/event resources and the .tens_p component of
   .tens_args[] are unmodified (.prmn_p and .const_mem_entry are released). **/
{
  const unsigned int TWO_BITS_SET = 3; // two right bits are set: {0:D,1:M,2:T,3:K}
  unsigned int bts, coh, s_d_same;
  int i, ret_stat, errc;
  syclTensArg_t *tens_arg;

  if (sycl_task == nullptr)
    return -1;
  if (sycl_task->task_error < 0)
    return 1; // unfinished or empty SYCL task cannot be finalized
  if (sycl_task->gpu_id < 0 || sycl_task->gpu_id >= MAX_GPUS_PER_NODE)
    return 2; // invalid GPU id or empty
  if (sycl_task->num_args > MAX_TENSOR_OPERANDS)
    return 3; // invalid number of tensor arguments
  ret_stat = 0;
  coh = sycl_task->coherence;
  // Release resources for tensor arguments:
  for (i = sycl_task->num_args - 1; i >= 0; i--) { // last argument corresponds to the first (minor) two bits
    bts = (coh) & (TWO_BITS_SET);
    tens_arg = &(sycl_task->tens_args[i]);
    if (tens_arg->tens_p != nullptr) { // pointer to the tensor block associated with this argument
      if (tens_arg->tens_p->src_rsc == nullptr)
        return -2; // source must always be present
      if (tens_arg->tens_p->dst_rsc != nullptr) {
        if (tens_arg->tens_p->src_rsc->dev_id == tens_arg->tens_p->dst_rsc->dev_id) {
          s_d_same = YEP;
        } else {
          s_d_same = NOPE;
        };
      } else {
        if (sycl_task->task_error == 0)
          return -3; // destination resource must be present for successfully completed SYCL tasks
        s_d_same = NOPE; // no destination resource (failed SYCL tasks only)
      }
      // Release temporary resources (always):
      if (tens_arg->tens_p->tmp_rsc != nullptr) {
        errc = tensDevRsc_release_all(tens_arg->tens_p->tmp_rsc);
        if (errc) {
          if (VERBOSE)
            printf("#ERROR(INT-TAL:sycl_task_finalize): tmp_rsc resource release error %d\n", errc);
          ret_stat = NOT_CLEAN;
        }
      }
      // Release source/destination resources if needed:
      if (tens_arg->tens_p->dst_rsc == tens_arg->tens_p->src_rsc)
        tens_arg->tens_p->dst_rsc = nullptr;
      if (sycl_task->task_error == 0) { // coherence control for successfully completed SYCL tasks
        if (bts < 2) {
          if (s_d_same == NOPE) {
            errc = tensDevRsc_release_all(tens_arg->tens_p->src_rsc);
            if (errc) {
              if (VERBOSE)
                printf("#ERROR(INT-TAL:sycl_task_finalize): src_rsc resource release error %d\n", errc);
              ret_stat = NOT_CLEAN;
            }
          }
          if (bts == 0 && tens_arg->tens_p->dst_rsc != nullptr) {
            errc = tensDevRsc_release_all(tens_arg->tens_p->dst_rsc);
            if (errc) {
              if (VERBOSE)
                printf("#ERROR(INT-TAL:sycl_task_finalize): dst_rsc resource release error %d\n", errc);
              ret_stat = NOT_CLEAN;
            }
          }
        } else if (bts == 2) {
          if (s_d_same == NOPE && tens_arg->tens_p->dst_rsc != nullptr) {
            errc = tensDevRsc_release_all(tens_arg->tens_p->dst_rsc);
            if (errc) {
              if (VERBOSE)
                printf("#ERROR(INT-TAL:sycl_task_finalize): dst_rsc resource release error %d\n", errc);
              ret_stat = NOT_CLEAN;
            }
          }
        }
      } else { // failed SYCL task
        if (tens_arg->tens_p->dst_rsc != nullptr) {
          errc = tensDevRsc_release_all(tens_arg->tens_p->dst_rsc);
          if (errc) {
            if (VERBOSE)
              printf("#ERROR(INT-TAL:sycl_task_finalize): dst_rsc resource release error %d\n", errc);
            ret_stat = NOT_CLEAN;
          }
        }
      }
      // Release multi-index entries if any:
      if (tens_arg->prmn_p != nullptr) { // if .prmn_p is not from the internal
                                         // pinned slab nothing will be done:
        if (mi_entry_pinned(tens_arg->prmn_p) == YEP) {
          errc = mi_entry_release(tens_arg->prmn_p);
          if (errc) {
            if (VERBOSE)
              printf("#ERROR(INT-TAL:sycl_task_finalize): permutation entry release error %d\n", errc);
            ret_stat = NOT_CLEAN;
          }
          tens_arg->prmn_p = nullptr;
        }
      }
      if (tens_arg->const_mem_entry >= 0) {
        errc =
            const_args_entry_free(sycl_task->gpu_id, tens_arg->const_mem_entry);
        if (errc) {
          if (VERBOSE)
            printf("#ERROR(INT-TAL:sycl_task_finalize): constant memory resource release error %d\n", errc);
          ret_stat = NOT_CLEAN;
        }
        tens_arg->const_mem_entry = 0;
      }
      // printf("\n#DEBUG(INT-TAL::sycl_task_finalize): tensBlck_t argument %d
      // end state:\n",i); tensBlck_print(tens_arg->tens_p); //debug
    } else {
      if (sycl_task->task_error == 0)
        return -4; // successfully completed SYCL tasks must have all tensor arguments associated
    }
    coh = coh >> 2; // select the 2-bits for the next argument
  }
  // Release prefactor resource, if needed:
  if (sycl_task->pref_ptr != nullptr) {
    errc = slab_entry_release(&prefactors, sycl_task->pref_ptr);
    if (errc) {
      if (VERBOSE)
        printf("#ERROR(INT-TAL:sycl_task_finalize): prefactor release error %d\n", errc);
      ret_stat = NOT_CLEAN;
    }
    sycl_task->pref_ptr = nullptr;
  }
  return ret_stat;
}
//-------------------------------------------------
// EXPORTED FUNCTIONS (callable from C/C++/Fortran):
//---------------------------------------------------------------------------
// MATRIX MULTIPLICATION 'TN' (blocking, slow):
template <typename T>
int gpu_matrix_multiply_tn(size_t ll, size_t lr, size_t lc, const T *lmat, const T *rmat, T *dmat)
/** dmat(0:ll-1,0:lr-1)+=lmat(0:lc-1,0:ll-1)*rmat(0:lc-1,0:lr-1)
    All matrices are in Host memory. Executed on the currently set GPU device. **/
  try {
    size_t dsize, lsize, rsize;
    T *dptr, *lptr, *rptr;
    int bx, by, err_code;
    const char *err_msg;

    if (lc > 0 && ll > 0 && lr > 0 && lmat != nullptr && rmat != nullptr && dmat != nullptr) {
      dsize = ll * lr * sizeof(T);
      lsize = lc * ll * sizeof(T);
      rsize = lc * lr * sizeof(T);
      err_code = gpu_mem_alloc((void **)&dptr, dsize);
      if (err_code != 0)
        return 1;
      err_code = gpu_mem_alloc((void **)&lptr, lsize);
      if (err_code != 0)
        return 2;
      err_code = gpu_mem_alloc((void **)&rptr, rsize);
      if (err_code != 0)
        return 3;
      talsh::get_queue().memcpy((void *)dptr, (void *)dmat, dsize).wait();
      talsh::get_queue().memcpy((void *)lptr, (void *)lmat, lsize).wait();
      talsh::get_queue().memcpy((void *)rptr, (void *)rmat, rsize).wait();
      err_code = gpu_get_error_count();
      bx = 1 + (ll - 1) / MAT_MULT_TILE_DIMX;
      by = 1 + (lr - 1) / MAT_MULT_TILE_DIMY;
      limit_sycl_workgroups2d(MAX_SYCL_BLOCKS, &bx, &by);
      cl::sycl::range<2> blcks(bx, by);
      cl::sycl::range<2> thrds(MAT_MULT_TILE_DIMX, MAT_MULT_TILE_DIMY);

      talsh::get_queue().submit([&](cl::sycl::handler &cgh) {
        auto gpu_error_count_ptr = gpu_error_count.get_ptr();

        cl::sycl::range<2> buf1_range(17 /*MAT_MULT_TILE_DIMX+1*/,
                                      17 /*MAT_MULT_TILE_DIMX+1*/);
        cl::sycl::range<2> buf2_range(17 /*MAT_MULT_TILE_DIMY+1*/,
                                      17 /*MAT_MULT_TILE_DIMX+1*/);
        local_accessor<T, 2> buf1_acc(buf1_range, cgh);
        local_accessor<T, 2> buf2_acc(buf2_range, cgh);
        auto global_range = blcks * thrds;

        cgh.parallel_for(cl::sycl::nd_range<2>(global_range, thrds),
                         [=](cl::sycl::nd_item<2> item) {
                           gpu_matrix_multiply_tn__(
                               ll, lr, lc, lptr, rptr, dptr, (T)(1.0), item,
                               gpu_error_count_ptr, buf1_acc, buf2_acc);
                         });
      });
      talsh::get_current_device().queues_wait_and_throw();
      if (gpu_get_error_count() > err_code)
        return 9;
      talsh::get_queue().memcpy((void *)dmat, (void *)dptr, dsize).wait();
      talsh::get_current_device().queues_wait_and_throw();
      err_code = gpu_mem_free((void *)rptr);
      if (err_code != 0)
        return 12;
      err_code = gpu_mem_free((void *)lptr);
      if (err_code != 0)
        return 13;
      err_code = gpu_mem_free((void *)dptr);
      if (err_code != 0)
        return 14;
      talsh::get_current_device().queues_wait_and_throw();
    } else {
      return 16;
    }

    return 0;
  } catch (cl::sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file: " << __FILE__
	      << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

//-----------------------------------------------------------------------------------------------------------------------------
// TENSOR BODY CLONING (non-blocking):
int gpu_tensor_block_place(tensBlck_t *ctens, int gpu_id, unsigned int coh_ctrl,
                           cudaTask_t *sycl_task, void *dev_mem)
/** Copies/moves the tensor body to a different GPU (gpu_id >= 0) or Host
    (gpu_id < 0). If <dev_mem> is a valid target device memory pointer, it
    will be used for storage, otherwise buffer memory will be allocated. A
    non-zero return status indicates an error. If the error code is negative,
    the SYCL task was not recorded. For positive error codes, the SYCL task
    was recorded. If the source device where the tensor body resides
    coincides with the destination device, no transfer will be scheduled.
    The source tensor body must reside either on Host or on Nvidia GPU. **/
{
    int j, tds, gpu_ex, src_gpu, devk, cur_gpu, devid, nclean, errc;
    size_t tvol, tsize;
    cl::sycl::queue **sycl_queue;
    cl::sycl::event *sycl_event;
    // sycl::event *sycl_start, *sycl_comput, *sycl_output, *sycl_finish,
    // *dep_event; std::chrono::time_point<std::chrono::high_resolution_clock>
    // sycl_start_ct;
    const char *err_msg;

    errc = 0;
    nclean = 0;
    // Argument check:
    if (ctens == nullptr)
      return -1;
    if (sycl_task == nullptr)
      return -2;
    if (sycl_task_gpu_id(sycl_task) >= 0)
      return -3; // SYCL task is not clean (destruct/clean it first)
    if (tens_valid_data_kind(ctens->data_kind, &tds) != YEP)
      return -4;
    if (tensBlck_present(ctens, DEV_NULL, DEV_INTEL_GPU) == YEP ||
        tensBlck_present(ctens, DEV_NULL, DEV_HOST) == YEP) {
      // Determine the id of the transfer executing GPU:
      cur_gpu = gpu_in_focus();                    // save the current GPU
      gpu_ex = DEV_NULL;                           // executing GPU
      src_gpu = tensBlck_src_dev_id(ctens, &devk); // source GPU (or Host)
      if (devk == DEV_HOST) {
        src_gpu = DEV_NULL;
      } else {
        if (devk != DEV_INTEL_GPU)
          return -5;
      } // src_gpu: source GPU (-1:Host)
      if (gpu_id >= 0 && gpu_id < MAX_GPUS_PER_NODE) { // destination is a GPU
        gpu_ex = gpu_id;
        if (gpu_is_mine(gpu_ex) <= GPU_OFF)
          return -6;
      } else if (gpu_id < 0) { // destination is Host
        if (src_gpu >= 0) {
          gpu_ex = src_gpu;
          if (gpu_is_mine(gpu_ex) <= GPU_OFF)
            return -7;
        }
      } else {
        return -8; // invalid gpu_id
      }
      // Construct the SYCL task:
      if (gpu_ex < 0) { // Host-to-self transfer requested (no transfer)
        errc = sycl_task_construct(sycl_task);
      } else { // Host-to-GPU, GPU-to-Host, GPU-to-GPU
        gpu_stats[gpu_ex].tasks_submitted++;
        // Check peer access if appropriate:
        if (src_gpu >= 0 && src_gpu != gpu_ex)
          return DEVICE_UNABLE; // peer access impossible for this GPU device
      }
      // Activate the transfer executing GPU:
      if (gpu_ex != cur_gpu) {
        j = gpu_activate(gpu_ex);
        if (j) {
          j = gpu_activate(cur_gpu);
          return -9;
        }
      } // activate the target GPU
      if (err != 0) {
        ++nclean;
        err = 0; // clear the GPU error status (sets NOT_CLEAN on exit)
      }
      errc = sycl_task_construct(sycl_task, gpu_ex);
      if (errc)
        j = gpu_activate(cur_gpu);
    }
  if (errc) {
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      return errc;
    } else {
      return -10;
    }
  }

  // *** From this point all error codes must be positive and the SYCL task must
  // be recorded! *** Set the SYCL task argument(s):
  errc = sycl_task_set_arg(sycl_task, 0, ctens);
  if (errc) {
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      j = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
      j = gpu_activate(
          cur_gpu); // not an error if TRY_LATER or DEVICE_UNABLE are returned
      return errc;
    } else {
      j = sycl_task_record(sycl_task, coh_ctrl, 1);
      j = gpu_activate(cur_gpu);
      return 1;
    }
  }
  // Determine the volume/size of the tensor block:
  tvol = tensBlck_volume(ctens);
  tsize = tvol * tds;
  if (tvol == 0) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 2);
    errc = gpu_activate(cur_gpu);
    return 2;
  }
  // Associate SYCL queue and event pointers locally for convenience:
  sycl_queue = sycl_queue_ptr(sycl_task->gpu_id, sycl_task->queue_hl);
  if (sycl_queue == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 3);
    errc = gpu_activate(cur_gpu);
    return 3;
  }
  sycl_event = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_hl);
  if (sycl_event == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 4);
    errc = gpu_activate(cur_gpu);
    return 4;
  }

  // Acquire global memory resources (destination resource):
  if (gpu_id >= 0) {
    devid = encode_device_id(DEV_INTEL_GPU, gpu_id);
  } else {
    devid = encode_device_id(DEV_HOST, 0);
  } // flat device id of the destination
  if (ctens->dst_rsc == ctens->src_rsc)
    ctens->dst_rsc = nullptr;
  if (gpu_ex >= 0 &&
      gpu_id != src_gpu) { // data is on a different GPU device or Host
    if (ctens->dst_rsc == nullptr) {
      errc = tensDevRsc_create(&(ctens->dst_rsc));
      if (errc) {
        j = sycl_task_record(sycl_task, coh_ctrl, 8);
        j = gpu_activate(cur_gpu);
        return 8;
      }
    } else {
      if (tensDevRsc_is_empty(ctens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(ctens->dst_rsc);
        if (errc)
          ++nclean;
      }
    }
    if (dev_mem == nullptr) {
      errc = tensDevRsc_allocate_mem(
          ctens->dst_rsc, devid, tsize,
          YEP); // device memory is allocated in the device argument buffer
    } else {
      errc = tensDevRsc_attach_mem(ctens->dst_rsc, devid,
                                   dev_mem); // externally provided device
                                             // memory will be used for storage
    }
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        j = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        j = gpu_activate(cur_gpu);
        return errc;
      } else {
        j = sycl_task_record(sycl_task, coh_ctrl, 9);
        j = gpu_activate(cur_gpu);
        return 9;
      }
    }
  } else {
    if (ctens->dst_rsc != nullptr) {
      if (tensDevRsc_is_empty(ctens->dst_rsc) == NOPE) {
        j = tensDevRsc_release_all(ctens->dst_rsc);
        if (j)
          ++nclean;
      }
    }
    ctens->dst_rsc = ctens->src_rsc; // destination and source resources are the same (because
    // the data is already on the executing GPU or Host)
  }
  // Record the start event:
  cuda_start_ct = std::chrono::high_resolution_clock::now();
  // Schedule the data transfer:
  if (gpu_ex >= 0 && gpu_id != src_gpu) {
    // Make sure the data transfer does not begin before the data transfer from
    // the previous task has finished:
    if (LastTask[gpu_ex] != nullptr) { //`This should be done atomically for thread safety
      dep_event = sycl_event_ptr(LastTask[gpu_ex]->gpu_id, LastTask[gpu_ex]->event_comput_hl);
      err = (*dep_event.wait(),
             0); // input transfers should only begin after the previous task
                 // input transfers have completed
    }
    // Transfer:
    (*sycl_queue)
        ->memcpy(ctens->dst_rsc->gmem_p, ctens->src_rsc->gmem_p, tsize);
    if (gpu_id >= 0) { // incoming traffic
      gpu_stats[gpu_ex].traffic_in += tsize;
    } else { // outgoing traffic (to Host)
      gpu_stats[gpu_ex].traffic_out += tsize;
    }
  }
  // Record other events:
  cuda_finish_ct = std::chrono::high_resolution_clock::now();
  // Record the successfully scheduled SYCL task, update the Last Task, and
  // restore the original GPU:
  errc = sycl_task_record(sycl_task, coh_ctrl, 0);
  if (gpu_ex >= 0 && gpu_ex != src_gpu)
    LastTask[gpu_ex] = sycl_task;
  if (gpu_ex >= 0 && gpu_ex != cur_gpu)
    j = gpu_activate(cur_gpu);
}
else {
  return -11; // tensor block is neither present on Host nor on any Nvidia GPU
}
if (nclean > 0 && errc == 0)
  errc = NOT_CLEAN;
return errc;
}

//------------------------------------------------------------------------------------------
// TENSOR INITIALIZATION (non-blocking):
int gpu_tensor_block_init(tensBlck_t *dtens, double val_real, double val_imag,
                          unsigned int coh_ctrl, cudaTask_t *sycl_task, int gpu_id)
/**
   dtens(:)=scalar_value
   INPUT:
   # (val_real,val_imag) - initialization value;
   # coh_ctrl - one of the COPY_X parameters regulating the data presence
   for each tensor argument; # sycl_task - pointer to an empty (clean) CUDA
   task; # gpu_id - suggested GPU ID on which the operation is to be scheduled
   (-1: defaults to the optimal one); OUTPUT: # dtens - initialized destination
   tensor; # sycl_task - recorded SYCL task (either successfully scheduled or
   failed). NOTES: # If the tensor operation has been scheduled successfully, a
   recorded (active) SYCL task will be returned along with zero return status.
   A scheduling error results in either a negative (at early stages) or
   positive (at later stages) return status. In the former case the SYCL task
   is left clean, while at the latter case it will be recorded as failed
   (error). # Special return statuses TRY_LATER and DEVICE_UNABLE are not
   errors but merely indicators of the current or permanent lack of resources,
   respectively. However, the SYCL task status in these cases will still be set
   to an error (always check the function return status!). # If <gpu_id> is out
   of the legitimate GPU range, it will be replaced by an optimal one, based on
   argument residence and the current load of GPU(s).
**/
  try {
    int i, j, drank, tds_d, gpu_d, gpu_num, cur_gpu, targ_dev, bx, errc, stat;
    size_t vol_d, dsize;
    unsigned int coh;
    const unsigned int TWO_BITS_SET = 3; // two right bits are set
    void *darg;
    float fval;
    sycl::queue **sycl_queue;
    sycl::event *dep_event;
    const char *err_msg;

    // if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_intel:gpu_tensor_block_init):
    // GPU Tensor Initialization:\n"); //debug
    stat = 0; // return status in case of successful scheduling
    // Check function arguments:
    if (dtens == nullptr || sycl_task == nullptr)
      return -1;
    if (tensBlck_present(dtens) != YEP)
      return -2; // tensor block must reside in some device memory
    if (sycl_task_gpu_id(sycl_task) >= 0)
      return -3; // SYCL task is not clean (destruct/clean it first)
    // Check tensor arguments:
    drank = (dtens->shape).num_dim; // destination tensor rank
    if (drank < 0 || drank > MAX_TENSOR_RANK)
      return -4;
    if (tens_valid_data_kind(dtens->data_kind, &tds_d) != YEP)
      return -5; // tds_d: destination tensor element size in bytes
    if (dtens->data_kind <= 0)
      return -6; // tensor must have been previsously allocated with a certain
    // data kind
    if (dtens->src_rsc == nullptr)
      return -7; // source resource must always be present
    if (tensDevRsc_is_empty(dtens->src_rsc) != NOPE)
      return -8; // source resource must be present (tensor body)
    // Activate the right GPU:
    if (gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE) {
      gpu_num = tens_op_best_gpu(dtens);
    } else {
      gpu_num = gpu_id;
    }
    if (gpu_is_mine(gpu_num) <= GPU_OFF)
      return -28; // GPU is not mine or error
    gpu_stats[gpu_num].tasks_submitted++;
    gpu_d = decode_device_id(dtens->src_rsc->dev_id, &j);
    if (gpu_d < 0)
      return -29; // destination tensor source device id
    if (j == DEV_INTEL_GPU) {
      if (gpu_d != gpu_num) {
        // todo: cudaDeviceCanAccessPeer ????
        err = (*&j = 0, 0);
        if (err != 0 || j == 0)
          return DEVICE_UNABLE; // peer access impossible for this GPU device
      }
    } else if (j == DEV_HOST) {
      gpu_d = -1; // data is in Host memory
    } else {
      return DEVICE_UNABLE; // data is not in Host or GPU memory
    }
    cur_gpu = gpu_in_focus(); // save the current GPU
    if (gpu_num != cur_gpu) {
      errc = gpu_activate(gpu_num);
      if (errc) {
        errc = gpu_activate(cur_gpu);
        return -32;
      }
    } // activate the target GPU
    err = 0;
    err = 0; // clear the GPU error status
    targ_dev = encode_device_id(DEV_INTEL_GPU, gpu_num); // flat device id
    // Construct a SYCL task (acquire CUDA resources) for the target GPU:
    errc = sycl_task_construct(sycl_task, gpu_num);
    if (errc) {
      i = gpu_activate(cur_gpu);
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        return errc;
      } else {
        return -33;
      }
    }

    // *** From this point all error codes must be positive and the SYCL task
    // must be recorded! *** Set up tensor arguments (allocates additional
    // resources for each tensor argument): Destination argument:
    errc = sycl_task_set_arg(sycl_task, 0, dtens);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(
            cur_gpu); // not an error if TRY_LATER or DEVICE_UNABLE are returned
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 1);
        i = gpu_activate(cur_gpu);
        return 1;
      }
    }

    vol_d = tensBlck_volume(dtens); // tensor block volume
    dsize = vol_d * tds_d;          // tensor argument size in bytes
    // Acquire global memory resources for tensor arguments if needed:
    // Set up destination memory resources in all tensors:
    // Destination tensor:
    if (dtens->dst_rsc == dtens->src_rsc)
      dtens->dst_rsc =
          nullptr; // destination resource was pointing to the source resource
    if (gpu_d != gpu_num) { // data is on a different GPU device or Host
      if (dtens->dst_rsc == nullptr) {
        errc = tensDevRsc_create(&(dtens->dst_rsc));
        if (errc) {
          i = sycl_task_record(sycl_task, coh_ctrl, 11);
          i = gpu_activate(cur_gpu);
          return 11;
        }
      } else {
        if (tensDevRsc_is_empty(dtens->dst_rsc) == NOPE) {
          errc = tensDevRsc_release_all(dtens->dst_rsc);
          if (errc)
            stat = NOT_CLEAN;
        }
      }
      errc = tensDevRsc_allocate_mem(dtens->dst_rsc, targ_dev, dsize, YEP);
      if (errc) {
        if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
          i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
          i = gpu_activate(cur_gpu);
          return errc;
        } else {
          i = sycl_task_record(sycl_task, coh_ctrl, 12);
          i = gpu_activate(cur_gpu);
          return 12;
        }
      }
    } else {
      if (dtens->dst_rsc != nullptr) {
        if (tensDevRsc_is_empty(dtens->dst_rsc) == NOPE) {
          errc = tensDevRsc_release_all(dtens->dst_rsc);
          if (errc)
            stat = NOT_CLEAN;
        }
      }
      dtens->dst_rsc =
          dtens->src_rsc; // destination and source resources are the same
      // (because the data is already on the computing GPU)
    }
    // Start scheduling SYCL calls:
    if (LastTask[gpu_num] !=
        nullptr) { //`This should be done atomically for thread safety
      dep_event = sycl_event_ptr(LastTask[gpu_num]->gpu_id,
                                 LastTask[gpu_num]->event_comput_hl);
      try {
        dep_event
            ->wait_and_throw(); // input transfers should only begin after the
                                // previous task input transfers have completed
      } catch (cl::sycl::exception const &exc) {
        if (VERBOSE)
          std::cerr << "#ERROR: Unable to create a task dependency: "
                    << exc.what() << ", at " << __FILE__
                    << ", line:" << __LINE__ << std::endl;
        errc = sycl_task_record(sycl_task, coh_ctrl, 24);
        errc = gpu_activate(cur_gpu);
        return 24;
      }
    }
    // Schedule forward data transfers for all tensors if needed:
    // Destination tensor:
    if (sycl_task->tens_args[0].const_mem_entry >= 0) {
      // GPU constant memory entry will contain tensor dimension extents and the
      // matricization permutation (if any)
      try {
        (*sycl_queue)
            ->memcpy(
                (char *)(const_args_dims.get_ptr()) +
                    sizeof(int) *
                        ((size_t)(MAX_TENSOR_RANK *
                                  (sycl_task->tens_args[0].const_mem_entry))),
                (void *)(dtens->shape.dims),
                sizeof(int) * ((size_t)drank)); // tensor dimension extents
      } catch (cl::sycl::exception const &exc) {
        if (VERBOSE)
          std::cerr << "#ERROR: Destination tensor dims H2D copy failed: "
                    << exc.what() << ", at " << __FILE__
                    << ", line:" << __LINE__ << std::endl;
        errc = sycl_task_record(sycl_task, coh_ctrl, 33);
        errc = gpu_activate(cur_gpu);
        return 33;
      }
      gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)drank);
      if (gpu_d != gpu_num) { // data is not on the computing GPU
        try {
          (*sycl_queue)
              ->memcpy(dtens->dst_rsc->gmem_p, dtens->src_rsc->gmem_p, dsize);
        } catch (cl::sycl::exception const &exc) {
          if (VERBOSE)
            std::cerr << "#ERROR: Destination tensor body copy failed: "
                      << exc.what() << ", at " << __FILE__
                      << ", line:" << __LINE__ << std::endl;
          errc = sycl_task_record(sycl_task, coh_ctrl, 35);
          errc = gpu_activate(cur_gpu);
          return 35;
        }
        gpu_stats[gpu_num].traffic_in += dsize;
      }
    } else {
      errc = sycl_task_record(sycl_task, coh_ctrl, 36);
      errc = gpu_activate(cur_gpu);
      return 36;
    }

    // Destination tensor argument does not need transposing:
    darg = dtens->dst_rsc->gmem_p;

    // Initialization kernel:
    bx = 1 + (vol_d - 1) / THRDS_ARRAY_INIT;
    if (bx > MAX_SYCL_BLOCKS)
      bx = MAX_SYCL_BLOCKS;
    switch (dtens->data_kind) {
    case R4:
      fval = (float)val_real;
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_INIT, THRDS_ARRAY_INIT),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_init__(vol_d, (float *)darg, fval, item);
            });
      });
      break;
    case R8:
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_INIT, THRDS_ARRAY_INIT),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_init__(vol_d, (double *)darg, val_real, item);
            });
      });
      break;
    case C4:
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_INIT, THRDS_ARRAY_INIT),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_init__(
                  vol_d, (talshComplex4 *)darg,
                  talshComplex4Set((float)val_real, (float)val_imag), item);
            });
      });
      break;
    case C8:
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_INIT, THRDS_ARRAY_INIT),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_init__(vol_d, (talshComplex8 *)darg,
                               talshComplex8Set(val_real, val_imag), item);
            });
      });
      break;
    default:
      errc = sycl_task_record(sycl_task, coh_ctrl, 48);
      errc = gpu_activate(cur_gpu);
      return 48;
    }

    // Transfer back the updated destination tensor if needed ("T","K" coherence
    // control):
    coh = (coh_ctrl) &
          (TWO_BITS_SET); // select bits 0,1 (destination tensor coherence)
    if (gpu_d != gpu_num && coh >= 2) { // data is not on the computing GPU and
                                        // coherence control = 2("T") or (3)"K":
      try {
        (*sycl_queue)
            ->memcpy(dtens->src_rsc->gmem_p, dtens->dst_rsc->gmem_p, dsize);
      } catch (cl::sycl::exception const &exc) {
        if (VERBOSE)
          std::cerr << "#ERROR: Destination tensor body back copy failed: "
                    << exc.what() << ", at " << __FILE__
                    << ", line:" << __LINE__ << std::endl;
        errc = sycl_task_record(sycl_task, coh_ctrl, 57);
        errc = gpu_activate(cur_gpu);
        return 57;
      }
      gpu_stats[gpu_num].traffic_out += dsize;
    }

    // Record the successfully scheduled SYCL task and update the Last Task:
    errc = sycl_task_record(sycl_task, coh_ctrl, 0);
    LastTask[gpu_num] = sycl_task;
    if (gpu_num != cur_gpu)
      errc = gpu_activate(cur_gpu);
    return stat; // either 0 (success) or NOT_CLEAN (warning)
} catch (cl::sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//-------------------------------------------------------------------------------------------------------------
// TENSOR SLICING (non-blocking):
int gpu_tensor_block_slice(tensBlck_t *ltens, tensBlck_t *dtens,
                           const int *offsets, unsigned int coh_ctrl,
                           cudaTask_t *sycl_task, int gpu_id,
                           int accumulative) {
  //`Implement
  printf("\n#FATAL(tensor_algebra_gpu_intel:gpu_tensor_block_slice): Operation "
         "not implemented!\n");
  return -1;
}
//--------------------------------------------------------------------------------------------------------------
// TENSOR INSERTION (non-blocking):
int gpu_tensor_block_insert(tensBlck_t *ltens, tensBlck_t *dtens,
                            const int *offsets, unsigned int coh_ctrl,
                            cudaTask_t *sycl_task, int gpu_id,
                            int accumulative) {
  //`Implement
  printf("\n#FATAL(tensor_algebra_gpu_intel:gpu_tensor_block_insert): "
         "Operation not implemented!\n");
  return -1;
}
//---------------------------------------------------------------------------------------------------------------
// TENSOR COPY/PERMUTATION (non-blocking):
int gpu_tensor_block_copy(const int *cptrn, tensBlck_t *ltens,
                          tensBlck_t *dtens, unsigned int coh_ctrl,
                          cudaTask_t *sycl_task, int gpu_id, int conj_bits)
    /**
       dtens(:)=ltens(:permuted)
       INPUT:
       # cptrn(1:lrank) - permutation pattern (O2N): Position correspondence:
       Uncontracted indices are positive, no contracted indices;
       # ltens - left tensor argument (initialized!);
       # dtens - destination tensor argument;
       # coh_ctrl - one of the COPY_XX parameters regulating the data presence
       for each tensor argument; # sycl_task - pointer to an empty (clean) CUDA
       task; # gpu_id - suggested GPU ID on which the operation is to be
    scheduled
       (-1: defaults to the optimal one); # conj_bits - tensor argument complex
       conjugation bits, one bit per argument: {0:D,1:L}; OUTPUT: # dtens -
    updated destination tensor; # sycl_task - recorded SYCL task (either
    successfully scheduled or failed). NOTES: # If the tensor operation has been
    scheduled successfully, a recorded (active) SYCL task will be returned along
    with zero return status. A scheduling error results in either a negative (at
    early stages) or positive (at later stages) return status. In the former
    case the SYCL task is left clean, while at the latter case it will be
    recorded as failed (error). # Special return statuses TRY_LATER and
    DEVICE_UNABLE are not errors but merely indicators of the current or
    permanent lack of resources, respectively. However, the SYCL task status in
    these cases will still be set to an error (always check the function return
    status!). # If <gpu_id> is out of the legitimate GPU range, it will be
    replaced by an optimal one, based on argument residence and the current load
    of GPU(s).
    **/
    try {
  int i, j, drank, lrank, tds_d, tds_l, gpu_d, gpu_l, gpu_num, cur_gpu,
      targ_dev, bx, errc, stat, conj_l;
  int lprm[1 + MAX_TENSOR_RANK];
  size_t vol_d, vol_l, dsize, lsize;
  unsigned int coh;
  const unsigned int TWO_BITS_SET = 3; // two right bits are set
  void *darg, *larg;
  cl::sycl::queue **sycl_queue;
  cl::sycl::event *cuda_start, *cuda_comput, *cuda_output, *cuda_finish,
      *dep_event;
  std::chrono::time_point<std::chrono::high_resolution_clock> cuda_start_ct;
  std::chrono::time_point<std::chrono::high_resolution_clock> cuda_comput_ct;
  std::chrono::time_point<std::chrono::high_resolution_clock> cuda_output_ct;
  std::chrono::time_point<std::chrono::high_resolution_clock> cuda_finish_ct;
  int err;

  // if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_intel:gpu_tensor_block_copy):
  // GPU Tensor Copy:\n"); //debug
  stat = 0; // return status in case of successful scheduling
  // Check function arguments:
  if (cptrn == nullptr || dtens == nullptr || ltens == nullptr ||
      sycl_task == nullptr)
    return -1;
  if (tensBlck_present(dtens) != YEP || tensBlck_present(ltens) != YEP)
    return -2; // tensor blocks must reside in some device memory
  if (sycl_task_gpu_id(sycl_task) >= 0)
    return -3; // SYCL task is not clean (destruct/clean it first)
  // Check tensor arguments:
  drank = (dtens->shape).num_dim; // destination tensor rank
  lrank = (ltens->shape).num_dim; // left tensor rank
  if (drank < 0 || drank > MAX_TENSOR_RANK || lrank < 0 ||
      lrank > MAX_TENSOR_RANK)
    return -4;
  if (tens_valid_data_kind(dtens->data_kind, &tds_d) !=
          YEP || // tds_d: destination tensor element size in bytes
      tens_valid_data_kind(ltens->data_kind, &tds_l) != YEP)
    return -5; // tds_l: left tensor element size in bytes
  if (!(dtens->data_kind > 0 && ltens->data_kind == dtens->data_kind))
    return -6; // data kind mismatch
  if (dtens->src_rsc == nullptr || ltens->src_rsc == nullptr)
    return -7; // source resource must always be present
  if (tensDevRsc_is_empty(dtens->src_rsc) != NOPE)
    return -8; // source resource must be present (tensor body)
  if (tensDevRsc_is_empty(ltens->src_rsc) != NOPE)
    return -9; // source resource must be present (tensor body)
  // Check the contraction pattern and dimension extent correspondence:
  for (i = 0; i < drank; i++)
    lprm[i] = 0;
  for (i = 0; i < lrank; i++) { // position in ltens
    j = cptrn[i];
    if (j > 0) { // position in dtens
      if (j > drank)
        return -11;
      if ((dtens->shape).dims[j - 1] != (ltens->shape).dims[i])
        return -12;
      if (lprm[j - 1] == 0) {
        lprm[j - 1] = 1;
      } else {
        return -13;
      }
    } else {
      return -18;
    }
  }
  for (i = 0; i < drank; i++)
    if (lprm[i] != 1)
      return -27;
  // Check argument complex conjugation bits:
  conj_bits = conj_bits &
              3; // keep only first two bits, one per tensor argument {0:D,1:L}
  if (conj_bits & 1) { // destination tensor argument conjugation = inverse
    // conjugation of the left argument
    conj_bits = conj_bits ^ 3; // XOR with 0b11 will invert bits
  }
  conj_l = 0;
  if ((conj_bits & 2) != 0)
    conj_l = 1; // left argument conjugation flag
  // Activate the right GPU:
  if (gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE) {
    gpu_num = tens_op_best_gpu(dtens, ltens);
  } else {
    gpu_num = gpu_id;
  }
  if (gpu_is_mine(gpu_num) <= GPU_OFF)
    return -28; // GPU is not mine or error
  gpu_stats[gpu_num].tasks_submitted++;
  gpu_d = decode_device_id(dtens->src_rsc->dev_id, &j);
  if (gpu_d < 0)
    return -29; // destination tensor source device id
  if (j == DEV_INTEL_GPU) {
    if (gpu_d != gpu_num) {
      // todo: peer access ???
      err = (*&j = 0, 0);
      if (err != 0 || j == 0)
        return DEVICE_UNABLE; // peer access impossible for this GPU device
    }
  } else if (j == DEV_HOST) {
    gpu_d = -1; // data is in Host memory
  } else {
    return DEVICE_UNABLE; // data is not in Host or GPU memory
  }
  gpu_l = decode_device_id(ltens->src_rsc->dev_id, &j);
  if (gpu_l < 0)
    return -30; // left tensor source device id
  if (j == DEV_INTEL_GPU) {
    if (gpu_l != gpu_num) {
      // todo: peer access ???
      err = (*&j = 0, 0);
      if (err != 0 || j == 0)
        return DEVICE_UNABLE; // peer access impossible for this GPU device
    }
  } else if (j == DEV_HOST) {
    gpu_l = -1; // data is in Host memory
  } else {
    return DEVICE_UNABLE; // data is not in Host or GPU memory
  }
  cur_gpu = gpu_in_focus(); // save the current GPU
  if (gpu_num != cur_gpu) {
    errc = gpu_activate(gpu_num);
    if (errc) {
      errc = gpu_activate(cur_gpu);
      return -32;
    }
  } // activate the target GPU

  targ_dev = encode_device_id(DEV_INTEL_GPU, gpu_num); // flat device id
  // Construct a SYCL task (acquire CUDA resources) for the target GPU:
  errc = sycl_task_construct(sycl_task, gpu_num);
  if (errc) {
    i = gpu_activate(cur_gpu);
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      return errc;
    } else {
      return -33;
    }
  }

  // *** From this point all error codes must be positive and the SYCL task must
  // be recorded! *** Set up tensor arguments (allocates additional resources
  // for each tensor argument): Destination argument:
  errc = sycl_task_set_arg(sycl_task, 0, dtens);
  if (errc) {
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
      i = gpu_activate(
          cur_gpu); // not an error if TRY_LATER or DEVICE_UNABLE are returned
      return errc;
    } else {
      i = sycl_task_record(sycl_task, coh_ctrl, 1);
      i = gpu_activate(cur_gpu);
      return 1;
    }
  }
  // Left argument:
  errc = sycl_task_set_arg(sycl_task, 1, ltens);
  if (errc) {
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
      i = gpu_activate(
          cur_gpu); // not an error if TRY_LATER or DEVICE_UNABLE are returned
      return errc;
    } else {
      i = sycl_task_record(sycl_task, coh_ctrl, 2);
      i = gpu_activate(cur_gpu);
      return 2;
    }
  }
  // Associate SYCL queue and event pointers locally for convenience:
  sycl_queue = sycl_queue_ptr(sycl_task->gpu_id, sycl_task->queue_hl);
  if (sycl_queue == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 4);
    errc = gpu_activate(cur_gpu);
    return 4;
  }
  cuda_start = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_start_hl);
  if (cuda_start == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 5);
    errc = gpu_activate(cur_gpu);
    return 5;
  }
  cuda_comput = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_comput_hl);
  if (cuda_comput == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 6);
    errc = gpu_activate(cur_gpu);
    return 6;
  }
  cuda_output = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_output_hl);
  if (cuda_output == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 7);
    errc = gpu_activate(cur_gpu);
    return 7;
  }
  cuda_finish = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_finish_hl);
  if (cuda_finish == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 8);
    errc = gpu_activate(cur_gpu);
    return 8;
  }
  // Determine the volume and required matricization permutation for each tensor
  // argument:
  for (i = 0; i < drank; i++)
    sycl_task->tens_args[0].prmn_p[i] = (1 + i); // trivial permutation
  for (i = 0; i < lrank; i++)
    sycl_task->tens_args[1].prmn_p[i] = cptrn[i]; // required O2N permutation
  vol_d = tensBlck_volume(dtens);
  vol_l = tensBlck_volume(ltens); // tensor block volumes
  dsize = vol_d * tds_d;
  lsize = vol_l * tds_l; // tensor argument sizes in bytes
  // Acquire global memory resources for tensor arguments if needed:
  // Set up destination memory resources in all tensors:
  //  Destination tensor:
  if (dtens->dst_rsc == dtens->src_rsc)
    dtens->dst_rsc =
        nullptr; // destination resource was pointing to the source resource
  if (gpu_d != gpu_num) { // data is on a different GPU device or Host
    if (dtens->dst_rsc == nullptr) {
      errc = tensDevRsc_create(&(dtens->dst_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 11);
        i = gpu_activate(cur_gpu);
        return 11;
      }
    } else {
      if (tensDevRsc_is_empty(dtens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(dtens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(dtens->dst_rsc, targ_dev, dsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 12);
        i = gpu_activate(cur_gpu);
        return 12;
      }
    }
  } else {
    if (dtens->dst_rsc != nullptr) {
      if (tensDevRsc_is_empty(dtens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(dtens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    dtens->dst_rsc =
        dtens->src_rsc; // destination and source resources are the same
                        // (because the data is already on the computing GPU)
  }
  //  Left tensor:
  if (ltens->dst_rsc == ltens->src_rsc)
    ltens->dst_rsc =
        nullptr; // destination resource was pointing to the source resource
  if (gpu_l != gpu_num) { // data is on a different GPU device or Host
    if (ltens->dst_rsc == nullptr) {
      errc = tensDevRsc_create(&(ltens->dst_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 13);
        i = gpu_activate(cur_gpu);
        return 13;
      }
    } else {
      if (tensDevRsc_is_empty(ltens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(ltens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(ltens->dst_rsc, targ_dev, lsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 14);
        i = gpu_activate(cur_gpu);
        return 14;
      }
    }
  } else {
    if (ltens->dst_rsc != nullptr) {
      if (tensDevRsc_is_empty(ltens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(ltens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    ltens->dst_rsc =
        ltens->src_rsc; // destination and source resources are the same
                        // (because the data is already on the computing GPU)
  }
  // Start scheduling CUDA calls:
  cuda_start_ct = std::chrono::high_resolution_clock::now();
  err = 0;
  if (err != 0) {
    if (VERBOSE)
      printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_copy): Unable "
             "to record the start event: %s\n",
             err_msg);
    errc = sycl_task_record(sycl_task, coh_ctrl, 23);
    errc = gpu_activate(cur_gpu);
    return 23;
  }
  if (LastTask[gpu_num] !=
      nullptr) { //`This should be done atomically for thread safety
    dep_event = sycl_event_ptr(LastTask[gpu_num]->gpu_id,
                               LastTask[gpu_num]->event_comput_hl);
    err = (*dep_event.wait(), 0); // input
    // transfers should only begin after the previous task input transfers have
    // completed
    if (err != 0) {
      if (VERBOSE)
        printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_copy): "
               "Unable to create a task dependency: %s\n",
               err_msg);
      errc = sycl_task_record(sycl_task, coh_ctrl, 24);
      errc = gpu_activate(cur_gpu);
      return 24;
    }
  }
  // Schedule forward data transfers for all tensors if needed:
  // Left tensor:
  if (sycl_task->tens_args[1].const_mem_entry >= 0) {
    // GPU constant memory entry will contain tensor dimension extents
    // and the matricization permutation (if any)
    try {
      (*sycl_queue)
          ->memcpy(
              (char *)(const_args_dims.get_ptr()) +
                  sizeof(int) *
                      ((size_t)(MAX_TENSOR_RANK *
                                (sycl_task->tens_args[1].const_mem_entry))),
              (void *)(ltens->shape.dims),
              sizeof(int) * ((size_t)lrank)); // tensor dimension extents
    } catch (cl::sycl::exception const &exc) {
      if (VERBOSE)
        std::cerr << "#ERROR: tensor dims H2D copy failed: " << exc.what()
                  << ", at " << __FILE__ << ", line:" << __LINE__ << std::endl;
      errc = sycl_task_record(sycl_task, coh_ctrl, 25);
      errc = gpu_activate(cur_gpu);
      return 25;
    }
    gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)lrank);
    try {
      (*sycl_queue)
          ->memcpy(
              (char *)(const_args_prmn.get_ptr()) +
                  sizeof(int) *
                      ((size_t)(MAX_TENSOR_RANK *
                                (sycl_task->tens_args[1].const_mem_entry))),
              (void *)(sycl_task->tens_args[1].prmn_p),
              sizeof(int) * ((size_t)lrank)); // tensor permutation
    } catch (cl::sycl::exception const &exc) {
      if (VERBOSE)
        std::cerr << "#ERROR: tensor prmn H2D copy failed: " << exc.what()
                  << ", at " << __FILE__ << ", line:" << __LINE__ << std::endl;
      errc = sycl_task_record(sycl_task, coh_ctrl, 26);
      errc = gpu_activate(cur_gpu);
      return 26;
    }
    gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)lrank);
    if (gpu_l != gpu_num) { // data is not on the computing GPU
      try {
        (*sycl_queue)
            ->memcpy(ltens->dst_rsc->gmem_p, ltens->src_rsc->gmem_p, lsize);
      } catch (cl::sycl::exception const &exc) {
        if (VERBOSE)
          std::cerr << "#ERROR: Left tensor body copy failed: " << exc.what()
                    << ", at " << __FILE__ << ", line:" << __LINE__
                    << std::endl;
        errc = sycl_task_record(sycl_task, coh_ctrl, 27);
        errc = gpu_activate(cur_gpu);
        return 27;
      }
      gpu_stats[gpu_num].traffic_in += lsize;
    }
  } else {
    errc = sycl_task_record(sycl_task, coh_ctrl, 28);
    errc = gpu_activate(cur_gpu);
    return 28;
  }
  // Use the destination resource pointers for each tensor argument:
  // Record a SYCL queue:
  cuda_comput_ct = std::chrono::high_resolution_clock::now();
  err = 0;
  if (err != 0) {
    if (VERBOSE)
      printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_copy): Unable "
             "to record the compute event: %s\n",
             err_msg);
    errc = sycl_task_record(sycl_task, coh_ctrl, 37);
    errc = gpu_activate(cur_gpu);
    return 37;
  }
  darg = dtens->dst_rsc->gmem_p;
  larg = ltens->dst_rsc->gmem_p;
  // Schedule the appropriate computation kernel:
  // Permutation kernel:
  if (TRANS_SHMEM == EFF_TRN_ON) {
    bx = 1 + (vol_l - 1) / THRDS_TENSOR_COPY;
    if (bx > MAX_SYCL_BLOCKS)
      bx = MAX_SYCL_BLOCKS;
    switch (ltens->data_kind) {
    case R4:
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        auto gpu_error_count_ptr = gpu_error_count.get_ptr();

        local_accessor<T, 1> buf0_acc(
            cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
        local_accessor<float, 0> val_acc(cgh);
        local_accessor<size_t, 1> base_in_acc(
            cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> base_out_acc(
            cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> ftb_acc(
            cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<size_t, 1> gtb_acc(
            cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> htb_acc(
            cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> stb_acc(
            cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> dim_in_acc(
            cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> dim_out_acc(
            cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> o2n_acc(cl::sycl::range(32 /*MAX_TENSOR_RANK*/),
                                       cgh);
        local_accessor<int, 1> n2o_acc(cl::sycl::range(32 /*MAX_TENSOR_RANK*/),
                                       cgh);
        local_accessor<int, 1> pri_acc(cl::sycl::range(32 /*MAX_TENSOR_RANK*/),
                                       cgh);
        local_accessor<int, 1> tmp0_acc(cl::sycl::range(32 /*MAX_TENSOR_RANK*/),
                                        cgh);
        local_accessor<int, 0> err_code_acc(cgh);
        local_accessor<int, 0> minor_acc(cgh);
        local_accessor<int, 0> minor_in_acc(cgh);
        local_accessor<int, 0> minor_out_acc(cgh);
        local_accessor<int, 0> s1_ind_acc(cgh);
        local_accessor<int, 0> s1_ond_acc(cgh);
        local_accessor<int, 0> s1_step_acc(cgh);
        local_accessor<int, 0> s1_dim_acc(cgh);
        local_accessor<int, 0> s2_ind_acc(cgh);
        local_accessor<int, 0> s2_ond_acc(cgh);
        local_accessor<int, 0> s2_step_acc(cgh);
        local_accessor<int, 0> s2_dim_acc(cgh);
        local_accessor<int, 0> ns1_acc(cgh);
        local_accessor<int, 0> ns2_acc(cgh);
        local_accessor<size_t, 0> vol_acc(cgh);
        local_accessor<size_t, 0> vol_ext_acc(cgh);

        auto const_args_dims_acc = const_args_dims.get_access(cgh);
        auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
        auto sycl_task_tens_args_const_mem_entry3 =
            sycl_task->tens_args[1].const_mem_entry;

        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
            [=](cl::sycl::nd_item<1> item) {
              gpu_tensor_block_copy_dlf__(
                  0, 0, lrank, sycl_task_tens_args_const_mem_entry3,
                  (float *)(larg), (float *)(darg), item, const_args_dims_acc,
                  const_args_prmn_acc, gpu_error_count_ptr,
                  buf0_acc.get_pointer(), val_acc.get_pointer(),
                  base_in_acc.get_pointer(), base_out_acc.get_pointer(),
                  ftb_acc.get_pointer(), gtb_acc.get_pointer(),
                  htb_acc.get_pointer(), stb_acc.get_pointer(),
                  dim_in_acc.get_pointer(), dim_out_acc.get_pointer(),
                  o2n_acc.get_pointer(), n2o_acc.get_pointer(),
                  pri_acc.get_pointer(), tmp0_acc.get_pointer(),
                  err_code_acc.get_pointer(), minor_acc.get_pointer(),
                  minor_in_acc.get_pointer(), minor_out_acc.get_pointer(),
                  s1_ind_acc.get_pointer(), s1_ond_acc.get_pointer(),
                  s1_step_acc.get_pointer(), s1_dim_acc.get_pointer(),
                  s2_ind_acc.get_pointer(), s2_ond_acc.get_pointer(),
                  s2_step_acc.get_pointer(), s2_dim_acc.get_pointer(),
                  ns1_acc.get_pointer(), ns2_acc.get_pointer(),
                  vol_acc.get_pointer(), vol_ext_acc.get_pointer());
            });
      });
      break;
    case R8:
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        auto gpu_error_count_ptr = gpu_error_count.get_ptr();

        local_accessor<T, 1> buf0_acc(
            cl::sycl::range<1>(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
        local_accessor<float, 0> val_acc(cgh);
        local_accessor<size_t, 1> base_in_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> base_out_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> ftb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<size_t, 1> gtb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> htb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> stb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> dim_in_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> dim_out_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> o2n_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> n2o_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> pri_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> tmp0_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 0> err_code_acc(cgh);
        local_accessor<int, 0> minor_acc(cgh);
        local_accessor<int, 0> minor_in_acc(cgh);
        local_accessor<int, 0> minor_out_acc(cgh);
        local_accessor<int, 0> s1_ind_acc(cgh);
        local_accessor<int, 0> s1_ond_acc(cgh);
        local_accessor<int, 0> s1_step_acc(cgh);
        local_accessor<int, 0> s1_dim_acc(cgh);
        local_accessor<int, 0> s2_ind_acc(cgh);
        local_accessor<int, 0> s2_ond_acc(cgh);
        local_accessor<int, 0> s2_step_acc(cgh);
        local_accessor<int, 0> s2_dim_acc(cgh);
        local_accessor<int, 0> ns1_acc(cgh);
        local_accessor<int, 0> ns2_acc(cgh);
        local_accessor<size_t, 0> vol_acc(cgh);
        local_accessor<size_t, 0> vol_ext_acc(cgh);

        auto const_args_dims_acc = const_args_dims.get_access(cgh);
        auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
        auto sycl_task_tens_args_const_mem_entry3 =
            sycl_task->tens_args[1].const_mem_entry;

        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
            [=](cl::sycl::nd_item<1> item) {
              gpu_tensor_block_copy_dlf__(
                  0, 0, lrank, sycl_task_tens_args_const_mem_entry3,
                  (double *)(larg), (double *)(darg), item, const_args_dims_acc,
                  const_args_prmn_acc, gpu_error_count_ptr,
                  buf0_acc.get_pointer(), val_acc.get_pointer(),
                  base_in_acc.get_pointer(), base_out_acc.get_pointer(),
                  ftb_acc.get_pointer(), gtb_acc.get_pointer(),
                  htb_acc.get_pointer(), stb_acc.get_pointer(),
                  dim_in_acc.get_pointer(), dim_out_acc.get_pointer(),
                  o2n_acc.get_pointer(), n2o_acc.get_pointer(),
                  pri_acc.get_pointer(), tmp0_acc.get_pointer(),
                  err_code_acc.get_pointer(), minor_acc.get_pointer(),
                  minor_in_acc.get_pointer(), minor_out_acc.get_pointer(),
                  s1_ind_acc.get_pointer(), s1_ond_acc.get_pointer(),
                  s1_step_acc.get_pointer(), s1_dim_acc.get_pointer(),
                  s2_ind_acc.get_pointer(), s2_ond_acc.get_pointer(),
                  s2_step_acc.get_pointer(), s2_dim_acc.get_pointer(),
                  ns1_acc.get_pointer(), ns2_acc.get_pointer(),
                  vol_acc.get_pointer(), vol_ext_acc.get_pointer());
            });
      });
      break;
    case C4:
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        auto gpu_error_count_ptr = gpu_error_count.get_ptr();

        local_accessor<T, 1> buf0_acc(
            cl::sycl::range<1>(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
        local_accessor<float, 0> val_acc(cgh);
        local_accessor<size_t, 1> base_in_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> base_out_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> ftb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<size_t, 1> gtb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> htb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> stb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> dim_in_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> dim_out_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> o2n_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> n2o_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> pri_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> tmp0_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 0> err_code_acc(cgh);
        local_accessor<int, 0> minor_acc(cgh);
        local_accessor<int, 0> minor_in_acc(cgh);
        local_accessor<int, 0> minor_out_acc(cgh);
        local_accessor<int, 0> s1_ind_acc(cgh);
        local_accessor<int, 0> s1_ond_acc(cgh);
        local_accessor<int, 0> s1_step_acc(cgh);
        local_accessor<int, 0> s1_dim_acc(cgh);
        local_accessor<int, 0> s2_ind_acc(cgh);
        local_accessor<int, 0> s2_ond_acc(cgh);
        local_accessor<int, 0> s2_step_acc(cgh);
        local_accessor<int, 0> s2_dim_acc(cgh);
        local_accessor<int, 0> ns1_acc(cgh);
        local_accessor<int, 0> ns2_acc(cgh);
        local_accessor<size_t, 0> vol_acc(cgh);
        local_accessor<size_t, 0> vol_ext_acc(cgh);

        auto const_args_dims_acc = const_args_dims.get_access(cgh);
        auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
        auto sycl_task_tens_args_const_mem_entry3 =
            sycl_task->tens_args[1].const_mem_entry;

        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
            [=](cl::sycl::nd_item<1> item) {
              gpu_tensor_block_copy_dlf__(
                  0, 0, lrank, sycl_task_tens_args_const_mem_entry3,
                  (talshComplex4 *)(larg), (talshComplex4 *)(darg), item,
                  const_args_dims_acc, const_args_prmn_acc, gpu_error_count_ptr,
                  buf0_acc.get_pointer(), val_acc.get_pointer(),
                  base_in_acc.get_pointer(), base_out_acc.get_pointer(),
                  ftb_acc.get_pointer(), gtb_acc.get_pointer(),
                  htb_acc.get_pointer(), stb_acc.get_pointer(),
                  dim_in_acc.get_pointer(), dim_out_acc.get_pointer(),
                  o2n_acc.get_pointer(), n2o_acc.get_pointer(),
                  pri_acc.get_pointer(), tmp0_acc.get_pointer(),
                  err_code_acc.get_pointer(), minor_acc.get_pointer(),
                  minor_in_acc.get_pointer(), minor_out_acc.get_pointer(),
                  s1_ind_acc.get_pointer(), s1_ond_acc.get_pointer(),
                  s1_step_acc.get_pointer(), s1_dim_acc.get_pointer(),
                  s2_ind_acc.get_pointer(), s2_ond_acc.get_pointer(),
                  s2_step_acc.get_pointer(), s2_dim_acc.get_pointer(),
                  ns1_acc.get_pointer(), ns2_acc.get_pointer(),
                  vol_acc.get_pointer(), vol_ext_acc.get_pointer());
            });
      });
      break;
    case C8:
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        auto gpu_error_count_ptr = gpu_error_count.get_ptr();

        local_accessor<T, 1> buf0_acc(
            cl::sycl::range<1>(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
        local_accessor<float, 0> val_acc(cgh);
        local_accessor<size_t, 1> base_in_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> base_out_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> ftb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<size_t, 1> gtb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> htb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> stb_acc(
            cl::sycl::range<1>(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
        local_accessor<int, 1> dim_in_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> dim_out_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> o2n_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> n2o_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> pri_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 1> tmp0_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<int, 0> err_code_acc(cgh);
        local_accessor<int, 0> minor_acc(cgh);
        local_accessor<int, 0> minor_in_acc(cgh);
        local_accessor<int, 0> minor_out_acc(cgh);
        local_accessor<int, 0> s1_ind_acc(cgh);
        local_accessor<int, 0> s1_ond_acc(cgh);
        local_accessor<int, 0> s1_step_acc(cgh);
        local_accessor<int, 0> s1_dim_acc(cgh);
        local_accessor<int, 0> s2_ind_acc(cgh);
        local_accessor<int, 0> s2_ond_acc(cgh);
        local_accessor<int, 0> s2_step_acc(cgh);
        local_accessor<int, 0> s2_dim_acc(cgh);
        local_accessor<int, 0> ns1_acc(cgh);
        local_accessor<int, 0> ns2_acc(cgh);
        local_accessor<size_t, 0> vol_acc(cgh);
        local_accessor<size_t, 0> vol_ext_acc(cgh);

        auto const_args_dims_acc = const_args_dims.get_access(cgh);
        auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
        auto sycl_task_tens_args_const_mem_entry3 =
            sycl_task->tens_args[1].const_mem_entry;

        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
            [=](cl::sycl::nd_item<1> item) {
              gpu_tensor_block_copy_dlf__(
                  0, 0, lrank, sycl_task_tens_args_const_mem_entry3,
                  (talshComplex8 *)(larg), (talshComplex8 *)(darg), item,
                  const_args_dims_acc, const_args_prmn_acc, gpu_error_count_ptr,
                  buf0_acc.get_pointer(), val_acc.get_pointer(),
                  base_in_acc.get_pointer(), base_out_acc.get_pointer(),
                  ftb_acc.get_pointer(), gtb_acc.get_pointer(),
                  htb_acc.get_pointer(), stb_acc.get_pointer(),
                  dim_in_acc.get_pointer(), dim_out_acc.get_pointer(),
                  o2n_acc.get_pointer(), n2o_acc.get_pointer(),
                  pri_acc.get_pointer(), tmp0_acc.get_pointer(),
                  err_code_acc.get_pointer(), minor_acc.get_pointer(),
                  minor_in_acc.get_pointer(), minor_out_acc.get_pointer(),
                  s1_ind_acc.get_pointer(), s1_ond_acc.get_pointer(),
                  s1_step_acc.get_pointer(), s1_dim_acc.get_pointer(),
                  s2_ind_acc.get_pointer(), s2_ond_acc.get_pointer(),
                  s2_step_acc.get_pointer(), s2_dim_acc.get_pointer(),
                  ns1_acc.get_pointer(), ns2_acc.get_pointer(),
                  vol_acc.get_pointer(), vol_ext_acc.get_pointer());
            });
      });
      break;
    default:
      errc = sycl_task_record(sycl_task, coh_ctrl, 40);
      errc = gpu_activate(cur_gpu);
      return 40;
    }
  } else if (TRANS_SHMEM == EFF_TRN_ON_CUTT) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 68);
    errc = gpu_activate(cur_gpu);
    return 68;
  } else if (TRANS_SHMEM == EFF_TRN_OFF) {
    bx = 1 + (vol_l - 1) / THRDS_TENSOR_COPY_SCAT;
    if (bx > MAX_SYCL_BLOCKS)
      bx = MAX_SYCL_BLOCKS;
    switch (ltens->data_kind) {
    case R4:
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

        local_accessor<int, 1> n2o_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 0> vol_acc(cgh);
        local_accessor<size_t, 1> base_in_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> base_out_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);

        auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
        auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
        auto sycl_task_tens_args_const_mem_entry_ct3 =
            sycl_task->tens_args[1].const_mem_entry;

        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                  THRDS_TENSOR_COPY_SCAT),
            [=](cl::sycl::nd_item<1> item) {
              gpu_tensor_block_copy_scatter_dlf__(
                  0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                  (float *)(larg), (float *)(darg), item,
                  const_args_dims_acc_ct, const_args_prmn_acc_ct,
                  gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                  vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                  base_out_acc_ct.get_pointer());
            });
      });
      break;
    case R8:
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

        local_accessor<int, 1> n2o_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 0> vol_acc(cgh);
        local_accessor<size_t, 1> base_in_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> base_out_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);

        auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
        auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
        auto sycl_task_tens_args_const_mem_entry_ct3 =
            sycl_task->tens_args[1].const_mem_entry;

        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                  THRDS_TENSOR_COPY_SCAT),
            [=](cl::sycl::nd_item<1> item) {
              gpu_tensor_block_copy_scatter_dlf__(
                  0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                  (double *)(larg), (double *)(darg), item,
                  const_args_dims_acc_ct, const_args_prmn_acc_ct,
                  gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                  vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                  base_out_acc_ct.get_pointer());
            });
      });
      break;
    case C4:
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

        local_accessor<int, 1> n2o_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 0> vol_acc(cgh);
        local_accessor<size_t, 1> base_in_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> base_out_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);

        auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
        auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
        auto sycl_task_tens_args_const_mem_entry_ct3 =
            sycl_task->tens_args[1].const_mem_entry;

        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                  THRDS_TENSOR_COPY_SCAT),
            [=](cl::sycl::nd_item<1> item) {
              gpu_tensor_block_copy_scatter_dlf__(
                  0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                  (talshComplex4 *)(larg), (talshComplex4 *)(darg), item,
                  const_args_dims_acc_ct, const_args_prmn_acc_ct,
                  gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                  vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                  base_out_acc_ct.get_pointer());
            });
      });
      break;
    case C8:
      (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
        auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

        local_accessor<int, 1> n2o_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 0> vol_acc(cgh);
        local_accessor<size_t, 1> base_in_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);
        local_accessor<size_t, 1> base_out_acc(
            cl::sycl::range<1>(32 /*MAX_TENSOR_RANK*/), cgh);

        auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
        auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
        auto sycl_task_tens_args_const_mem_entry_ct3 =
            sycl_task->tens_args[1].const_mem_entry;

        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                  THRDS_TENSOR_COPY_SCAT),
            [=](cl::sycl::nd_item<1> item) {
              gpu_tensor_block_copy_scatter_dlf__(
                  0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                  (talshComplex8 *)(larg), (talshComplex8 *)(darg), item,
                  const_args_dims_acc_ct, const_args_prmn_acc_ct,
                  gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                  vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                  base_out_acc_ct.get_pointer());
            });
      });
      break;
    default:
      errc = sycl_task_record(sycl_task, coh_ctrl, 41);
      errc = gpu_activate(cur_gpu);
      return 41;
    }
  } else {
    errc = sycl_task_record(sycl_task, coh_ctrl, 60);
    errc = gpu_activate(cur_gpu);
    return 60;
  }
  // Transfer back the updated destination tensor if needed ("T","K" coherence control):
  coh = (coh_ctrl >> 2) &
        (TWO_BITS_SET); // select bits 2,3 (destination tensor coherence)
  if (gpu_d != gpu_num && coh >= 2) { // data is not on the computing GPU and
                                      // coherence control = 2("T") or (3)"K":
    try {
      (*sycl_queue)
          ->memcpy(dtens->src_rsc->gmem_p, dtens->dst_rsc->gmem_p, dsize);
    } catch (cl::sycl::exception const &exc) {
      if (VERBOSE)
        std::cerr << "#ERROR: Destination tensor body back copy failed: "
                  << exc.what() << ", at " << __FILE__ << ", line:" << __LINE__
                  << std::endl;
      errc = sycl_task_record(sycl_task, coh_ctrl, 57);
      errc = gpu_activate(cur_gpu);
      return 57;
    }
    gpu_stats[gpu_num].traffic_out += dsize;
  }
  // Record the successfully scheduled SYCL task and update the Last Task:
  errc = sycl_task_record(sycl_task, coh_ctrl, 0);
  LastTask[gpu_num] = sycl_task;
  if (gpu_num != cur_gpu)
    errc = gpu_activate(cur_gpu);
  return stat; // either 0 (success) or NOT_CLEAN (warning)
} catch (cl::sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
//-----------------------------------------------------------------------------------------------------------------------
// TENSOR ADDITION (non-blocking):
int gpu_tensor_block_add(const int *cptrn, tensBlck_t *ltens, tensBlck_t *dtens,
                         unsigned int coh_ctrl, cudaTask_t *sycl_task,
                         int gpu_id, double scale_real, double scale_imag,
                         int conj_bits)
    /**
       dtens(:)+=ltens(:)*scalar
       INPUT:
       # cptrn(1:lrank) - addition pattern: Position correspondence:
       Uncontracted indices are positive, no contracted indices;
       # ltens - left tensor argument (initialized!);
       # dtens - destination tensor argument (initialized!);
       # coh_ctrl - one of the COPY_XX parameters regulating the data presence
    for each tensor argument; # sycl_task - pointer to an empty (clean) CUDA
    task; # gpu_id - suggested GPU ID on which the operation is to be scheduled
    (-1: defaults to the optimal one); # scale_real - real part of the GEMM
    alpha coefficient (defaults to 1.0); # scale_imag - imaginary part of the
    GEMM alpha coefficient (defaults to 0.0); # conj_bits - tensor argument
    complex conjugation bits, one bit per argument: {0:D,1:L}; OUTPUT: # dtens -
    updated destination tensor; # sycl_task - recorded SYCL task (either
    successfully scheduled or failed). NOTES: # If the tensor operation has been
    scheduled successfully, a recorded (active) SYCL task will be returned along
    with zero return status. A scheduling error results in either a negative (at
    early stages) or positive (at later stages) return status. In the former
    case the SYCL task is left clean, while at the latter case it will be
    recorded as failed (error). # Special return statuses TRY_LATER and
    DEVICE_UNABLE are not errors but merely indicators of the current or
    permanent lack of resources, respectively. However, the SYCL task status in
    these cases will still be set to an error (always check the function return
    status!). # If <gpu_id> is out of the legitimate GPU range, it will be
    replaced by an optimal one, based on argument residence and the current load
    of GPU(s).
    **/
    try {
  int i, j, drank, lrank, tds_d, tds_l, gpu_d, gpu_l, perm_d, perm_l, ncd, nlu,
      nru, gpu_num, cur_gpu, targ_dev, bx, errc, stat, conj_l;
  int dprm[1 + MAX_TENSOR_RANK], lprm[1 + MAX_TENSOR_RANK],
      rprm[1]; // the 1st element is the sign of the permutation
  size_t vol_d, vol_l, dsize, lsize;
  unsigned int coh;
  const unsigned int TWO_BITS_SET = 3; // two right bits are set
  void *darg, *larg;
  talshComplex4 scale_cmplx4;
  talshComplex8 scale_cmplx8;

  sycl::queue **sycl_stream;
  sycl::event *sycl_start, *sycl_comput, *sycl_output, *sycl_finish, *dep_event;
  std::chrono::time_point<std::chrono::high_resolution_clock> sycl_start_ct;
  std::chrono::time_point<std::chrono::high_resolution_clock> sycl_comput_ct;
  std::chrono::time_point<std::chrono::high_resolution_clock> sycl_output_ct;
  std::chrono::time_point<std::chrono::high_resolution_clock> sycl_finish_ct;

  int err;
  const char *err_msg;

  // if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_intel:gpu_tensor_block_add):
  // GPU Tensor Addition:\n"); //debug
  stat = 0; // return status in case of successful scheduling
  // Check function arguments:
  if (cptrn == nullptr || dtens == nullptr || ltens == nullptr ||
      sycl_task == nullptr)
    return -1;
  if (tensBlck_present(dtens) != YEP || tensBlck_present(ltens) != YEP)
    return -2; // tensor blocks must reside in some device memory
  if (sycl_task_gpu_id(sycl_task) >= 0)
    return -3; // SYCL task is not clean (destruct/clean it first)
  // Check tensor arguments:
  drank = (dtens->shape).num_dim; // destination tensor rank
  lrank = (ltens->shape).num_dim; // left tensor rank
  if (drank < 0 || drank > MAX_TENSOR_RANK || lrank < 0 ||
      lrank > MAX_TENSOR_RANK)
    return -4;
  if (tens_valid_data_kind(dtens->data_kind, &tds_d) !=
          YEP || // tds_d: destination tensor element size in bytes
      tens_valid_data_kind(ltens->data_kind, &tds_l) != YEP)
    return -5; // tds_l: left tensor element size in bytes
  if (!(dtens->data_kind > 0 && ltens->data_kind == dtens->data_kind))
    return -6; // data kind mismatch
  if (dtens->src_rsc == nullptr || ltens->src_rsc == nullptr)
    return -7; // source resource must always be present
  if (tensDevRsc_is_empty(dtens->src_rsc) != NOPE)
    return -8; // source resource must be present (tensor body)
  if (tensDevRsc_is_empty(ltens->src_rsc) != NOPE)
    return -9; // source resource must be present (tensor body)
  // Check the contraction pattern and dimension extent correspondence:
  for (i = 0; i < drank; i++)
    dprm[i] = 0;
  for (i = 0; i < lrank; i++)
    lprm[i] = 0;
  for (i = 0; i < lrank; i++) { // position in ltens
    j = cptrn[i];
    if (j > 0) { // position in dtens
      if (j > drank)
        return -11;
      if ((dtens->shape).dims[j - 1] != (ltens->shape).dims[i])
        return -12;
      if (dprm[j - 1] == 0) {
        dprm[j - 1] = 1;
      } else {
        return -13;
      }
    } else {
      return -18;
    }
  }
  for (i = 0; i < drank; i++)
    if (dprm[i] != 1)
      return -27;
  // Check argument complex conjugation bits:
  conj_bits = conj_bits &
              3; // keep only first two bits, one per tensor argument {0:D,1:L}
  if (conj_bits & 1) { // destination tensor argument conjugation = inverse
                       // conjugation of the left argument
    conj_bits = conj_bits ^ 3; // XOR with 0b11 will invert bits
  }
  conj_l = 0;
  if ((conj_bits & 2) != 0)
    conj_l = 1; // left argument conjugation flag
  // Activate the right GPU:
  if (gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE) {
    gpu_num = tens_op_best_gpu(dtens, ltens);
  } else {
    gpu_num = gpu_id;
  }
  if (gpu_is_mine(gpu_num) <= GPU_OFF)
    return -28; // GPU is not mine or error
  gpu_stats[gpu_num].tasks_submitted++;
  gpu_d = decode_device_id(dtens->src_rsc->dev_id, &j);
  if (gpu_d < 0)
    return -29; // destination tensor source device id
  if (j == DEV_INTEL_GPU) {
    if (gpu_d != gpu_num) {
      // abb TODO: place a msg here
      return DEVICE_UNABLE; // peer access impossible for this GPU device
    }
  } else if (j == DEV_HOST) {
    gpu_d = -1; // data is in Host memory
  } else {
    return DEVICE_UNABLE; // data is not in Host or GPU memory
  }
  gpu_l = decode_device_id(ltens->src_rsc->dev_id, &j);
  if (gpu_l < 0)
    return -30; // left tensor source device id
  if (j == DEV_INTEL_GPU) {
    if (gpu_l != gpu_num) {
      // abb TODO: place a msg here
      return DEVICE_UNABLE; // peer access impossible for this GPU device
    }
  } else if (j == DEV_HOST) {
    gpu_l = -1; // data is in Host memory
  } else {
    return DEVICE_UNABLE; // data is not in Host or GPU memory
  }
  cur_gpu = gpu_in_focus(); // save the current GPU
  if (gpu_num != cur_gpu) {
    errc = gpu_activate(gpu_num);
    if (errc) {
      errc = gpu_activate(cur_gpu);
      return -32;
    }
  } // activate the target GPU
  targ_dev = encode_device_id(DEV_INTEL_GPU, gpu_num); // flat device id
  // Construct a SYCL task (acquire CUDA resources) for the target GPU:
  errc = sycl_task_construct(sycl_task, gpu_num);
  if (errc) {
    i = gpu_activate(cur_gpu);
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      return errc;
    } else {
      return -33;
    }
  }

  // *** From this point all error codes must be positive and the SYCL task must
  // be recorded! ***
  // Set up tensor arguments (allocates additional resources for each tensor
  // argument):
  // Destination argument:
  errc = sycl_task_set_arg(sycl_task, 0, dtens);
  if (errc) {
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
      i = gpu_activate(
          cur_gpu); // not an error if TRY_LATER or DEVICE_UNABLE are returned
      return errc;
    } else {
      i = sycl_task_record(sycl_task, coh_ctrl, 1);
      i = gpu_activate(cur_gpu);
      return 1;
    }
  }
  // Left argument:
  errc = sycl_task_set_arg(sycl_task, 1, ltens);
  if (errc) {
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
      i = gpu_activate(
          cur_gpu); // not an error if TRY_LATER or DEVICE_UNABLE are returned
      return errc;
    } else {
      i = sycl_task_record(sycl_task, coh_ctrl, 2);
      i = gpu_activate(cur_gpu);
      return 2;
    }
  }
  // Associate SYCL stream and event pointers locally for convenience:
  sycl_stream = sycl_stream_ptr(sycl_task->gpu_id, sycl_task->queue_hl);
  if (sycl_stream == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 4);
    errc = gpu_activate(cur_gpu);
    return 4;
  }
  sycl_start = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_start_hl);
  if (sycl_start == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 5);
    errc = gpu_activate(cur_gpu);
    return 5;
  }
  sycl_comput = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_comput_hl);
  if (sycl_comput == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 6);
    errc = gpu_activate(cur_gpu);
    return 6;
  }
  sycl_output = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_output_hl);
  if (sycl_output == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 7);
    errc = gpu_activate(cur_gpu);
    return 7;
  }
  sycl_finish = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_finish_hl);
  if (sycl_finish == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 8);
    errc = gpu_activate(cur_gpu);
    return 8;
  }

  // Determine the volume and required matricization permutation for each tensor
  // argument:
  get_contr_permutations(1, 0, lrank, 0, cptrn, 0, dprm, lprm, rprm, &ncd, &nlu,
                         &nru, &errc); // permutations and numbers of dimensions
  if (errc) {
    i = sycl_task_record(sycl_task, coh_ctrl, 9);
    i = gpu_activate(cur_gpu);
    return 9;
  }
  for (i = 0; i < drank; i++)
    sycl_task->tens_args[0].prmn_p[i] =
        dprm[1 + i]; // ignore the permutaion sign
  perm_d =
      non_trivial_prmn(drank, sycl_task->tens_args[0].prmn_p); // trivial or not
  for (i = 0; i < lrank; i++)
    sycl_task->tens_args[1].prmn_p[i] =
        lprm[1 + i]; // ignore the permutaion sign
  perm_l =
      non_trivial_prmn(lrank, sycl_task->tens_args[1].prmn_p); // trivial or not
  vol_d = tensBlck_volume(dtens);
  vol_l = tensBlck_volume(ltens); // tensor block volumes
  dsize = vol_d * tds_d;
  lsize = vol_l * tds_l; // tensor argument sizes in bytes
  // Acquire global memory resources for tensor arguments if needed:
  // Set up destination memory resources in all tensors:
  //  Destination tensor:
  if (dtens->dst_rsc == dtens->src_rsc)
    dtens->dst_rsc =
        nullptr; // destination resource was pointing to the source resource
  if (gpu_d != gpu_num) { // data is on a different GPU device or Host
    if (dtens->dst_rsc == nullptr) {
      errc = tensDevRsc_create(&(dtens->dst_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 11);
        i = gpu_activate(cur_gpu);
        return 11;
      }
    } else {
      if (tensDevRsc_is_empty(dtens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(dtens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(dtens->dst_rsc, targ_dev, dsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 12);
        i = gpu_activate(cur_gpu);
        return 12;
      }
    }
  } else {
    if (dtens->dst_rsc != nullptr) {
      if (tensDevRsc_is_empty(dtens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(dtens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    dtens->dst_rsc =
        dtens->src_rsc; // destination and source resources are the same
    // (because the data is already on the computing GPU)
  }
  //  Left tensor:
  if (ltens->dst_rsc == ltens->src_rsc)
    ltens->dst_rsc =
        nullptr; // destination resource was pointing to the source resource
  if (gpu_l != gpu_num) { // data is on a different GPU device or Host
    if (ltens->dst_rsc == nullptr) {
      errc = tensDevRsc_create(&(ltens->dst_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 13);
        i = gpu_activate(cur_gpu);
        return 13;
      }
    } else {
      if (tensDevRsc_is_empty(ltens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(ltens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(ltens->dst_rsc, targ_dev, lsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 14);
        i = gpu_activate(cur_gpu);
        return 14;
      }
    }
  } else {
    if (ltens->dst_rsc != nullptr) {
      if (tensDevRsc_is_empty(ltens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(ltens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    ltens->dst_rsc =
        ltens->src_rsc; // destination and source resources are the same
    // (because the data is already on the computing GPU)
  }
  // Set up temporary memory resources in all tensors if needed (because of
  // out-of-place tensor transpose):
  //  Destination tensor:
  if (perm_d == YEP) {
    if (dtens->tmp_rsc == nullptr) {
      errc = tensDevRsc_create(&(dtens->tmp_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 17);
        i = gpu_activate(cur_gpu);
        return 17;
      }
    } else {
      if (tensDevRsc_is_empty(dtens->tmp_rsc) == NOPE) {
        errc = tensDevRsc_release_all(dtens->tmp_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(dtens->tmp_rsc, targ_dev, dsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 18);
        i = gpu_activate(cur_gpu);
        return 18;
      }
    }
  }
  //  Left tensor:
  if (perm_l == YEP) {
    if (ltens->tmp_rsc == nullptr) {
      errc = tensDevRsc_create(&(ltens->tmp_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 19);
        i = gpu_activate(cur_gpu);
        return 19;
      }
    } else {
      if (tensDevRsc_is_empty(ltens->tmp_rsc) == NOPE) {
        errc = tensDevRsc_release_all(ltens->tmp_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(ltens->tmp_rsc, targ_dev, lsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 20);
        i = gpu_activate(cur_gpu);
        return 20;
      }
    }
  }
  // Start scheduling SYCL calls:
  cuda_start_ct = std::chrono::high_resolution_clock::now();

  if (LastTask[gpu_num] !=
      nullptr) { //`This should be done atomically for thread safety
    dep_event = sycl_event_ptr(LastTask[gpu_num]->gpu_id,
                               LastTask[gpu_num]->event_comput_hl);
    dep_event->wait(); // input transfers should only begin after the previous
                       // task input transfers have completed
    if (err != 0) {
      if (VERBOSE)
        printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_add): "
               "Unable to create a task dependency: %s\n",
               err_msg);
      errc = sycl_task_record(sycl_task, coh_ctrl, 24);
      errc = gpu_activate(cur_gpu);
      return 24;
    }
  }
  // Schedule forward data transfers for all tensors if needed:
  // Left tensor:
  if (sycl_task->tens_args[1].const_mem_entry >= 0) {
    // GPU constant memory entry will contain tensor dimension extents
    // and the matricization permutation (if any)
    try {
      (*sycl_queue)
          ->memcpy(
              (char *)(const_args_dims.get_ptr()) +
                  sizeof(int) *
                      ((size_t)(MAX_TENSOR_RANK *
                                (sycl_task->tens_args[1].const_mem_entry))),
              (void *)(ltens->shape.dims),
              sizeof(int) * ((size_t)lrank)); // tensor dimension extents
    } catch (cl::sycl::exception const &exc) {
      if (VERBOSE)
        std::cerr << "#ERROR: tensor dims H2D copy failed: " << exc.what()
                  << ", at " << __FILE__ << ", line:" << __LINE__ << std::endl;
      errc = sycl_task_record(sycl_task, coh_ctrl, 25);
      errc = gpu_activate(cur_gpu);
      return 25;
    }
    gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)lrank);
    if (perm_l == YEP) {
      try {
        (*sycl_queue)
            ->memcpy(
                (char *)(const_args_prmn.get_ptr()) +
                    sizeof(int) *
                        ((size_t)(MAX_TENSOR_RANK *
                                  (sycl_task->tens_args[1].const_mem_entry))),
                (void *)(sycl_task->tens_args[1].prmn_p),
                sizeof(int) *
                    ((size_t)lrank)); // tensor matricization permutation
      } catch (cl::sycl::exception const &exc) {
        if (VERBOSE)
          std::cerr << "#ERROR: Left tensor prmn H2D copy failed: "
                    << exc.what() << ", at " << __FILE__
                    << ", line:" << __LINE__ << std::endl;
        errc = sycl_task_record(sycl_task, coh_ctrl, 26);
        errc = gpu_activate(cur_gpu);
        return 26;
      }
      gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)lrank);
    }
    if (gpu_l != gpu_num) { // data is not on the computing GPU
      try {
        (*sycl_queue)
            ->memcpy(ltens->dst_rsc->gmem_p, ltens->src_rsc->gmem_p, lsize);
      } catch (cl::sycl::exception const &exc) {
        if (VERBOSE)
          std::cerr << "#ERROR: Left tensor body copy failed: " << exc.what()
                    << ", at " << __FILE__ << ", line:" << __LINE__
                    << std::endl;
        errc = sycl_task_record(sycl_task, coh_ctrl, 27);
        errc = gpu_activate(cur_gpu);
        return 27;
      }
      gpu_stats[gpu_num].traffic_in += lsize;
    }
  } else {
    errc = sycl_task_record(sycl_task, coh_ctrl, 28);
    errc = gpu_activate(cur_gpu);
    return 28;
  }
  // Destination tensor:
  if (sycl_task->tens_args[0].const_mem_entry >= 0) {
    // GPU constant memory entry will contain tensor dimension extents
    // and the matricization permutation (if any)
    try {
      (*sycl_queue)
          ->memcpy(
              (char *)(const_args_dims.get_ptr()) +
                  sizeof(int) *
                      ((size_t)(MAX_TENSOR_RANK *
                                (sycl_task->tens_args[0].const_mem_entry))),
              (void *)(dtens->shape.dims),
              sizeof(int) * ((size_t)drank)); // tensor dimension extents
    } catch (cl::sycl::exception const &exc) {
      if (VERBOSE)
        std::cerr << "#ERROR: Destination tensor dims H2D copy failed: "
                  << exc.what() << ", at " << __FILE__ << ", line:" << __LINE__
                  << std::endl;
      errc = sycl_task_record(sycl_task, coh_ctrl, 33);
      errc = gpu_activate(cur_gpu);
      return 33;
    }
    gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)drank);
    if (perm_d == YEP) {
      try {
        (*sycl_queue)
            ->memcpy(
                (char *)(const_args_prmn.get_ptr()) +
                    sizeof(int) *
                        ((size_t)(MAX_TENSOR_RANK *
                                  (sycl_task->tens_args[0].const_mem_entry))),
                (void *)(sycl_task->tens_args[0].prmn_p),
                sizeof(int) *
                    ((size_t)drank)); // tensor matricization permutation
      } catch (cl::sycl::exception const &exc) {
        if (VERBOSE)
          std::cerr << "#ERROR: Destination tensor prmn H2D copy failed: "
                    << exc.what() << ", at " << __FILE__
                    << ", line:" << __LINE__ << std::endl;
        errc = sycl_task_record(sycl_task, coh_ctrl, 34);
        errc = gpu_activate(cur_gpu);
        return 34;
      }
      gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)drank);
    }
    if (gpu_d != gpu_num) { // data is not on the computing GPU
      try {
        (*sycl_queue)
            ->memcpy(dtens->dst_rsc->gmem_p, dtens->src_rsc->gmem_p, dsize);
      } catch (cl::sycl::exception const &exc) {
        if (VERBOSE)
          std::cerr << "#ERROR: Destination tensor body copy failed: "
                    << exc.what() << ", at " << __FILE__
                    << ", line:" << __LINE__ << std::endl;
        errc = sycl_task_record(sycl_task, coh_ctrl, 35);
        errc = gpu_activate(cur_gpu);
        return 35;
      }
      gpu_stats[gpu_num].traffic_in += dsize;
    }
  } else {
    errc = sycl_task_record(sycl_task, coh_ctrl, 36);
    errc = gpu_activate(cur_gpu);
    return 36;
  }
  // Schedule tensor transposes if needed:
  // Record a SYCL queue:
  cuda_comput_ct = std::chrono::high_resolution_clock::now();
  err = 0;
  if (err != 0) {
    if (VERBOSE)
      printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_add): Unable "
             "to record the compute event: %s\n",
             err_msg);
    errc = sycl_task_record(sycl_task, coh_ctrl, 37);
    errc = gpu_activate(cur_gpu);
    return 37;
  }
  // Destination tensor transpose (it should not happen actually):
  if (perm_d == YEP) {
    if (TRANS_SHMEM == EFF_TRN_ON) {
      bx = 1 + (vol_d - 1) / THRDS_TENSOR_COPY;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (dtens->data_kind) {
      case R4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry3 =
              sycl_task->tens_args[0].const_mem_entry;
          auto dtens_dst_rsc_gmem_p4 = (float *)(dtens->dst_rsc->gmem_p);
          auto dtens_tmp_rsc_gmem_p5 = (float *)(dtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 1, drank, sycl_task_tens_args_const_mem_entry3,
                    dtens_dst_rsc_gmem_p4, dtens_tmp_rsc_gmem_p5, item,
                    const_args_dims_acc, const_args_prmn_acc,
                    gpu_error_count_ptr, buf0_acc.get_pointer(),
                    val_acc.get_pointer(), base_in_acc.get_pointer(),
                    base_out_acc.get_pointer(), ftb_acc.get_pointer(),
                    gtb_acc.get_pointer(), htb_acc.get_pointer(),
                    stb_acc.get_pointer(), dim_in_acc.get_pointer(),
                    dim_out_acc.get_pointer(), o2n_acc.get_pointer(),
                    n2o_acc.get_pointer(), pri_acc.get_pointer(),
                    tmp0_acc.get_pointer(), err_code_acc.get_pointer(),
                    minor_acc.get_pointer(), minor_in_acc.get_pointer(),
                    minor_out_acc.get_pointer(), s1_ind_acc.get_pointer(),
                    s1_ond_acc.get_pointer(), s1_step_acc.get_pointer(),
                    s1_dim_acc.get_pointer(), s2_ind_acc.get_pointer(),
                    s2_ond_acc.get_pointer(), s2_step_acc.get_pointer(),
                    s2_dim_acc.get_pointer(), ns1_acc.get_pointer(),
                    ns2_acc.get_pointer(), vol_acc.get_pointer(),
                    vol_ext_acc.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry3 =
              sycl_task->tens_args[0].const_mem_entry;
          auto dtens_dst_rsc_gmem_p4 = (double *)(dtens->dst_rsc->gmem_p);
          auto dtens_tmp_rsc_gmem_p5 = (double *)(dtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 1, drank, sycl_task_tens_args_const_mem_entry3,
                    dtens_dst_rsc_gmem_p4, dtens_tmp_rsc_gmem_p5, item,
                    const_args_dims_acc, const_args_prmn_acc,
                    gpu_error_count_ptr, buf0_acc.get_pointer(),
                    val_acc.get_pointer(), base_in_acc.get_pointer(),
                    base_out_acc.get_pointer(), ftb_acc.get_pointer(),
                    gtb_acc.get_pointer(), htb_acc.get_pointer(),
                    stb_acc.get_pointer(), dim_in_acc.get_pointer(),
                    dim_out_acc.get_pointer(), o2n_acc.get_pointer(),
                    n2o_acc.get_pointer(), pri_acc.get_pointer(),
                    tmp0_acc.get_pointer(), err_code_acc.get_pointer(),
                    minor_acc.get_pointer(), minor_in_acc.get_pointer(),
                    minor_out_acc.get_pointer(), s1_ind_acc.get_pointer(),
                    s1_ond_acc.get_pointer(), s1_step_acc.get_pointer(),
                    s1_dim_acc.get_pointer(), s2_ind_acc.get_pointer(),
                    s2_ond_acc.get_pointer(), s2_step_acc.get_pointer(),
                    s2_dim_acc.get_pointer(), ns1_acc.get_pointer(),
                    ns2_acc.get_pointer(), vol_acc.get_pointer(),
                    vol_ext_acc.get_pointer());
              });
        });
        break;
      case C4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry3 =
              sycl_task->tens_args[0].const_mem_entry;
          auto dtens_dst_rsc_gmem_p4 =
              (talshComplex4 *)(dtens->dst_rsc->gmem_p);
          auto dtens_tmp_rsc_gmem_p5 =
              (talshComplex4 *)(dtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 1, drank, sycl_task_tens_args_const_mem_entry3,
                    dtens_dst_rsc_gmem_p4, dtens_tmp_rsc_gmem_p5, item,
                    const_args_dims_acc, const_args_prmn_acc,
                    gpu_error_count_ptr, buf0_acc.get_pointer(),
                    val_acc.get_pointer(), base_in_acc.get_pointer(),
                    base_out_acc.get_pointer(), ftb_acc.get_pointer(),
                    gtb_acc.get_pointer(), htb_acc.get_pointer(),
                    stb_acc.get_pointer(), dim_in_acc.get_pointer(),
                    dim_out_acc.get_pointer(), o2n_acc.get_pointer(),
                    n2o_acc.get_pointer(), pri_acc.get_pointer(),
                    tmp0_acc.get_pointer(), err_code_acc.get_pointer(),
                    minor_acc.get_pointer(), minor_in_acc.get_pointer(),
                    minor_out_acc.get_pointer(), s1_ind_acc.get_pointer(),
                    s1_ond_acc.get_pointer(), s1_step_acc.get_pointer(),
                    s1_dim_acc.get_pointer(), s2_ind_acc.get_pointer(),
                    s2_ond_acc.get_pointer(), s2_step_acc.get_pointer(),
                    s2_dim_acc.get_pointer(), ns1_acc.get_pointer(),
                    ns2_acc.get_pointer(), vol_acc.get_pointer(),
                    vol_ext_acc.get_pointer());
              });
        });
        break;
      case C8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry3 =
              sycl_task->tens_args[0].const_mem_entry;
          auto dtens_dst_rsc_gmem_p4 =
              (talshComplex8 *)(dtens->dst_rsc->gmem_p);
          auto dtens_tmp_rsc_gmem_p5 =
              (talshComplex8 *)(dtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 1, drank, sycl_task_tens_args_const_mem_entry3,
                    dtens_dst_rsc_gmem_p4, dtens_tmp_rsc_gmem_p5, item,
                    const_args_dims_acc, const_args_prmn_acc,
                    gpu_error_count_ptr, buf0_acc.get_pointer(),
                    val_acc.get_pointer(), base_in_acc.get_pointer(),
                    base_out_acc.get_pointer(), ftb_acc.get_pointer(),
                    gtb_acc.get_pointer(), htb_acc.get_pointer(),
                    stb_acc.get_pointer(), dim_in_acc.get_pointer(),
                    dim_out_acc.get_pointer(), o2n_acc.get_pointer(),
                    n2o_acc.get_pointer(), pri_acc.get_pointer(),
                    tmp0_acc.get_pointer(), err_code_acc.get_pointer(),
                    minor_acc.get_pointer(), minor_in_acc.get_pointer(),
                    minor_out_acc.get_pointer(), s1_ind_acc.get_pointer(),
                    s1_ond_acc.get_pointer(), s1_step_acc.get_pointer(),
                    s1_dim_acc.get_pointer(), s2_ind_acc.get_pointer(),
                    s2_ond_acc.get_pointer(), s2_step_acc.get_pointer(),
                    s2_dim_acc.get_pointer(), ns1_acc.get_pointer(),
                    ns2_acc.get_pointer(), vol_acc.get_pointer(),
                    vol_ext_acc.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 38);
        errc = gpu_activate(cur_gpu);
        return 38;
      }
    } else if (TRANS_SHMEM == EFF_TRN_ON_CUTT) {
      errc = sycl_task_record(sycl_task, coh_ctrl, 65);
      errc = gpu_activate(cur_gpu);
      return 65;
    } else if (TRANS_SHMEM == EFF_TRN_OFF) {
      bx = 1 + (vol_d - 1) / THRDS_TENSOR_COPY_SCAT;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (dtens->data_kind) {
      case R4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_dst_rsc_gmem_p_ct4 = (float *)(dtens->dst_rsc->gmem_p);
          auto dtens_tmp_rsc_gmem_p_ct5 = (float *)(dtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_dst_rsc_gmem_p_ct4 = (double *)(dtens->dst_rsc->gmem_p);
          auto dtens_tmp_rsc_gmem_p_ct5 = (double *)(dtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case C4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_dst_rsc_gmem_p_ct4 = (talshComplex4 *)(dtens->dst_rsc->gmem_p);
          auto dtens_tmp_rsc_gmem_p_ct5 = (talshComplex4 *)(dtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case C8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_dst_rsc_gmem_p_ct4 = (talshComplex8 *)(dtens->dst_rsc->gmem_p);
          auto dtens_tmp_rsc_gmem_p_ct5 = (talshComplex8 *)(dtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 39);
        errc = gpu_activate(cur_gpu);
        return 39;
      }
    } else {
      errc = sycl_task_record(sycl_task, coh_ctrl, 59);
      errc = gpu_activate(cur_gpu);
      return 59;
    }
    darg = dtens->tmp_rsc->gmem_p;
  } else {
    darg = dtens->dst_rsc->gmem_p;
  }
  // Left tensor transpose:
  if (perm_l == YEP) {
    if (TRANS_SHMEM == EFF_TRN_ON) {
      bx = 1 + (vol_l - 1) / THRDS_TENSOR_COPY;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (ltens->data_kind) {
      case R4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p_ct4 = (float *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p_ct5 = (float *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                    ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p_ct4 = (double *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p_ct5 = (double *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                    ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case C4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p_ct4 = (talshComplex4 *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p_ct5 = (talshComplex4 *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                    ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case C8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p_ct4 = (talshComplex8 *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p_ct5 = (talshComplex8 *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                    ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 40);
        errc = gpu_activate(cur_gpu);
        return 40;
      }
    } else if (TRANS_SHMEM == EFF_TRN_ON_CUTT) {
      errc = sycl_task_record(sycl_task, coh_ctrl, 68);
      errc = gpu_activate(cur_gpu);
      return 68;
    } else if (TRANS_SHMEM == EFF_TRN_OFF) {
      bx = 1 + (vol_l - 1) / THRDS_TENSOR_COPY_SCAT;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (ltens->data_kind) {
      case R4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p_ct4 = (float *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p_ct5 = (float *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                    ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p_ct4 = (double *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p_ct5 = (double *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                    ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case C4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p_ct4 = (talshComplex4 *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p_ct5 = (talshComplex4 *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                    ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case C8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p_ct4 = (talshComplex8 *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p_ct5 = (talshComplex8 *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                    ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 41);
        errc = gpu_activate(cur_gpu);
        return 41;
      }
    } else {
      errc = sycl_task_record(sycl_task, coh_ctrl, 60);
      errc = gpu_activate(cur_gpu);
      return 60;
    }
    larg = ltens->tmp_rsc->gmem_p;
  } else {
    larg = ltens->dst_rsc->gmem_p;
  }
// Schedule the appropriate computation kernel:
  // Addition kernel:
  bx = 1 + (vol_d - 1) / THRDS_ARRAY_ADD;
  if (bx > MAX_SYCL_BLOCKS)
    bx = MAX_SYCL_BLOCKS;
  switch (dtens->data_kind) {
  case R4:
    (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for(
          cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
          [=](cl::sycl::nd_item<1> item) {
            gpu_array_add__(vol_d, (float *)darg, (float *)larg,
                            (float)scale_real, item, 0);
          });
    });
    gpu_stats[gpu_num].flops += 2.0 * ((double)(dsize)); // 1 mul, 1 add SP
    break;
  case R8:
    (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for(
          cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
          [=](cl::sycl::nd_item<1> item) {
            gpu_array_add__(vol_d, (double *)darg, (double *)larg, scale_real,
                            item, 0);
          });
    });
    gpu_stats[gpu_num].flops += 2.0 * ((double)(dsize)); // 1 mul, 1 add DP
    break;
  case C4:
    scale_cmplx4 = talshComplex4Set((float)scale_real, (float)scale_imag);
    (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for(
          cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
          [=](cl::sycl::nd_item<1> item) {
            gpu_array_add__(vol_d, (talshComplex4 *)darg, (talshComplex4 *)larg,
                            scale_cmplx4, item, conj_l);
          });
    });
    gpu_stats[gpu_num].flops += 8.0 * ((double)(dsize)); // 4 mul, 4 add SP
    break;
  case C8:
    scale_cmplx8 = talshComplex8Set(scale_real, scale_imag);
    (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for(
          cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
          [=](cl::sycl::nd_item<1> item) {
            gpu_array_add__(vol_d, (talshComplex8 *)darg, (talshComplex8 *)larg,
                            scale_cmplx8, item, conj_l);
          });
    });
    gpu_stats[gpu_num].flops += 8.0 * ((double)(dsize)); // 4 mul, 4 add DP
    break;
  default:
    errc = sycl_task_record(sycl_task, coh_ctrl, 48);
    errc = gpu_activate(cur_gpu);
    return 48;
  }
  // Schedule the inverse tensor transpose for the destination tensor (should
  // not happen actually):
  if (perm_d == YEP) {
    if (TRANS_SHMEM == EFF_TRN_ON) {
      bx = 1 + (vol_d - 1) / THRDS_TENSOR_COPY;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (dtens->data_kind) {
      case R4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (float *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (float *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (double *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (double *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case C4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (talshComplex4 *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (talshComplex4 *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case C8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (talshComplex8 *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (talshComplex8 *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 54);
        errc = gpu_activate(cur_gpu);
        return 54;
      }
    } else if (TRANS_SHMEM == EFF_TRN_ON_CUTT) {
      errc = sycl_task_record(sycl_task, coh_ctrl, 65);
      errc = gpu_activate(cur_gpu);
      return 65;
    } else if (TRANS_SHMEM == EFF_TRN_OFF) {
      bx = 1 + (vol_d - 1) / THRDS_TENSOR_COPY_SCAT;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (dtens->data_kind) {
      case R4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (float *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (float *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (double *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (double *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case C4:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (talshComplex4 *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (talshComplex4 *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case C8:
        (*sycl_queue)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (talshComplex8 *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (talshComplex8 *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 55);
        errc = gpu_activate(cur_gpu);
        return 55;
      }
    } else {
      errc = sycl_task_record(sycl_task, coh_ctrl, 62);
      errc = gpu_activate(cur_gpu);
      return 62;
    }
  }
  // Record a SYCL queue (output ready on GPU):
  cuda_output_ct = std::chrono::high_resolution_clock::now();
  err = 0;
  if (err != 0) {
    if (VERBOSE)
      printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_add): Unable "
             "to record the output event: %s\n",
             err_msg);
    errc = sycl_task_record(sycl_task, coh_ctrl, 56);
    errc = gpu_activate(cur_gpu);
    return 56;
  }
  // Transfer back the updated destination tensor if needed ("T","K" coherence
  // control):
  coh = (coh_ctrl >> 2) &
        (TWO_BITS_SET); // select bits 2,3 (destination tensor coherence)
  if (gpu_d != gpu_num && coh >= 2) { // data is not on the computing GPU and
                                      // coherence control = 2("T") or (3)"K":
    (*sycl_queue)
        ->memcpy(dtens->src_rsc->gmem_p, dtens->dst_rsc->gmem_p, dsize);
    if (err != 0) {
      if (VERBOSE)
        printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_add): "
               "Destination tensor body back copy failed: %s\n",
               err_msg);
      errc = sycl_task_record(sycl_task, coh_ctrl, 57);
      errc = gpu_activate(cur_gpu);
      return 57;
    }
    gpu_stats[gpu_num].traffic_out += dsize;
  }
  // Record a SYCL queue (task finished):
  cuda_finish_ct = std::chrono::high_resolution_clock::now();
  err = 0;
  if (err != 0) {
    if (VERBOSE)
      printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_add): Unable "
             "to record the finish event: %s\n",
             err_msg);
    errc = sycl_task_record(sycl_task, coh_ctrl, 58);
    errc = gpu_activate(cur_gpu);
    return 58;
  }
  // Record the successfully scheduled SYCL task and update the Last Task:
  errc = sycl_task_record(sycl_task, coh_ctrl, 0);
  LastTask[gpu_num] = sycl_task;
  if (gpu_num != cur_gpu)
    errc = gpu_activate(cur_gpu);
  return stat; // either 0 (success) or NOT_CLEAN (warning)
} catch (cl::sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
//-------------------------------------------------------------------------------------------------------------------
// TENSOR CONTRACTION (non-blocking):
int gpu_tensor_block_contract_dlf(const int *cptrn, tensBlck_t *ltens,
                                  tensBlck_t *rtens, tensBlck_t *dtens,
                                  unsigned int coh_ctrl, cudaTask_t *sycl_task,
                                  int gpu_id, double scale_real,
                                  double scale_imag, int conj_bits,
                                  int accumulative)
    /**
       dtens(:)+=ltens(:)*rtens(:)
       INPUT:
       # cptrn(1:lrank+rrank) - contraction pattern: Position correspondence:
       Uncontracted indices are positive, contracted are negative;
       # ltens - left tensor argument (initialized!);
       # rtens - right tensor argument (initialized!);
       # dtens - destination tensor argument (initialized!);
       # coh_ctrl - one of the COPY_XXX parameters regulating the data presence
    for each tensor argument; # sycl_task - pointer to an empty (clean) CUDA
    task; # gpu_id - suggested GPU ID on which the operation is to be scheduled
    (-1: defaults to the optimal one); # scale_real - real part of the GEMM
    alpha coefficient (defaults to 1.0); # scale_imag - imaginary part of the
    GEMM alpha coefficient (defaults to 0.0); # conj_bits - tensor argument
    complex conjugation bits, one bit per argument: {0:D,1:L,2:R}; #
    accumulative - accumulate in (default) VS overwrite destination tensor:
    [YEP|NOPE]; OUTPUT: # dtens - updated destination tensor; # sycl_task -
    recorded SYCL task (either successfully scheduled or failed). NOTES: # If
    the tensor operation has been scheduled successfully, a recorded (active)
    SYCL task will be returned along with zero return status. A scheduling error
    results in either a negative (at early stages) or positive (at later stages)
    return status. In the former case the SYCL task is left clean, while at the
    latter case it will be recorded as failed (error). # Special return statuses
    TRY_LATER and DEVICE_UNABLE are not errors but merely indicators of the
    current or permanent lack of resources, respectively. However, the SYCL task
    status in these cases will still be set to an error (always check the
    function return status!). # If <gpu_id> is out of the legitimate GPU range,
    it will be replaced by an optimal one, based on argument residence and the
    current load of GPU(s).
    **/
    try {
  int i, j, drank, lrank, rrank, tds_d, tds_l, tds_r, gpu_d, gpu_l, gpu_r,
      perm_d, perm_l, perm_r;
  int ncd, nlu, nru, gpu_num, cur_gpu, targ_dev, bx, by, errc, stat, conj_l,
      conj_r, fast_math;
  int dprm[1 + MAX_TENSOR_RANK], lprm[1 + MAX_TENSOR_RANK],
      rprm[1 +
           MAX_TENSOR_RANK]; // the 1st element is the sign of the permutation
  size_t vol_d, vol_l, vol_r, dsize, lsize, rsize, lc, ll, lr, pofs;
  unsigned int coh;
  const unsigned int TWO_BITS_SET = 3; // two right bits are set
  void *darg, *larg, *rarg, *alpha_plus_p, *alpha_minus_p, *beta_p, *beta_one_p;
  talshComplex4 scale_cmplx4;
  talshComplex8 scale_cmplx8;
  sycl::queue **sycl_stream;
  sycl::event *sycl_start, *sycl_comput, *sycl_output, *sycl_finish, *dep_event;
  std::chrono::time_point<std::chrono::high_resolution_clock> sycl_start_ct;
  std::chrono::time_point<std::chrono::high_resolution_clock> sycl_comput_ct;
  std::chrono::time_point<std::chrono::high_resolution_clock> sycl_output_ct;
  std::chrono::time_point<std::chrono::high_resolution_clock> sycl_finish_ct;
  int err;
  const char *err_msg;
#ifndef NO_BLAS
  int err_onemkl;
  oneapi::mkl::transpose left_conj, right_conj;
#endif

  // if(DEBUG)
  // printf("\n#DEBUG(tensor_algebra_gpu_intel:gpu_tensor_block_contract_dlf):
  // GPU Tensor Contraction:\n"); //debug
  stat = 0; // return status in case of successful scheduling
  // Check function arguments:
  if (cptrn == nullptr || dtens == nullptr || ltens == nullptr ||
      rtens == nullptr || sycl_task == nullptr)
    return -1;
  if (tensBlck_present(dtens) != YEP || tensBlck_present(ltens) != YEP ||
      tensBlck_present(rtens) != YEP)
    return -2; // tensor blocks must reside in some device memory
  if (sycl_task_gpu_id(sycl_task) >= 0)
    return -3; // SYCL task is not clean (destruct/clean it first)
  // Check tensor arguments:
  drank = (dtens->shape).num_dim; // destination tensor rank
  lrank = (ltens->shape).num_dim; // left tensor rank
  rrank = (rtens->shape).num_dim; // right tensor rank
  if (drank < 0 || drank > MAX_TENSOR_RANK || lrank < 0 ||
      lrank > MAX_TENSOR_RANK || rrank < 0 || rrank > MAX_TENSOR_RANK)
    return -4;
  if (tens_valid_data_kind(dtens->data_kind, &tds_d) !=
          YEP || // tds_d: destination tensor element size in bytes
      tens_valid_data_kind(ltens->data_kind, &tds_l) !=
          YEP || // tds_l: left tensor element size in bytes
      tens_valid_data_kind(rtens->data_kind, &tds_r) != YEP)
    return -5; // tds_r: right tensor element size in bytes
  if (!(dtens->data_kind > 0 && ltens->data_kind == dtens->data_kind &&
        rtens->data_kind == dtens->data_kind))
    return -6; // data kind mismatch
  if (dtens->src_rsc == nullptr || ltens->src_rsc == nullptr ||
      rtens->src_rsc == nullptr)
    return -7; // source resource must always be present
  if (tensDevRsc_is_empty(dtens->src_rsc) != NOPE)
    return -8; // source resource must be present (tensor body)
  if (tensDevRsc_is_empty(ltens->src_rsc) != NOPE)
    return -9; // source resource must be present (tensor body)
  if (tensDevRsc_is_empty(rtens->src_rsc) != NOPE)
    return -10; // source resource must be present (tensor body)
  // Check the contraction pattern and tensor dimension extent correspondence:
  for (i = 0; i < drank; i++)
    dprm[i] = 0;
  for (i = 0; i < lrank; i++)
    lprm[i] = 0;
  for (i = 0; i < rrank; i++)
    rprm[i] = 0;
  for (i = 0; i < lrank; i++) { // position in ltens
    j = cptrn[i];
    if (j > 0) { // position in dtens
      if (j > drank)
        return -11;
      if ((dtens->shape).dims[j - 1] != (ltens->shape).dims[i])
        return -12;
      if (dprm[j - 1] == 0) {
        dprm[j - 1] = 1;
      } else {
        return -13;
      }
    } else if (j < 0) { // position in rtens
      if (-j > rrank)
        return -14;
      if ((rtens->shape).dims[-j - 1] != (ltens->shape).dims[i])
        return -15;
      if (cptrn[lrank + (-j - 1)] != -(i + 1))
        return -16;
      if (rprm[-j - 1] == 0) {
        rprm[-j - 1] = 1;
      } else {
        return -17;
      }
    } else {
      return -18;
    }
  }
  for (i = 0; i < rrank; i++) { // position in rtens
    j = cptrn[lrank + i];
    if (j > 0) { // position in dtens
      if (j > drank)
        return -19;
      if ((dtens->shape).dims[j - 1] != (rtens->shape).dims[i])
        return -20;
      if (dprm[j - 1] == 0) {
        dprm[j - 1] = 1;
      } else {
        return -21;
      }
    } else if (j < 0) { // position in ltens
      if (-j > lrank)
        return -22;
      if ((ltens->shape).dims[-j - 1] != (rtens->shape).dims[i])
        return -23;
      if (cptrn[-j - 1] != -(i + 1))
        return -24;
      if (lprm[-j - 1] == 0) {
        lprm[-j - 1] = 1;
      } else {
        return -25;
      }
    } else {
      return -26;
    }
  }
  for (i = 0; i < drank; i++)
    if (dprm[i] != 1)
      return -27;
// Check argument complex conjugation bits:
#ifndef NO_BLAS
  left_conj = oneapi::mkl::transpose::trans;
  right_conj = oneapi::mkl::transpose::nontrans; // default is TN GEMM
#endif
  conj_bits =
      conj_bits &
      7; // keep only first three bits, one per tensor argument {0:D,1:L,2:R}
  if (conj_bits & 1) { // destination tensor argument conjugation = inverse
                       // conjugation of left and right tensor arguments
    conj_bits = conj_bits ^ 7; // XOR with 0b111 will invert bits
  }
  if (dtens->data_kind == C4 ||
      dtens->data_kind == C8) { // conjugation may apply to complex data kinds
    conj_l = 0;
    if ((conj_bits & 2) != 0)
      conj_l = 1; // left tensor argument conjugation flag
    conj_r = 0;
    if ((conj_bits & 4) != 0)
      conj_r = 1; // right tensor argument conjugation flag
#ifndef NO_BLAS
    if (conj_l != 0)
      left_conj = oneapi::mkl::transpose::conjtrans;
    if (conj_r != 0)
      right_conj = oneapi::mkl::transpose::conjtrans;
#endif
  } else {
    conj_bits = 0;
    conj_l = 0;
    conj_r = 0; // no conjugation for real data kinds
  }
  // Activate the right GPU:
  if (gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE) {
    gpu_num = tens_op_best_gpu(dtens, ltens, rtens);
  } else {
    gpu_num = gpu_id;
  }
  if (gpu_is_mine(gpu_num) <= GPU_OFF)
    return -28; // GPU is not mine or error
  gpu_stats[gpu_num].tasks_submitted++;
  gpu_d = decode_device_id(dtens->src_rsc->dev_id, &j);
  if (gpu_d < 0)
    return -29; // destination tensor source device id
  if (j == DEV_INTEL_GPU) {
    if (gpu_d != gpu_num) {
      return DEVICE_UNABLE; // peer access impossible for this GPU device
    }
  } else if (j == DEV_HOST) {
    gpu_d = -1; // data is in Host memory
  } else {
    return DEVICE_UNABLE; // data is not in Host or GPU memory
  }
  gpu_l = decode_device_id(ltens->src_rsc->dev_id, &j);
  if (gpu_l < 0)
    return -30; // left tensor source device id
  if (j == DEV_INTEL_GPU) {
    if (gpu_l != gpu_num) {
      return DEVICE_UNABLE; // peer access impossible for this GPU device
    }
  } else if (j == DEV_HOST) {
    gpu_l = -1; // data is in Host memory
  } else {
    return DEVICE_UNABLE; // data is not in Host or GPU memory
  }
  gpu_r = decode_device_id(rtens->src_rsc->dev_id, &j);
  if (gpu_r < 0)
    return -31; // right tensor source device id
  if (j == DEV_INTEL_GPU) {
    if (gpu_r != gpu_num) {
      return DEVICE_UNABLE; // peer access impossible for this GPU device
    }
  } else if (j == DEV_HOST) {
    gpu_r = -1; // data is in Host memory
  } else {
    return DEVICE_UNABLE; // data is not in Host or GPU memory
  }
  cur_gpu = gpu_in_focus(); // save the current GPU
  if (gpu_num != cur_gpu) {
    errc = gpu_activate(gpu_num);
    if (errc) {
      errc = gpu_activate(cur_gpu);
      return -32;
    }
  }        // activate the target GPU
  err = 0; // abb: clear the GPU error status
  targ_dev = encode_device_id(DEV_INTEL_GPU, gpu_num); // flat device id
  // Construct a SYCL task (acquire CUDA resources) for the target GPU:
  errc = sycl_task_construct(sycl_task, gpu_num);
  if (errc) {
    i = gpu_activate(cur_gpu);
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      return errc;
    } else {
      return -33;
    }
  }

  // *** From this point all error codes must be positive and the SYCL task must
  // be recorded! ***
  // Set up tensor arguments (allocates additional resources for each tensor
  // argument):
  // Destination argument:
  errc = sycl_task_set_arg(sycl_task, 0, dtens);
  if (errc) {
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
      i = gpu_activate(
          cur_gpu); // not an error if TRY_LATER or DEVICE_UNABLE are returned
      return errc;
    } else {
      i = sycl_task_record(sycl_task, coh_ctrl, 1);
      i = gpu_activate(cur_gpu);
      return 1;
    }
  }
  // Left argument:
  errc = sycl_task_set_arg(sycl_task, 1, ltens);
  if (errc) {
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
      i = gpu_activate(
          cur_gpu); // not an error if TRY_LATER or DEVICE_UNABLE are returned
      return errc;
    } else {
      i = sycl_task_record(sycl_task, coh_ctrl, 2);
      i = gpu_activate(cur_gpu);
      return 2;
    }
  }
  // Right argument:
  errc = sycl_task_set_arg(sycl_task, 2, rtens);
  if (errc) {
    if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
      i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
      i = gpu_activate(
          cur_gpu); // not an error if TRY_LATER or DEVICE_UNABLE are returned
      return errc;
    } else {
      i = sycl_task_record(sycl_task, coh_ctrl, 3);
      i = gpu_activate(cur_gpu);
      return 3;
    }
  }
  // Associate SYCL queue and event pointers locally for convenience:
  sycl_stream = sycl_stream_ptr(sycl_task->gpu_id, sycl_task->queue_hl);
  if (sycl_stream == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 4);
    errc = gpu_activate(cur_gpu);
    return 4;
  }
  sycl_start = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_start_hl);
  if (sycl_start == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 5);
    errc = gpu_activate(cur_gpu);
    return 5;
  }
  sycl_comput = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_comput_hl);
  if (sycl_comput == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 6);
    errc = gpu_activate(cur_gpu);
    return 6;
  }
  sycl_output = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_output_hl);
  if (sycl_output == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 7);
    errc = gpu_activate(cur_gpu);
    return 7;
  }
  sycl_finish = sycl_event_ptr(sycl_task->gpu_id, sycl_task->event_finish_hl);
  if (sycl_finish == nullptr) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 8);
    errc = gpu_activate(cur_gpu);
    return 8;
  }
  // Determine the volume and required matricization permutation for each tensor
  // argument:
  if (drank > 0 && lrank > 0 && rrank > 0 &&
      drank <
          (lrank + rrank)) { // GEMM mapped tensor contraction: {TN, NT, NN, TT}
    get_contr_permutations(1, 0, lrank, rrank, cptrn, conj_bits, dprm, lprm,
                           rprm, &ncd, &nlu, &nru,
                           &errc); // permutations and numbers of dimensions
  } else { // custom kernel mapped tensor contraction (complex conjugation does
           // not require modified permutations)
    get_contr_permutations(1, 0, lrank, rrank, cptrn, 0, dprm, lprm, rprm, &ncd,
                           &nlu, &nru,
                           &errc); // permutations and numbers of dimensions
  }
  // Get permutations:
  if (errc) {
    i = sycl_task_record(sycl_task, coh_ctrl, 11);
    i = gpu_activate(cur_gpu);
    return 11;
  }
  for (i = 0; i < drank; i++)
    sycl_task->tens_args[0].prmn_p[i] =
        dprm[1 + i]; // ignore the permutaion sign
  perm_d =
      non_trivial_prmn(drank, sycl_task->tens_args[0].prmn_p); // trivial or not
  for (i = 0; i < lrank; i++)
    sycl_task->tens_args[1].prmn_p[i] =
        lprm[1 + i]; // ignore the permutaion sign
  perm_l =
      non_trivial_prmn(lrank, sycl_task->tens_args[1].prmn_p); // trivial or not
  for (i = 0; i < rrank; i++)
    sycl_task->tens_args[2].prmn_p[i] =
        rprm[1 + i]; // ignore the permutaion sign
  perm_r =
      non_trivial_prmn(rrank, sycl_task->tens_args[2].prmn_p); // trivial or not
  // Get tensor volumes, sizes and matrix attributes:
  vol_d = tensBlck_volume(dtens);
  vol_l = tensBlck_volume(ltens);
  vol_r = tensBlck_volume(rtens); // tensor block volumes
  lc = 1;
  ll = 1;
  for (i = 0; i < lrank; i++) {
    if (sycl_task->tens_args[1].prmn_p[i] <= ncd) {
      lc *= ((ltens->shape).dims[i]);
    } else {
      ll *= ((ltens->shape).dims[i]);
    }
  }
  lr = vol_d / ll;
  if (vol_l <= 0 || vol_r <= 0 || vol_d <= 0 || vol_d % ll != 0 ||
      vol_r % lr != 0 || vol_r / lr != lc) {
    i = sycl_task_record(sycl_task, coh_ctrl, 12);
    i = gpu_activate(cur_gpu);
    return 12; // invalid matrix dimensions obtained
  }
  dsize = vol_d * tds_d;
  lsize = vol_l * tds_l;
  rsize = vol_r * tds_r; // tensor argument sizes in bytes
  // Check fast math requirements:
  fast_math = NOPE;

  if (gpu_query_fast_math(gpu_num) == YEP) {
    if (dtens->data_kind == R4 ||
        dtens->data_kind == C4) { //`Will require extension if new hardware
      if (lr % WMMA_ALIGN == 0 && ll % WMMA_ALIGN == 0 &&
          lc % WMMA_ALIGN == 0) {
        if (TRANS_SHMEM == EFF_TRN_ON) {
          if (dtens->data_kind == C4 ||
              dtens->data_kind ==
                  C8) { // complex data types will require real/imag split
            if (scale_real == 1.0 &&
                scale_imag == 0.0) { //`Lift this restriction in future
                                     //(requires better handling of prefactors)
              perm_d = YEP;
              perm_l = YEP;
              perm_r = YEP;
              fast_math = YEP;
            }
          } else {
            fast_math = YEP;
          }
        }
      }
    }
  }

  // Acquire global memory resources for tensor arguments if needed:
  // Set up destination memory resources in all tensors:
  //  Destination tensor:
  if (dtens->dst_rsc == dtens->src_rsc)
    dtens->dst_rsc =
        nullptr; // destination resource was pointing to the source resource
  if (gpu_d != gpu_num) { // data is on a different GPU device or Host
    if (dtens->dst_rsc == nullptr) {
      errc = tensDevRsc_create(&(dtens->dst_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 13);
        i = gpu_activate(cur_gpu);
        return 13;
      }
    } else {
      if (tensDevRsc_is_empty(dtens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(dtens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(dtens->dst_rsc, targ_dev, dsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 14);
        i = gpu_activate(cur_gpu);
        return 14;
      }
    }
  } else {
    if (dtens->dst_rsc != nullptr) {
      if (tensDevRsc_is_empty(dtens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(dtens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    dtens->dst_rsc =
        dtens->src_rsc; // destination and source resources are the same
                        // (because the data is already on the computing GPU)
  }
  //  Left tensor:
  if (ltens->dst_rsc == ltens->src_rsc)
    ltens->dst_rsc =
        nullptr; // destination resource was pointing to the source resource
  if (gpu_l != gpu_num) { // data is on a different GPU device or Host
    if (ltens->dst_rsc == nullptr) {
      errc = tensDevRsc_create(&(ltens->dst_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 15);
        i = gpu_activate(cur_gpu);
        return 15;
      }
    } else {
      if (tensDevRsc_is_empty(ltens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(ltens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(ltens->dst_rsc, targ_dev, lsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 16);
        i = gpu_activate(cur_gpu);
        return 16;
      }
    }
  } else {
    if (ltens->dst_rsc != nullptr) {
      if (tensDevRsc_is_empty(ltens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(ltens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    ltens->dst_rsc =
        ltens->src_rsc; // destination and source resources are the same
                        // (because the data is already on the computing GPU)
  }
  //  Right tensor:
  if (rtens->dst_rsc == rtens->src_rsc)
    rtens->dst_rsc =
        nullptr; // destination resource was pointing to the source resource
  if (gpu_r != gpu_num) { // data is on a different GPU device or Host
    if (rtens->dst_rsc == nullptr) {
      errc = tensDevRsc_create(&(rtens->dst_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 17);
        i = gpu_activate(cur_gpu);
        return 17;
      }
    } else {
      if (tensDevRsc_is_empty(rtens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(rtens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(rtens->dst_rsc, targ_dev, rsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 18);
        i = gpu_activate(cur_gpu);
        return 18;
      }
    }
  } else {
    if (rtens->dst_rsc != nullptr) {
      if (tensDevRsc_is_empty(rtens->dst_rsc) == NOPE) {
        errc = tensDevRsc_release_all(rtens->dst_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    rtens->dst_rsc =
        rtens->src_rsc; // destination and source resources are the same
                        // (because the data is already on the computing GPU)
  }
  // Set up temporary memory resources in all tensors if needed (because of
  // out-of-place tensor transpose):
  //  Destination tensor:
  if (perm_d == YEP) {
    if (dtens->tmp_rsc == nullptr) {
      errc = tensDevRsc_create(&(dtens->tmp_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 19);
        i = gpu_activate(cur_gpu);
        return 19;
      }
    } else {
      if (tensDevRsc_is_empty(dtens->tmp_rsc) == NOPE) {
        errc = tensDevRsc_release_all(dtens->tmp_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(dtens->tmp_rsc, targ_dev, dsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 20);
        i = gpu_activate(cur_gpu);
        return 20;
      }
    }
  }
  //  Left tensor:
  if (perm_l == YEP) {
    if (ltens->tmp_rsc == nullptr) {
      errc = tensDevRsc_create(&(ltens->tmp_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 21);
        i = gpu_activate(cur_gpu);
        return 21;
      }
    } else {
      if (tensDevRsc_is_empty(ltens->tmp_rsc) == NOPE) {
        errc = tensDevRsc_release_all(ltens->tmp_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(ltens->tmp_rsc, targ_dev, lsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 22);
        i = gpu_activate(cur_gpu);
        return 22;
      }
    }
  }
  //  Right tensor:
  if (perm_r == YEP) {
    if (rtens->tmp_rsc == nullptr) {
      errc = tensDevRsc_create(&(rtens->tmp_rsc));
      if (errc) {
        i = sycl_task_record(sycl_task, coh_ctrl, 23);
        i = gpu_activate(cur_gpu);
        return 23;
      }
    } else {
      if (tensDevRsc_is_empty(rtens->tmp_rsc) == NOPE) {
        errc = tensDevRsc_release_all(rtens->tmp_rsc);
        if (errc)
          stat = NOT_CLEAN;
      }
    }
    errc = tensDevRsc_allocate_mem(rtens->tmp_rsc, targ_dev, rsize, YEP);
    if (errc) {
      if (errc == TRY_LATER || errc == DEVICE_UNABLE) {
        i = sycl_task_record(sycl_task, coh_ctrl, NVTAL_DEFERRED);
        i = gpu_activate(cur_gpu);
        return errc;
      } else {
        i = sycl_task_record(sycl_task, coh_ctrl, 24);
        i = gpu_activate(cur_gpu);
        return 24;
      }
    }
  }
  // Start scheduling CUDA calls:
  sycl_start_ct = std::chrono::high_resolution_clock::now();
  errc = sycl_task_record(sycl_task, coh_ctrl, 25);
  errc = gpu_activate(cur_gpu);
  return 25;

  if (LastTask[gpu_num] !=
      nullptr) { //`This should be done atomically for thread safety
    dep_event = sycl_event_ptr(LastTask[gpu_num]->gpu_id,
                               LastTask[gpu_num]->event_comput_hl);
    err = (*dep_event.wait(),
           0); // input
    // transfers
    // should
    // only
    // begin
    // after
    // the
    // previous
    // task
    // input
    // transfers
    // have
    // completed
    if (err != 0) {
      if (VERBOSE)
        printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_"
               "dlf): Unable to create a task dependency: %s\n",
               err_msg);
      errc = sycl_task_record(sycl_task, coh_ctrl, 26);
      errc = gpu_activate(cur_gpu);
      return 26;
    }
  }
  // Schedule forward data transfers for all tensors if needed:
  // Left tensor:
  if (sycl_task->tens_args[1].const_mem_entry >=
      0) { // GPU constant memory entry will contain tensor dimension extents
           // and the matricization permutation (if any)
    err = (*sycl_stream->memcpy(
               (char *)(const_args_dims.get_ptr()) +
                   sizeof(int) *
                       ((size_t)(MAX_TENSOR_RANK *
                                 (sycl_task->tens_args[1].const_mem_entry))),
               (void *)(ltens->shape.dims), sizeof(int) * ((size_t)lrank)),
           0); // tensor
    // dimension
    // extents
    if (err != 0) {
      if (VERBOSE)
        printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_"
               "dlf): Left tensor dims H2D copy failed: %s\n",
               err_msg);
      errc = sycl_task_record(sycl_task, coh_ctrl, 27);
      errc = gpu_activate(cur_gpu);
      return 27;
    }
    gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)lrank);
    if (perm_l == YEP) {
      *sycl_stream->memcpy((char *)(const_args_prmn.get_ptr()) +
			   sizeof(int) *
			   ((size_t)(MAX_TENSOR_RANK *
				     (sycl_task->tens_args[1].const_mem_entry))),
			   (void *)(sycl_task->tens_args[1].prmn_p),
			   sizeof(int) * ((size_t)lrank)); // tensor matricization permutation
      if (err != 0) {
        if (VERBOSE)
          printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_"
                 "dlf): Left tensor prmn H2D copy failed: %s\n",
                 err_msg);
        errc = sycl_task_record(sycl_task, coh_ctrl, 28);
        errc = gpu_activate(cur_gpu);
        return 28;
      }
      gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)lrank);
    }
    if (gpu_l != gpu_num) { // data is not on the computing GPU
      err = (*sycl_stream->memcpy(ltens->dst_rsc->gmem_p,
                                  ltens->src_rsc->gmem_p, lsize),
             0);
      if (err != 0) {
        if (VERBOSE)
          printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_"
                 "dlf): Left tensor body copy failed: %s\n",
                 err_msg);
        errc = sycl_task_record(sycl_task, coh_ctrl, 29);
        errc = gpu_activate(cur_gpu);
        return 29;
      }
      gpu_stats[gpu_num].traffic_in += lsize;
    }
  } else {
    errc = sycl_task_record(sycl_task, coh_ctrl, 30);
    errc = gpu_activate(cur_gpu);
    return 30;
  }
  // Right tensor:
  if (sycl_task->tens_args[2].const_mem_entry >=
      0) { // GPU constant memory entry will contain tensor dimension extents
           // and the matricization permutation (if any)
    err = (*sycl_stream->memcpy(
               (char *)(const_args_dims.get_ptr()) +
                   sizeof(int) *
                       ((size_t)(MAX_TENSOR_RANK *
                                 (sycl_task->tens_args[2].const_mem_entry))),
               (void *)(rtens->shape.dims), sizeof(int) * ((size_t)rrank)),
           0); // tensor
    // dimension
    // extents
    if (err != 0) {
      if (VERBOSE)
        printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_"
               "dlf): Right tensor dims H2D copy failed: %s\n",
               err_msg);
      errc = sycl_task_record(sycl_task, coh_ctrl, 31);
      errc = gpu_activate(cur_gpu);
      return 31;
    }
    gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)rrank);
    if (perm_r == YEP) {
      err = (*sycl_stream->memcpy(
                 (char *)(const_args_prmn.get_ptr()) +
                     sizeof(int) *
                         ((size_t)(MAX_TENSOR_RANK *
                                   (sycl_task->tens_args[2].const_mem_entry))),
                 (void *)(sycl_task->tens_args[2].prmn_p),
                 sizeof(int) * ((size_t)rrank)),
             0); // tensor matricization permutation
      if (err != 0) {
        if (VERBOSE)
          printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_"
                 "dlf): Right tensor prmn H2D copy failed: %s\n",
                 err_msg);
        errc = sycl_task_record(sycl_task, coh_ctrl, 32);
        errc = gpu_activate(cur_gpu);
        return 32;
      }
      gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)rrank);
    }
    if (gpu_r != gpu_num) { // data is not on the computing GPU
      err = (*sycl_stream->memcpy(rtens->dst_rsc->gmem_p,
                                  rtens->src_rsc->gmem_p, rsize),
             0);
      if (err != 0) {
        if (VERBOSE)
          printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_"
                 "dlf): Right tensor body copy failed: %s\n",
                 err_msg);
        errc = sycl_task_record(sycl_task, coh_ctrl, 33);
        errc = gpu_activate(cur_gpu);
        return 33;
      }
      gpu_stats[gpu_num].traffic_in += rsize;
    }
  } else {
    errc = sycl_task_record(sycl_task, coh_ctrl, 34);
    errc = gpu_activate(cur_gpu);
    return 34;
  }
  // Destination tensor:
  if (sycl_task->tens_args[0].const_mem_entry >=
      0) { // GPU constant memory entry will contain tensor dimension extents
           // and the matricization permutation (if any)
    err = (*sycl_stream->memcpy(
               (char *)(const_args_dims.get_ptr()) +
                   sizeof(int) *
                       ((size_t)(MAX_TENSOR_RANK *
                                 (sycl_task->tens_args[0].const_mem_entry))),
               (void *)(dtens->shape.dims), sizeof(int) * ((size_t)drank)),
           0); // tensor
    // dimension
    // extents
    if (err != 0) {
      if (VERBOSE)
        printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_"
               "dlf): Destination tensor dims H2D copy failed: %s\n",
               err_msg);
      errc = sycl_task_record(sycl_task, coh_ctrl, 35);
      errc = gpu_activate(cur_gpu);
      return 35;
    }
    gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)drank);
    if (perm_d == YEP) {
      err = (*sycl_stream->memcpy(
                 (char *)(const_args_prmn.get_ptr()) +
                     sizeof(int) *
                         ((size_t)(MAX_TENSOR_RANK *
                                   (sycl_task->tens_args[0].const_mem_entry))),
                 (void *)(sycl_task->tens_args[0].prmn_p),
                 sizeof(int) * ((size_t)drank)),
             0); // tensor matricization permutation
      if (err != 0) {
        if (VERBOSE)
          printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_"
                 "dlf): Destination tensor prmn H2D copy failed: %s\n",
                 err_msg);
        errc = sycl_task_record(sycl_task, coh_ctrl, 36);
        errc = gpu_activate(cur_gpu);
        return 36;
      }
      gpu_stats[gpu_num].traffic_in += sizeof(int) * ((size_t)drank);
    }
    if (gpu_d != gpu_num &&
        accumulative != NOPE) { // data is not on the computing GPU
      err = (*sycl_stream->memcpy(dtens->dst_rsc->gmem_p,
                                  dtens->src_rsc->gmem_p, dsize),
             0);
      if (err != 0) {
        if (VERBOSE)
          printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_"
                 "dlf): Destination tensor body copy failed: %s\n",
                 err_msg);
        errc = sycl_task_record(sycl_task, coh_ctrl, 37);
        errc = gpu_activate(cur_gpu);
        return 37;
      }
      gpu_stats[gpu_num].traffic_in += dsize;
    }
  } else {
    errc = sycl_task_record(sycl_task, coh_ctrl, 38);
    errc = gpu_activate(cur_gpu);
    return 38;
  }
  // Schedule tensor transposes if needed:
  // Record a SYCL queue:
  sycl_comput_ct = std::chrono::high_resolution_clock::now();
  err = 0;
  if (err != 0) {
    if (VERBOSE)
      printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_dlf):"
             " Unable to record the compute event: %s\n",
             err_msg);
    errc = sycl_task_record(sycl_task, coh_ctrl, 39);
    errc = gpu_activate(cur_gpu);
    return 39;
  }
  // Destination tensor transpose:
  if (perm_d == YEP) {
    if (accumulative != NOPE) {
      if (TRANS_SHMEM == EFF_TRN_ON) {
        bx = 1 + (vol_d - 1) / THRDS_TENSOR_COPY;
        if (bx > MAX_SYCL_BLOCKS)
          bx = MAX_SYCL_BLOCKS;
        switch (dtens->data_kind) {
        case R4:
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

            local_accessor<T, 1> buf0_acc(
                cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
            local_accessor<float, 0> val_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> ftb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<size_t, 1> gtb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> htb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> stb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> dim_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> dim_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> o2n_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> pri_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> tmp0_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 0> err_code_acc(cgh);
            local_accessor<int, 0> minor_acc(cgh);
            local_accessor<int, 0> minor_in_acc(cgh);
            local_accessor<int, 0> minor_out_acc(cgh);
            local_accessor<int, 0> s1_ind_acc(cgh);
            local_accessor<int, 0> s1_ond_acc(cgh);
            local_accessor<int, 0> s1_step_acc(cgh);
            local_accessor<int, 0> s1_dim_acc(cgh);
            local_accessor<int, 0> s2_ind_acc(cgh);
            local_accessor<int, 0> s2_ond_acc(cgh);
            local_accessor<int, 0> s2_step_acc(cgh);
            local_accessor<int, 0> s2_dim_acc(cgh);
            local_accessor<int, 0> ns1_acc(cgh);
            local_accessor<int, 0> ns2_acc(cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 0> vol_ext_acc(cgh);

            auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
            auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
            auto dtens_dst_rsc_gmem_p_ct4 = (float *)(dtens->dst_rsc->gmem_p);
            auto dtens_tmp_rsc_gmem_p_ct5 = (float *)(dtens->tmp_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY,
                                      THRDS_TENSOR_COPY),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_dlf__(
                      0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                      dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5, item,
                      const_args_dims_acc_ct, const_args_prmn_acc_ct,
                      gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                      val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                      base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                      gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                      stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                      dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                      n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                      tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                      minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                      minor_out_acc_ct.get_pointer(),
                      s1_ind_acc_ct.get_pointer(), s1_ond_acc_ct.get_pointer(),
                      s1_step_acc_ct.get_pointer(), s1_dim_acc_ct.get_pointer(),
                      s2_ind_acc_ct.get_pointer(), s2_ond_acc_ct.get_pointer(),
                      s2_step_acc_ct.get_pointer(), s2_dim_acc_ct.get_pointer(),
                      ns1_acc_ct.get_pointer(), ns2_acc_ct.get_pointer(),
                      vol_acc_ct.get_pointer(), vol_ext_acc_ct.get_pointer());
                });
          });
          break;
        case R8:
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

            local_accessor<T, 1> buf0_acc(
                cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
            local_accessor<float, 0> val_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> ftb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<size_t, 1> gtb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> htb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> stb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> dim_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> dim_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> o2n_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> pri_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> tmp0_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 0> err_code_acc(cgh);
            local_accessor<int, 0> minor_acc(cgh);
            local_accessor<int, 0> minor_in_acc(cgh);
            local_accessor<int, 0> minor_out_acc(cgh);
            local_accessor<int, 0> s1_ind_acc(cgh);
            local_accessor<int, 0> s1_ond_acc(cgh);
            local_accessor<int, 0> s1_step_acc(cgh);
            local_accessor<int, 0> s1_dim_acc(cgh);
            local_accessor<int, 0> s2_ind_acc(cgh);
            local_accessor<int, 0> s2_ond_acc(cgh);
            local_accessor<int, 0> s2_step_acc(cgh);
            local_accessor<int, 0> s2_dim_acc(cgh);
            local_accessor<int, 0> ns1_acc(cgh);
            local_accessor<int, 0> ns2_acc(cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 0> vol_ext_acc(cgh);

            auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
            auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
            auto dtens_dst_rsc_gmem_p_ct4 = (double *)(dtens->dst_rsc->gmem_p);
            auto dtens_tmp_rsc_gmem_p_ct5 = (double *)(dtens->tmp_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY,
                                      THRDS_TENSOR_COPY),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_dlf__(
                      0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                      dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5, item,
                      const_args_dims_acc_ct, const_args_prmn_acc_ct,
                      gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                      val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                      base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                      gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                      stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                      dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                      n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                      tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                      minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                      minor_out_acc_ct.get_pointer(),
                      s1_ind_acc_ct.get_pointer(), s1_ond_acc_ct.get_pointer(),
                      s1_step_acc_ct.get_pointer(), s1_dim_acc_ct.get_pointer(),
                      s2_ind_acc_ct.get_pointer(), s2_ond_acc_ct.get_pointer(),
                      s2_step_acc_ct.get_pointer(), s2_dim_acc_ct.get_pointer(),
                      ns1_acc_ct.get_pointer(), ns2_acc_ct.get_pointer(),
                      vol_acc_ct.get_pointer(), vol_ext_acc_ct.get_pointer());
                });
          });
          break;
        case C4:
          if (fast_math == YEP) {
            (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
              auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

              local_accessor<T, 1> buf0_acc(
                  cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
              local_accessor<float, 0> val_acc(cgh);
              local_accessor<size_t, 1> base_in_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<size_t, 1> base_out_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<size_t, 1> ftb_acc(
                  cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
              local_accessor<size_t, 1> gtb_acc(
                  cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
              local_accessor<int, 1> htb_acc(
                  cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
              local_accessor<int, 1> stb_acc(
                  cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
              local_accessor<int, 1> dim_in_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 1> dim_out_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 1> o2n_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 1> n2o_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 1> pri_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 1> tmp0_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 0> err_code_acc(cgh);
              local_accessor<int, 0> minor_acc(cgh);
              local_accessor<int, 0> minor_in_acc(cgh);
              local_accessor<int, 0> minor_out_acc(cgh);
              local_accessor<int, 0> s1_ind_acc(cgh);
              local_accessor<int, 0> s1_ond_acc(cgh);
              local_accessor<int, 0> s1_step_acc(cgh);
              local_accessor<int, 0> s1_dim_acc(cgh);
              local_accessor<int, 0> s2_ind_acc(cgh);
              local_accessor<int, 0> s2_ond_acc(cgh);
              local_accessor<int, 0> s2_step_acc(cgh);
              local_accessor<int, 0> s2_dim_acc(cgh);
              local_accessor<int, 0> ns1_acc(cgh);
              local_accessor<int, 0> ns2_acc(cgh);
              local_accessor<size_t, 0> vol_acc(cgh);
              local_accessor<size_t, 0> vol_ext_acc(cgh);

              auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
              auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
              auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
              auto dtens_dst_rsc_gmem_p_ct4 = (talshComplex4 *)(dtens->dst_rsc->gmem_p);
              auto dtens_tmp_rsc_gmem_p_ct5 = (talshComplex4 *)(dtens->tmp_rsc->gmem_p);

              cgh.parallel_for(
                  cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY,
                                        THRDS_TENSOR_COPY),
                  [=](cl::sycl::nd_item<1> item) {
                    gpu_tensor_block_copy_cmplx_split_out_dlf__(
                        0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                        dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5,
                        item, const_args_dims_acc_ct, const_args_prmn_acc_ct,
                        gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                        val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                        base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                        gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                        stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                        dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                        n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                        tmp0_acc_ct.get_pointer(),
                        err_code_acc_ct.get_pointer(),
                        minor_acc_ct.get_pointer(),
                        minor_in_acc_ct.get_pointer(),
                        minor_out_acc_ct.get_pointer(),
                        s1_ind_acc_ct.get_pointer(),
                        s1_ond_acc_ct.get_pointer(),
                        s1_step_acc_ctget_pointer(), s1_dim_acc_ctget_pointer(),
                        s2_ind_acc_ct.get_pointer(),
                        s2_ond_acc_ct.get_pointer(),
                        s2_step_acc_ct.get_pointer(),
                        s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                        ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                        vol_ext_acc_ct.get_pointer());
                  });
            });
          } else {
            (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
              auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

              local_accessor<T, 1> buf0_acc(
                  cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
              local_accessor<float, 0> val_acc(cgh);
              local_accessor<size_t, 1> base_in_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<size_t, 1> base_out_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<size_t, 1> ftb_acc(
                  cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
              local_accessor<size_t, 1> gtb_acc(
                  cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
              local_accessor<int, 1> htb_acc(
                  cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
              local_accessor<int, 1> stb_acc(
                  cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
              local_accessor<int, 1> dim_in_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 1> dim_out_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 1> o2n_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 1> n2o_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 1> pri_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 1> tmp0_acc(
                  cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
              local_accessor<int, 0> err_code_acc(cgh);
              local_accessor<int, 0> minor_acc(cgh);
              local_accessor<int, 0> minor_in_acc(cgh);
              local_accessor<int, 0> minor_out_acc(cgh);
              local_accessor<int, 0> s1_ind_acc(cgh);
              local_accessor<int, 0> s1_ond_acc(cgh);
              local_accessor<int, 0> s1_step_acc(cgh);
              local_accessor<int, 0> s1_dim_acc(cgh);
              local_accessor<int, 0> s2_ind_acc(cgh);
              local_accessor<int, 0> s2_ond_acc(cgh);
              local_accessor<int, 0> s2_step_acc(cgh);
              local_accessor<int, 0> s2_dim_acc(cgh);
              local_accessor<int, 0> ns1_acc(cgh);
              local_accessor<int, 0> ns2_acc(cgh);
              local_accessor<size_t, 0> vol_acc(cgh);
              local_accessor<size_t, 0> vol_ext_acc(cgh);

              auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
              auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
              auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
              auto dtens_dst_rsc_gmem_p_ct4 = (talshComplex4 *)(dtens->dst_rsc->gmem_p);
              auto dtens_tmp_rsc_gmem_p_ct5 = (talshComplex4 *)(dtens->tmp_rsc->gmem_p);

              cgh.parallel_for(
                  cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY,
                                        THRDS_TENSOR_COPY),
                  [=](cl::sycl::nd_item<1> item) {
                    gpu_tensor_block_copy_dlf__(
                        0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                        dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5,
                        item, const_args_dims_acc_ct, const_args_prmn_acc_ct,
                        gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                        val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                        base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                        gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                        stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                        dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                        n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                        tmp0_acc_ct.get_pointer(),
                        err_code_acc_ct.get_pointer(),
                        minor_acc_ct.get_pointer(),
                        minor_in_acc_ct.get_pointer(),
                        minor_out_acc_ct.get_pointer(),
                        s1_ind_acc_ct.get_pointer(),
                        s1_ond_acc_ct.get_pointer(),
                        s1_step_acc_ct.get_pointer(),
                        s1_dim_acc_ct.get_pointer(),
                        s2_ind_acc_ct.get_pointer(),
                        s2_ond_acc_ct.get_pointer(),
                        s2_step_acc_ct.get_pointer(),
                        s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                        ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                        vol_ext_acc_ct.get_pointer());
                  });
            });
          }
          break;
        case C8:
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

            local_accessor<T, 1> buf0_acc(
                cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
            local_accessor<float, 0> val_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> ftb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<size_t, 1> gtb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> htb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> stb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> dim_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> dim_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> o2n_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> pri_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> tmp0_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 0> err_code_acc(cgh);
            local_accessor<int, 0> minor_acc(cgh);
            local_accessor<int, 0> minor_in_acc(cgh);
            local_accessor<int, 0> minor_out_acc(cgh);
            local_accessor<int, 0> s1_ind_acc(cgh);
            local_accessor<int, 0> s1_ond_acc(cgh);
            local_accessor<int, 0> s1_step_acc(cgh);
            local_accessor<int, 0> s1_dim_acc(cgh);
            local_accessor<int, 0> s2_ind_acc(cgh);
            local_accessor<int, 0> s2_ond_acc(cgh);
            local_accessor<int, 0> s2_step_acc(cgh);
            local_accessor<int, 0> s2_dim_acc(cgh);
            local_accessor<int, 0> ns1_acc(cgh);
            local_accessor<int, 0> ns2_acc(cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 0> vol_ext_acc(cgh);

            auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
            auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
            auto dtens_dst_rsc_gmem_p_ct4 = (talshComplex8 *)(dtens->dst_rsc->gmem_p);
            auto dtens_tmp_rsc_gmem_p_ct5 = (talshComplex8 *)(dtens->tmp_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY,
                                      THRDS_TENSOR_COPY),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_dlf__(
                      0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                      dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5, item,
                      const_args_dims_acc_ct, const_args_prmn_acc_ct,
                      gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                      val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                      base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                      gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                      stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                      dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                      n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                      tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                      minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                      minor_out_acc_ct.get_pointer(),
                      s1_ind_acc_ct.get_pointer(), s1_ond_acc_ct.get_pointer(),
                      s1_step_acc_ct.get_pointer(), s1_dim_acc_ct.get_pointer(),
                      s2_ind_acc_ct.get_pointer(), s2_ond_acc_ct.get_pointer(),
                      s2_step_acc_ct.get_pointer(), s2_dim_acc_ct.get_pointer(),
                      ns1_acc_ct.get_pointer(), ns2_acc_ct.get_pointer(),
                      vol_acc_ct.get_pointer(), vol_ext_acc_ct.get_pointer());
                });
          });
          break;
        default:
          errc = sycl_task_record(sycl_task, coh_ctrl, 40);
          errc = gpu_activate(cur_gpu);
          return 40;
        }
      } else if (TRANS_SHMEM == EFF_TRN_ON_CUTT) {
        errc = sycl_task_record(sycl_task, coh_ctrl, 43);
        errc = gpu_activate(cur_gpu);
        return 43;
      } else if (TRANS_SHMEM == EFF_TRN_OFF) {
        bx = 1 + (vol_d - 1) / THRDS_TENSOR_COPY_SCAT;
        if (bx > MAX_SYCL_BLOCKS)
          bx = MAX_SYCL_BLOCKS;
        switch (dtens->data_kind) {
        case R4:
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

            auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
            auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
            auto dtens_dst_rsc_gmem_p_ct4 = (float *)(dtens->dst_rsc->gmem_p);
            auto dtens_tmp_rsc_gmem_p_ct5 = (float *)(dtens->tmp_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                      THRDS_TENSOR_COPY_SCAT),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_scatter_dlf__(
                      0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                      dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5, item,
                      const_args_dims_acc_ct, const_args_prmn_acc_ct,
                      gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                      vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                      base_out_acc_ct.get_pointer());
                });
          });
          break;
        case R8:
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();
            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

            auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
            auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
            auto dtens_dst_rsc_gmem_p_ct4 = (double *)(dtens->dst_rsc->gmem_p);
            auto dtens_tmp_rsc_gmem_p_ct5 = (double *)(dtens->tmp_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                      THRDS_TENSOR_COPY_SCAT),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_scatter_dlf__(
                      0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                      dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5, item,
                      const_args_dims_acc_ct, const_args_prmn_acc_ct,
                      gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                      vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                      base_out_acc_ct.get_pointer());
                });
          });
          break;
        case C4:
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();
            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

            auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
            auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
            auto dtens_dst_rsc_gmem_p_ct4 = (talshComplex4 *)(dtens->dst_rsc->gmem_p);
            auto dtens_tmp_rsc_gmem_p_ct5 = (talshComplex4 *)(dtens->tmp_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                      THRDS_TENSOR_COPY_SCAT),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_scatter_dlf__(
                      0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                      dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5, item,
                      const_args_dims_acc_ct, const_args_prmn_acc_ct,
                      gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                      vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                      base_out_acc_ct.get_pointer());
                });
          });
          break;
        case C8:
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

            auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
            auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
            auto dtens_dst_rsc_gmem_p_ct4 = (talshComplex8 *)(dtens->dst_rsc->gmem_p);
            auto dtens_tmp_rsc_gmem_p_ct5 = (talshComplex8 *)(dtens->tmp_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                      THRDS_TENSOR_COPY_SCAT),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_scatter_dlf__(
                      0, 1, drank, sycl_task_tens_args_const_mem_entry_ct3,
                      dtens_dst_rsc_gmem_p_ct4, dtens_tmp_rsc_gmem_p_ct5, item,
                      const_args_dims_acc_ct, const_args_prmn_acc_ct,
                      gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                      vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                      base_out_acc_ct.get_pointer());
                });
          });
          break;
        default:
          errc = sycl_task_record(sycl_task, coh_ctrl, 44);
          errc = gpu_activate(cur_gpu);
          return 44;
        }
      } else {
        errc = sycl_task_record(sycl_task, coh_ctrl, 45);
        errc = gpu_activate(cur_gpu);
        return 45;
      }
    } else {
      (*sycl_stream)->memset(dtens->tmp_rsc->gmem_p, 0, dsize);
    }
    darg = dtens->tmp_rsc->gmem_p;
  } else {
    if (accumulative == NOPE) {
      (*sycl_stream)->memset(dtens->dst_rsc->gmem_p, 0, dsize);
    }
    darg = dtens->dst_rsc->gmem_p;
  }
  // Left tensor transpose:
  if (perm_l == YEP) {
    if (TRANS_SHMEM == EFF_TRN_ON) {
      bx = 1 + (vol_l - 1) / THRDS_TENSOR_COPY;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (ltens->data_kind) {
      case R4:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();
          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p_ct4 = (float *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p_ct5 = (float *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                    ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p_ct4 = (double *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p_ct5 = (double *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                    ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case C4:
        if (fast_math == YEP) {
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

            local_accessor<T, 1> buf0_acc(
                cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
            local_accessor<float, 0> val_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> ftb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<size_t, 1> gtb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> htb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> stb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> dim_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> dim_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> o2n_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> pri_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> tmp0_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 0> err_code_acc(cgh);
            local_accessor<int, 0> minor_acc(cgh);
            local_accessor<int, 0> minor_in_acc(cgh);
            local_accessor<int, 0> minor_out_acc(cgh);
            local_accessor<int, 0> s1_ind_acc(cgh);
            local_accessor<int, 0> s1_ond_acc(cgh);
            local_accessor<int, 0> s1_step_acc(cgh);
            local_accessor<int, 0> s1_dim_acc(cgh);
            local_accessor<int, 0> s2_ind_acc(cgh);
            local_accessor<int, 0> s2_ond_acc(cgh);
            local_accessor<int, 0> s2_step_acc(cgh);
            local_accessor<int, 0> s2_dim_acc(cgh);
            local_accessor<int, 0> ns1_acc(cgh);
            local_accessor<int, 0> ns2_acc(cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 0> vol_ext_acc(cgh);

            auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
            auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
            auto ltens_dst_rsc_gmem_p_ct4 = (talshComplex4 *)(ltens->dst_rsc->gmem_p);
            auto ltens_tmp_rsc_gmem_p_ct5 = (talshComplex4 *)(ltens->tmp_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY,
                                      THRDS_TENSOR_COPY),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_cmplx_split_out_dlf__(
                      0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                      ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                      const_args_dims_acc_ct, const_args_prmn_acc_ct,
                      gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                      val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                      base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                      gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                      stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                      dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                      n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                      tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                      minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                      minor_out_acc_ct.get_pointer(),
                      s1_ind_acc_ct.get_pointer(), s1_ond_acc_ct.get_pointer(),
                      s1_step_acc_ct.get_pointer(), s1_dim_acc_ct.get_pointer(),
                      s2_ind_acc_ct.get_pointer(), s2_ond_acc_ct.get_pointer(),
                      s2_step_acc_ct.get_pointer(), s2_dim_acc_ct.get_pointer(),
                      ns1_acc_ct.get_pointer(), ns2_acc_ct.get_pointer(),
                      vol_acc_ct.get_pointer(), vol_ext_acc_ct.get_pointer());
                });
          });
        } else {
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

            local_accessor<T, 1> buf0_acc(
                cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
            local_accessor<float, 0> val_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> ftb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<size_t, 1> gtb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> htb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> stb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> dim_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> dim_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> o2n_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> pri_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> tmp0_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 0> err_code_acc(cgh);
            local_accessor<int, 0> minor_acc(cgh);
            local_accessor<int, 0> minor_in_acc(cgh);
            local_accessor<int, 0> minor_out_acc(cgh);
            local_accessor<int, 0> s1_ind_acc(cgh);
            local_accessor<int, 0> s1_ond_acc(cgh);
            local_accessor<int, 0> s1_step_acc(cgh);
            local_accessor<int, 0> s1_dim_acc(cgh);
            local_accessor<int, 0> s2_ind_acc(cgh);
            local_accessor<int, 0> s2_ond_acc(cgh);
            local_accessor<int, 0> s2_step_acc(cgh);
            local_accessor<int, 0> s2_dim_acc(cgh);
            local_accessor<int, 0> ns1_acc(cgh);
            local_accessor<int, 0> ns2_acc(cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 0> vol_ext_acc(cgh);

            auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
            auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
            auto ltens_dst_rsc_gmem_p_ct4 = (talshComplex4 *)(ltens->dst_rsc->gmem_p);
            auto ltens_tmp_rsc_gmem_p_ct5 = (talshComplex4 *)(ltens->tmp_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY,
                                      THRDS_TENSOR_COPY),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_dlf__(
                      0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                      ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                      const_args_dims_acc_ct, const_args_prmn_acc_ct,
                      gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                      val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                      base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                      gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                      stb_acc_ct.get_pointer(), dim_in_acc_ctget_pointer(),
                      dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                      n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                      tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                      minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                      minor_out_acc_ct.get_pointer(),
                      s1_ind_acc_ct.get_pointer(), s1_ond_acc_ct.get_pointer(),
                      s1_step_acc_ct.get_pointer(), s1_dim_acc_ct.get_pointer(),
                      s2_ind_acc_ct.get_pointer(), s2_ond_acc_ct.get_pointer(),
                      s2_step_acc_ct.get_pointer(), s2_dim_acc_ct.get_pointer(),
                      ns1_acc_ct.get_pointer(), ns2_acc_ct.get_pointer(),
                      vol_acc_ct.get_pointer(), vol_ext_acc_ct.get_pointer());
                });
          });
        }
        break;
      case C8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p_ct4 = (talshComplex8 *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p_ct5 = (talshComplex8 *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry_ct3,
                    ltens_dst_rsc_gmem_p_ct4, ltens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 48);
        errc = gpu_activate(cur_gpu);
        return 48;
      }
    } else if (TRANS_SHMEM == EFF_TRN_ON_CUTT) {
      errc = sycl_task_record(sycl_task, coh_ctrl, 51);
      errc = gpu_activate(cur_gpu);
      return 51;
    } else if (TRANS_SHMEM == EFF_TRN_OFF) {
      bx = 1 + (vol_l - 1) / THRDS_TENSOR_COPY_SCAT;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (ltens->data_kind) {
      case R4:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry3 =
              sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p4 = (float *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p5 = (float *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry3,
                    ltens_dst_rsc_gmem_p4, ltens_tmp_rsc_gmem_p5, item,
                    const_args_dims_acc, const_args_prmn_acc,
                    gpu_error_count_ptr, n2o_acc.get_pointer(),
                    vol_acc.get_pointer(), base_in_acc.get_pointer(),
                    base_out_acc.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry3 =
              sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p4 = (double *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p5 = (double *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry3,
                    ltens_dst_rsc_gmem_p4, ltens_tmp_rsc_gmem_p5, item,
                    const_args_dims_acc, const_args_prmn_acc,
                    gpu_error_count_ptr, n2o_acc.get_pointer(),
                    vol_acc.get_pointer(), base_in_acc.get_pointer(),
                    base_out_acc.get_pointer());
              });
        });
        break;
      case C4:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry3 =
              sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p4 =
              (talshComplex4 *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p5 =
              (talshComplex4 *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry3,
                    ltens_dst_rsc_gmem_p4, ltens_tmp_rsc_gmem_p5, item,
                    const_args_dims_acc, const_args_prmn_acc,
                    gpu_error_count_ptr, n2o_acc.get_pointer(),
                    vol_acc.get_pointer(), base_in_acc.get_pointer(),
                    base_out_acc.get_pointer());
              });
        });
        break;
      case C8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry3 =
              sycl_task->tens_args[1].const_mem_entry;
          auto ltens_dst_rsc_gmem_p4 =
              (talshComplex8 *)(ltens->dst_rsc->gmem_p);
          auto ltens_tmp_rsc_gmem_p5 =
              (talshComplex8 *)(ltens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, lrank, sycl_task_tens_args_const_mem_entry3,
                    ltens_dst_rsc_gmem_p4, ltens_tmp_rsc_gmem_p5, item,
                    const_args_dims_acc, const_args_prmn_acc,
                    gpu_error_count_ptr, n2o_acc.get_pointer(),
                    vol_acc.get_pointer(), base_in_acc.get_pointer(),
                    base_out_acc.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 52);
        errc = gpu_activate(cur_gpu);
        return 52;
      }
    } else {
      errc = sycl_task_record(sycl_task, coh_ctrl, 53);
      errc = gpu_activate(cur_gpu);
      return 53;
    }
    larg = ltens->tmp_rsc->gmem_p;
  } else {
    larg = ltens->dst_rsc->gmem_p;
  }
  // Right tensor transpose:
  if (perm_r == YEP) {
    if (TRANS_SHMEM == EFF_TRN_ON) {
      bx = 1 + (vol_r - 1) / THRDS_TENSOR_COPY;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (rtens->data_kind) {
      case R4:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[2].const_mem_entry;
          auto rtens_dst_rsc_gmem_p_ct4 = (float *)(rtens->dst_rsc->gmem_p);
          auto rtens_tmp_rsc_gmem_p_ct5 = (float *)(rtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 0, rrank, sycl_task_tens_args_const_mem_entry_ct3,
                    rtens_dst_rsc_gmem_p_ct4, rtens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[2].const_mem_entry;
          auto rtens_dst_rsc_gmem_p_ct4 = (double *)(rtens->dst_rsc->gmem_p);
          auto rtens_tmp_rsc_gmem_p_ct5 = (double *)(rtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 0, rrank, sycl_task_tens_args_const_mem_entry_ct3,
                    rtens_dst_rsc_gmem_p_ct4, rtens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case C4:
        if (fast_math == YEP) {
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

            local_accessor<T, 1> buf0_acc(
                cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
            local_accessor<float, 0> val_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> ftb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<size_t, 1> gtb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> htb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> stb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> dim_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> dim_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> o2n_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> pri_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> tmp0_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 0> err_code_acc(cgh);
            local_accessor<int, 0> minor_acc(cgh);
            local_accessor<int, 0> minor_in_acc(cgh);
            local_accessor<int, 0> minor_out_acc(cgh);
            local_accessor<int, 0> s1_ind_acc(cgh);
            local_accessor<int, 0> s1_ond_acc(cgh);
            local_accessor<int, 0> s1_step_acc(cgh);
            local_accessor<int, 0> s1_dim_acc(cgh);
            local_accessor<int, 0> s2_ind_acc(cgh);
            local_accessor<int, 0> s2_ond_acc(cgh);
            local_accessor<int, 0> s2_step_acc(cgh);
            local_accessor<int, 0> s2_dim_acc(cgh);
            local_accessor<int, 0> ns1_acc(cgh);
            local_accessor<int, 0> ns2_acc(cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 0> vol_ext_acc(cgh);

            auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
            auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[2].const_mem_entry;
            auto rtens_dst_rsc_gmem_p_ct4 = (talshComplex4 *)(rtens->dst_rsc->gmem_p);
            auto rtens_tmp_rsc_gmem_p_ct5 = (talshComplex4 *)(rtens->tmp_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range(cl::sycl::range(bx * THRDS_TENSOR_COPY),
                                   cl::sycl::range(THRDS_TENSOR_COPY)),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_cmplx_split_out_dlf__(
                      0, 0, rrank, sycl_task_tens_args_const_mem_entry_ct3,
                      rtens_dst_rsc_gmem_p_ct4, rtens_tmp_rsc_gmem_p_ct5, item,
                      const_args_dims_acc_ct, const_args_prmn_acc_ct,
                      gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                      val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                      base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                      gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                      stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                      dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                      n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                      tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                      minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                      minor_out_acc_ct.get_pointer(),
                      s1_ind_acc_ct.get_pointer(), s1_ond_acc_ct.get_pointer(),
                      s1_step_acc_ct.get_pointer(), s1_dim_acc_ct.get_pointer(),
                      s2_ind_acc_ct.get_pointer(), s2_ond_acc_ct.get_pointer(),
                      s2_step_acc_ct.get_pointer(), s2_dim_acc_ct.get_pointer(),
                      ns1_acc_ct.get_pointer(), ns2_acc_ct.get_pointer(),
                      vol_acc_ct.get_pointer(), vol_ext_acc_ct.get_pointer());
                });
          });
        } else {
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

            local_accessor<T, 1> buf0_acc(
                cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
            local_accessor<float, 0> val_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> ftb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<size_t, 1> gtb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> htb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> stb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> dim_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> dim_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> o2n_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> pri_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> tmp0_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 0> err_code_acc(cgh);
            local_accessor<int, 0> minor_acc(cgh);
            local_accessor<int, 0> minor_in_acc(cgh);
            local_accessor<int, 0> minor_out_acc(cgh);
            local_accessor<int, 0> s1_ind_acc(cgh);
            local_accessor<int, 0> s1_ond_acc(cgh);
            local_accessor<int, 0> s1_step_acc(cgh);
            local_accessor<int, 0> s1_dim_acc(cgh);
            local_accessor<int, 0> s2_ind_acc(cgh);
            local_accessor<int, 0> s2_ond_acc(cgh);
            local_accessor<int, 0> s2_step_acc(cgh);
            local_accessor<int, 0> s2_dim_acc(cgh);
            local_accessor<int, 0> ns1_acc(cgh);
            local_accessor<int, 0> ns2_acc(cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 0> vol_ext_acc(cgh);

            auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
            auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[2].const_mem_entry;
            auto rtens_dst_rsc_gmem_p_ct4 = (talshComplex4 *)(rtens->dst_rsc->gmem_p);
            auto rtens_tmp_rsc_gmem_p_ct5 = (talshComplex4 *)(rtens->tmp_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY,
                                      THRDS_TENSOR_COPY),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_dlf__(
                      0, 0, rrank, sycl_task_tens_args_const_mem_entry_ct3,
                      rtens_dst_rsc_gmem_p_ct4, rtens_tmp_rsc_gmem_p_ct5, item,
                      const_args_dims_acc_ct, const_args_prmn_acc_ct,
                      gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                      val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                      base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                      gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                      stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                      dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                      n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                      tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                      minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                      minor_out_acc_ct.get_pointer(),
                      s1_ind_acc_ct.get_pointer(), s1_ond_acc_ct.get_pointer(),
                      s1_step_acc_ct.get_pointer(), s1_dim_acc_ct.get_pointer(),
                      s2_ind_acc_ct.get_pointer(), s2_ond_acc_ct.get_pointer(),
                      s2_step_acc_ct.get_pointer(), s2_dim_acc_ct.get_pointer(),
                      ns1_acc_ct.get_pointer(), ns2_acc_ct.get_pointer(),
                      vol_acc_ct.get_pointer(), vol_ext_acc_ct.get_pointer());
                });
          });
        }
        break;
      case C8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[2].const_mem_entry;
          auto rtens_dst_rsc_gmem_p_ct4 = (talshComplex8 *)(rtens->dst_rsc->gmem_p);
          auto rtens_tmp_rsc_gmem_p_ct5 = (talshComplex8 *)(rtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    0, 0, rrank, sycl_task_tens_args_const_mem_entry_ct3,
                    rtens_dst_rsc_gmem_p_ct4, rtens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 54);
        errc = gpu_activate(cur_gpu);
        return 54;
      }
    } else if (TRANS_SHMEM == EFF_TRN_ON_CUTT) {
      errc = sycl_task_record(sycl_task, coh_ctrl, 57);
      errc = gpu_activate(cur_gpu);
      return 57;
    } else if (TRANS_SHMEM == EFF_TRN_OFF) {
      bx = 1 + (vol_r - 1) / THRDS_TENSOR_COPY_SCAT;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (rtens->data_kind) {
      case R4:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[2].const_mem_entry;
          auto rtens_dst_rsc_gmem_p_ct4 = (float *)(rtens->dst_rsc->gmem_p);
          auto rtens_tmp_rsc_gmem_p_ct5 = (float *)(rtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range(cl::sycl::range(bx * THRDS_TENSOR_COPY_SCAT),
                                 cl::sycl::range(THRDS_TENSOR_COPY_SCAT)),
              [=](cl::sycl::nd_item<3> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, rrank, sycl_task_tens_args_const_mem_entry_ct3,
                    rtens_dst_rsc_gmem_p_ct4, rtens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[2].const_mem_entry;
          auto rtens_dst_rsc_gmem_p_ct4 = (double *)(rtens->dst_rsc->gmem_p);
          auto rtens_tmp_rsc_gmem_p_ct5 = (double *)(rtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range(cl::sycl::range(bx * THRDS_TENSOR_COPY_SCAT),
                                 cl::sycl::range(THRDS_TENSOR_COPY_SCAT)),
              [=](cl::sycl::nd_item<3> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, rrank, sycl_task_tens_args_const_mem_entry_ct3,
                    rtens_dst_rsc_gmem_p_ct4, rtens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case C4:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[2].const_mem_entry;
          auto rtens_dst_rsc_gmem_p_ct4 = (talshComplex4 *)(rtens->dst_rsc->gmem_p);
          auto rtens_tmp_rsc_gmem_p_ct5 = (talshComplex4 *)(rtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range(cl::sycl::range(bx * THRDS_TENSOR_COPY_SCAT),
                                 cl::sycl::range(THRDS_TENSOR_COPY_SCAT)),
              [=](cl::sycl::nd_item<3> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, rrank, sycl_task_tens_args_const_mem_entry_ct3,
                    rtens_dst_rsc_gmem_p_ct4, rtens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case C8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);

          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);
          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[2].const_mem_entry;
          auto rtens_dst_rsc_gmem_p_ct4 = (talshComplex8 *)(rtens->dst_rsc->gmem_p);
          auto rtens_tmp_rsc_gmem_p_ct5 = (talshComplex8 *)(rtens->tmp_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range(cl::sycl::range(bx * THRDS_TENSOR_COPY_SCAT),
                                 cl::sycl::range(THRDS_TENSOR_COPY_SCAT)),
              [=](cl::sycl::nd_item<3> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    0, 0, rrank, sycl_task_tens_args_const_mem_entry_ct3,
                    rtens_dst_rsc_gmem_p_ct4, rtens_tmp_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 58);
        errc = gpu_activate(cur_gpu);
        return 58;
      }
    } else {
      errc = sycl_task_record(sycl_task, coh_ctrl, 59);
      errc = gpu_activate(cur_gpu);
      return 59;
    }
    rarg = rtens->tmp_rsc->gmem_p;
  } else {
    rarg = rtens->dst_rsc->gmem_p;
  }
  // Schedule the appropriate computation kernel:
  // Set up the scaling prefactor (in mapped Host memory):
  errc = 0;
  switch (dtens->data_kind) {

  case R4:
    if (scale_real != 1.0 || scale_imag != 0.0) {
      errc = sycl_task_set_prefactor(
          sycl_task, talshComplex4Set((float)scale_real, (float)scale_imag));
      if (errc) {
        j = sycl_task_record(sycl_task, coh_ctrl, 60);
        j = gpu_activate(cur_gpu);
        return 60;
      }
      j = slab_get_entry_offset(&prefactors, sycl_task->pref_ptr, &pofs);
      if (j != 0)
        errc++;
      alpha_plus_p = (void *)&(((char *)(gpu_prefs_base_ptr))[pofs]);
    } else {
      alpha_plus_p = static_cast<void *>(&sgemm_alpha_plus);
      alpha_minus_p = static_cast<void *>(&sgemm_alpha_minus);
    }
    if (accumulative == NOPE) {
      beta_p = static_cast<void *>(&sgemm_beta_zero);
    } else {
      beta_p = static_cast<void *>(&sgemm_beta_one);
    }
    beta_one_p = static_cast<void *>(&sgemm_beta_one);
    break;

  case R8:
    if (scale_real != 1.0 || scale_imag != 0.0) {
      errc = sycl_task_set_prefactor(sycl_task,
                                     talshComplex8Set(scale_real, scale_imag));
      if (errc) {
        j = sycl_task_record(sycl_task, coh_ctrl, 61);
        j = gpu_activate(cur_gpu);
        return 61;
      }
      j = slab_get_entry_offset(&prefactors, sycl_task->pref_ptr, &pofs);
      if (j != 0)
        errc++;
      alpha_plus_p = (void *)&(((char *)(gpu_prefs_base_ptr))[pofs]);
    } else {
      alpha_plus_p = static_cast<void *>(&dgemm_alpha_plus);
      alpha_minus_p = static_cast<void *>(&dgemm_alpha_minus);
    }
    if (accumulative == NOPE) {
      beta_p = static_cast<void *>(&dgemm_beta_zero);
    } else {
      beta_p = static_cast<void *>(&dgemm_beta_one);
    }
    beta_one_p = static_cast<void *>(&dgemm_beta_one);
    break;

  case C4:
    if (scale_real != 1.0 || scale_imag != 0.0) {
      errc = sycl_task_set_prefactor(
          sycl_task, talshComplex4Set((float)scale_real, (float)scale_imag));
      if (errc) {
        j = sycl_task_record(sycl_task, coh_ctrl, 62);
        j = gpu_activate(cur_gpu);
        return 62;
      }
      j = slab_get_entry_offset(&prefactors, sycl_task->pref_ptr, &pofs);
      if (j != 0)
        errc++;
      alpha_plus_p = (void *)&(((char *)(gpu_prefs_base_ptr))[pofs]);
    } else {
      alpha_plus_p = static_cast<void *>(&cgemm_alpha_plus);
      alpha_minus_p = static_cast<void *>(&cgemm_alpha_minus);
    }
    if (accumulative == NOPE) {
      beta_p = static_cast<void *>(&cgemm_beta_zero);
    } else {
      beta_p = static_cast<void *>(&cgemm_beta_one);
    }
    beta_one_p = static_cast<void *>(&cgemm_beta_one);
    break;

  case C8:
    if (scale_real != 1.0 || scale_imag != 0.0) {
      errc = sycl_task_set_prefactor(sycl_task,
                                     talshComplex8Set(scale_real, scale_imag));
      if (errc) {
        j = sycl_task_record(sycl_task, coh_ctrl, 63);
        j = gpu_activate(cur_gpu);
        return 63;
      }
      j = slab_get_entry_offset(&prefactors, sycl_task->pref_ptr, &pofs);
      if (j != 0)
        errc++;
      alpha_plus_p = static_cast<void *>(&(((char *)(gpu_prefs_base_ptr))[pofs]));
    } else {
      alpha_plus_p = static_cast<void *>(&zgemm_alpha_plus);
      alpha_minus_p = static_cast<void *>(&zgemm_alpha_minus);
    }
    if (accumulative == NOPE) {
      beta_p = static_cast<void *>(&zgemm_beta_zero);
    } else {
      beta_p = static_cast<void *>(&zgemm_beta_one);
    }
    beta_one_p = static_cast<void *>(&zgemm_beta_one);
    break;

  default:
    errc++;
  }
  if (errc) {
    errc = sycl_task_record(sycl_task, coh_ctrl, 64);
    errc = gpu_activate(cur_gpu);
    return 64;
  }

  // Scalar multiplication:
  if (drank == 0 && lrank == 0 && rrank == 0) {
    switch (dtens->data_kind) {
    case R4:
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.single_task<class scalarMultiply>([=]() {
          gpu_scalar_multiply__((float *)larg, (float *)rarg, (float *)darg,
                                (float)scale_real);
        });
      });
      break;
    case R8:
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.single_task<class scalarMultiply>([=]() {
          gpu_scalar_multiply__((double *)larg, (double *)rarg, (double *)darg,
                                (double)scale_real);
        });
      });
      break;
    case C4:
      scale_cmplx4 = talshComplex4Set((float)scale_real, (float)scale_imag);
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.single_task<class scalarMultiply>([=]() {
          gpu_scalar_multiply__((talshComplex4 *)larg, (talshComplex4 *)rarg,
                                (talshComplex4 *)darg,
                                (talshComplex4)scale_cmplx4);
        });
      });
      break;
    case C8:
      scale_cmplx8 = talshComplex8Set(scale_real, scale_imag);
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.single_task<class scalarMultiply>([=]() {
          gpu_scalar_multiply__((talshComplex8 *)larg, (talshComplex8 *)rarg,
                                (talshComplex8 *)darg,
                                (talshComplex8)scale_cmplx8);
        });
      });
      break;
    default:
      errc = sycl_task_record(sycl_task, coh_ctrl, 67);
      errc = gpu_activate(cur_gpu);
      return 67;
    }
    // Right tensor rescaled addition:
  } else if (lrank == 0 && rrank > 0) {
    bx = 1 + (vol_d - 1) / THRDS_ARRAY_ADD;
    if (bx > MAX_SYCL_BLOCKS)
      bx = MAX_SYCL_BLOCKS;
    switch (dtens->data_kind) {
    case R4:
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_add__(vol_d, (float *)(darg), (float *)(rarg),
                              (float *)(larg), (float)scale_real, item, 0);
            });
      });
      break;
    case R8:
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_add__(vol_d, (double *)(darg), (double *)(rarg),
                              (double *)(larg), (double)scale_real, item, 0);
            });
      });
      break;
    case C4:
      scale_cmplx4 = talshComplex4Set((float)scale_real, (float)scale_imag);
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_add__(vol_d, (talshComplex4 *)(darg),
                              (talshComplex4 *)(rarg), (talshComplex4 *)(larg),
                              (talshComplex4)scale_cmplx4, item, conj_r);
            });
      });
      break;
    case C8:
      scale_cmplx8 = talshComplex8Set(scale_real, scale_imag);
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_add__(vol_d, (talshComplex8 *)(darg),
                              (talshComplex8 *)(rarg), (talshComplex8 *)(larg),
                              (talshComplex8)scale_cmplx8, item, conj_r);
            });
      });
      break;
    default:
      errc = sycl_task_record(sycl_task, coh_ctrl, 68);
      errc = gpu_activate(cur_gpu);
      return 68;
    }
    // Left tensor rescaled addition:
  } else if (lrank > 0 && rrank == 0) {
    bx = 1 + (vol_d - 1) / THRDS_ARRAY_ADD;
    if (bx > MAX_SYCL_BLOCKS)
      bx = MAX_SYCL_BLOCKS;
    switch (dtens->data_kind) {
    case R4:
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_add__(vol_d, (float *)(darg), (float *)(larg),
                              (float *)(rarg), (float)scale_real, item, 0);
            });
      });
      break;
    case R8:
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_add__(vol_d, (double *)(darg), (double *)(larg),
                              (double *)(rarg), (double)scale_real, item, 0);
            });
      });
      break;
    case C4:
      scale_cmplx4 = talshComplex4Set((float)scale_real, (float)scale_imag);
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_add__(vol_d, (talshComplex4 *)(darg),
                              (talshComplex4 *)(larg), (talshComplex4 *)(rarg),
                              (talshComplex4)scale_cmplx4, item, conj_l);
            });
      });
      break;
    case C8:
      scale_cmplx8 = talshComplex8Set(scale_real, scale_imag);
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::nd_range<1>(bx * THRDS_ARRAY_ADD, THRDS_ARRAY_ADD),
            [=](cl::sycl::nd_item<1> item) {
              gpu_array_add__(vol_d, (talshComplex8 *)(darg),
                              (talshComplex8 *)(larg), (talshComplex8 *)(rarg),
                              (talshComplex8)scale_cmplx8, item, conj_l);
            });
      });
      break;
    default:
      errc = sycl_task_record(sycl_task, coh_ctrl, 69);
      errc = gpu_activate(cur_gpu);
      return 69;
    }
    // Full tensor contraction (via vector dot-product):
  } else if (drank == 0 && lrank > 0 && rrank == lrank) {
    bx = 1 + (vol_l - 1) / THRDS_ARRAY_SCALE;
    if (bx > MAX_SYCL_BLOCKS)
      bx = MAX_SYCL_BLOCKS;
    switch (ltens->data_kind) {
    case R4:
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        auto dot_product_wr_lock_ptr_ct = dot_product_wr_lock.get_ptr();

        local_accessor<uint8_t, 1> local_acc(
            cl::sycl::range(THRDS_ARRAY_SCALE * sizeof(float)), cgh);

        cgh.parallel_for(
            cl::sycl::nd_range(cl::sycl::range(1, 1, bx) *
                                   cl::sycl::range(1, 1, THRDS_ARRAY_SCALE),
                               cl::sycl::range(1, 1, THRDS_ARRAY_SCALE)),
            [=](cl::sycl::nd_item<3> item) {
              gpu_array_dot_product__(vol_l, (float *)larg, (float *)rarg,
                                      (float *)darg, (float)scale_real, item,
                                      local_acc.get_pointer(),
                                      dot_product_wr_lock_ptr_ct, 0, 0);
            });
      });
      break;
    case R8:
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        auto dot_product_wr_lock_ptr_ct = dot_product_wr_lock.get_ptr();

        local_accessor<uint8_t, 1> local_acc(
            cl::sycl::range(THRDS_ARRAY_SCALE * sizeof(double)), cgh);

        cgh.parallel_for(
            cl::sycl::nd_range(cl::sycl::range(1, 1, bx) *
                                   cl::sycl::range(1, 1, THRDS_ARRAY_SCALE),
                               cl::sycl::range(1, 1, THRDS_ARRAY_SCALE)),
            [=](cl::sycl::nd_item<3> item) {
              gpu_array_dot_product__(vol_l, (double *)larg, (double *)rarg,
                                      (double *)darg, scale_real, item,
                                      local_acc.get_pointer(),
                                      dot_product_wr_lock_ptr_ct, 0, 0);
            });
      });
      break;
    case C4:
      scale_cmplx4 = talshComplex4Set((float)scale_real, (float)scale_imag);
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        auto dot_product_wr_lock_ptr_ct = dot_product_wr_lock.get_ptr();

        local_accessor<uint8_t, 1> local_acc(
            cl::sycl::range(THRDS_ARRAY_SCALE * sizeof(talshComplex4)), cgh);

        cgh.parallel_for(
            cl::sycl::nd_range(cl::sycl::range(1, 1, bx) *
                                   cl::sycl::range(1, 1, THRDS_ARRAY_SCALE),
                               cl::sycl::range(1, 1, THRDS_ARRAY_SCALE)),
            [=](cl::sycl::nd_item<3> item) {
              gpu_array_dot_product__(
                  vol_l, (talshComplex4 *)larg, (talshComplex4 *)rarg,
                  (talshComplex4 *)darg, scale_cmplx4, item,
                  local_acc.get_pointer(), dot_product_wr_lock_ptr_ct, conj_l,
                  conj_r);
            });
      });
      break;
    case C8:
      scale_cmplx8 = talshComplex8Set(scale_real, scale_imag);
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        auto dot_product_wr_lock_ptr_ct = dot_product_wr_lock.get_ptr();

        local_accessor<uint8_t, 1> local_acc(
            cl::sycl::range(THRDS_ARRAY_SCALE * sizeof(talshComplex8)), cgh);

        cgh.parallel_for(
            cl::sycl::nd_range(cl::sycl::range(1, 1, bx) *
                                   cl::sycl::range(1, 1, THRDS_ARRAY_SCALE),
                               cl::sycl::range(1, 1, THRDS_ARRAY_SCALE)),
            [=](cl::sycl::nd_item<3> item) {
              gpu_array_dot_product__(
                  vol_l, (talshComplex8 *)larg, (talshComplex8 *)rarg,
                  (talshComplex8 *)darg, scale_cmplx8, item,
                  local_acc.get_pointer(), dot_product_wr_lock_ptr_ct, conj_l,
                  conj_r);
            });
      });
      break;
    default:
      errc = sycl_task_record(sycl_task, coh_ctrl, 70);
      errc = gpu_activate(cur_gpu);
      return 70;
    }
    // Tensor product (no contracted indices):
  } else if (drank > 0 && drank == lrank + rrank) {
    bx = 1 + (vol_l - 1) / THRDS_ARRAY_PRODUCT;
    by = 1 + (vol_r - 1) / THRDS_ARRAY_PRODUCT;
    limit_sycl_workgroups2d(MAX_SYCL_BLOCKS, &bx, &by);
    sycl::range blcks(bx, by, 1);

    switch (dtens->data_kind) {
    case R4:
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        local_accessor<T, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<T, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT*/), cgh);
        local_accessor<talshComplex4, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<talshComplex4, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT*/), cgh);
        local_accessor<talshComplex8, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<talshComplex8, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT*/), cgh);

        cgh.parallel_for(
            cl::sycl::nd_range(
                cl::sycl::range(blcks.get(2), blcks.get(1), blcks.get(0)) *
                    cl::sycl::range(1, 1, THRDS_ARRAY_PRODUCT),
                cl::sycl::range(1, 1, THRDS_ARRAY_PRODUCT)),
            [=](cl::sycl::nd_item<3> item) {
              gpu_array_product__(
                  vol_l, (float *)larg, vol_r, (float *)rarg, (float *)darg,
                  (float)scale_real, item, lbuf_acc_ct.get_pointer(),
                  rbuf_acc_ct.get_pointer(), lbuf_acc_ct.get_pointer(),
                  rbuf_acc_ct.get_pointer(), lbuf_acc_ct.get_pointer(),
                  rbuf_acc_ct.get_pointer(), 0, 0);
            });
      });
      break;
    case R8:
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        local_accessor<T, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<T, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT*/), cgh);
        local_accessor<talshComplex4, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<talshComplex4, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT  */), cgh);
        local_accessor<talshComplex8, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<talshComplex8, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT  */), cgh);

        cgh.parallel_for(
            cl::sycl::nd_range(
                cl::sycl::range(blcks.get(2), blcks.get(1), blcks.get(0)) *
                    cl::sycl::range(1, 1, THRDS_ARRAY_PRODUCT),
                cl::sycl::range(1, 1, THRDS_ARRAY_PRODUCT)),
            [=](cl::sycl::nd_item<3> item) {
              gpu_array_product__(
                  vol_l, (double *)larg, vol_r, (double *)rarg, (double *)darg,
                  scale_real, item, lbuf_acc_ct.get_pointer(),
                  rbuf_acc_ct.get_pointer(), lbuf_acc_ct.get_pointer(),
                  rbuf_acc_ct.get_pointer(), lbuf_acc_ct.get_pointer(),
                  rbuf_acc_ct.get_pointer(), 0, 0);
            });
      });
      break;
    case C4:
      scale_cmplx4 = talshComplex4Set((float)scale_real, (float)scale_imag);
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        local_accessor<T, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<T, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT*/), cgh);
        local_accessor<talshComplex4, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<talshComplex4, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT  */), cgh);
        local_accessor<talshComplex8, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<talshComplex8, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT  */), cgh);

        cgh.parallel_for(
            cl::sycl::nd_range(
                cl::sycl::range(blcks.get(2), blcks.get(1), blcks.get(0)) *
                    cl::sycl::range(1, 1, THRDS_ARRAY_PRODUCT),
                cl::sycl::range(1, 1, THRDS_ARRAY_PRODUCT)),
            [=](cl::sycl::nd_item<3> item) {
              gpu_array_product__(
                  vol_l, (talshComplex4 *)larg, vol_r, (talshComplex4 *)rarg,
                  (talshComplex4 *)darg, scale_cmplx4, item,
                  lbuf_acc_ct.get_pointer(), rbuf_acc_ct.get_pointer(),
                  lbuf_acc_ct.get_pointer(), rbuf_acc_ct.get_pointer(),
                  lbuf_acc_ct.get_pointer(), rbuf_acc_ct.get_pointer(), conj_l,
                  conj_r);
            });
      });
      break;
    case C8:
      scale_cmplx8 = talshComplex8Set(scale_real, scale_imag);
      (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
        local_accessor<T, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<T, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT  */), cgh);
        local_accessor<talshComplex4, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<talshComplex4, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT  */), cgh);
        local_accessor<talshComplex8, 1> lbuf_acc(
            cl::sycl::range(257 /*THRDS_ARRAY_PRODUCT+1*/), cgh);
        local_accessor<talshComplex8, 1> rbuf_acc(
            cl::sycl::range(256 /*THRDS_ARRAY_PRODUCT  */), cgh);

        cgh.parallel_for(
            cl::sycl::nd_range(
                cl::sycl::range(blcks.get(2), blcks.get(1), blcks.get(0)) *
                    cl::sycl::range(1, 1, THRDS_ARRAY_PRODUCT),
                cl::sycl::range(1, 1, THRDS_ARRAY_PRODUCT)),
            [=](cl::sycl::nd_item<3> item) {
              gpu_array_product__(
                  vol_l, (talshComplex8 *)larg, vol_r, (talshComplex8 *)rarg,
                  (talshComplex8 *)darg, scale_cmplx8, item,
                  lbuf_acc_ct.get_pointer(), rbuf_acc_ct.get_pointer(),
                  lbuf_acc_ct.get_pointer(), rbuf_acc_ct.get_pointer(),
                  lbuf_acc_ct.get_pointer(), rbuf_acc_ct.get_pointer(), conj_l,
                  conj_r);
            });
      });
      break;
    default:
      errc = sycl_task_record(sycl_task, coh_ctrl, 71);
      errc = gpu_activate(cur_gpu);
      return 71;
    }
    // Partial tensor contraction (via TN matrix multiplication):
  } else {
#ifndef NO_BLAS
    if (DISABLE_BLAS == 0 &&
        gpu_is_mine(gpu_num) >= GPU_MINE_ONEMKL) { // BLAS is enabled
      cublas_handle[gpu_num] = *sycl_stream;
      // abb: check the following commented out region
      // if (err_onemkl != 0) {
      //   errc = sycl_task_record(sycl_task, coh_ctrl, 72);
      //   errc = gpu_activate(cur_gpu);
      //   return 72;
      // }

      switch (dtens->data_kind) {
      case R4:
	oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
				left_conj, right_conj,
				static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
				*static_cast<float *>(alpha_plus_p),
				static_cast<float *>(larg), static_cast<std::int64_t>(lc),
				static_cast<float *>(rarg), static_cast<std::int64_t>(lc),
				*static_cast<float *>(beta_p),
				static_cast<float *>(darg), static_cast<std::int64_t>(ll));
        break;

      case R8:
	oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
				left_conj, right_conj,
				static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
				*static_cast<double *>(alpha_plus_p),
				static_cast<double *>(larg), static_cast<std::int64_t>(lc),
				static_cast<double *>(rarg), static_cast<std::int64_t>(lc),
				*static_cast<double *>(beta_p),
				static_cast<double *>(darg), static_cast<std::int64_t>(ll));
        break;

      case C4:
        if (fast_math == YEP) {
          if (conj_r) {
                oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
					left_conj, right_conj,
					static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
					*static_cast<float *>(alpha_plus_p),
					&(((float *)larg)[0]), static_cast<std::int64_t>(lc),
					&(((float *)rarg)[0]), static_cast<std::int64_t>(lr),
					*static_cast<float *>(beta_p),
					&(((float *)darg)[0]), static_cast<std::int64_t>(ll));
                oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
					left_conj, right_conj,
					static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
					*static_cast<float *>(alpha_minus_p),
					&(((float *)larg)[vol_l]), static_cast<std::int64_t>(lc),
					&(((float *)rarg)[vol_r]), static_cast<std::int64_t>(lr),
					*static_cast<float *>(beta_one_p),
					&(((float *)darg)[0]), static_cast<std::int64_t>(ll));
                oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
					left_conj, right_conj,
					static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
					*static_cast<float *>(alpha_plus_p),
					&(((float *)larg)[vol_l]), static_cast<std::int64_t>(lc),
					&(((float *)rarg)[0]), static_cast<std::int64_t>(lr),
					*static_cast<float *>(beta_p),
					&(((float *)darg)[vol_d]), static_cast<std::int64_t>(ll));
                oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
					left_conj, right_conj,
					static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
					*static_cast<float *>(alpha_plus_p),
					&(((float *)larg)[0]), static_cast<std::int64_t>(lc),
					&(((float *)rarg)[vol_r]), static_cast<std::int64_t>(lr),
					*static_cast<float *>(beta_one_p),
					&(((float *)darg)[vol_d]), static_cast<std::int64_t>(ll));
          } else {
	    oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
				    left_conj, right_conj,
				    static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
				    *static_cast<float *>(alpha_plus_p),
				    &(((float *)larg)[0]), static_cast<std::int64_t>(lc),
				    &(((float *)rarg)[0]), static_cast<std::int64_t>(lc),
				    *static_cast<float *>(beta_p),
				    &(((float *)darg)[0]), static_cast<std::int64_t>(ll));
	    oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
				    left_conj, right_conj,
				    static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
				    *static_cast<float *>(alpha_minus_p),
				    &(((float *)larg)[vol_l]), static_cast<std::int64_t>(lc),
				    &(((float *)rarg)[vol_r]), static_cast<std::int64_t>(lc),
				    *static_cast<float *>(beta_one_p),
				    &(((float *)darg)[0]), static_cast<std::int64_t>(ll));
	    oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
				    left_conj, right_conj,
				    static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
				    *static_cast<float *>(alpha_plus_p),
				    &(((float *)larg)[vol_l]), static_cast<std::int64_t>(lc),
				    &(((float *)rarg)[0]), static_cast<std::int64_t>(lc),
				    *static_cast<float *>(beta_p),
				    &(((float *)darg)[vol_d]), static_cast<std::int64_t>(ll));
	    oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
				    left_conj, right_conj,
				    static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
				    *static_cast<float *>(alpha_plus_p),
				    &(((float *)larg)[0]), static_cast<std::int64_t>(lc),
				    &(((float *)rarg)[vol_r]), static_cast<std::int64_t>(lc),
				    *static_cast<float *>(beta_one_p),
				    &(((float *)darg)[vol_d]), static_cast<std::int64_t>(ll));
          }
        } else {
          if (conj_r) {
	    oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
				    left_conj, right_conj,
				    static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
				    *static_cast<talshComplex4 *>(alpha_plus_p),
				    static_cast<talshComplex4 *>(larg), static_cast<std::int64_t>(lc),
				    static_cast<talshComplex4 *>(rarg), static_cast<std::int64_t>(lr),
				    *static_cast<talshComplex4 *>(beta_p),
				    static_cast<talshComplex4 *>(darg), static_cast<std::int64_t>(ll));
          } else {
	    oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
				    left_conj, right_conj,
				    static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
				    *static_cast<talshComplex4 *>(alpha_plus_p),
				    static_cast<talshComplex4 *>(larg), static_cast<std::int64_t>(lc),
				    static_cast<talshComplex4 *>(rarg), static_cast<std::int64_t>(lc),
				    *static_cast<talshComplex4 *>(beta_p),
				    static_cast<talshComplex4 *>(darg), static_cast<std::int64_t>(ll));
          }
        }
        break;

      case C8:
        if (conj_r) {
	  oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
				  left_conj, right_conj,
				  static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
				  *static_cast<talshComplex8 *>(alpha_plus_p),
				  static_cast<talshComplex8 *>(larg), static_cast<std::int64_t>(lc),
				  static_cast<talshComplex8 *>(rarg), static_cast<std::int64_t>(lr),
				  *static_cast<talshComplex8 *>(beta_p),
				  static_cast<talshComplex8 *>(darg), static_cast<std::int64_t>(ll));
        } else {
	  oneapi::mkl::blas::gemm(*cublas_handle[gpu_num],
				  left_conj, right_conj,
				  static_cast<std::int64_t>(ll), static_cast<std::int64_t>(lr), static_cast<std::int64_t>(lc),
				  *static_cast<talshComplex8 *>(alpha_plus_p),
				  static_cast<talshComplex8 *>(larg), static_cast<std::int64_t>(lc),
				  static_cast<talshComplex8 *>(rarg), static_cast<std::int64_t>(lc),
				  *static_cast<talshComplex8 *>(beta_p),
				  static_cast<talshComplex8 *>(darg), static_cast<std::int64_t>(ll));
        }
        break;

      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 73);
        errc = gpu_activate(cur_gpu);
        return 73;
      }
      if (err_onemkl != 0) {
        errc = sycl_task_record(sycl_task, coh_ctrl, 74);
        errc = gpu_activate(cur_gpu);
        return 74;
      }
    } else { // BLAS is disabled
#endif       /*NO_BLAS*/
      bx = 1 + (vol_l - 1) / MAT_MULT_TILE_DIMX;
      by = 1 + (vol_r - 1) / MAT_MULT_TILE_DIMY;
      limit_sycl_workgroups2d(MAX_SYCL_BLOCKS, &bx, &by);
      // if(DEBUG)
      // printf("\n#DEBUG(tensor_algebra_gpu_intel:gpu_tensor_block_contract_dlf):
      // CUDA exec conf: %d %d %d
      // %d\n",bx,by,MAT_MULT_TILE_DIMX,MAT_MULT_TILE_DIMY); //debug
      cl::sycl::range<2> blcks(bx, by);
      cl::sycl::range<2> thrds(MAT_MULT_TILE_DIMX, MAT_MULT_TILE_DIMY);
      auto global_range = blcks * thrds;
      cl::sycl::range<2> buf1_range(17 /*MAT_MULT_TILE_DIMX+1*/,
                                    17 /*MAT_MULT_TILE_DIMX+1*/);
      cl::sycl::range<2> buf2_range(17 /*MAT_MULT_TILE_DIMY+1*/,
                                    17 /*MAT_MULT_TILE_DIMX+1*/);

      switch (dtens->data_kind) {
      case R4:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          local_accessor<T, 2> buf1_acc(buf1_range, cgh);
          local_accessor<T, 2> buf2_acc(buf2_range, cgh);

          auto gpu_error_count_ptr = gpu_error_count.get_ptr();

          cgh.parallel_for(cl::sycl::nd_range<2>(global_range, thrds),
                           [=](cl::sycl::nd_item<2> item) {
                             gpu_matrix_multiply_tn__(
                                 ll, lr, lc, (float *)larg, (float *)rarg,
                                 (float *)darg, (float)scale_real, item,
                                 gpu_error_count_ptr, buf1_acc, buf2_acc);
                           });
        });
        break;
      case R8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          local_accessor<T, 2> buf1_acc(buf1_range, cgh);
          local_accessor<T, 2> buf2_acc(buf2_range, cgh);

          auto gpu_error_count_ptr = gpu_error_count.get_ptr();

          cgh.parallel_for(cl::sycl::nd_range<2>(global_range, thrds),
                           [=](cl::sycl::nd_item<2> item) {
                             gpu_matrix_multiply_tn__(
                                 ll, lr, lc, (double *)larg, (double *)rarg,
                                 (double *)darg, (double)scale_real, item,
                                 gpu_error_count_ptr, buf1_acc, buf2_acc);
                           });
        });
        break;
      default: //`Add complex cases with and without conjugation
        errc = sycl_task_record(sycl_task, coh_ctrl, 75);
        errc = gpu_activate(cur_gpu);
        return 75;
      }
#ifndef NO_BLAS
    }
#endif
  }
  switch (dtens->data_kind) {
  case R4:
    gpu_stats[gpu_num].flops += 2.0 * ((double)(lc)) * ((double)(ll)) * ((double)(lr));
    break;
  case R8:
    gpu_stats[gpu_num].flops += 2.0 * ((double)(lc)) * ((double)(ll)) * ((double)(lr));
    break;
  case C4:
    gpu_stats[gpu_num].flops += 8.0 * ((double)(lc)) * ((double)(ll)) * ((double)(lr));
    break;
  case C8:
    gpu_stats[gpu_num].flops += 8.0 * ((double)(lc)) * ((double)(ll)) * ((double)(lr));
    break;
  }
  // Schedule the inverse tensor transpose for the destination tensor:
  if (perm_d == YEP) {
    if (TRANS_SHMEM == EFF_TRN_ON) {
      bx = 1 + (vol_d - 1) / THRDS_TENSOR_COPY;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (dtens->data_kind) {
      case R4:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);
          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);

          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (float *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (float *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);
          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);

          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (double *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (double *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      case C4:
        if (fast_math == YEP) {
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr = gpu_error_count.get_ptr();

            local_accessor<T, 1> buf0_acc(
                cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
            local_accessor<float, 0> val_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> ftb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<size_t, 1> gtb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> htb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> stb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> dim_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> dim_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> o2n_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> pri_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> tmp0_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 0> err_code_acc(cgh);
            local_accessor<int, 0> minor_acc(cgh);
            local_accessor<int, 0> minor_in_acc(cgh);
            local_accessor<int, 0> minor_out_acc(cgh);
            local_accessor<int, 0> s1_ind_acc(cgh);
            local_accessor<int, 0> s1_ond_acc(cgh);
            local_accessor<int, 0> s1_step_acc(cgh);
            local_accessor<int, 0> s1_dim_acc(cgh);
            local_accessor<int, 0> s2_ind_acc(cgh);
            local_accessor<int, 0> s2_ond_acc(cgh);
            local_accessor<int, 0> s2_step_acc(cgh);
            local_accessor<int, 0> s2_dim_acc(cgh);
            local_accessor<int, 0> ns1_acc(cgh);
            local_accessor<int, 0> ns2_acc(cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 0> vol_ext_acc(cgh);
            auto const_args_dims_acc = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc = const_args_prmn.get_access(cgh);

            auto sycl_task_tens_args_const_mem_entry3 =
                sycl_task->tens_args[0].const_mem_entry;
            auto dtens_tmp_rsc_gmem_p4 =
                (talshComplex4 *)(dtens->tmp_rsc->gmem_p);
            auto dtens_dst_rsc_gmem_p5 =
                (talshComplex4 *)(dtens->dst_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY,
                                      THRDS_TENSOR_COPY),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_cmplx_split_in_dlf__(
                      1, 0, drank, sycl_task_tens_args_const_mem_entry3,
                      dtens_tmp_rsc_gmem_p4, dtens_dst_rsc_gmem_p5, item,
                      const_args_dims_acc, const_args_prmn_acc,
                      gpu_error_count_ptr, buf0_acc.get_pointer(),
                      val_acc.get_pointer(), base_in_acc.get_pointer(),
                      base_out_acc.get_pointer(), ftb_acc.get_pointer(),
                      gtb_acc.get_pointer(), htb_acc.get_pointer(),
                      stb_acc.get_pointer(), dim_in_acc.get_pointer(),
                      dim_out_acc.get_pointer(), o2n_acc.get_pointer(),
                      n2o_acc.get_pointer(), pri_acc.get_pointer(),
                      tmp0_acc.get_pointer(), err_code_acc.get_pointer(),
                      minor_acc.get_pointer(), minor_in_acc.get_pointer(),
                      minor_out_acc.get_pointer(), s1_ind_acc.get_pointer(),
                      s1_ond_acc.get_pointer(), s1_step_acc.get_pointer(),
                      s1_dim_acc.get_pointer(), s2_ind_acc.get_pointer(),
                      s2_ond_acc.get_pointer(), s2_step_acc.get_pointer(),
                      s2_dim_acc.get_pointer(), ns1_acc.get_pointer(),
                      ns2_acc.get_pointer(), vol_acc.get_pointer(),
                      vol_ext_acc.get_pointer());
                });
          });
        } else {
          (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
            auto gpu_error_count_ptr = gpu_error_count.get_ptr();

            local_accessor<T, 1> buf0_acc(
                cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
            local_accessor<float, 0> val_acc(cgh);
            local_accessor<size_t, 1> base_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> base_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<size_t, 1> ftb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<size_t, 1> gtb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> htb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> stb_acc(
                cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
            local_accessor<int, 1> dim_in_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> dim_out_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> o2n_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> n2o_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> pri_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 1> tmp0_acc(
                cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
            local_accessor<int, 0> err_code_acc(cgh);
            local_accessor<int, 0> minor_acc(cgh);
            local_accessor<int, 0> minor_in_acc(cgh);
            local_accessor<int, 0> minor_out_acc(cgh);
            local_accessor<int, 0> s1_ind_acc(cgh);
            local_accessor<int, 0> s1_ond_acc(cgh);
            local_accessor<int, 0> s1_step_acc(cgh);
            local_accessor<int, 0> s1_dim_acc(cgh);
            local_accessor<int, 0> s2_ind_acc(cgh);
            local_accessor<int, 0> s2_ond_acc(cgh);
            local_accessor<int, 0> s2_step_acc(cgh);
            local_accessor<int, 0> s2_dim_acc(cgh);
            local_accessor<int, 0> ns1_acc(cgh);
            local_accessor<int, 0> ns2_acc(cgh);
            local_accessor<size_t, 0> vol_acc(cgh);
            local_accessor<size_t, 0> vol_ext_acc(cgh);
            auto const_args_dims_acc = const_args_dims.get_access(cgh);
            auto const_args_prmn_acc = const_args_prmn.get_access(cgh);

            auto sycl_task_tens_args_const_mem_entry3 =
                sycl_task->tens_args[0].const_mem_entry;
            auto dtens_tmp_rsc_gmem_p4 =
                (talshComplex4 *)(dtens->tmp_rsc->gmem_p);
            auto dtens_dst_rsc_gmem_p5 =
                (talshComplex4 *)(dtens->dst_rsc->gmem_p);

            cgh.parallel_for(
                cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY,
                                      THRDS_TENSOR_COPY),
                [=](cl::sycl::nd_item<1> item) {
                  gpu_tensor_block_copy_dlf__(
                      1, 0, drank, sycl_task_tens_args_const_mem_entry3,
                      dtens_tmp_rsc_gmem_p4, dtens_dst_rsc_gmem_p5, item,
                      const_args_dims_acc, const_args_prmn_acc,
                      gpu_error_count_ptr, buf0_acc.get_pointer(),
                      val_acc.get_pointer(), base_in_acc.get_pointer(),
                      base_out_acc.get_pointer(), ftb_acc.get_pointer(),
                      gtb_acc.get_pointer(), htb_acc.get_pointer(),
                      stb_acc.get_pointer(), dim_in_acc.get_pointer(),
                      dim_out_acc.get_pointer(), o2n_acc.get_pointer(),
                      n2o_acc.get_pointer(), pri_acc.get_pointer(),
                      tmp0_acc.get_pointer(), err_code_acc.get_pointer(),
                      minor_acc.get_pointer(), minor_in_acc.get_pointer(),
                      minor_out_acc.get_pointer(), s1_ind_acc.get_pointer(),
                      s1_ond_acc.get_pointer(), s1_step_acc.get_pointer(),
                      s1_dim_acc.get_pointer(), s2_ind_acc.get_pointer(),
                      s2_ond_acc.get_pointer(), s2_step_acc.get_pointer(),
                      s2_dim_acc.get_pointer(), ns1_acc.get_pointer(),
                      ns2_acc.get_pointer(), vol_acc.get_pointer(),
                      vol_ext_acc.get_pointer());
                });
          });
        }
        break;
      case C8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<T, 1> buf0_acc(
              cl::sycl::range(1536 /*TENS_TRANSP_BUF_SIZE*/), cgh);
          local_accessor<float, 0> val_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> ftb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<size_t, 1> gtb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> htb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> stb_acc(
              cl::sycl::range(69 /*TENS_TRANSP_TAB_SIZE*/), cgh);
          local_accessor<int, 1> dim_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> dim_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> o2n_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> pri_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 1> tmp0_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<int, 0> err_code_acc(cgh);
          local_accessor<int, 0> minor_acc(cgh);
          local_accessor<int, 0> minor_in_acc(cgh);
          local_accessor<int, 0> minor_out_acc(cgh);
          local_accessor<int, 0> s1_ind_acc(cgh);
          local_accessor<int, 0> s1_ond_acc(cgh);
          local_accessor<int, 0> s1_step_acc(cgh);
          local_accessor<int, 0> s1_dim_acc(cgh);
          local_accessor<int, 0> s2_ind_acc(cgh);
          local_accessor<int, 0> s2_ond_acc(cgh);
          local_accessor<int, 0> s2_step_acc(cgh);
          local_accessor<int, 0> s2_dim_acc(cgh);
          local_accessor<int, 0> ns1_acc(cgh);
          local_accessor<int, 0> ns2_acc(cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 0> vol_ext_acc(cgh);
          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);

          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (talshComplex8 *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (talshComplex8 *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY, THRDS_TENSOR_COPY),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, buf0_acc_ct.get_pointer(),
                    val_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer(), ftb_acc_ct.get_pointer(),
                    gtb_acc_ct.get_pointer(), htb_acc_ct.get_pointer(),
                    stb_acc_ct.get_pointer(), dim_in_acc_ct.get_pointer(),
                    dim_out_acc_ct.get_pointer(), o2n_acc_ct.get_pointer(),
                    n2o_acc_ct.get_pointer(), pri_acc_ct.get_pointer(),
                    tmp0_acc_ct.get_pointer(), err_code_acc_ct.get_pointer(),
                    minor_acc_ct.get_pointer(), minor_in_acc_ct.get_pointer(),
                    minor_out_acc_ct.get_pointer(), s1_ind_acc_ct.get_pointer(),
                    s1_ond_acc_ct.get_pointer(), s1_step_acc_ct.get_pointer(),
                    s1_dim_acc_ct.get_pointer(), s2_ind_acc_ct.get_pointer(),
                    s2_ond_acc_ct.get_pointer(), s2_step_acc_ct.get_pointer(),
                    s2_dim_acc_ct.get_pointer(), ns1_acc_ct.get_pointer(),
                    ns2_acc_ct.get_pointer(), vol_acc_ct.get_pointer(),
                    vol_ext_acc_ct.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 77);
        errc = gpu_activate(cur_gpu);
        return 77;
      }
    } else if (TRANS_SHMEM == EFF_TRN_ON_CUTT) {
      errc = sycl_task_record(sycl_task, coh_ctrl, 80);
      errc = gpu_activate(cur_gpu);
      return 80;
    } else if (TRANS_SHMEM == EFF_TRN_OFF) {
      bx = 1 + (vol_d - 1) / THRDS_TENSOR_COPY_SCAT;
      if (bx > MAX_SYCL_BLOCKS)
        bx = MAX_SYCL_BLOCKS;
      switch (dtens->data_kind) {
      case R4:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);

          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (float *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (float *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              sycl::nd_range(cl::sycl::range(bx * THRDS_TENSOR_COPY_SCAT),
                             sycl::range(THRDS_TENSOR_COPY_SCAT)),
              [=](cl::sycl::nd_item<3> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case R8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);

          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (double *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (double *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              sycl::nd_range(cl::sycl::range(bx * THRDS_TENSOR_COPY_SCAT),
                             sycl::range(THRDS_TENSOR_COPY_SCAT)),
              [=](cl::sycl::nd_item<3> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case C4:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);

          auto sycl_task_tens_args_const_mem_entry_ct3 = sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 = (talshComplex4 *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 = (talshComplex4 *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range(cl::sycl::range(bx * THRDS_TENSOR_COPY_SCAT),
                                 sycl::range(THRDS_TENSOR_COPY_SCAT)),
              [=](cl::sycl::nd_item<3> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      case C8:
        (*sycl_stream)->submit([&](cl::sycl::handler &cgh) {
          auto gpu_error_count_ptr_ct = gpu_error_count.get_ptr();

          local_accessor<int, 1> n2o_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 0> vol_acc(cgh);
          local_accessor<size_t, 1> base_in_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          local_accessor<size_t, 1> base_out_acc(
              cl::sycl::range(32 /*MAX_TENSOR_RANK*/), cgh);
          auto const_args_dims_acc_ct = const_args_dims.get_access(cgh);
          auto const_args_prmn_acc_ct = const_args_prmn.get_access(cgh);

          auto sycl_task_tens_args_const_mem_entry_ct3 =
              sycl_task->tens_args[0].const_mem_entry;
          auto dtens_tmp_rsc_gmem_p_ct4 =
              (talshComplex8 *)(dtens->tmp_rsc->gmem_p);
          auto dtens_dst_rsc_gmem_p_ct5 =
              (talshComplex8 *)(dtens->dst_rsc->gmem_p);

          cgh.parallel_for(
              cl::sycl::nd_range<1>(bx * THRDS_TENSOR_COPY_SCAT,
                                    THRDS_TENSOR_COPY_SCAT),
              [=](cl::sycl::nd_item<1> item) {
                gpu_tensor_block_copy_scatter_dlf__(
                    1, 0, drank, sycl_task_tens_args_const_mem_entry_ct3,
                    dtens_tmp_rsc_gmem_p_ct4, dtens_dst_rsc_gmem_p_ct5, item,
                    const_args_dims_acc_ct, const_args_prmn_acc_ct,
                    gpu_error_count_ptr_ct, n2o_acc_ct.get_pointer(),
                    vol_acc_ct.get_pointer(), base_in_acc_ct.get_pointer(),
                    base_out_acc_ct.get_pointer());
              });
        });
        break;
      default:
        errc = sycl_task_record(sycl_task, coh_ctrl, 81);
        errc = gpu_activate(cur_gpu);
        return 81;
      }
    } else {
      errc = sycl_task_record(sycl_task, coh_ctrl, 82);
      errc = gpu_activate(cur_gpu);
      return 82;
    }
  }
  // Record a SYCL queue (output ready on GPU):
  sycl_output_ct = std::chrono::high_resolution_clock::now();
  err = 0;
  if (err != 0) {
    if (VERBOSE)
      printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_dlf):"
             " Unable to record the output event: %s\n",
             err_msg);
    errc = sycl_task_record(sycl_task, coh_ctrl, 83);
    errc = gpu_activate(cur_gpu);
    return 83;
  }
  // Transfer back the updated destination tensor if needed ("T","K" coherence
  // control):
  coh = (coh_ctrl >> 4) &
        (TWO_BITS_SET); // select bits 4,5 (destination tensor coherence)
  if (gpu_d != gpu_num && coh >= 2) { // data is not on the computing GPU and
                                      // coherence control = 2("T") or (3)"K":
    err = (*sycl_stream->memcpy(dtens->src_rsc->gmem_p, dtens->dst_rsc->gmem_p,
                                dsize),
           0);
    if (err != 0) {
      if (VERBOSE)
        printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_"
               "dlf): Destination tensor body back copy failed: %s\n",
               err_msg);
      errc = sycl_task_record(sycl_task, coh_ctrl, 84);
      errc = gpu_activate(cur_gpu);
      return 84;
    }
    gpu_stats[gpu_num].traffic_out += dsize;
  }
  // Record a SYCL queue (task finished):
  sycl_finish_ct = std::chrono::high_resolution_clock::now();
  err = 0;
  if (err != 0) {
    if (VERBOSE)
      printf("\n#ERROR(tensor_algebra_gpu_intel:gpu_tensor_block_contract_dlf):"
             " Unable to record the finish event: %s\n",
             err_msg);
    errc = sycl_task_record(sycl_task, coh_ctrl, 85);
    errc = gpu_activate(cur_gpu);
    return 85;
  }
  // Record the successfully scheduled SYCL task and update the Last Task:
  errc = sycl_task_record(sycl_task, coh_ctrl, 0);
  LastTask[gpu_num] = sycl_task;
  if (gpu_num != cur_gpu)
    errc = gpu_activate(cur_gpu);
  return stat; // either 0 (success) or NOT_CLEAN (warning)
} catch (cl::sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int gpu_tensor_block_decompose_svd(const char absorb, tensBlck_t *dtens,
                                   tensBlck_t *ltens, tensBlck_t *rtens,
                                   tensBlck_t *stens, int gpu_id) {
  //`Finish
  return -1;
}

#endif /*NO_GPU*/
