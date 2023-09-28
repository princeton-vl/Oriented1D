// Convolutional Networks with Oriented 1D Kernels
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CUTLASS_PRAGMA_UNROLL
#define CUTLASS_PRAGMA_UNROLL #pragma unroll
#endif


// Based on the NVIDIA Cutlass repository: https://github.com/NVIDIA/cutlass

/******************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
 
template <typename T>
__global__ void nab_to_nba_kernel(T *output, 
                                    const T *input, 
                                    const int n,
                                    const int a,
                                    const int b) {
  const int ab = a*b;
  __shared__ T shbuf[32 * (32 + 1)]; 
  const int32_t tid  = threadIdx.y*blockDim.x + threadIdx.x;
  const int32_t wid  = tid / 32; 
  const int32_t lid  = tid % 32; 
  const int32_t ni   = blockIdx.z;
  const int32_t ai0  = blockIdx.y * 32;  
  const int32_t bi0 = blockIdx.x * 32;  

  const size_t input_idx = ni * ab + (ai0 + wid) * b + bi0;
  const T *A = input + input_idx;
  if (bi0 + lid < b) {
    const int lid_x_33 = lid * 33;
    if ((ai0 + 32) <= a) {
      int ai = wid;  // between 0 and 7
      CUTLASS_PRAGMA_UNROLL
      for (int bLoopIdx = 0; bLoopIdx < 4; bLoopIdx++) { 
        shbuf[lid_x_33 + ai] = A[lid];
        A                     = &A[8 * b];
        ai += 8;
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int ai = wid; ai < 32; ai += 8) { 
        if ((ai + ai0) < a) {
          shbuf[lid_x_33 + ai] = A[lid];
        }
        A = &A[8 * b];
      }
    }
  }
  __syncthreads();

  const int32_t aiOut = ai0 + lid;
  output = &output[ni * ab + aiOut];
  if (aiOut < a) {
    if (bi0 + 32 < b) {
      int bI = wid;
      CUTLASS_PRAGMA_UNROLL
      for (int aLoopIdx = 0; aLoopIdx < 4; ++aLoopIdx) {
        output[(bi0 + bI) * a] = shbuf[(bI)*33 + lid];
        bI += 8;
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int bI = wid; bI < 32; bI += 8) {
        if (bi0 + bI < b) {
          output[(bi0 + bI) * a] = shbuf[(bI)*33 + lid];
        }
      }
    }
  }
}

template <typename T>void nab_to_nba(int n, int a, int b, T* output, T* input, cudaStream_t stream = 0) {
  dim3 grid((b + 31)/32, (a + 31)/32, n);
  dim3 block(32, 8);
  nab_to_nba_kernel<<<grid, block, 0, stream>>>(output, input, n, a, b);
}