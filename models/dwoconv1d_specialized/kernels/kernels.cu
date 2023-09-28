// Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using Element = float;
using Params = ProblemParams<Element>;

template<int H, int W, int S, int stride, int pad>
inline torch::Tensor template_fprop_fp32(torch::Tensor x, torch::Tensor w, torch::Tensor theta) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    const int N = x.size(0), C = x.size(1), P = H/stride, Q=W/stride;
    TORCH_CHECK(x.size(2) == H && x.size(3) == W && w.size(2) == 1 && w.size(3) == S, "Input and kernel template specialization mismatch");
    
    using Problem = SpecializedFpropProblem<H, W, S, stride, pad>;
    using Kernel = SpecializedFpropKernel<Element, Problem, Params>;
    torch::Tensor y = torch::empty({N, C, P, Q}, x.options());
    Kernel::run((float*)y.data_ptr(), (float*)w.data_ptr(), (float*)x.data_ptr(), Params((float*)theta.data_ptr(), N, C), at::cuda::getCurrentCUDAStream().stream());
    return y;
}

template<int H, int W, int S, int stride, int pad>
inline torch::Tensor template_dgrad_fp32(torch::Tensor dy, torch::Tensor w, torch::Tensor theta) {
    CHECK_INPUT(dy);
    CHECK_INPUT(w);
    const int N = dy.size(0), C = dy.size(1), P = H/stride, Q=W/stride;
    TORCH_CHECK(dy.size(2) == P && dy.size(3) == Q && w.size(2) == 1 && w.size(3) == S, "Input and kernel template specialization mismatch");
    
    using Problem = SpecializedDgradProblem<H, W, S, stride, pad>;
    using Kernel = SpecializedDgradKernel<Element, Problem, Params>;
    torch::Tensor dx = torch::empty({N, C, H, W}, dy.options());
    Kernel::run((float*)dx.data_ptr(), (float*)w.data_ptr(), (float*)dy.data_ptr(), Params((float*)theta.data_ptr(), N, C), at::cuda::getCurrentCUDAStream().stream());
    return dx;
}

template<int H, int W, int S, int stride, int pad>
inline torch::Tensor template_wgrad_fp32(torch::Tensor dy, torch::Tensor x, torch::Tensor theta) {
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    const int N = dy.size(0), C = dy.size(1), P = H/stride, Q=W/stride;
    TORCH_CHECK(dy.size(2) == P && dy.size(3) == Q && x.size(2) == H && x.size(3) == W, "Input and kernel template specialization mismatch");
    
    using Problem = SpecializedWgradProblem<H, W, S, stride, pad>;
    using Kernel = SpecializedWgradKernel<Element, Problem, Params>;
    torch::Tensor dw = torch::empty({C, 1, 1, S}, dy.options());
    Kernel::run((float*)dw.data_ptr(), (float*)dy.data_ptr(), (float*)x.data_ptr(), Params((float*)theta.data_ptr(), N, C), at::cuda::getCurrentCUDAStream().stream());
    return dw;
}

template<typename T>torch::Tensor template_nhwc_to_nchw(torch::Tensor x) {
    int n = x.size(0), h = x.size(1), w = x.size(2), c = x.size(3);
    torch::Tensor y = torch::empty({n,c,h,w}, x.options());
    nab_to_nba(n, h*w, c, (T*)y.data_ptr(), (T*)x.data_ptr(), at::cuda::getCurrentCUDAStream().stream());
    return y;
}

template<typename T>torch::Tensor template_nchw_to_nhwc(torch::Tensor x) {
    int n = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
    torch::Tensor y = torch::empty({n,h,w,c}, x.options());
    nab_to_nba(n, c, h*w, (T*)y.data_ptr(), (T*)x.data_ptr(), at::cuda::getCurrentCUDAStream().stream());
    return y;
}