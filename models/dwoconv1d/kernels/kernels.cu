// Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename T, int S, int Sb, int stride, int pad>
inline torch::Tensor template_fprop(torch::Tensor x, torch::Tensor w, torch::Tensor theta) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3), P = H/stride, Q=W/stride;
    TORCH_CHECK(H%stride == 0 && W%stride == 0, "Invalid input size: H,W not divisible by stride");
    
    using Params = GeneralFpropParams<T, S, Sb, stride>;
    using Kernel = GeneralFpropKernel<T, S, Sb, stride>;
    torch::Tensor y = torch::empty({N, C, P, Q}, x.options());
    Kernel::run((T*)y.data_ptr(), (T*)w.data_ptr(), (T*)x.data_ptr(), Params((T*)theta.data_ptr(), N, C, H, W, P, Q, pad), at::cuda::getCurrentCUDAStream().stream());
    return y;
}

template<typename T, int S, int Sb, int stride, int pad>
inline torch::Tensor template_dgrad(torch::Tensor dy, torch::Tensor w, torch::Tensor theta) {
    CHECK_INPUT(dy);
    CHECK_INPUT(w);
    const int N = dy.size(0), C = dy.size(1), P = dy.size(2), Q = dy.size(3), H=P*stride, W=Q*stride;
    
    using Params = GeneralDgradParams<T, S, Sb, stride>;
    using Kernel = GeneralDgradKernel<T, S, Sb, stride>;
    torch::Tensor dx = torch::empty({N, C, H, W}, dy.options());
    Kernel::run((T*)dx.data_ptr(), (T*)w.data_ptr(), (T*)dy.data_ptr(), Params((T*)theta.data_ptr(), N, C, H, W, P, Q, pad), at::cuda::getCurrentCUDAStream().stream());
    return dx;
}

template<typename T, int S, int Sb, int stride, int pad>
inline torch::Tensor template_wgrad(torch::Tensor dy, torch::Tensor x, torch::Tensor theta) {
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3), P = H/stride, Q=W/stride;
    TORCH_CHECK(H%stride == 0 && W%stride == 0, "Invalid input size: H,W not divisible by stride");
    
    using Params = GeneralWgradParams<T, S, Sb, stride>;
    using Kernel = GeneralWgradKernel<T, S, Sb, stride>;
    torch::Tensor dw = torch::empty({C, 1, 1, S}, dy.options());
    Kernel::run((T*)dw.data_ptr(), (T*)dy.data_ptr(), (T*)x.data_ptr(), Params((T*)theta.data_ptr(), N, C, H, W, P, Q, pad), at::cuda::getCurrentCUDAStream().stream());
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
