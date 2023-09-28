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

inline int get(int n, int c, int h, int w, int N, int C, int H, int W) {
    return n*C*H*W + c*H*W + h*W + w;
}

template<typename T>
inline torch::Tensor template_fprop(torch::Tensor x_, torch::Tensor w_, torch::Tensor theta_, int S, int stride, int pad) {
    CHECK_INPUT(x_);
    CHECK_INPUT(w_);
    const int N = x_.size(0), C = x_.size(1), H = x_.size(2), W = x_.size(3), P = H/stride, Q=W/stride;
    TORCH_CHECK(H%stride == 0 && W%stride == 0, "Invalid input size: H,W not divisible by stride");

    torch::Tensor y_ = torch::zeros({N, C, P, Q}, x_.options());

    printf("fprop\n");

    auto device = x_.device();
    y_ = y_.cpu(), x_ = x_.cpu(), w_ = w_.cpu(), theta_ = theta_.cpu();
    T *y__ = (T*)y_.data_ptr(), *x__ = (T*)x_.data_ptr(), *w__ = (T*)w_.data_ptr();
    float *theta__ = (float*)theta_.data_ptr();

    for(int c = 0; c < C; c++) {
        float sin_ = sin(theta__[c]), cos_ = cos(theta__[c]);
        for(int n = 0; n < N; n++) {
            for(int p = 0; p < P; p++) {
                for(int q = 0; q < Q; q++) {
                    T sum = 0;
                    for(int s = 0 ; s < S; s++) {
                        int h = p*stride + int(-sin_*(s-pad));
                        int w = q*stride + int( cos_*(s-pad));
                        if(h >= 0 && h < H && w >= 0 && w < W)
                            sum += (x__[get(n, c, h, w, N, C, H, W)]*w__[get(c, 0, 0, s, C, 1 ,1 ,S)]);
                    }
                    y__[get(n, c, p, q, N, C, P, Q)] = sum;
                }
            }
        }
    }

    return y_.to(device);
}

template<typename T>
inline torch::Tensor template_dgrad(torch::Tensor dy_, torch::Tensor w_, torch::Tensor theta_, int S, int stride, int pad) {
    CHECK_INPUT(dy_);
    CHECK_INPUT(w_);
    const int N = dy_.size(0), C = dy_.size(1), P = dy_.size(2), Q = dy_.size(3), H=P*stride, W=Q*stride;
    
    torch::Tensor dx_ = torch::zeros({N, C, H, W}, dy_.options());

    printf("dgrad\n");
    auto device = dy_.device();
    dy_ = dy_.cpu(), dx_ = dx_.cpu(), w_ = w_.cpu(), theta_ = theta_.cpu();
    T *dy__ = (T*)dy_.data_ptr(), *dx__ = (T*)dx_.data_ptr(), *w__ = (T*)w_.data_ptr();
    float *theta__ = (float*)theta_.data_ptr();

    for(int c = 0; c < C; c++) {
        float sin_ = sin(theta__[c]), cos_ = cos(theta__[c]);
        for(int n = 0; n < N; n++) {
            for(int h = 0; h < H; h++) {
                for(int w = 0; w < W; w++) {
                    T sum = 0;
                    for(int s = 0 ; s < S; s++) {
                        int pp = h - int(-sin_*(s-pad));
                        int qq = w - int( cos_*(s-pad));
                        if(pp%stride != 0 || qq%stride != 0)
                            continue;
                        int p = pp/stride, q = qq/stride;
                        if(p >= 0 && p < P && q >= 0 && q < Q)
                            sum += dy__[get(n, c, p, q, N, C, P, Q)]*w__[get(c, 0, 0, s, C, 1, 1, S)];
                    }
                    dx__[get(n, c, h, w, N, C, H, W)] = sum;
                }
            }
        }
    }

    return dx_.to(device);
}

template<typename T>
inline torch::Tensor template_wgrad(torch::Tensor dy_, torch::Tensor x_, torch::Tensor theta_, int S, int stride, int pad) {
    CHECK_INPUT(dy_);
    CHECK_INPUT(x_);
    const int N = x_.size(0), C = x_.size(1), H = x_.size(2), W = x_.size(3), P = H/stride, Q=W/stride;
    TORCH_CHECK(H%stride == 0 && W%stride == 0, "Invalid input size: H,W not divisible by stride");
    
    torch::Tensor dw_ = torch::zeros({C, 1, 1, S}, dy_.options());

    printf("wgrad\n");

    auto device = dy_.device();
    dy_ = dy_.cpu(), dw_ = dw_.cpu(), x_ = x_.cpu(), theta_ = theta_.cpu();

    T *dw__ = (T*)dw_.data_ptr(), *dy__ = (T*)dy_.data_ptr(), *x__ = (T*)x_.data_ptr();
    float *theta__ = (float*)theta_.data_ptr();
    for(int c = 0; c < C; c++) {
        float sin_ = sin(theta__[c]), cos_ = cos(theta__[c]);
        for(int s = 0 ; s < S; s++) {
            T sum = 0;
            for(int p = 0; p < P; p++) {
                for(int q = 0; q < Q; q++) {
                    int h = p*stride + int(-sin_*(s-pad));
                    int w = q*stride + int( cos_*(s-pad));
                    if(h >= 0 && h < H && w >= 0 && w < W)
                        for(int n = 0; n < N; n++) 
                            sum += x__[get(n, c, h, w, N, C, H, W)]*dy__[get(n, c, p, q, N, C, P, Q)];
                }
            }
            dw__[get(c, 0, 0, s, C, 1, 1, S)] = sum;
        }
    }

    return dw_.to(device);
}

using namespace torch::autograd;

template<typename T>torch::Tensor template_nhwc_to_nchw(torch::Tensor x) {
    return x.permute({0,3,1,2}).contiguous();
}

template<typename T>torch::Tensor template_nchw_to_nhwc(torch::Tensor x) {
    return x.permute({0,2,3,1}).contiguous();
}

template<typename T>
class _DepthwiseOrientedConv1d : public Function<_DepthwiseOrientedConv1d<T>> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor x, torch::Tensor w, torch::Tensor theta, torch::Tensor bias, torch::Tensor resid, bool with_bias, bool with_resid, bool nhwc, int S, int stride, int pad) {
    ctx->set_materialize_grads(false);
    ctx->saved_data["nhwc"] = nhwc;
    ctx->saved_data["S"] = S;
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["pad"] = pad;
    ctx->saved_data["with_bias"] = with_bias;

    if(with_resid)
        w = w+resid;
    
    if(nhwc)
        x = template_nhwc_to_nchw<T>(x);

    ctx->save_for_backward({x, w, theta});

    auto y = template_fprop<T>(x, w, theta, S, stride, pad);
    if(with_bias)
        y = y + bias.to(y).view({1, -1, 1, 1});
    if(nhwc)
        y = template_nchw_to_nhwc<T>(y);

    return y;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto dy = grad_outputs[0];

    if(!dy.defined())
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};

    auto saved = ctx->get_saved_variables();
    const auto x = saved[0], w = saved[1], theta = saved[2];
    
    bool nhwc = ctx->saved_data["nhwc"].toBool();
    bool with_bias = ctx->saved_data["with_bias"].toBool();
    int S = ctx->saved_data["S"].toInt();
    int stride = ctx->saved_data["stride"].toInt();
    int pad = ctx->saved_data["pad"].toInt();
    
    if(nhwc)
        dy = template_nhwc_to_nchw<T>(dy.contiguous());
    else 
        dy = dy.contiguous();

    auto dx = template_dgrad<T>(dy, w, theta, S, stride, pad);
    const auto dw = template_wgrad<T>(dy, x, theta, S, stride, pad);
    const auto db = (with_bias) ? dy.sum({0,2,3}) : torch::Tensor();

    if(nhwc)
        dx = template_nchw_to_nhwc<T>(dx);
    
    return {dx, dw, torch::Tensor(), db, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

template<typename T>
inline torch::Tensor template_dwoconv1d(torch::Tensor x, torch::Tensor w, torch::Tensor theta, torch::Tensor bias, torch::Tensor resid, bool with_bias, bool with_resid, bool nhwc, int S, int stride, int pad) {
    return _DepthwiseOrientedConv1d<T>::apply(x, w, theta, bias, resid, with_bias, with_resid, nhwc, S, stride, pad);
}