// Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CUTLASS_HOST_DEVICE
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#endif
#ifndef CUTLASS_PRAGMA_UNROLL
#define CUTLASS_PRAGMA_UNROLL #pragma unroll
#endif

enum class ConvOperation {
    FPROP,
    DGRAD,
    WGRAD
};

template<typename T>__device__ T atomic_add(T* dst, T const& val) {
    atomicAdd(dst, val);
}

template<typename T, int offset0=16>__device__ T warp_reduce_sum(T val) {
    if constexpr(offset0 == 1)
        return val;
    if constexpr(offset0 > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int offset = offset0; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
        return val;
    }
}

constexpr int constexpr_sqrt(int res, int l, int r){
    if(l == r){
        return r;
    } else {
        const auto mid = (r + l) / 2;

        if(mid * mid >= res){
            return constexpr_sqrt(res, l, mid);
        } else {
            return constexpr_sqrt(res, mid + 1, r);
        }
    }
}

constexpr int constexpr_sqrt(int res){
    return constexpr_sqrt(res, 1, res);
}

template<typename T> constexpr
T const& constexpr_min(T const& a, T const& b) {
  return a < b ? a : b;
}

template<typename T> constexpr
T const& constexpr_max(T const& a, T const& b) {
  return a > b ? a : b;
}

template<typename T, int S_, int Sb_, int STR_>
struct GeneralFpropParams {
    static constexpr const int STR = STR_, S = S_, Sb = Sb_, SS = (S+Sb-1)/Sb;
    static constexpr const ConvOperation OPS = ConvOperation::FPROP;
    static constexpr const int WARP = 32, K = sizeof(float4) / sizeof(T); 
    static constexpr const int L = 4;
    
    T *theta;
    int N, C, H, W, P, Q, PAD,HW, PQ, RS;
    int DST_H, DST_W, SRC_H, SRC_W, DST_HW, SRC_HW;
    int N_THREADS, N_BLOCKS, BLOCK_H, BLOCK_HW; 

    CUTLASS_HOST_DEVICE GeneralFpropParams() {}
    CUTLASS_HOST_DEVICE GeneralFpropParams(T* theta, int N, int C, int H, int W, int P, int Q, int PAD) : theta(theta), N(N), C(C), H(H), W(W), P(P), Q(Q), PAD(PAD) {
        PQ = P*Q, RS=S, HW=H*W;
        compute_params();
    }

    CUTLASS_HOST_DEVICE T angle(int c) const {
        return theta[c];
    }

    CUTLASS_HOST_DEVICE void compute_params() {
        DST_H = P;
        DST_W = Q;
        SRC_H = H;
        SRC_W = W;
        DST_HW = DST_H*DST_W;
        SRC_HW = SRC_H*SRC_W;
        N_THREADS = max(min(STR*int(constexpr_sqrt((SRC_HW+WARP*K-1)/(WARP*K)))*WARP, 256), 32);
        N_BLOCKS = max((DST_HW+L*N_THREADS-1)/(L*N_THREADS), 1);
        BLOCK_H = min(SS+(N_THREADS*L*STR+DST_W-1)/DST_W, SRC_H);
        BLOCK_HW = BLOCK_H*SRC_W + (((SRC_W%K) == 0) ? 0 : K);
    }

    CUTLASS_HOST_DEVICE dim3 compute_grid_size() const {
        return dim3(N_BLOCKS, N, C);
    }
    CUTLASS_HOST_DEVICE dim3 compute_block_size() const {
        return dim3(N_THREADS, 1, 1);
    }
    CUTLASS_HOST_DEVICE int shared_mem() const {
        return BLOCK_HW + SS*Sb;
    }
};

template<typename T, int S_, int Sb_, int STR_>
struct GeneralDgradParams {
    static constexpr const int STR = STR_, S = S_, Sb = Sb_, SS = (S+Sb-1)/Sb;
    static constexpr const ConvOperation OPS = ConvOperation::DGRAD;
    static constexpr const int WARP = 32, K = sizeof(float4) / sizeof(T); 
    static constexpr const int L = 8;
    
    T *theta;
    int N, C, H, W, P, Q, PAD, HW, PQ, RS;
    int DST_H, DST_W, SRC_H, SRC_W, DST_HW, SRC_HW;
    int N_THREADS, N_BLOCKS, BLOCK_H, BLOCK_HW; 

    CUTLASS_HOST_DEVICE GeneralDgradParams() {}
    CUTLASS_HOST_DEVICE GeneralDgradParams(T* theta, int N, int C, int H, int W, int P, int Q, int PAD) : theta(theta), N(N), C(C), H(H), W(W), P(P), Q(Q), PAD(PAD) {
        PQ = P*Q, RS=S, HW=H*W;
        compute_params();
    }

    CUTLASS_HOST_DEVICE T angle(int c) const {
        return theta[c];
    }

    CUTLASS_HOST_DEVICE void compute_params() {
        DST_H = H;
        DST_W = W;
        SRC_H = P;
        SRC_W = Q;
        DST_HW = DST_H*DST_W;
        SRC_HW = SRC_H*SRC_W;
        N_THREADS = max(min(STR*int(constexpr_sqrt((SRC_HW+WARP*K-1)/(WARP*K)))*WARP, 128), 32);
        N_BLOCKS = max((DST_HW+L*N_THREADS-1)/(L*N_THREADS), 1);
        BLOCK_H = min(SS+(N_THREADS*L+DST_W*STR-1)/(DST_W*STR), SRC_H);
        BLOCK_HW = BLOCK_H*SRC_W + (((SRC_W%K) == 0) ? 0 : K);
    }

    CUTLASS_HOST_DEVICE dim3 compute_grid_size() const {
        return dim3(N_BLOCKS, N, C);
    }
    CUTLASS_HOST_DEVICE dim3 compute_block_size() const {
        return dim3(N_THREADS, 1, 1);
    }
    CUTLASS_HOST_DEVICE int shared_mem() const {
        return BLOCK_HW + SS*Sb;
    }
};

template<typename T, int S_, int Sb_, int STR_>
struct GeneralWgradParams {
    static constexpr const int STR = STR_, S = S_, Sb = Sb_, SS = (S+Sb-1)/Sb;
    static constexpr const ConvOperation OPS = ConvOperation::WGRAD;

    static constexpr const int LL = (STR == 4) ? 1 : 8/STR;
    static constexpr const int LLL = 1;
    static constexpr const int WARP = 32, K = sizeof(float4) / sizeof(T); 

    T *theta;
    int N, C, H, W, P, Q, PAD, HW, PQ, RS;
    int DST_H, DST_W, SRC_H, SRC_W, DST_HW, SRC_HW;
    int N_THREADS, N_WARPS, N_BLOCKS, BLOCK_H, BLOCK_P, BLOCK_HW, BLOCK_PQ; 

    CUTLASS_HOST_DEVICE GeneralWgradParams() {}
    CUTLASS_HOST_DEVICE GeneralWgradParams(T* theta, int N, int C, int H, int W, int P, int Q, int PAD) : theta(theta), N(N), C(C), H(H), W(W), P(P), Q(Q), PAD(PAD) {
        PQ = P*Q, RS=S, HW=H*W;
        compute_params();
    }

    CUTLASS_HOST_DEVICE T angle(int c) const {
        return theta[c];
    }

    CUTLASS_HOST_DEVICE void compute_params() {
        N_THREADS = max(STR*min(int(constexpr_sqrt((HW+WARP*K-1)/(WARP*K)))*WARP, 128), 32);
        N_WARPS = N_THREADS/WARP;

        N_BLOCKS = max((PQ+LL*N_THREADS-1)/(LL*N_THREADS), 1);
        BLOCK_H = min(SS+(N_THREADS*LL*STR+Q-1)/Q, H);
        BLOCK_P = min((N_THREADS*LL+Q-1)/Q, P);
        BLOCK_HW = BLOCK_H*W + (((W%K) == 0) ? 0 : K);
        BLOCK_PQ = BLOCK_P*Q + (((PQ%K) == 0) ? 0 : K);

        DST_W = W;
    }

    CUTLASS_HOST_DEVICE dim3 compute_grid_size() const {
        return dim3(N_BLOCKS, N, C);
    }
    CUTLASS_HOST_DEVICE dim3 compute_block_size() const {
        return dim3(N_THREADS, 1, 1);
    }
    CUTLASS_HOST_DEVICE int shared_mem() const {
        return BLOCK_HW + BLOCK_PQ + N_WARPS*RS*LL;
    }
};

template<typename Element, typename Params>
static __global__ void general_fprop_kernel(Element* dst, Element* flt, Element* src, Params params) {
    using Vector = float4;
    const int n   = blockIdx.y,
            c   = blockIdx.z,
            nc  = n*params.C+c;

    dst = dst + (nc * params.DST_HW); 
    src = src + (nc * params.SRC_HW);
    flt = flt + ( c * params.RS);

    const int READ_SIZE = params.N_THREADS*Params::K, N_READS = (params.BLOCK_HW+READ_SIZE-1)/READ_SIZE;
    const int COMP_SIZE = params.N_THREADS*params.N_BLOCKS, N_COMPS = (params.DST_HW+COMP_SIZE-1)/COMP_SIZE;
    const int WARP_COMP_SIZE = Params::WARP*N_COMPS;
    const int BLOCK_COMP_SIZE = params.N_THREADS*N_COMPS;

    extern __shared__ Element shared[];
    Element *shared_src = shared, *shared_flt = shared + params.BLOCK_HW;

    float cos, sin;
    sincosf(params.angle(c), &sin, &cos);

    const int cid = params.N_THREADS*blockIdx.x+threadIdx.x;
    const int block_base_comp0 = (cid/params.N_THREADS)*BLOCK_COMP_SIZE;
    const int rid = threadIdx.x;
    const int warp_base_read  = (rid/Params::WARP)*Params::WARP*N_READS*Params::K + (rid%Params::WARP)*Params::K;
    const int warp_base_read1 = (rid/Params::WARP+1)*Params::WARP*N_READS*Params::K;
    const int warp_base_comp  = (cid/Params::WARP)*WARP_COMP_SIZE + (cid%Params::WARP); 
    const int warp_base_comp1 = (cid/Params::WARP)*WARP_COMP_SIZE + WARP_COMP_SIZE;
    
    if(rid < params.Sb*params.SS) shared_flt[rid] = (rid < params.S) ? flt[rid] : Element(0);

    CUTLASS_PRAGMA_UNROLL
    for(int ss = 0; ss < Params::S; ss+=Params::SS) {
        const int base_h = max((block_base_comp0/params.DST_W)*Params::STR + min( int(-sin*(ss-params.PAD)),  int(-sin*(ss+Params::SS-1-params.PAD))),0);
        const int base_hw = (base_h*params.SRC_W) - ((nc*params.SRC_HW+base_h*params.SRC_W)%Params::K);

        if(base_hw+warp_base_read1 <= params.SRC_HW && warp_base_read1 <= params.BLOCK_HW) {
            for(int i = 0; i < N_READS; i++) {
                const int hw = warp_base_read+i*Params::WARP*Params::K;
                *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
            }
        }
        else if(base_hw+warp_base_read < params.SRC_HW && warp_base_read < params.BLOCK_HW){
            for(int i = 0; i < N_READS; i++) {
                const int hw = warp_base_read+i*Params::WARP*Params::K;
                if(hw < params.BLOCK_HW && base_hw+hw < params.SRC_HW) 
                    *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
            }
        }

        __syncthreads();
    
        if(warp_base_comp1 <= params.DST_HW) {
            for(int i = 0; i < N_COMPS; i++) {
                const int pq = warp_base_comp + i*Params::WARP, h0 = pq/params.DST_W, w0 = pq%params.DST_W;
                Element sum(0);
                CUTLASS_PRAGMA_UNROLL
                for(int s = 0; s < Params::SS; s++) {
                    const int h = h0*Params::STR+int(-sin*(ss+s-params.PAD)), w=w0*Params::STR+int(cos*(ss+s-params.PAD));
                    sum += (h >= 0 && h < params.SRC_H && w >= 0 && w < params.SRC_W) ? shared_src[h*params.SRC_W+w-base_hw]*shared_flt[ss+s] : Element(0);
                }
                dst[pq] = (ss == 0) ? sum : dst[pq] + sum;
            }
        }
        else if(warp_base_comp < params.DST_HW){
            for(int i = 0; i < N_COMPS; i++) {
                const int pq = warp_base_comp + i*Params::WARP, h0 = pq/params.DST_W, w0 = pq%params.DST_W;
                if(pq < params.DST_HW) {
                    Element sum(0);
                    CUTLASS_PRAGMA_UNROLL
                    for(int s = 0; s < Params::SS; s++) {
                        const int h = h0*Params::STR+int(-sin*(ss+s-params.PAD)), w=w0*Params::STR+int(cos*(ss+s-params.PAD));
                        sum += (h >= 0 && h < params.SRC_H && w >= 0 && w < params.SRC_W) ? shared_src[h*params.SRC_W+w-base_hw]*shared_flt[ss+s] : Element(0);
                    }
                    dst[pq] = (ss == 0) ? sum : dst[pq] + sum;
                }
            }
        }
    }
}

template<typename Element, typename Params>
static __global__ void general_dgrad_kernel(Element* dst, Element* flt, Element* src, Params params) {
    using Vector = float4;

    const int n = blockIdx.y,
              c = blockIdx.z,
             nc = n*params.C+c;

    dst = dst + (nc * params.DST_HW); 
    src = src + (nc * params.SRC_HW);
    flt = flt + ( c * params.RS);

    const int READ_SIZE = params.N_THREADS*Params::K, N_READS = (params.BLOCK_HW+READ_SIZE-1)/READ_SIZE;
    const int COMP_SIZE = params.N_THREADS*params.N_BLOCKS, N_COMPS = (params.DST_HW+COMP_SIZE-1)/COMP_SIZE;
    const int WARP_COMP_SIZE = Params::WARP*N_COMPS;
    const int BLOCK_COMP_SIZE = params.N_THREADS*N_COMPS;

    extern __shared__ Element shared[];
    Element *shared_src = shared, *shared_flt = shared + params.BLOCK_HW;

    float cos, sin;
    sincosf(params.angle(c), &sin, &cos);

    const int cid = params.N_THREADS*blockIdx.x+threadIdx.x;
    const int block_base_comp0 = (cid/params.N_THREADS)*BLOCK_COMP_SIZE;
    const int rid = threadIdx.x;
    const int warp_base_read  = (rid/Params::WARP)*Params::WARP*N_READS*Params::K + (rid%Params::WARP)*Params::K;
    const int warp_base_read1 = (rid/Params::WARP+1)*Params::WARP*N_READS*Params::K;
    const int warp_base_comp  = (cid/Params::WARP)*WARP_COMP_SIZE + (cid%Params::WARP); 
    const int warp_base_comp1 = (cid/Params::WARP)*WARP_COMP_SIZE + WARP_COMP_SIZE;

    if(rid < params.Sb*params.SS) shared_flt[rid] = (rid < params.S) ? flt[rid] : Element(0);

    CUTLASS_PRAGMA_UNROLL
    for(int ss = 0; ss < Params::S; ss+=Params::SS) {
        const int base_h = max((block_base_comp0/params.DST_W) + min(-int(-sin*(ss-params.PAD)), -int(-sin*(ss+Params::SS-1-params.PAD))),0)/Params::STR;
        const int base_hw = (base_h*params.SRC_W) - ((nc*params.SRC_HW+base_h*params.SRC_W)%Params::K);

        if(base_hw+warp_base_read1 <= params.SRC_HW && warp_base_read1 <= params.BLOCK_HW) {
            for(int i = 0; i < N_READS; i++) {
                const int hw = warp_base_read+i*Params::WARP*Params::K;
                *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
            }
        }
        else if(base_hw+warp_base_read < params.SRC_HW && warp_base_read < params.BLOCK_HW){
            for(int i = 0; i < N_READS; i++) {
                const int hw = warp_base_read+i*Params::WARP*Params::K;
                if(hw < params.BLOCK_HW && base_hw+hw < params.SRC_HW) 
                    *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
            }
        }

        __syncthreads();

        if(warp_base_comp1 <= params.DST_HW) {
            for(int i = 0; i < N_COMPS; i++) {
                const int hw = warp_base_comp + i*Params::WARP, h = hw/params.DST_W, w = hw%params.DST_W;

                Element sum(0);
                CUTLASS_PRAGMA_UNROLL
                for(int s = 0; s < Params::SS; s++) {
                    const int p_str = h-int(-sin*(ss+s-params.PAD)),         q_str = w-int(cos*(ss+s-params.PAD));
                    const int p_rem = p_str%Params::STR, q_rem = q_str%Params::STR,
                              p     = p_str/Params::STR, q     = q_str/Params::STR; 
                    sum += (p_str >= 0 && q_str >= 0 && p < params.SRC_H && q < params.SRC_W && p_rem == 0 && q_rem == 0) ? shared_src[p*params.SRC_W+q-base_hw]*shared_flt[ss+s] : Element(0);
                }
                dst[hw] = (ss == 0) ? sum : dst[hw] + sum;
            }
        }
        else if(warp_base_comp < params.DST_HW){
            for(int i = 0; i < N_COMPS; i++) {
                const int hw = warp_base_comp + i*Params::WARP, h = hw/params.DST_W, w = hw%params.DST_W;

                if(hw < params.DST_HW) {
                    Element sum(0);
                    CUTLASS_PRAGMA_UNROLL
                    for(int s = 0; s < Params::SS; s++) {
                        const int p_str = h-int(-sin*(ss+s-params.PAD)),         q_str = w-int(cos*(ss+s-params.PAD));
                        const int p_rem = p_str%Params::STR, q_rem = q_str%Params::STR,
                                  p     = p_str/Params::STR, q     = q_str/Params::STR; 
                        sum += (p_str >= 0 && q_str >= 0 && p < params.SRC_H && q < params.SRC_W && p_rem == 0 && q_rem == 0) ? shared_src[p*params.SRC_W+q-base_hw]*shared_flt[ss+s] : Element(0);
                    }
                    dst[hw] = (ss == 0) ? sum : dst[hw] + sum;
                }
            }
        }
    }
}


template<typename Element, typename Params>
__global__ void general_wgrad_kernel(Element* dst, Element* grd, Element* src, Params params) {
    using Vector = float4;

    const int n   = blockIdx.y,
              c   = blockIdx.z,
              nc  = n*params.C+c;
    dst = dst + ( c * params.RS);
    src = src + (nc * params.HW);
    grd = grd + (nc * params.PQ); 

    const int cid = params.N_THREADS*blockIdx.x+threadIdx.x;
    if(cid < Params::S && n == 0)
        dst[cid] = Element(0);

    const int N_WARPS = params.N_WARPS;
    const int READ_SIZE = params.N_THREADS*Params::K;
    const int N_READS_HW = (params.BLOCK_HW+READ_SIZE-1)/READ_SIZE;
    const int N_READS_PQ = (params.BLOCK_PQ+READ_SIZE-1)/READ_SIZE;
    
    const int COMP_SIZE = params.N_THREADS*params.N_BLOCKS, N_COMPS = (params.PQ+COMP_SIZE-1)/COMP_SIZE;
    const int WARP_COMP_SIZE = Params::WARP*N_COMPS;
    const int BLOCK_COMP_SIZE = params.N_THREADS*N_COMPS;

    extern __shared__ Element shared[];
    Element *shared_src = shared, *shared_grd = shared + params.BLOCK_HW, *shared_dst = shared + (params.BLOCK_HW + params.BLOCK_PQ);

    float cos, sin;
    sincosf(params.angle(c), &sin, &cos);

    const int tid = threadIdx.x, lane = tid%Params::WARP, warp = tid/Params::WARP;
    const int rid = threadIdx.x;
    const int warp_base_read_hw  = (rid/Params::WARP)*Params::WARP*N_READS_HW*Params::K + (rid%Params::WARP)*Params::K;
    const int warp_base_read_hw1 = (rid/Params::WARP+1)*Params::WARP*N_READS_HW*Params::K;
    const int warp_base_read_pq  = (rid/Params::WARP)*Params::WARP*N_READS_PQ*Params::K + (rid%Params::WARP)*Params::K;
    const int warp_base_read_pq1 = (rid/Params::WARP+1)*Params::WARP*N_READS_PQ*Params::K;
    const int block_base_comp0 = (cid/params.N_THREADS)*BLOCK_COMP_SIZE;

    const int warp_base_comp0 = (cid/Params::WARP)*WARP_COMP_SIZE;
    const int warp_base_comp1 = (1+cid/Params::WARP)*WARP_COMP_SIZE;
    const int warp_base_comp  = warp_base_comp0 + (cid%Params::WARP); 

    CUTLASS_PRAGMA_UNROLL
    for(int ss = 0; ss < Params::S; ss += Params::SS) {
        const int base_h = max((block_base_comp0/params.Q)*Params::STR + min( int(-sin*(-params.PAD)),  int(-sin*(ss+Params::SS-1-params.PAD))),0);
        const int base_pq = block_base_comp0 -  (((nc * params.PQ) + block_base_comp0)%Params::K);
        const int base_hw = base_h*params.W - (((nc * params.HW) + base_h*params.W)%Params::K);

        if(base_hw+warp_base_read_hw1 <= params.HW && warp_base_read_hw1 <= params.BLOCK_HW) {
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < N_READS_HW; i++) {
                const int hw = warp_base_read_hw+i*Params::WARP*Params::K;
                *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
            }
        }
        else if(base_hw+warp_base_read_hw < params.HW && warp_base_read_hw < params.BLOCK_HW){
            for(int i = 0; i < N_READS_HW; i++) {
                const int hw = warp_base_read_hw+i*Params::WARP*Params::K;
                if(hw < params.BLOCK_HW && base_hw+hw < params.HW) {
                    *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
                }
            }
        }

        if(base_pq+warp_base_read_pq1 <= params.PQ && warp_base_read_pq1 <= params.BLOCK_PQ) {
            for(int i = 0; i < N_READS_PQ; i++) {
                const int pq = warp_base_read_pq+i*Params::WARP*Params::K;
                CUTLASS_PRAGMA_UNROLL
                for(int j = 0; j < Params::K; j++)
                    shared_grd[pq+j] = grd[base_pq+pq+j];
            }
        }
        else if(base_pq+warp_base_read_pq < params.PQ && warp_base_read_pq < params.BLOCK_PQ){
            for(int i = 0; i < N_READS_PQ; i++) {
                const int pq = warp_base_read_pq+i*Params::WARP*Params::K;
                if(pq < params.BLOCK_PQ && base_pq+pq < params.PQ) {
                    CUTLASS_PRAGMA_UNROLL
                    for(int j = 0; j < Params::K; j++)
                        shared_grd[pq+j] = grd[base_pq+pq+j];
                }
            }
        }

        __syncthreads();

        CUTLASS_PRAGMA_UNROLL
        for(int s = 0; s < Params::SS; s++) {
            Element sum(0);
            const int h1 = int(-sin*(ss+s-params.PAD)), w1 = int(cos*(ss+s-params.PAD));
            for(int i = 0; i < N_COMPS; i++) {
                const int pq = warp_base_comp + i*Params::WARP;
                const int h = h1+(pq/params.Q)*Params::STR, w = w1+(pq%params.Q)*Params::STR;
                sum += (pq < params.PQ && h >= 0 && h < params.H && w >= 0 && w < params.W) ? shared_src[h*params.W+w-base_hw]*shared_grd[pq-base_pq] : Element(0);
            }
            sum = warp_reduce_sum<Element, 16/Params::LL>(sum);
            if((lane%(32/Params::LL)) == 0)
                shared_dst[(s+ss)*N_WARPS*Params::LL+warp*Params::LL+lane/(32/Params::LL)] = sum;
        }
    }

    __syncthreads();

    if(tid < Params::LLL*Params::S) {
        Element sum_(0);
        const int LLL_SIZE = N_WARPS*Params::LL/Params::LLL;
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < LLL_SIZE; i++)
            sum_ += shared_dst[tid*LLL_SIZE+i];
        if constexpr(Params::LLL == 1)
            atomic_add(dst+tid, sum_);
        else {
            sum_ = warp_reduce_sum<Element, Params::LLL/2>(sum_);
            if(lane%Params::LLL == 0)
                atomic_add(dst+(tid/Params::LLL), sum_);
        }
    }
}

template<typename Element, int S_ = 15, int Sb_ = 1, int STR_ = 1>
struct GeneralFpropKernel {
    template<typename Params>static void run(Element* dst, Element* flt, Element* src, Params const& params, cudaStream_t stream = 0) {
        const dim3 grid = params.compute_grid_size();
        const dim3 block = params.compute_block_size();
        const int shared_mem = params.shared_mem()*sizeof(Element);
        if(shared_mem >= 48 * 1024) {
            cudaFuncSetAttribute(general_fprop_kernel<Element, Params>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
        }
        general_fprop_kernel<Element, Params><<<grid, block, shared_mem, stream>>>(dst, flt, src, params);
    }
};

template<typename Element, int S_ = 15, int Sb_ = 1, int STR_ = 1>
struct GeneralDgradKernel {
    template<typename Params>static void run(Element* dst, Element* flt, Element* src, Params const& params, cudaStream_t stream = 0) {
        const dim3 grid = params.compute_grid_size();
        const dim3 block = params.compute_block_size();
        const int shared_mem = params.shared_mem()*sizeof(Element);
        if(shared_mem >= 48 * 1024) {
            cudaFuncSetAttribute(general_dgrad_kernel<Element, Params>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
        }
        general_dgrad_kernel<Element, Params><<<grid, block, shared_mem, stream>>>(dst, flt, src, params);
    }
};

template<typename Element, int S_ = 15, int Sb_ = 1, int STR_ = 1>
struct GeneralWgradKernel {
    template<typename Params>static void run(Element* dst, Element* flt, Element* src, Params const& params, cudaStream_t stream = 0) {
        const dim3 grid = params.compute_grid_size();
        const dim3 block = params.compute_block_size();
        const int shared_mem = params.shared_mem()*sizeof(Element);
        if(shared_mem >= 48 * 1024) {
            cudaFuncSetAttribute(general_wgrad_kernel<Element, Params>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
        }
        general_wgrad_kernel<Element, Params><<<grid, block, shared_mem, stream>>>(dst, flt, src, params);
    }
};