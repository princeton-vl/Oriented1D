// Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

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

template<typename T>
struct ProblemParams {
    T *theta;
    int N, C;

    CUTLASS_HOST_DEVICE ProblemParams() {}
    CUTLASS_HOST_DEVICE ProblemParams(T* theta, int N, int C) : theta(theta), N(N), C(C) {}

    CUTLASS_HOST_DEVICE T angle(int c) const {
        return theta[c];
    }
};

template<int DIV>
struct FastDiv {
    static CUTLASS_HOST_DEVICE int divide(int n) { return n / DIV; }
    static CUTLASS_HOST_DEVICE int modulo(int n) { return n % DIV; }
};
template<>
struct FastDiv<1> {
    static CUTLASS_HOST_DEVICE int divide(int n) { return n; }
    static CUTLASS_HOST_DEVICE int modulo(int n) { return 0; }
};
template<>
struct FastDiv<2> {
    static CUTLASS_HOST_DEVICE int divide(int n) { return n >> 1; }
    static CUTLASS_HOST_DEVICE int modulo(int n) { return n&1; }
};
template<>
struct FastDiv<4> {
    static CUTLASS_HOST_DEVICE int divide(int n) { return n >> 2; }
    static CUTLASS_HOST_DEVICE int modulo(int n) { return n&3; }
};


// https://baptiste-wicht.com/posts/2014/07/compile-integer-square-roots-at-compile-time-in-cpp.html
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

template<int branch, typename T>CUTLASS_HOST_DEVICE T constexpr_ternary(T const& a, T const& b) {
    if constexpr(branch == 1)
        return a;
    if constexpr(branch == 0)
        return b;
} 

template<typename T, int offset0=16>CUTLASS_HOST_DEVICE T warp_reduce_sum(T val) {
    CUTLASS_PRAGMA_UNROLL
    for (int offset = offset0; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


template<int H_, int W_, int S_, int STR_, int PAD_>
struct SpecializedFpropProblem {
    static constexpr const int H = H_, W = W_, R = 1, S = S_, PAD = PAD_, STR = STR_, P=H/STR, Q=W/STR, HW=H*W, PQ=P*Q, RS=R*S;
    static_assert(STR == 1 || STR == 2 || STR == 4, "Unimplemented");

    static constexpr const int DST_H = P;
    static constexpr const int DST_W = Q;
    static constexpr const int SRC_H = H;
    static constexpr const int SRC_W = W;
    static constexpr const int DST_HW = DST_H*DST_W;
    static constexpr const int SRC_HW = SRC_H*SRC_W;

    static constexpr const int WARP = 32;
    static constexpr const int K = 4;
    static constexpr const int N_THREADS = constexpr_max(constexpr_min(constexpr_sqrt((SRC_HW+WARP*K-1)/(WARP*K))*WARP, 128), 64);

    static constexpr const int L = (STR == 4) ? 1 : constexpr_min(32/constexpr_min(S*STR, 32), 4);

    static constexpr const int N_BLOCKS = constexpr_max((DST_HW+L*N_THREADS-1)/(L*N_THREADS), 1);

    static constexpr const int BLOCK_H = constexpr_min(S+(N_THREADS*L*STR+DST_W-1)/DST_W, SRC_H);
    static constexpr const int BLOCK_HW = BLOCK_H*SRC_W + (((SRC_W%K) == 0) ? 0 : K);

    template<typename Params>static CUTLASS_HOST_DEVICE dim3 compute_grid_size(Params const& params) {
        return dim3(N_BLOCKS, params.N, params.C);
    }
    static CUTLASS_HOST_DEVICE dim3 compute_block_size() {
        return dim3(N_THREADS, 1, 1);
    }
    static int shared_mem() {
        return BLOCK_HW + RS;
    }
};

template<int H_, int W_, int S_, int STR_, int PAD_>
struct SpecializedDgradProblem {
    static constexpr const int H = H_, W = W_, R = 1, S = S_, PAD = PAD_, STR = STR_, P=H/STR, Q=W/STR, HW=H*W, PQ=P*Q, RS=R*S;
    static_assert(STR == 1 || STR == 2 || STR == 4, "Unimplemented");

    static constexpr const int DST_H = H;
    static constexpr const int DST_W = W;
    static constexpr const int SRC_H = P;
    static constexpr const int SRC_W = Q;
    static constexpr const int DST_HW = DST_H*DST_W;
    static constexpr const int SRC_HW = SRC_H*SRC_W;

    static constexpr const int WARP = 32;
    static constexpr const int K = 4;
    static constexpr const int N_THREADS = constexpr_max(constexpr_min(constexpr_sqrt((SRC_HW+WARP*K-1)/(WARP*K))*WARP, 128), 64);

    static constexpr const int L = constexpr_min(32/constexpr_min(S, 32), 4);
    static constexpr const int N_BLOCKS = constexpr_max((DST_HW+L*N_THREADS-1)/(L*N_THREADS), 1);

    static constexpr const int BLOCK_H = constexpr_min(S+(N_THREADS*L+DST_W*STR-1)/(DST_W*STR), SRC_H);
    static constexpr const int BLOCK_HW = BLOCK_H*SRC_W + (((SRC_W%K) == 0) ? 0 : K);

    template<typename Params>static CUTLASS_HOST_DEVICE dim3 compute_grid_size(Params const& params) {
        return dim3(N_BLOCKS, params.N, params.C);
    }
    static CUTLASS_HOST_DEVICE dim3 compute_block_size() {
        return dim3(N_THREADS, 1, 1);
    }
    static int shared_mem() {
        return BLOCK_HW + RS;
    }
};

template<int H_, int W_, int S_, int STR_, int PAD_>
struct SpecializedWgradProblem {
    static constexpr const int H = H_, W = W_, R = 1, S = S_, PAD = PAD_, STR = STR_, P=H/STR, Q=W/STR, HW=H*W, PQ=P*Q, RS=R*S;
    static_assert(STR == 1 || STR == 2 || STR == 4, "Unimplemented");

    static constexpr const int WARP = 32;
    static constexpr const int K = 4;
    static constexpr const int N_THREADS = constexpr_max(constexpr_min(constexpr_sqrt((HW+WARP*K-1)/(WARP*K))*WARP, 128), 64);
    static constexpr const int N_WARPS = N_THREADS/WARP;

    static constexpr const int LL = (STR == 4) ? 1 : 8/STR;
    static constexpr const int LLL = 1;
    static_assert(LL >= LLL);

    static constexpr const int N_BLOCKS = constexpr_max((PQ+LL*N_THREADS-1)/(LL*N_THREADS), 1);
    static constexpr const int COMP_SIZE = N_THREADS*N_BLOCKS, N_COMPS = (PQ+COMP_SIZE-1)/COMP_SIZE;
    static_assert(LL >= N_COMPS);
    static constexpr const int BLOCK_H = constexpr_min(S+(N_THREADS*LL*STR+Q-1)/Q, H);
    static constexpr const int BLOCK_P = constexpr_min((N_THREADS*LL+Q-1)/Q, P);
    static constexpr const int BLOCK_HW = BLOCK_H*W + (((W%K) == 0) ? 0 : K);
    static constexpr const int BLOCK_PQ = BLOCK_P*Q + (((PQ%K) == 0) ? 0 : K);

    template<typename Params>static CUTLASS_HOST_DEVICE dim3 compute_grid_size(Params const& params) {
        return dim3(N_BLOCKS, params.N, params.C);
    }
    static CUTLASS_HOST_DEVICE dim3 compute_block_size() {
        return dim3(N_THREADS, 1, 1);
    }
    static CUTLASS_HOST_DEVICE int shared_mem() {
        return BLOCK_HW + BLOCK_PQ + N_WARPS*RS*LL;
    }
};

// We use vectorized reads to enable fast memory reads and use shared memory for faster computations 
template<typename Element, typename Problem, typename Params>
__global__ void specialized_fprop_kernel(Element* dst, Element* flt, Element* src, Params params) {
    using Vector = float4;

    const int n   = blockIdx.y,
              c   = blockIdx.z,
              nc  = n*params.C+c;

    dst = dst + (nc * Problem::DST_HW); 
    src = src + (nc * Problem::SRC_HW);
    flt = flt + ( c * Problem::RS);

    constexpr const int READ_SIZE = Problem::N_THREADS*Problem::K,        N_READS = (Problem::BLOCK_HW+READ_SIZE-1)/READ_SIZE;
    constexpr const int COMP_SIZE = Problem::N_THREADS*Problem::N_BLOCKS, N_COMPS = (Problem::DST_HW+COMP_SIZE-1)/COMP_SIZE;
    constexpr const int WARP_COMP_SIZE = Problem::WARP*N_COMPS;
    constexpr const int BLOCK_COMP_SIZE = Problem::N_THREADS*N_COMPS;

    extern __shared__ Element shared[];
    Element *shared_src = shared, *shared_flt = shared + Problem::BLOCK_HW;

    float cos, sin;
    sincosf(params.angle(c), &sin, &cos);

    const int cid = Problem::N_THREADS*blockIdx.x+threadIdx.x;
    const int block_base_comp0 = (cid/Problem::N_THREADS)*BLOCK_COMP_SIZE;
    const int base_h = max((block_base_comp0/Problem::DST_W)*Problem::STR + min( int(-(-Problem::PAD)),  int(-(Problem::S-1-Problem::PAD))),0);
    const int base_hw = (base_h*Problem::SRC_W) - ((nc*Problem::SRC_HW+base_h*Problem::SRC_W)%Problem::K);

    // read data
    const int rid = threadIdx.x;
    const int warp_base_read  = (rid/Problem::WARP)*Problem::WARP*N_READS*Problem::K + (rid%Problem::WARP)*Problem::K;
    const int warp_base_read1 = (rid/Problem::WARP+1)*Problem::WARP*N_READS*Problem::K;
    
    if(rid < Problem::RS) shared_flt[rid] = flt[rid];
    
    // split in two cases to avoid as most if conditions as possible 
    // -> avoid warp divergence / non-coalesced memory accesses
    // vectoized reads with Vector
    if(base_hw+warp_base_read1 <= Problem::SRC_HW && warp_base_read1 <= Problem::BLOCK_HW) {
        // specialized implementation allows for unrolling which speeds up data reads
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < N_READS; i++) {
            const int hw = warp_base_read+i*Problem::WARP*Problem::K;
            *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
        }
    }
    else if(base_hw+warp_base_read < Problem::SRC_HW && warp_base_read < Problem::BLOCK_HW){
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < N_READS; i++) {
            const int hw = warp_base_read+i*Problem::WARP*Problem::K;
            if(hw < Problem::BLOCK_HW && base_hw+hw < Problem::SRC_HW) 
                *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
        }
    }

    __syncthreads();
    
    // compute convolution and store in global memory
    const int warp_base_comp  = (cid/Problem::WARP)*WARP_COMP_SIZE + (cid%Problem::WARP); 
    const int warp_base_comp1 = (cid/Problem::WARP)*WARP_COMP_SIZE + WARP_COMP_SIZE;
    if(warp_base_comp1 <= Problem::DST_HW) {
        // specialized implementation allows for unrolling which speeds up computation
        CUTLASS_PRAGMA_UNROLL 
        for(int i = 0; i < N_COMPS; i++) {
            const int pq = warp_base_comp + i*Problem::WARP, h0 = (pq/Problem::DST_W)*Problem::STR, w0 = (pq%Problem::DST_W)*Problem::STR;
            Element sum(0);
            CUTLASS_PRAGMA_UNROLL
            for(int s = 0; s < Problem::S; s++) {
                const int h = h0+int(-sin*(s-Problem::PAD)), w=w0+int(cos*(s-Problem::PAD));
                // conditional memory reads are much less costly when reading from shared memory...
                sum += (h >= 0 && h < Problem::SRC_H && w >= 0 && w < Problem::SRC_W) ? shared_src[h*Problem::SRC_W+w-base_hw]*shared_flt[s] : 0;
            }
            dst[pq] = sum;
        }
    }
    else if(warp_base_comp < Problem::DST_HW){
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < N_COMPS; i++) {
            const int pq = warp_base_comp + i*Problem::WARP, h0 = (pq/Problem::DST_W)*Problem::STR, w0 = (pq%Problem::DST_W)*Problem::STR;
            if(pq < Problem::DST_HW) {
                Element sum(0);
                CUTLASS_PRAGMA_UNROLL
                for(int s = 0; s < Problem::S; s++) {
                    const int h = h0+int(-sin*(s-Problem::PAD)), w=w0+int(cos*(s-Problem::PAD));
                    sum += (h >= 0 && h < Problem::SRC_H && w >= 0 && w < Problem::SRC_W) ? shared_src[h*Problem::SRC_W+w-base_hw]*shared_flt[s] : 0;
                }
                dst[pq] = sum;
            }
        }
    }
}

// dgrad implementation is very similar to fprop because the formulas are similar
// the only difference is that we're summing over (h,w) instead of (p,q)
template<typename Element, typename Problem, typename Params>
__global__ void specialized_dgrad_kernel(Element* dst, Element* flt, Element* src, Params params) {
    using Vector = float4;

    const int n   = blockIdx.y,
              c   = blockIdx.z,
              nc  = n*params.C+c;

    dst = dst + (nc * Problem::DST_HW); 
    src = src + (nc * Problem::SRC_HW);
    flt = flt + ( c * Problem::RS);

    constexpr const int READ_SIZE = Problem::N_THREADS*Problem::K,        N_READS = (Problem::BLOCK_HW+READ_SIZE-1)/READ_SIZE;
    constexpr const int COMP_SIZE = Problem::N_THREADS*Problem::N_BLOCKS, N_COMPS = (Problem::DST_HW+COMP_SIZE-1)/COMP_SIZE;
    constexpr const int WARP_COMP_SIZE = Problem::WARP*N_COMPS;
    constexpr const int BLOCK_COMP_SIZE = Problem::N_THREADS*N_COMPS;

    extern __shared__ Element shared[];
    Element *shared_src = shared, *shared_flt = shared + Problem::BLOCK_HW;

    float cos, sin;
    sincosf(params.angle(c), &sin, &cos);

    const int cid = Problem::N_THREADS*blockIdx.x+threadIdx.x;
    const int block_base_comp0 = (cid/Problem::N_THREADS)*BLOCK_COMP_SIZE;
    const int base_h = max((block_base_comp0/Problem::DST_W) + min(-int(-(-Problem::PAD)), -int(-(Problem::S-1-Problem::PAD))),0)/Problem::STR;
    const int base_hw = (base_h*Problem::SRC_W) - ((nc*Problem::SRC_HW+base_h*Problem::SRC_W)%Problem::K);

    const int rid = threadIdx.x;
    const int warp_base_read  = (rid/Problem::WARP)*Problem::WARP*N_READS*Problem::K + (rid%Problem::WARP)*Problem::K;
    const int warp_base_read1 = (rid/Problem::WARP+1)*Problem::WARP*N_READS*Problem::K;
    
    if(rid < Problem::RS) shared_flt[rid] = flt[rid];
    
    // read data
    if(base_hw+warp_base_read1 <= Problem::SRC_HW && warp_base_read1 <= Problem::BLOCK_HW) {
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < N_READS; i++) {
            const int hw = warp_base_read+i*Problem::WARP*Problem::K;
            *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
        }
    }
    else if(base_hw+warp_base_read < Problem::SRC_HW && warp_base_read < Problem::BLOCK_HW){
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < N_READS; i++) {
            const int hw = warp_base_read+i*Problem::WARP*Problem::K;
            if(hw < Problem::BLOCK_HW && base_hw+hw < Problem::SRC_HW) 
                *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
        }
    }

    __syncthreads();
    
    // compute convolution
    const int warp_base_comp  = (cid/Problem::WARP)*WARP_COMP_SIZE + (cid%Problem::WARP); 
    const int warp_base_comp1 = (cid/Problem::WARP)*WARP_COMP_SIZE + WARP_COMP_SIZE;
    if(warp_base_comp1 <= Problem::DST_HW) {
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < N_COMPS; i++) {
            const int hw = warp_base_comp + i*Problem::WARP, h = (hw/Problem::DST_W), w = (hw%Problem::DST_W);
            Element sum(0);
            CUTLASS_PRAGMA_UNROLL
            for(int s = 0; s < Problem::S; s++) {
                const int p_str = h-int(-sin*(s-Problem::PAD)),         q_str = w-int(cos*(s-Problem::PAD));
                const int p_rem = FastDiv<Problem::STR>::modulo(p_str), q_rem = FastDiv<Problem::STR>::modulo(q_str); 
                const int p     = FastDiv<Problem::STR>::divide(p_str), q     = FastDiv<Problem::STR>::divide(q_str);
                sum += (p_str >= 0 && q_str >= 0 && p < Problem::SRC_H && q < Problem::SRC_W && p_rem == 0 && q_rem == 0) ? shared_src[p*Problem::SRC_W+q-base_hw]*shared_flt[s] : 0;
            }
            dst[hw] = sum;
        }
    }
    else if(warp_base_comp < Problem::DST_HW){
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < N_COMPS; i++) {
            const int hw = warp_base_comp + i*Problem::WARP, h = (hw/Problem::DST_W), w = (hw%Problem::DST_W);
            if(hw < Problem::DST_HW) {
                Element sum(0);
                CUTLASS_PRAGMA_UNROLL
                for(int s = 0; s < Problem::S; s++) {
                    const int p_str = h-int(-sin*(s-Problem::PAD)),         q_str = w-int(cos*(s-Problem::PAD));
                    const int p_rem = FastDiv<Problem::STR>::modulo(p_str), q_rem = FastDiv<Problem::STR>::modulo(q_str); 
                    const int p     = FastDiv<Problem::STR>::divide(p_str), q     = FastDiv<Problem::STR>::divide(q_str);
                    sum += (p_str >= 0 && q_str >= 0 && p < Problem::SRC_H && q < Problem::SRC_W && p_rem == 0 && q_rem == 0) ? shared_src[p*Problem::SRC_W+q-base_hw]*shared_flt[s] : 0;
                }
                dst[hw] = sum;
            }
        }
    }
}

// wgrad implementation is very different from fprop / dgrad 
// because we sum a lot of terms for a small number of output parameters
// We use warps and atomics to speed up the computation
template<typename Element, typename Problem, typename Params>
__global__ void specialized_wgrad_kernel(Element* dst, Element* grd, Element* src, Params params) {
    using Vector = float4;

    const int n   = blockIdx.y,
              c   = blockIdx.z,
              nc  = n*params.C+c;
    dst = dst + ( c * Problem::RS);
    src = src + (nc * Problem::HW);
    grd = grd + (nc * Problem::PQ); 

    const int cid = Problem::N_THREADS*blockIdx.x+threadIdx.x;
    if(cid < Problem::S && n == 0)
        dst[cid] = 0;

    constexpr const int N_WARPS = Problem::N_WARPS;
    constexpr const int READ_SIZE = Problem::N_THREADS*Problem::K;
    constexpr const int N_READS_HW = (Problem::BLOCK_HW+READ_SIZE-1)/READ_SIZE;
    constexpr const int N_READS_PQ = (Problem::BLOCK_PQ+READ_SIZE-1)/READ_SIZE;
    constexpr const int COMP_SIZE = Problem::N_THREADS*Problem::N_BLOCKS, N_COMPS = (Problem::PQ+COMP_SIZE-1)/COMP_SIZE;
    constexpr const int WARP_COMP_SIZE = Problem::WARP*N_COMPS;
    constexpr const int BLOCK_COMP_SIZE = Problem::N_THREADS*N_COMPS;

    extern __shared__ Element shared[];
    Element *shared_src = shared, *shared_grd = shared + Problem::BLOCK_HW, *shared_dst = shared + (Problem::BLOCK_HW + Problem::BLOCK_PQ);

    float cos, sin;
    sincosf(params.angle(c), &sin, &cos);

    const int block_base_comp0 = (cid/Problem::N_THREADS)*BLOCK_COMP_SIZE;
    const int base_h = max((block_base_comp0/Problem::Q)*Problem::STR + min( int(-(-Problem::PAD)),  int(-(Problem::S-1-Problem::PAD))),0);
    const int base_pq = block_base_comp0 -  (((nc * Problem::PQ) + block_base_comp0)%Problem::K);
    const int base_hw = base_h*Problem::W - (((nc * Problem::HW) + base_h*Problem::W)%Problem::K);
    // read data
    {
        const int rid = threadIdx.x;
        const int warp_base_read_hw  = (rid/Problem::WARP)*Problem::WARP*N_READS_HW*Problem::K + (rid%Problem::WARP)*Problem::K;
        const int warp_base_read_hw1 = (rid/Problem::WARP+1)*Problem::WARP*N_READS_HW*Problem::K;
        const int warp_base_read_pq  = (rid/Problem::WARP)*Problem::WARP*N_READS_PQ*Problem::K + (rid%Problem::WARP)*Problem::K;
        const int warp_base_read_pq1 = (rid/Problem::WARP+1)*Problem::WARP*N_READS_PQ*Problem::K;
        
        if(base_hw+warp_base_read_hw1 <= Problem::HW && warp_base_read_hw1 <= Problem::BLOCK_HW) {
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < N_READS_HW; i++) {
                const int hw = warp_base_read_hw+i*Problem::WARP*Problem::K;
                *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
            }
        }
        else if(base_hw+warp_base_read_hw < Problem::HW && warp_base_read_hw < Problem::BLOCK_HW){
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < N_READS_HW; i++) {
                const int hw = warp_base_read_hw+i*Problem::WARP*Problem::K;
                if(hw < Problem::BLOCK_HW && base_hw+hw < Problem::HW) {
                    *reinterpret_cast<Vector*>(shared_src+hw) = *(reinterpret_cast<Vector*>(src+base_hw+hw));
                }
            }
        }

        if(base_pq+warp_base_read_pq1 <= Problem::PQ && warp_base_read_pq1 <= Problem::BLOCK_PQ) {
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < N_READS_PQ; i++) {
                const int pq = warp_base_read_pq+i*Problem::WARP*Problem::K;
                *reinterpret_cast<Vector*>(shared_grd+pq) = *(reinterpret_cast<Vector*>(grd+base_pq+pq));
            }
        }
        else if(base_pq+warp_base_read_pq < Problem::PQ && warp_base_read_pq < Problem::BLOCK_PQ){
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < N_READS_PQ; i++) {
                const int pq = warp_base_read_pq+i*Problem::WARP*Problem::K;
                if(pq < Problem::BLOCK_PQ && base_pq+pq < Problem::PQ) {
                    // to avoid bug, have to set each element individualy...
                    CUTLASS_PRAGMA_UNROLL
                    for(int j = 0; j < Problem::K; j++)
                        shared_grd[pq+j] = grd[base_pq+pq+j];
                }
            }
        }
    }
    __syncthreads();

    // compute individual contributions and store in shared memory

    const int tid = threadIdx.x, lane = tid%Problem::WARP, warp = tid/Problem::WARP;
    {
        const int warp_base_comp0 = (cid/Problem::WARP)*WARP_COMP_SIZE;
        const int warp_base_comp1 = (1+cid/Problem::WARP)*WARP_COMP_SIZE;
        const int warp_base_comp  = warp_base_comp0 + (cid%Problem::WARP); 

        CUTLASS_PRAGMA_UNROLL
        for(int s = 0; s < Problem::S; s++) {
            Element sum(0);
            const int h1 = int(-sin*(s-Problem::PAD)), w1 = int(cos*(s-Problem::PAD));
            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < N_COMPS; i++) {
                const int pq = warp_base_comp + i*Problem::WARP, h0 = (pq/Problem::Q)*Problem::STR, w0 = (pq%Problem::Q)*Problem::STR;
                const int h = h0+h1, w=w0+w1;
                sum += (pq < Problem::PQ && h >= 0 && h < Problem::H && w >= 0 && w < Problem::W) ? shared_src[h*Problem::W+w-base_hw]*shared_grd[pq-base_pq] : 0;
            }
            sum = warp_reduce_sum<Element, 16/Problem::LL>(sum);
            if((lane%(32/Problem::LL)) == 0)
                shared_dst[s*N_WARPS*Problem::LL+warp*Problem::LL+lane/(32/Problem::LL)] = sum;
        }
    }

    __syncthreads();

    // aggregate individual contributions and store in global memory using atomic operations
    if(tid < Problem::LLL*Problem::S) {
        Element sum_(0);
        const int LLL_SIZE = N_WARPS*Problem::LL/Problem::LLL;
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < LLL_SIZE; i++)
            sum_ += shared_dst[tid*LLL_SIZE+i];
        if constexpr(Problem::LLL == 1)
            atomicAdd(dst+tid, sum_);
        else {
            sum_ = warp_reduce_sum<Element, Problem::LLL/2>(sum_);
            if(lane%Problem::LLL == 0)
                atomicAdd(dst+(tid/Problem::LLL), sum_);
        }
    }
}

template<typename Element, typename Problem, typename Params>
struct SpecializedFpropKernel {
    static void run(Element* dst, Element* flt, Element* src, Params const& params, cudaStream_t stream = 0) {
        const dim3 grid = Problem::compute_grid_size(params);
        const dim3 block = Problem::compute_block_size();
        const int shared_mem = Problem::shared_mem()*sizeof(Element);
        if(shared_mem >= 48 * 1024) {
            cudaFuncSetAttribute(specialized_fprop_kernel<Element, Problem, Params>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
        }
        specialized_fprop_kernel<Element, Problem, Params><<<grid, block, shared_mem, stream>>>(dst, flt, src, params);
    }
};

template<typename Element, typename Problem, typename Params>
struct SpecializedDgradKernel {
    static void run(Element* dst, Element* flt, Element* src, Params const& params, cudaStream_t stream = 0) {
        const dim3 grid = Problem::compute_grid_size(params);
        const dim3 block = Problem::compute_block_size();
        const int shared_mem = Problem::shared_mem()*sizeof(Element);
        if(shared_mem >= 48 * 1024) {
            cudaFuncSetAttribute(specialized_dgrad_kernel<Element, Problem, Params>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
        }
        specialized_dgrad_kernel<Element, Problem, Params><<<grid, block, shared_mem, stream>>>(dst, flt, src, params);
    }
};

template<typename Element, typename Problem, typename Params>
struct SpecializedWgradKernel {
    static void run(Element* dst, Element* grd, Element* src, Params const& params, cudaStream_t stream = 0) {
        const dim3 grid = Problem::compute_grid_size(params);
        const dim3 block = Problem::compute_block_size();
        const int shared_mem = Problem::shared_mem()*sizeof(Element);
        if(shared_mem >= 48 * 1024) {
            cudaFuncSetAttribute(specialized_wgrad_kernel<Element, Problem, Params>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);
        }
        specialized_wgrad_kernel<Element, Problem, Params><<<grid, block, shared_mem, stream>>>(dst, grd, src, params);
    }
};