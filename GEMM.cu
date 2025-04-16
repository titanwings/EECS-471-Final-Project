#include "pyt_all_reduce_kernel.hh"
// #include <cuda_fp16.h> // Removed for FP32 version

namespace eecs471 {


#define TILE_WIDTH 16
#define WARP_SIZE 32
#define MAX_K 7

// 将卷积索引映射到 GEMM 索引的宏
// 输出索引 (n,p,q,k) 映射到 GEMM C(i,j)
// 其中 i = n*P*Q + p*Q + q, j = k
#define GEMM_C_INDEX(n, p, q, k, P, Q) ((n)*(P)*(Q) + (p)*(Q) + (q)), (k)

// 输入和核权重索引映射到 GEMM K 维度
// 其中 gemm_k = c*R*S + r*S + s
#define GEMM_K_INDEX(c, r, s, R, S) ((c)*(R)*(S) + (r)*(S) + (s))

// 从 GEMM_K 反向映射到卷积索引
#define GEMM_K_TO_CRS(gemm_k, R, S, c, r, s) \
    do { \
        c = (gemm_k) / ((R)*(S)); \
        int rs_idx = (gemm_k) % ((R)*(S)); \
        r = rs_idx / (S); \
        s = rs_idx % (S); \
    } while(0)

// 卷积中输入元素坐标计算
#define INPUT_COORD(p, q, r, s, stride_h, stride_w, pad_h, pad_w, h, w) \
    do { \
        h = (p) * (stride_h) - (pad_h) + (r); \
        w = (q) * (stride_w) - (pad_w) + (s); \
    } while(0)

// 使用 __ldg 提高全局内存读取效率
template <typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
    return __ldg(ptr);
}

__global__ void forward_kernel_implicit_gemm(
    float *y, const float *x, const float *w, 
    const int B, const int M, const int C, const int H, const int W, const int K,
    const int P, const int Q, const int stride_h, const int stride_w, 
    const int pad_h, const int pad_w) {

    // GEMM 维度定义
    // GEMM_M = B * P * Q (输出像素数)
    // GEMM_N = M (输出通道数)
    // GEMM_K = C * K * K (输入通道 * 卷积核大小)
    
    // 定义共享内存
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH]; // 输入数据块
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH]; // 权重数据块
    
    // 计算当前线程处理的 GEMM C 矩阵的坐标
    const int tx = threadIdx.x; // 块内 x 坐标
    const int ty = threadIdx.y; // 块内 y 坐标
    
    // 当前线程块处理的 GEMM 输出块的起始位置
    const int block_m = blockIdx.x * TILE_WIDTH; // C 矩阵的行索引起点
    const int block_n = blockIdx.y * TILE_WIDTH; // C 矩阵的列索引起点
    
    // 计算当前线程对应的输出元素 C(i,j) 的全局坐标
    const int i = block_m + ty;
    const int j = block_n + tx;
    
    // 计算 C(i,j) 映射回卷积输出 y(n,p,q,k) 的坐标
    const int out_channel = j; // GEMM_N 维对应输出通道 k (重命名避免与参数K冲突)
    
    // 将 GEMM 行索引 i 映射回 n,p,q
    int n = i / (P * Q);
    int npq_residual = i % (P * Q);
    int p = npq_residual / Q;
    int q = npq_residual % Q;
    
    // 累加结果
    float acc = 0.0f;
    
    // 循环处理 GEMM_K 维度的分块
    const int K_tiles = (C * K * K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int tile_k = 0; tile_k < K_tiles; ++tile_k) {
        // 当前 K 维度的起始索引
        const int k_start = tile_k * TILE_WIDTH;
        
        // 将当前 GEMM 元素 C(i,j) 对应的 A 行和 B 列加载到共享内存
        // 每个线程负责加载一个元素到共享内存
        
        // 加载 A(i,k_idx) 到共享内存
        if (i < B * P * Q && (k_start + tx) < C * K * K) {
            // 将 GEMM_K 维的索引 k_idx 反向映射到 c,r,s
            int k_idx = k_start + tx;
            int c, r, s;
            GEMM_K_TO_CRS(k_idx, K, K, c, r, s);
            
            // 计算对应的输入坐标 h,w
            int h, w;
            INPUT_COORD(p, q, r, s, stride_h, stride_w, pad_h, pad_w, h, w);
            
            // 检查边界并加载数据
            if (n < B && c < C && h >= 0 && h < H && w >= 0 && w < W) {
                sh_A[ty][tx] = ldg(&x[(n * C * H * W) + (c * H * W) + (h * W) + w]);
            } else {
                sh_A[ty][tx] = 0.0f;
            }
        } else {
            sh_A[ty][tx] = 0.0f;
        }
        
        // 加载 B(k_idx,j) 到共享内存
        if ((k_start + ty) < C * K * K && j < M) {
            // 将 GEMM_K 维的索引 k_idx 反向映射到 c,r,s
            int k_idx = k_start + ty;
            int c, r, s;
            GEMM_K_TO_CRS(k_idx, K, K, c, r, s);
            
            // 加载卷积核权重 (修复参数，w是权重指针，不是out_channel，替换k变量)
            sh_B[ty][tx] = ldg(&w[(out_channel * C * K * K) + (c * K * K) + (r * K) + s]);
        } else {
            sh_B[ty][tx] = 0.0f;
        }
        
        // 同步以确保所有数据都加载完成
        __syncthreads();
        
        // 计算当前分块的矩阵乘法
        #pragma unroll
        for (int k_idx = 0; k_idx < TILE_WIDTH; ++k_idx) {
            acc += sh_A[ty][k_idx] * sh_B[k_idx][tx];
        }
        
        // 同步以确保计算完成再加载下一个分块
        __syncthreads();
    }
    
    // 将结果写回全局内存
    if (n < B && p < P && q < Q && out_channel < M) {
        y[(n * M * P * Q) + (out_channel * P * Q) + (p * Q) + q] = acc;
    }
}

torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &w, int64_t M) {
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = w.size(3); // 卷积核大小
    
    // 计算输出尺寸
    const int P = H - K + 1; // 输出高度
    const int Q = W - K + 1; // 输出宽度
    
    // 默认参数 (后续可以扩展函数接口支持自定义参数)
    const int stride_h = 1;
    const int stride_w = 1;
    const int pad_h = 0;
    const int pad_w = 0;
    
    // 创建输出张量
    auto y = torch::empty({B, M, P, Q}, x.options());
    
    // GEMM 维度
    const int GEMM_M = B * P * Q;
    const int GEMM_N = M;
    
    // 计算网格和块维度
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(
        (GEMM_M + TILE_WIDTH - 1) / TILE_WIDTH,
        (GEMM_N + TILE_WIDTH - 1) / TILE_WIDTH
    );
    
    // 启动内核
    forward_kernel_implicit_gemm<<<gridDim, blockDim>>>(
        y.data_ptr<float>(), 
        x.data_ptr<float>(), 
        w.data_ptr<float>(),
        B, M, C, H, W, K,
        P, Q, stride_h, stride_w, pad_h, pad_w
    );
    
    // 同步设备
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    
    return y;
}

}; 

