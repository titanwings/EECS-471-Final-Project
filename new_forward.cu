#include "pyt_all_reduce_kernel.hh"

namespace eecs471 {
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // 定义数组访问宏
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    // 定义共享内存
    __shared__ float tile[32 * 32];  // 32x32的共享内存块
    __shared__ float kernel_tile[7 * 7];  // 最大7x7的卷积核
    
    // 线程索引
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // 块索引
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    // 计算输出位置
    const int row = by * 32 + ty;
    const int col = bx * 32 + tx;
    const int b = bz;
    
    // 输出值
    float acc = 0.0f;
    
    // 确保在边界内
    if (row < H_out && col < W_out && b < B) {
        // 对每个输出通道
        for (int m = 0; m < M; m++) {
            acc = 0.0f;
            
            // 对每个输入通道
            for (int c = 0; c < C; c++) {
                // 加载输入数据到共享内存
                if (ty < K && tx < K) {
                    const int x_row = row + ty;
                    const int x_col = col + tx;
                    if (x_row < H && x_col < W) {
                        tile[ty * 32 + tx] = x4d(b, c, x_row, x_col);
                    }
                }
                __syncthreads();
                
                // 加载卷积核到共享内存
                if (ty < K && tx < K) {
                    kernel_tile[ty * 7 + tx] = k4d(m, c, ty, tx);
                }
                __syncthreads();
                
                // 计算卷积 - 使用循环展开
                #pragma unroll
                for (int p = 0; p < K; p++) {
                    #pragma unroll
                    for (int q = 0; q < K; q++) {
                        const int tile_row = ty + p;
                        const int tile_col = tx + q;
                        if (tile_row < 32 && tile_col < 32) {
                            acc += tile[tile_row * 32 + tile_col] * kernel_tile[p * 7 + q];
                        }
                    }
                }
                __syncthreads();
            }
            
            // 写入结果
            y4d(b, m, row, col) = acc;
        }
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &w, int64_t M) {
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = w.size(3);
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    auto y = torch::empty({B, M, H_out, W_out}, x.options());

    // 计算网格大小
    dim3 gridDim((W_out + 31) / 32,  // 32x32的线程块
                 (H_out + 31) / 32,
                 B);
    dim3 blockDim(32, 32, 1);  // 32x32的线程块
    
    forward_kernel<<<gridDim, blockDim>>>(
        y.data_ptr<float>(), x.data_ptr<float>(),
        w.data_ptr<float>(), B, M, C, H, W, K);

    return y;
}
}; // namespace eecs471
