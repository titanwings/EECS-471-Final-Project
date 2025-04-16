#include "pyt_all_reduce_kernel.hh"


namespace eecs471 {

#define TILE_WIDTH 32
#define TILE_WIDTH_16 18
#define MAX_K 7 

// 定义常量内存为一维数组，使用与PyTorch相同的内存布局
__constant__ float const_kernel[24*12*MAX_K*MAX_K];

__global__ void forward_kernel_c1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
// 添加常量内存访问宏，与k4d保持一致的索引方式
#define ck4d(i3, i2, i1, i0) const_kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int X_tile_width = TILE_WIDTH_16 + K - 1;
    // 只需要共享内存用于输入数据
    __shared__ float shared_input[TILE_WIDTH_16 + MAX_K - 1][TILE_WIDTH_16 + MAX_K - 1];

    int W_grid = (W_out + TILE_WIDTH_16 - 1) / TILE_WIDTH_16;

    int m = blockIdx.x;
    int h_base = (blockIdx.y / W_grid) * TILE_WIDTH_16;
    int w_base = (blockIdx.y % W_grid) * TILE_WIDTH_16;
    int h = h_base + threadIdx.y;
    int w = w_base + threadIdx.x;
    int b = blockIdx.z;

    // 累加器使用float
    float acc = 0.0f;

    // C=1情况，直接使用通道索引0，无需循环
    const int c = 0;  // 固定通道索引为0
    
    // Load input tile into shared memory (as float)
    for (int i = threadIdx.y; i < X_tile_width; i += blockDim.y) {
        for (int j = threadIdx.x; j < X_tile_width; j += blockDim.x) {
            int row_in = h_base + i;
            int col_in = w_base + j;
            if (row_in < H && col_in < W) { // Add boundary checks
                shared_input[i][j] = __ldg(&x4d(b, c, row_in, col_in));
            } else {
                // Pad with zero
                shared_input[i][j] = 0.0f;
            }
        }
    }

    __syncthreads(); 

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (h < H_out && w < W_out) {
        // 使用一维常量内存，索引方式与原始k4d保持一致
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * ck4d(m, c, 0, 0);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * ck4d(m, c, 0, 1);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * ck4d(m, c, 0, 2);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * ck4d(m, c, 0, 3);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * ck4d(m, c, 0, 4);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * ck4d(m, c, 0, 5);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * ck4d(m, c, 0, 6);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * ck4d(m, c, 1, 0);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * ck4d(m, c, 1, 1);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * ck4d(m, c, 1, 2);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * ck4d(m, c, 1, 3);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * ck4d(m, c, 1, 4);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * ck4d(m, c, 1, 5);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * ck4d(m, c, 1, 6);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * ck4d(m, c, 2, 0);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * ck4d(m, c, 2, 1);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * ck4d(m, c, 2, 2);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * ck4d(m, c, 2, 3);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * ck4d(m, c, 2, 4);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * ck4d(m, c, 2, 5);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * ck4d(m, c, 2, 6);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * ck4d(m, c, 3, 0);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * ck4d(m, c, 3, 1);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * ck4d(m, c, 3, 2);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * ck4d(m, c, 3, 3);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * ck4d(m, c, 3, 4);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * ck4d(m, c, 3, 5);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * ck4d(m, c, 3, 6);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * ck4d(m, c, 4, 0);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * ck4d(m, c, 4, 1);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * ck4d(m, c, 4, 2);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * ck4d(m, c, 4, 3);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * ck4d(m, c, 4, 4);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * ck4d(m, c, 4, 5);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * ck4d(m, c, 4, 6);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * ck4d(m, c, 5, 0);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * ck4d(m, c, 5, 1);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * ck4d(m, c, 5, 2);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * ck4d(m, c, 5, 3);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * ck4d(m, c, 5, 4);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * ck4d(m, c, 5, 5);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * ck4d(m, c, 5, 6);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * ck4d(m, c, 6, 0);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * ck4d(m, c, 6, 1);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * ck4d(m, c, 6, 2);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * ck4d(m, c, 6, 3);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * ck4d(m, c, 6, 4);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * ck4d(m, c, 6, 5);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * ck4d(m, c, 6, 6);
    }

    // Write final accumulated result (float)
    if(h < H_out && w < W_out){
        y4d(b, m, h, w) = acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
    #undef ck4d
}

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int X_tile_width = TILE_WIDTH + K - 1;
    // 恢复使用共享内存存储卷积核和输入数据
    __shared__ float shared_kernel[MAX_K][MAX_K];
    __shared__ float shared_input[TILE_WIDTH + MAX_K - 1][TILE_WIDTH + MAX_K - 1];

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

    int m = blockIdx.x;
    int h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.y % W_grid) * TILE_WIDTH;
    int h = h_base + threadIdx.y;
    int w = w_base + threadIdx.x;
    int b = blockIdx.z;

    // 累加器使用float
    float acc = 0.0f;

    // Removed #pragma unroll from c loop
    for(int c = 0; c < C; c++){
        // Load input tile into shared memory (as float)
        // Removed #pragma unroll from loading loops
        for (int i = threadIdx.y; i < X_tile_width; i += blockDim.y) {
            for (int j = threadIdx.x; j < X_tile_width; j += blockDim.x) {
                int row_in = h_base + i;
                int col_in = w_base + j;
                if (row_in < H && col_in < W) { // Add boundary checks
                    shared_input[i][j] = x4d(b, c, row_in, col_in);
                } else {
                    // Pad with zero
                    shared_input[i][j] = 0.0f;
                }
            }
        }

        // 恢复加载卷积核到共享内存的代码
        if (threadIdx.y < K && threadIdx.x < K) {
            shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, c, threadIdx.y, threadIdx.x);
        }

        __syncthreads(); 

        // Perform computation within the tile
        // Check if the current thread's output pixel (h, w) is within bounds
        if (h < H_out && w < W_out) {
            // 改回使用共享内存中的卷积核
            acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
            acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
            acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
            acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
            acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
            acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
            acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
            acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
            acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
            acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
            acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
            acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
            acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
            acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
            acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
            acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
            acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
            acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
            acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
            acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
            acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
            acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
            acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
            acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
            acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
            acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
            acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
            acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
            acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
            acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
            acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
            acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
            acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
            acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
            acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
            acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
            acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
            acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
            acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
            acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
            acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
            acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
            acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
            acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
            acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
            acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
            acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
            acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
            acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];
        }
        __syncthreads(); 
    }

    // Write final accumulated result (float)
    if(h < H_out && w < W_out){
        y4d(b, m, h, w) = acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &w, int64_t M) {
    // 使用原始FP32计算，不进行精度转换

    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = w.size(3); 

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    auto y = torch::empty({B, M, H_out, W_out}, x.options()); // options should be kFloat from input x

    // 将卷积核复制到常量内存
    // 直接使用data_ptr获取指向线性内存的指针，与一维常量数组内存布局匹配

    if(C == 1){
        // 使用16x16的块大小
        int W_grid = (W_out + TILE_WIDTH_16 - 1) / TILE_WIDTH_16;
        int H_grid = (H_out + TILE_WIDTH_16 - 1) / TILE_WIDTH_16;
        int Y = W_grid * H_grid; // Total tiles in H-W plane

        cudaMemcpyToSymbol(const_kernel, w.data_ptr<float>(), M * C * K * K * sizeof(float));

        dim3 gridDim(M, Y, B);
        dim3 blockDim(TILE_WIDTH_16, TILE_WIDTH_16, 1);
        forward_kernel_c1<<<gridDim, blockDim>>>(
            y.data_ptr<float>(),
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            B, M, C, H, W, K);
    } else {
        // 使用32x32的块大小
        int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
        int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
        int Y = W_grid * H_grid; // Total tiles in H-W plane

        dim3 gridDim(M, Y, B);
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        forward_kernel<<<gridDim, blockDim>>>(
            y.data_ptr<float>(),
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            B, M, C, H, W, K);
    }

    // Synchronize device after kernel launch
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return y; // Return float tensor
}
}; 

