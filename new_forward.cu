#include "pyt_all_reduce_kernel.hh"

namespace eecs471 {

/*Self-defined Parameters*/
#define TILE_WIDTH 16
// #define MAXKernelLength 24*12*7*7
#define MAX_K 7
// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {

//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.
//     We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     */

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

// // An example use of these macros:
// // float a = y4d(0,0,0,0)
// // y4d(0,0,0,0) = a
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     int b = blockDim.x * blockIdx.x + threadIdx.x;

//     if (b < B) // for each image in the batch
//     {
//         /*Original Version*/

//         for (int m = 0; m < M; m++)         // for each output feature maps
//             for (int h = 0; h < H_out; h++) // for each output element
//                 for (int w = 0; w < W_out; w++)
//                 {
//                     y4d(b, m, h, w) = 0;
//                     for (int c = 0; c < C; c++)     // sum over all input feature maps
//                         for (int p = 0; p < K; p++) // KxK filter
//                             for (int q = 0; q < K; q++)
//                                 y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
//                 }

//     }

// #undef y4d
// #undef x4d
// #undef k4d
// }



/*Working Version*/
// __constant__ float k[MAXKernelLength]; 


__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int X_tile_width = TILE_WIDTH + K - 1;
    // 声明共享内存，为每个通道分配空间
    __shared__ float shared_kernel[MAX_K][MAX_K]; 
    __shared__ float shared_input[TILE_WIDTH + MAX_K - 1][TILE_WIDTH + MAX_K - 1 +1];

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

    // int m = blockIdx.x / C;
    // int c = blockIdx.x % C;

    int m = blockIdx.x;
    int h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.y % W_grid) * TILE_WIDTH;
    int h = h_base + threadIdx.y;
    int w = w_base + threadIdx.x;
    int b = blockIdx.z;
    float acc = 0.0f;
    for(int c = 0; c < C; c++){
    // 协作加载输入数据到共享内存
    for (int i = threadIdx.y; i < X_tile_width; i += blockDim.y) {
        for (int j = threadIdx.x; j < X_tile_width; j += blockDim.x) {
            int row_in = h_base + i;
            int col_in = w_base + j;
            if (row_in < H && col_in < W) {
                shared_input[i][j] = x4d(b, c, row_in, col_in);
            }
        }
    }

        if (threadIdx.x < K && threadIdx.y < K) {
                shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, c, threadIdx.y, threadIdx.x);
        }
        __syncthreads();


        for (int p = 0; p < K; p++){
            for (int q = 0; q < K; q++){
                acc += shared_input[threadIdx.y + p][threadIdx.x + q] * shared_kernel[p][q];
            }
        }
        __syncthreads();
    }
    if(m < M && h < H_out && w < W_out){
        y4d(b, m, h, w) = acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &w, int64_t M) {
    /*Original Version*/
    // const int B = x.size(0);
    // const int C = x.size(1);
    // const int H = x.size(2);
    // const int W = x.size(3);
    // const int K = w.size(3);
    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;
    // auto y = torch::empty({B, M, H_out, W_out}, x.options());

    // dim3 gridDim((B + 511) / 512);
    // dim3 blockDim(512);
    

    // // C10_CUDA_CHECK(cudaDeviceSynchronize());
    // forward_kernel<<<gridDim, blockDim>>>(y.data_ptr<float>(), x.data_ptr<float>(),
    //                                       w.data_ptr<float>(), B, M, C, H, W, K);
    // // C10_CUDA_CHECK(cudaDeviceSynchronize());

    // return y;

    /*Current Version*/
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = w.size(3);
    
    // 检查维度是否在限制范围内
    // if (M > MAX_M || C > MAX_C || K > MAX_K) {
    //     throw std::runtime_error("Input dimensions exceed constant memory limits");
    // }
    // 将卷积核复制到常量内存
    // cudaMemcpyToSymbol(k, w.data_ptr<float>(), M * C * K * K * sizeof(float));
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    // 创建输出张量并初始化为0
    auto y = torch::empty({B, M, H_out, W_out}, x.options());

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Y = W_grid * H_grid;
    // int M_C = M * C;
    dim3 gridDim(M, Y, B);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    forward_kernel<<<gridDim, blockDim>>>(y.data_ptr<float>(), x.data_ptr<float>(), w.data_ptr<float>(), B, M, C, H, W, K);
    cudaDeviceSynchronize();

    return y;
}
}; // namespace eecs471

