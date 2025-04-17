#include "pyt_all_reduce_kernel.hh"

namespace eecs471 {

#define TILE_WIDTH 32
#define TILE_WIDTH_16 18
#define MAX_K 7 

__constant__ float const_kernel[24*12*MAX_K*MAX_K];

// 单通道卷积内核 - 每个通道在独立流中运行
__global__ void forward_kernel_single_channel(float *partial_y, const float *x, const float *k, 
                                            const int B, const int M, const int c, 
                                            const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) partial_y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (12 * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (12 * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int X_tile_width = TILE_WIDTH + K - 1;
    __shared__ float shared_kernel[MAX_K][MAX_K];
    __shared__ float shared_input[TILE_WIDTH + MAX_K - 1][TILE_WIDTH + MAX_K - 1];

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

    int m = blockIdx.x;
    int h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.y % W_grid) * TILE_WIDTH;
    int h = h_base + threadIdx.y;
    int w = w_base + threadIdx.x;
    int b = blockIdx.z;

    bool within_bounds = (h < H_out && w < W_out && m < M);
    
    // 只处理单个通道
    for (int i = threadIdx.y; i < X_tile_width; i += blockDim.y) {
        for (int j = threadIdx.x; j < X_tile_width; j += blockDim.x) {
            int row_in = h_base + i;
            int col_in = w_base + j;
            if (row_in < H && col_in < W) { 
                shared_input[i][j] = x4d(b, c, row_in, col_in);
            } else {
                shared_input[i][j] = 0.0f;
            }
        }
    }

    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, c, threadIdx.y, threadIdx.x);
    }

    __syncthreads();

    float acc = 0.0f;
    if (within_bounds) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                acc += shared_input[threadIdx.y + p][threadIdx.x + q] * shared_kernel[p][q];
            }
        }
        // 直接写入部分结果
        y4d(b, m, h, w) = acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

// 归约内核 - 将多个通道的部分结果合并
__global__ void reduction_kernel(float *y, float **partial_results, const int B, const int M, const int H_out, const int W_out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total_elements = B * M * H_out * W_out;

    // 每个线程处理多个元素
    for (int i = idx; i < total_elements; i += stride) {
        float sum = 0.0f;
        // 累加所有通道的部分结果
        #pragma unroll
        for (int c = 0; c < 12; c++) {
            sum += partial_results[c][i];
        }
        y[i] = sum;
    }
}

__global__ void forward_kernel_c1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define ck4d(i3, i2, i1, i0) const_kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int X_tile_width = TILE_WIDTH_16 + K - 1;

    __shared__ float shared_input[TILE_WIDTH_16 + MAX_K - 1][TILE_WIDTH_16 + MAX_K - 1];

    int W_grid = (W_out + TILE_WIDTH_16 - 1) / TILE_WIDTH_16;

    int m = blockIdx.x;
    int h_base = (blockIdx.y / W_grid) * TILE_WIDTH_16;
    int w_base = (blockIdx.y % W_grid) * TILE_WIDTH_16;
    int h = h_base + threadIdx.y;
    int w = w_base + threadIdx.x;
    int b = blockIdx.z;

    float acc = 0.0f;


    const int c = 0;  
    
    // Load input tile into shared memory (as float)
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
    __shared__ float shared_kernel[MAX_K][MAX_K];
    __shared__ float shared_input[TILE_WIDTH + MAX_K - 1][TILE_WIDTH + MAX_K - 1];

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

    int m = blockIdx.x;
    int h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.y % W_grid) * TILE_WIDTH;
    int h = h_base + threadIdx.y;
    int w = w_base + threadIdx.x;
    int b = blockIdx.z;

    float acc = 0.0f;

    for(int c = 0; c < C; c++){
        for (int i = threadIdx.y; i < X_tile_width; i += blockDim.y) {
            for (int j = threadIdx.x; j < X_tile_width; j += blockDim.x) {
                int row_in = h_base + i;
                int col_in = w_base + j;
                if (row_in < H && col_in < W) { 
                    shared_input[i][j] = x4d(b, c, row_in, col_in);
                } else {
                    shared_input[i][j] = 0.0f;
                }
            }
        }

        if (threadIdx.y < K && threadIdx.x < K) {
            shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, c, threadIdx.y, threadIdx.x);
        }

        __syncthreads(); 

        if (h < H_out && w < W_out) {
            
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
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = w.size(3); 

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    auto y = torch::empty({B, M, H_out, W_out}, x.options());

    if(C == 1){
        int W_grid = (W_out + TILE_WIDTH_16 - 1) / TILE_WIDTH_16;
        int H_grid = (H_out + TILE_WIDTH_16 - 1) / TILE_WIDTH_16;
        int Y = W_grid * H_grid; 

        cudaMemcpyToSymbol(const_kernel, w.data_ptr<float>(), M * C * K * K * sizeof(float));

        dim3 gridDim(M, Y, B);
        dim3 blockDim(TILE_WIDTH_16, TILE_WIDTH_16, 1);
        forward_kernel_c1<<<gridDim, blockDim>>>(
            y.data_ptr<float>(),
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            B, M, C, H, W, K);
    } else if (C == 12) {

        int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
        int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
        int Y = W_grid * H_grid;

        dim3 gridDim(M, Y, B);
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        
        // 分配设备内存和主机指针数组
        float** d_partial_results_host = new float*[12];
        float** d_partial_results_dev;
        cudaMalloc(&d_partial_results_dev, 12 * sizeof(float*));
        
        for (int c = 0; c < 12; c++) {
            cudaMalloc(&d_partial_results_host[c], B * M * H_out * W_out * sizeof(float));
            // 初始化为0
            cudaMemset(d_partial_results_host[c], 0, B * M * H_out * W_out * sizeof(float));
        }
        
        // 将主机指针数组复制到设备
        cudaMemcpy(d_partial_results_dev, d_partial_results_host, 12 * sizeof(float*), cudaMemcpyHostToDevice);
        
        // 创建流
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 设置每个通道的计算
        for (int c = 0; c < 12; c++) {
            forward_kernel_single_channel<<<gridDim, blockDim, 0, stream>>>(
                d_partial_results_host[c], 
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                B, M, c, H, W, K);
        }
        
        cudaDeviceSynchronize(); // 确保所有通道内核完成
        
        // 设置归约操作，在主流中执行
        int total_elements = B * M * H_out * W_out;
        int threads_per_block = 256;
        int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
     // 避免创建过多块
        
        // reduction_kernel<<<blocks, threads_per_block, 0, stream>>>(
        //     y.data_ptr<float>(), 
        //     d_partial_results_dev, 
        //     B, M, H_out, W_out);
        
        // 等待归约完成
        // cudaDeviceSynchronize();
        
        // 清理资源
        cudaStreamDestroy(stream);
        
        for (int c = 0; c < 12; c++) {
            cudaFree(d_partial_results_host[c]);
        }
        cudaFree(d_partial_results_dev);
        delete[] d_partial_results_host;
        
    } else {
        int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
        int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
        int Y = W_grid * H_grid;

        dim3 gridDim(M, Y, B);
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        forward_kernel<<<gridDim, blockDim>>>(
            y.data_ptr<float>(),
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            B, M, C, H, W, K);
    }

    // C10_CUDA_CHECK(cudaDeviceSynchronize());

    return y;
}
}; 

