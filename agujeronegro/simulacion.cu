// black_hole_simulation.cu
// C++ + OpenCV + custom CUDA kernels for high-performance black hole simulation

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <omp.h>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

// Simulation parameters
constexpr int WIDTH    = 1920;
constexpr int HEIGHT   = 1080;
constexpr int FPS      = 60;
constexpr int DURATION = 3600;  // 1 hour
constexpr int TOTAL_FRAMES = FPS * DURATION;

// Lens parameters
constexpr float THETA_E  = 150.0f;
constexpr float R_SHADOW = 100.0f;

// Disk parameters
constexpr int   N_DISK     = 5000;
constexpr float R_MIN_DISK = 200.0f;
constexpr float R_MAX_DISK = 400.0f;
constexpr float OMEGA0     = 10.88f;

// Motion blur weight
constexpr float MOTION_ALPHA = 0.3f;

// CUDA error check
#define CUDA_CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(e) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// Remap kernel: nearest neighbor
__global__ void remapKernel(const uchar3* in, uchar3* out, const float* map_x, const float* map_y) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x>=WIDTH||y>=HEIGHT) return;
    int idx = y*WIDTH + x;
    float mx = map_x[idx], my = map_y[idx];
    int sx = int(mx+0.5f), sy = int(my+0.5f);
    if(sx>=0 && sx<WIDTH && sy>=0 && sy<HEIGHT) out[idx] = in[sy*WIDTH+sx];
    else out[idx] = make_uchar3(0,0,0);
}

// Weighted add: out = a*inA + b*inB
__global__ void weightedAddKernel(const uchar3* A, const uchar3* B, uchar3* out, float a, float b) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= WIDTH*HEIGHT) return;
    uchar3 p1 = A[idx], p2 = B[idx];
    out[idx].x = uchar(min(255.0f, a*p1.x + b*p2.x));
    out[idx].y = uchar(min(255.0f, a*p1.y + b*p2.y));
    out[idx].z = uchar(min(255.0f, a*p1.z + b*p2.z));
}

// Simple 3x3 box blur
__global__ void boxBlur3(const uchar3* in, uchar3* out) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x>=WIDTH||y>=HEIGHT) return;
    int sumx=0, sumy=0, sumz=0, count=0;
    for(int dy=-1; dy<=1; ++dy) for(int dx=-1; dx<=1; ++dx) {
        int sx=x+dx, sy=y+dy;
        if(sx>=0&&sx<WIDTH&&sy>=0&&sy<HEIGHT) {
            uchar3 v = in[sy*WIDTH+sx];
            sumx += v.x; sumy += v.y; sumz += v.z; count++;
        }
    }
    int idx = y*WIDTH + x;
    out[idx].x = sumx/count;
    out[idx].y = sumy/count;
    out[idx].z = sumz/count;
}

int main(){
    // Prepare CPU background and lens maps
    Mat bg(HEIGHT, WIDTH, CV_8UC3, Scalar(0));
    mt19937 rng(12345);
    uniform_int_distribution<int> dx(0,WIDTH-1), dy(0,HEIGHT-1), db(0,255);
    for(int i=0;i<10000;++i){int x=dx(rng),y=dy(rng),b=db(rng);circle(bg,Point(x,y),1,Scalar(b,b,b),-1);}    

    Mat map_x_mat(HEIGHT,WIDTH,CV_32FC1), map_y_mat(HEIGHT,WIDTH,CV_32FC1);
    // ... fill map_x_mat,map_y_mat with gravitational lens mapping as earlier ...

    // Allocate GPU buffers
    size_t imgSize = WIDTH*HEIGHT*sizeof(uchar3);
    uchar3 *d_bg, *d_frame, *d_tmp, *d_warped;
    float *d_mx, *d_my;
    CUDA_CHECK(cudaMalloc(&d_bg, imgSize));
    CUDA_CHECK(cudaMalloc(&d_frame, imgSize));
    CUDA_CHECK(cudaMalloc(&d_tmp, imgSize));
    CUDA_CHECK(cudaMalloc(&d_warped, imgSize));
    CUDA_CHECK(cudaMalloc(&d_mx, WIDTH*HEIGHT*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_my, WIDTH*HEIGHT*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_bg, bg.ptr<uchar3>(), imgSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mx, map_x_mat.ptr<float>(), WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_my, map_y_mat.ptr<float>(), WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));

    // Setup VideoWriter and display window
    VideoWriter writer("black_hole_simulation.mp4", VideoWriter::fourcc('m','p','4','v'), FPS, Size(WIDTH,HEIGHT));
    namedWindow("Framebuffer", WINDOW_NORMAL);
    resizeWindow("Framebuffer", WIDTH/2, HEIGHT/2);

    dim3 block2D(16,16), grid2D((WIDTH+15)/16,(HEIGHT+15)/16);
    int total = WIDTH*HEIGHT;
    dim3 block1D(256), grid1D((total+255)/256);

    Mat frameMat(HEIGHT, WIDTH, CV_8UC3);

    for(int f=0; f<TOTAL_FRAMES; ++f){
        // 1) Warp background
        remapKernel<<<grid2D,block2D>>>(d_bg, d_tmp, d_mx, d_my);
        CUDA_CHECK(cudaDeviceSynchronize());
        // 2) Glow (box blur x2)
        boxBlur3<<<grid2D,block2D>>>(d_tmp, d_warped);
        boxBlur3<<<grid2D,block2D>>>(d_warped, d_tmp);
        // 3) Motion blur accumulation
        weightedAddKernel<<<grid1D,block1D>>>(d_tmp, d_frame, d_frame, MOTION_ALPHA, 1.0f-MOTION_ALPHA);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download, write and show
        CUDA_CHECK(cudaMemcpy(frameMat.ptr<uchar3>(), d_frame, imgSize, cudaMemcpyDeviceToHost));
        writer.write(frameMat);
        imshow("Framebuffer", frameMat);
        if(waitKey(1)==27) break;  // ESC to quit early
    }

    writer.release();
    destroyAllWindows();
    // Cleanup
    cudaFree(d_bg); cudaFree(d_frame); cudaFree(d_tmp); cudaFree(d_warped);
    cudaFree(d_mx); cudaFree(d_my);
    return 0;
}

