#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <omp.h>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

// Window dimensions
constexpr int WIDTH    = 1920;
constexpr int HEIGHT   = 1080;

// Number of particles
constexpr int NUM_PART = 1000;

// Simulation parameters
constexpr float DT          = 1.0f / 1250.0f;
constexpr float NEIGHBOR_TH = 200.0f;

// Collision parameters
constexpr float KE_THRESHOLD = 10000.0f;  // kinetic‐energy threshold
constexpr float RESTITUTION  = 0.8f;      // bounce energy retention [0..1]

struct Particle {
    float x, y;
    float vx, vy;
    float mass;
    unsigned char r, g, b;
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error: " << cudaGetErrorString(err) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// GPU kernel: integrate positions & velocities by local attraction
__global__ void updateParticles(Particle* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle me = p[i];
    // attract to neighbors
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        float dx = p[j].x - me.x;
        float dy = p[j].y - me.y;
        if (fabsf(dx) < NEIGHBOR_TH && fabsf(dy) < NEIGHBOR_TH) {
            float dist2 = dx*dx + dy*dy + 1e-6f;
            float invDist = 1.0f / sqrtf(dist2);
            float mass_term = p[j].mass + 1.0f;
            me.vx += dx * invDist * invDist * mass_term;
            me.vy += dy * invDist * invDist * mass_term;
        }
    }
    // friction
    if (fabsf(me.vx) > 1.0f) me.vx *= 0.5f;
    if (fabsf(me.vy) > 1.0f) me.vy *= 0.5f;
    // integrate
    me.x += me.vx * DT;
    me.y += me.vy * DT;
    p[i] = me;
}

int main() {
    // OpenCV
    namedWindow("Simulation", WINDOW_NORMAL);
    resizeWindow("Simulation", WIDTH, HEIGHT);

    // Host buffer
    vector<Particle> h_particles(NUM_PART);
    mt19937 rng((unsigned)time(nullptr));
    uniform_real_distribution<float> distX(0.0f, WIDTH);
    uniform_real_distribution<float> distY(0.0f, HEIGHT);
    uniform_real_distribution<float> distV(-0.5f, 0.5f);
    uniform_real_distribution<float> distM(5.0f, 105.0f);
    uniform_int_distribution<int>   distC(0, 255);

    for (int i = 0; i < NUM_PART; ++i) {
        float x = distX(rng), y = distY(rng);
        float ang = atan2f(HEIGHT/2.0f - y, WIDTH/2.0f - x)
                  + (distX(rng)/WIDTH) * 2.0f * CV_PI;
        h_particles[i].x    = x;
        h_particles[i].y    = y;
        h_particles[i].vx   = cosf(ang + CV_PI/2) * 1000.0f;
        h_particles[i].vy   = sinf(ang + CV_PI/2) * 1000.0f;
        h_particles[i].mass = distM(rng);
        h_particles[i].r    = distC(rng);
        h_particles[i].g    = distC(rng);
        h_particles[i].b    = distC(rng);
    }

    // Device buffer
    Particle* d_particles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_particles, NUM_PART * sizeof(Particle)));
    CUDA_CHECK(cudaMemcpy(
        d_particles,
        h_particles.data(),
        NUM_PART * sizeof(Particle),
        cudaMemcpyHostToDevice));

    // Trail & overlay
    Mat trail(HEIGHT, WIDTH, CV_8UC3, Scalar(0,0,0));
    Mat overlay(HEIGHT, WIDTH, CV_8UC3, Scalar(0,0,0));

    const int threadsPerBlock = 256;
    int currentN = NUM_PART;

    while (true) {
        // fade trail
        trail *= 0.99;
        overlay.setTo(Scalar(0,0,0));

        // GPU integrate
        int blocks = (currentN + threadsPerBlock - 1) / threadsPerBlock;
        updateParticles<<<blocks, threadsPerBlock>>>(d_particles, currentN);
        CUDA_CHECK(cudaDeviceSynchronize());

        // copy back
        CUDA_CHECK(cudaMemcpy(
            h_particles.data(),
            d_particles,
            currentN * sizeof(Particle),
            cudaMemcpyDeviceToHost));

        // collision with OpenMP
        vector<char> alive(currentN, 1);
        #pragma omp parallel
        {
            vector<int> local_kill;
            #pragma omp for schedule(dynamic,8)
            for (int i = 0; i < currentN; ++i) {
                if (!alive[i]) continue;
                auto &pi = h_particles[i];
                float ri = pi.mass / 40.0f;
                for (int j = i+1; j < currentN; ++j) {
                    if (!alive[j]) continue;
                    auto &pj = h_particles[j];
                    float dx = pj.x - pi.x, dy = pj.y - pi.y;
                    float rj = pj.mass / 40.0f, sumR = ri + rj;
                    if (dx*dx + dy*dy >= sumR*sumR) continue;
                    // KE test
                    float kei = 0.5f*pi.mass*(pi.vx*pi.vx + pi.vy*pi.vy);
                    float kej = 0.5f*pj.mass*(pj.vx*pj.vx + pj.vy*pj.vy);
                    if (kei + kej > KE_THRESHOLD) {
                        // bounce
                        float dist = sqrtf(dx*dx + dy*dy) + 1e-6f;
                        float nx = dx/dist, ny = dy/dist;
                        float rvx = pi.vx - pj.vx, rvy = pi.vy - pj.vy;
                        float vAlong = rvx*nx + rvy*ny;
                        if (vAlong > 0) continue;
                        float invMi = 1.0f/pi.mass, invMj = 1.0f/pj.mass;
                        float jimp = -(1+RESTITUTION)*vAlong/(invMi+invMj);
                        pi.vx +=  jimp*nx*invMi;  pi.vy +=  jimp*ny*invMi;
                        pj.vx -=  jimp*nx*invMj;  pj.vy -=  jimp*ny*invMj;
                    } else {
                        // fuse
                        float totalM = pi.mass + pj.mass;
                        pi.vx = (pi.vx*pi.mass + pj.vx*pj.mass)/totalM;
                        pi.vy = (pi.vy*pi.mass + pj.vy*pj.mass)/totalM;
                        pi.mass = totalM;
                        pi.r = (pi.r + pj.r)/2;
                        pi.g = (pi.g + pj.g)/2;
                        pi.b = (pi.b + pj.b)/2;
                        local_kill.push_back(j);
                    }
                }
            }
            #pragma omp critical
            for (int j : local_kill) alive[j] = 0;
        }

        // compact
        int writeIdx = 0;
        for (int i = 0; i < currentN; ++i)
            if (alive[i]) h_particles[writeIdx++] = h_particles[i];
        currentN = writeIdx;

        // draw
        for (int i = 0; i < currentN; ++i) {
            auto &pt = h_particles[i];
            int ix = int(pt.x), iy = int(pt.y);
            if (ix<0||ix>=WIDTH||iy<0||iy>=HEIGHT) continue;
            Vec3b col(pt.b,pt.g,pt.r);
            trail.at<Vec3b>(iy,ix) = col;
            circle(overlay, Point(ix,iy),
                   int(pt.mass/40.0f),
                   Scalar(pt.b,pt.g,pt.r),
                   -1, LINE_AA);
        }

        // upload active
        CUDA_CHECK(cudaMemcpy(
            d_particles,
            h_particles.data(),
            currentN * sizeof(Particle),
            cudaMemcpyHostToDevice));

        // show
        Mat disp;
        max(trail, overlay, disp);
        imshow("Simulation", disp);
        if (waitKey(1) == 27) break;
    }

    cudaFree(d_particles);
    return 0;
}

