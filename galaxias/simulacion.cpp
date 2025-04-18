#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <omp.h>               // OpenMP
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
constexpr int NUM_PART = 100000;

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

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error: " << cudaGetErrorString(err) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel: update velocities and positions based on local attraction
__global__ void updateParticles(Particle* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle me = p[i];

    // Attract to neighbors within threshold
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        float dx = p[j].x - me.x;
        float dy = p[j].y - me.y;
        if (fabsf(dx) < NEIGHBOR_TH && fabsf(dy) < NEIGHBOR_TH) {
            float dist2 = dx*dx + dy*dy + 1e-6f;
            float invDist = rsqrtf(dist2);
            float mass_term = p[j].mass + 1.0f;
            me.vx += dx * invDist * invDist * mass_term;
            me.vy += dy * invDist * invDist * mass_term;
        }
    }

    // Friction for runaway speeds
    if (fabsf(me.vx) > 1.0f) me.vx *= 0.5f;
    if (fabsf(me.vy) > 1.0f) me.vy *= 0.5f;

    // Integrate
    me.x += me.vx * DT;
    me.y += me.vy * DT;

    p[i] = me;
}

int main() {
    // --- OpenCV window setup ---
    namedWindow("Simulation", WINDOW_NORMAL);
    resizeWindow("Simulation", WIDTH, HEIGHT);

    // --- Host particle buffer initialization ---
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

    // --- Device buffer allocation & copy ---
    Particle* d_particles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_particles, NUM_PART * sizeof(Particle)));
    CUDA_CHECK(cudaMemcpy(
        d_particles,
        h_particles.data(),
        NUM_PART * sizeof(Particle),
        cudaMemcpyHostToDevice));

    // --- Trails & overlay images ---
    Mat trail(HEIGHT, WIDTH, CV_8UC3, Scalar(0,0,0));
    Mat overlay(HEIGHT, WIDTH, CV_8UC3, Scalar(0,0,0));

    const int threadsPerBlock = 256;
    int currentN = NUM_PART;

    while (true) {
        // 1) fade the trail
        trail *= 0.99;

        // 2) clear overlay
        overlay.setTo(Scalar(0,0,0));

        // 3) GPU integration step
        int blocks = (currentN + threadsPerBlock - 1) / threadsPerBlock;
        updateParticles<<<blocks, threadsPerBlock>>>(d_particles, currentN);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4) bring data back to host
        CUDA_CHECK(cudaMemcpy(
            h_particles.data(),
            d_particles,
            currentN * sizeof(Particle),
            cudaMemcpyDeviceToHost));

        // 5) HOST collision detection & response with OpenMP
        vector<char> alive(currentN, 1);

        #pragma omp parallel
        {
            int n = currentN;
            int tid = omp_get_thread_num();
            vector<int> local_kill;

            #pragma omp for schedule(dynamic, 8)
            for (int i = 0; i < n; ++i) {
                if (!alive[i]) continue;
                auto &pi = h_particles[i];
                float ri = pi.mass / 40.0f;

                for (int j = i + 1; j < n; ++j) {
                    if (!alive[j]) continue;
                    auto &pj = h_particles[j];
                    float dx = pj.x - pi.x;
                    float dy = pj.y - pi.y;
                    float rj = pj.mass / 40.0f;
                    float radSum = ri + rj;
                    if (dx*dx + dy*dy >= radSum*radSum) continue;

                    // compute total KE
                    float vix = pi.vx, viy = pi.vy;
                    float vjx = pj.vx, vjy = pj.vy;
                    float kei = 0.5f * pi.mass * (vix*vix + viy*viy);
                    float kej = 0.5f * pj.mass * (vjx*vjx + vjy*vjy);
                    float keSum = kei + kej;

                    if (keSum > KE_THRESHOLD) {
                        // Bounce impulse
                        float dist = sqrtf(dx*dx + dy*dy) + 1e-6f;
                        float nx = dx/dist, ny = dy/dist;
                        float rvx = vix - vjx, rvy = viy - vjy;
                        float velAlong = rvx*nx + rvy*ny;
                        if (velAlong > 0) continue;
                        float invMi = 1.0f / pi.mass;
                        float invMj = 1.0f / pj.mass;
                        float j_imp = -(1.0f + RESTITUTION) * velAlong
                                      / (invMi + invMj);
                        pi.vx +=  j_imp * nx * invMi;
                        pi.vy +=  j_imp * ny * invMi;
                        pj.vx -=  j_imp * nx * invMj;
                        pj.vy -=  j_imp * ny * invMj;
                    } else {
                        // Fuse j into i
                        float totalM = pi.mass + pj.mass;
                        pi.vx = (vix * pi.mass + vjx * pj.mass) / totalM;
                        pi.vy = (viy * pi.mass + vjy * pj.mass) / totalM;
                        pi.mass = totalM;
                        pi.r = (pi.r + pj.r)/2;
                        pi.g = (pi.g + pj.g)/2;
                        pi.b = (pi.b + pj.b)/2;
                        local_kill.push_back(j);
                    }
                }
            }

            // commit kills
            #pragma omp critical
            {
                for (int idx : local_kill) alive[idx] = 0;
            }
        }

        // 6) compact the arrays
        int writeIdx = 0;
        for (int i = 0; i < currentN; ++i) {
            if (alive[i]) {
                h_particles[writeIdx++] = h_particles[i];
            }
        }
        currentN = writeIdx;

        // 7) draw to trail & overlay
        for (int i = 0; i < currentN; ++i) {
            auto &pt = h_particles[i];
            int ix = int(pt.x), iy = int(pt.y);
            if (ix < 0 || ix >= WIDTH || iy < 0 || iy >= HEIGHT) 
                continue;
            Vec3b col(pt.b, pt.g, pt.r);
            trail.at<Vec3b>(iy, ix) = col;
            circle(overlay,
                   Point(ix, iy),
                   int(pt.mass / 40.0f),
                   Scalar(pt.b, pt.g, pt.r),
                   -1, LINE_AA);
        }

        // 8) upload back active particles
        CUDA_CHECK(cudaMemcpy(
            d_particles,
            h_particles.data(),
            currentN * sizeof(Particle),
            cudaMemcpyHostToDevice));

        // 9) blend & display
        Mat display;
        max(trail, overlay, display);
        imshow("Simulation", display);
        if (waitKey(1) == 27) break;  // ESC to exit
    }

    cudaFree(d_particles);
    return 0;
}

