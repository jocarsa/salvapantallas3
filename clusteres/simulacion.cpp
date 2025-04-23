#include <opencv2/opencv.hpp>
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
constexpr int NUM_PART = 10000;

// Simulation parameters
constexpr float DT          = 1.0f / 1250.0f;
constexpr float NEIGHBOR_TH = 200.0f;

// Collision parameters
constexpr float RESTITUTION  = 0.8f;      // bounce energy retention [0..1]

// Perlin noise parameters
constexpr float PERLIN_SCALE = 50.0f;
constexpr int PERLIN_OCTAVES = 4;

struct Particle {
    float x, y;
    float vx, vy;
    float mass;
    unsigned char r, g, b;
};

// Function to generate 3D Perlin noise
float perlinNoise3D(float x, float y, float z, int octaves, float scale) {
    // Implementation of 3D Perlin noise
    // This is a placeholder for the actual Perlin noise function
    // You can use a library like libnoise or implement your own Perlin noise function
    return 0.0f;
}

// Function to update velocities and positions based on local attraction
void updateParticles(vector<Particle>& particles, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        Particle& me = particles[i];

        // Attract to neighbors within threshold
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            float dx = particles[j].x - me.x;
            float dy = particles[j].y - me.y;
            if (fabsf(dx) < NEIGHBOR_TH && fabsf(dy) < NEIGHBOR_TH) {
                float dist2 = dx*dx + dy*dy + 1e-6f;
                float invDist = 1.0f / sqrtf(dist2);
                float mass_term = particles[j].mass + 1.0f;
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
    }
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
        float z = distX(rng); // Use z for 3D Perlin noise
        float noise = perlinNoise3D(x / PERLIN_SCALE, y / PERLIN_SCALE, z / PERLIN_SCALE, PERLIN_OCTAVES, PERLIN_SCALE);
        x += noise * PERLIN_SCALE;
        y += noise * PERLIN_SCALE;
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

    // --- Trails & overlay images ---
    Mat trail(HEIGHT, WIDTH, CV_8UC3, Scalar(0,0,0));
    Mat overlay(HEIGHT, WIDTH, CV_8UC3, Scalar(0,0,0));

    int currentN = NUM_PART;

    while (true) {
        // 1) fade the trail
        trail *= 0.99;

        // 2) clear overlay
        overlay.setTo(Scalar(0,0,0));

        // 3) CPU integration step with OpenMP
        updateParticles(h_particles, currentN);

        // 4) HOST collision detection & response with OpenMP
        vector<char> alive(currentN, 1);

        #pragma omp parallel
        {
            int n = currentN;
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

                    // Bounce impulse
                    float dist = sqrtf(dx*dx + dy*dy) + 1e-6f;
                    float nx = dx/dist, ny = dy/dist;
                    float rvx = pi.vx - pj.vx, rvy = pi.vy - pj.vy;
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
                }
            }

            // commit kills
            #pragma omp critical
            {
                for (int idx : local_kill) alive[idx] = 0;
            }
        }

        // 5) compact the arrays
        int writeIdx = 0;
        for (int i = 0; i < currentN; ++i) {
            if (alive[i]) {
                h_particles[writeIdx++] = h_particles[i];
            }
        }
        currentN = writeIdx;

        // 6) draw to trail & overlay
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

        // 7) blend & display
        Mat display;
        max(trail, overlay, display);
        imshow("Simulation", display);
        if (waitKey(1) == 27) break;  // ESC to exit
    }

    return 0;
}

