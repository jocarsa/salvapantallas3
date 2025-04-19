#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;
using Clock = chrono::high_resolution_clock;

struct Body {
    double x, y;
    double vx, vy;
    double mass;
    uchar r, g, b;
};

int main() {
    // Simulation parameters
    const int width = 1920;
    const int height = 1080;
    const int numParticles = 200;
    const int fps = 60;
    const int durationMin = 60;
    const int totalFrames = durationMin * 60 * fps;
    const double dt = 0.05;
    const double G = 0.2;

    // Random engine
    mt19937 rng((unsigned)time(nullptr));
    uniform_real_distribution<double> distX(0, width);
    uniform_real_distribution<double> distY(0, height);
    uniform_real_distribution<double> distM(1e3, 1e6);
    uniform_real_distribution<double> distV(-1.0, 1.0);
    uniform_int_distribution<int> distColor(150, 255);

    // Initialize particles
    vector<Body> particles(numParticles);
    for (auto &p : particles) {
        p.x = distX(rng);
        p.y = distY(rng);
        p.vx = distV(rng) * 0.1;
        p.vy = distV(rng) * 0.1;
        p.mass = distM(rng);
        p.r = (uchar)distColor(rng);
        p.g = (uchar)distColor(rng);
        p.b = (uchar)distColor(rng);
    }

    VideoWriter writer("galaxies.mp4", VideoWriter::fourcc('a','v','c','1'), fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Error: Cannot open video writer." << endl;
        return -1;
    }

    namedWindow("Framebuffer", WINDOW_NORMAL);
    resizeWindow("Framebuffer", width/2, height/2);

    auto startTime = Clock::now();
    Mat trail = Mat::zeros(height, width, CV_8UC3);

    for (int frameIdx = 1; frameIdx <= totalFrames; ++frameIdx) {
        // Snapshot previous positions
        auto prev = particles;

        // Update particles in parallel
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < numParticles; ++i) {
            double ax = 0, ay = 0;
            for (int j = 0; j < numParticles; ++j) {
                if (i == j) continue;
                double dx = prev[j].x - prev[i].x;
                double dy = prev[j].y - prev[i].y;
                double dist2 = dx*dx + dy*dy + 1e3;
                double invDist = 1.0 / sqrt(dist2);
                double force = G * prev[j].mass / dist2;
                ax += dx * invDist * force;
                ay += dy * invDist * force;
            }
            particles[i].vx += ax * dt;
            particles[i].vy += ay * dt;
            particles[i].x += particles[i].vx * dt;
            particles[i].y += particles[i].vy * dt;
            // Wrap around edges
            if (particles[i].x < 0) particles[i].x += width;
            if (particles[i].x >= width) particles[i].x -= width;
            if (particles[i].y < 0) particles[i].y += height;
            if (particles[i].y >= height) particles[i].y -= height;
        }

        // Fade trails
        trail *= 0.97;

        // Draw lines between prev and current positions, avoid wrap-around artifacts
        for (int i = 0; i < numParticles; ++i) {
            double dx = particles[i].x - prev[i].x;
            double dy = particles[i].y - prev[i].y;
            // if crossing border, skip line
            if (fabs(dx) > width / 2.0 || fabs(dy) > height / 2.0) continue;
            Point p0((int)prev[i].x, (int)prev[i].y);
            Point p1((int)particles[i].x, (int)particles[i].y);
            Scalar color(particles[i].b, particles[i].g, particles[i].r);
            line(trail, p0, p1, color, 1);
        }

        // Write and display
        writer.write(trail);
        Mat disp;
        resize(trail, disp, Size(width/2, height/2));
        imshow("Framebuffer", disp);
        if (waitKey(1) == 27) break;

        // Stats
        if (frameIdx % 100 == 0) {
            auto now = Clock::now();
            double elapsed = chrono::duration<double>(now - startTime).count();
            double pct = 100.0 * frameIdx / totalFrames;
            double eta = elapsed * (totalFrames - frameIdx) / frameIdx;
            cout << "Frame " << frameIdx << "/" << totalFrames
                 << " (" << fixed << setprecision(2) << pct << "% )"
                 << " - Elapsed: " << elapsed << "s"
                 << " - ETA: " << eta << "s" << endl;
        }
    }

    writer.release();
    destroyAllWindows();
    return 0;
}
