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

struct Particle {
    double x, y;
    double angle;
    uchar r, g, b;
    double v;
};

int main() {
    // Video parameters
    const int width = 1920;
    const int height = 1080;
    const int fps = 60;
    const int durationMin = 60;
    const int totalFrames = durationMin * 60 * fps;
    const int numParticles = 200;

    // Depth-of-field parameter: Gaussian blur sigma for background
    const double dofSigma = 5.0;  // adjust for stronger/weaker blur
    // Motion blur parameter: contribution of new frame
    const double motionBlurAlpha = 0.7; // 0 = no new frame, 1 = no trail

    // RNG setup
    mt19937 rng((unsigned)time(nullptr));
    uniform_real_distribution<double> distX(0, width);
    uniform_real_distribution<double> distY(0, height);
    uniform_real_distribution<double> distAng(0, 2 * M_PI);
    uniform_int_distribution<int> distColor(0, 255);
    uniform_real_distribution<double> distV(30, 130);

    // Initialize particles
    vector<Particle> particles(numParticles);
    for (auto &p : particles) {
        p.x = distX(rng);
        p.y = distY(rng);
        p.angle = distAng(rng);
        p.r = static_cast<uchar>(distColor(rng));
        p.g = static_cast<uchar>(distColor(rng));
        p.b = static_cast<uchar>(distColor(rng));
        p.v = distV(rng);
    }
    double worldAngle = distAng(rng);

    // Pre-allocate layers
    Mat background(height, width, CV_8UC3);
    Mat foreground(height, width, CV_8UC3);
    Mat frame(height, width, CV_8UC3);
    Mat composite(height, width, CV_8UC3, Scalar(0,0,0));

    // Video writer
    VideoWriter writer("particle_dof_motionblur.mp4",
                       VideoWriter::fourcc('a','v','c','1'),
                       fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Error: cannot open video writer" << endl;
        return -1;
    }

    // Display window
    namedWindow("Framebuffer", WINDOW_NORMAL);
    resizeWindow("Framebuffer", width/2, height/2);

    auto startTime = Clock::now();

    vector<int> idx(numParticles);
    iota(idx.begin(), idx.end(), 0);

    for (int frameIdx = 1; frameIdx <= totalFrames; ++frameIdx) {
        // Clear layers
        background.setTo(Scalar(0,0,0));
        foreground.setTo(Scalar(0,0,0));

        // Recycle out-of-bounds
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < numParticles; ++i) {
            auto &p = particles[i];
            if (p.x > width*2 || p.y > height*2 || p.x < -width || p.y < -height) {
                double oldV = p.v;
                p.x = distX(rng);
                p.y = distY(rng);
                p.angle = distAng(rng);
                p.r = static_cast<uchar>(distColor(rng));
                p.g = static_cast<uchar>(distColor(rng));
                p.b = static_cast<uchar>(distColor(rng));
                p.v = distV(rng);
                p.x -= cos(worldAngle) * oldV * 50;
                p.y -= sin(worldAngle) * oldV * 50;
            }
        }

        // Sort by speed for depth
        sort(idx.begin(), idx.end(), [&](int a, int b) {
            return particles[a].v < particles[b].v;
        });
        int split = numParticles/2;

        // Draw background (slower half)
        for (int k = 0; k < split; ++k) {
            auto &p = particles[idx[k]];
            p.x += cos(worldAngle) * p.v / 10;
            p.y += sin(worldAngle) * p.v / 10;
            Point pt1(int(p.x), int(p.y));
            Point pt2(int(p.x + cos(worldAngle)*p.v),
                      int(p.y + sin(worldAngle)*p.v));
            int thickness = max(1, int(p.v/3));
            Scalar color(p.b, p.g, p.r);
            line(background, pt1, pt2, color, thickness, LINE_AA);
        }
        // Draw foreground (faster half)
        for (int k = split; k < numParticles; ++k) {
            auto &p = particles[idx[k]];
            p.x += cos(worldAngle) * p.v / 10;
            p.y += sin(worldAngle) * p.v / 10;
            Point pt1(int(p.x), int(p.y));
            Point pt2(int(p.x + cos(worldAngle)*p.v),
                      int(p.y + sin(worldAngle)*p.v));
            int thickness = max(1, int(p.v/3));
            Scalar color(p.b, p.g, p.r);
            line(foreground, pt1, pt2, color, thickness, LINE_AA);
        }

        // DOF: blur background only
        GaussianBlur(background, background, Size(0,0), dofSigma);

        // Composite DOF layers
        addWeighted(background, 1.0, foreground, 1.0, 0.0, frame);

        // Motion blur: blend into persistent composite
        addWeighted(frame, motionBlurAlpha,
                    composite, 1.0 - motionBlurAlpha,
                    0.0, composite);

        // Output
        writer.write(composite);
        Mat disp;
        resize(composite, disp, Size(width/2, height/2));
        imshow("Framebuffer", disp);
        if (waitKey(1) == 27) break;

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

