#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace cv;
using namespace std;
using Clock = chrono::high_resolution_clock;

int main() {
    // Simulation parameters
    const int width = 1920;
    const int height = 1080;
    const int numParticles = 100;
    const int fps = 60;
    const int durationMin = 60;
    const int totalFrames = durationMin * 60 * fps;
    const double velocity = 3.0;

    // Random engine
    mt19937 rng((unsigned)time(nullptr));
    uniform_real_distribution<double> distX(0, width);
    uniform_real_distribution<double> distY(0, height);
    uniform_real_distribution<double> distAngle(0, 2 * M_PI);
    uniform_real_distribution<double> distTurn(-0.5, 0.5);
    uniform_int_distribution<int> distColor(0, 255);
    uniform_int_distribution<int> distDelta(-2, 2);

    // Particle state
    vector<double> px(numParticles), py(numParticles);
    vector<double> prevX(numParticles), prevY(numParticles);
    vector<double> angle(numParticles);
    vector<uchar> r(numParticles), g(numParticles), b(numParticles);

    for (int i = 0; i < numParticles; ++i) {
        px[i] = distX(rng);
        py[i] = distY(rng);
        prevX[i] = px[i];
        prevY[i] = py[i];
        angle[i] = distAngle(rng);
        r[i] = static_cast<uchar>(distColor(rng));
        g[i] = static_cast<uchar>(distColor(rng));
        b[i] = static_cast<uchar>(distColor(rng));
    }

    // Prepare persistent white canvas for trails
    Mat frame(height, width, CV_8UC3, Scalar(255, 255, 255));

    // Video writer
    VideoWriter writer("simulation.mp4", VideoWriter::fourcc('a','v','c','1'), fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Error: Cannot open video writer." << endl;
        return -1;
    }

    namedWindow("Framebuffer", WINDOW_NORMAL);
    resizeWindow("Framebuffer", width / 2, height / 2);

    auto startTime = Clock::now();

    for (int frameIdx = 1; frameIdx <= totalFrames; ++frameIdx) {
        // Store previous positions
        prevX = px;
        prevY = py;

        // Update particles in parallel
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < numParticles; ++i) {
            // Bounce if out of bounds
            if (px[i] < 0 || px[i] > width || py[i] < 0 || py[i] > height) {
                angle[i] += M_PI;
            }
            // Move
            px[i] += cos(angle[i]) * velocity;
            py[i] += sin(angle[i]) * velocity;
            // Turn variation
            angle[i] += 0.1 + distTurn(rng) * 0.2;
            // Color variation
            int dr = distDelta(rng), dg = distDelta(rng), db = distDelta(rng);
            r[i] = static_cast<uchar>(clamp<int>(r[i] + dr, 0, 255));
            g[i] = static_cast<uchar>(clamp<int>(g[i] + dg, 0, 255));
            b[i] = static_cast<uchar>(clamp<int>(b[i] + db, 0, 255));
        }

        // Draw new positions on persistent canvas
        for (int i = 0; i < numParticles; ++i) {
            Point center(static_cast<int>(px[i]), static_cast<int>(py[i]));
            Scalar color(b[i], g[i], r[i]);
            // Draw circle
            circle(frame, center, 2, color, FILLED, LINE_AA);
            // Draw line from prev to current if no wrap-around
            double dx = px[i] - prevX[i];
            double dy = py[i] - prevY[i];
            if (fabs(dx) < width / 2.0 && fabs(dy) < height / 2.0) {
                Point p0(static_cast<int>(prevX[i]), static_cast<int>(prevY[i]));
                line(frame, p0, center, color, 1, LINE_AA);
            }
        }

        // Write and display
        writer.write(frame);
        Mat disp;
        resize(frame, disp, Size(width / 2, height / 2));
        imshow("Framebuffer", disp);
        if (waitKey(1) == 27) break;

        // Stats every 100 frames
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
