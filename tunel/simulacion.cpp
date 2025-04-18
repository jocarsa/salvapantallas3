#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using Clock = chrono::high_resolution_clock;

struct Ring {
    double x, y;
    double radius;
    double angle;
};

int main() {
    // Video parameters
    const int width = 1920;
    const int height = 1080;
    const int fps = 60;
    const int durationMin = 60;
    const int totalFrames = durationMin * 60 * fps;

    // Tunnel parameters
    const int numSides = 20 + (rand() % 41); // random 10..50
    const double growthFactor = 1.08;
    const double spawnInterval = 1;         // frames per new ring
    const double cursorSpeed = 5.0;
    const double angleJitter = 0.7;
    const int lineWidth = 5;

    // RNG
    mt19937 rng((unsigned)time(nullptr));
    uniform_int_distribution<int> distSides(10, 50);
    uniform_real_distribution<double> distJitt(-0.5, 0.5);

    // Initialize cursor
    double cursorX = width/2.0;
    double cursorY = height/2.0;
    double cursorAngle = uniform_real_distribution<double>(0, 2*M_PI)(rng);
    int sides = distSides(rng);

    vector<Ring> rings;
    rings.reserve(1000);

    // Frame buffer
    Mat frame(height, width, CV_8UC3);

    // Video writer
    VideoWriter writer("wireframe_tunnel.mp4",
                       VideoWriter::fourcc('a','v','c','1'),
                       fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Error: cannot open video writer" << endl;
        return -1;
    }

    namedWindow("Framebuffer", WINDOW_NORMAL);
    resizeWindow("Framebuffer", width/2, height/2);

    auto startTime = Clock::now();
    
    for (int frameIdx = 1; frameIdx <= totalFrames; ++frameIdx) {
        // Clear background
        frame.setTo(Scalar(0,0,0));

        // Spawn new ring
        if (frameIdx % int(spawnInterval) == 0) {
            rings.push_back({cursorX, cursorY, 1.0, frameIdx / 300.0});
        }

        // Move cursor
        cursorX += cos(cursorAngle) * cursorSpeed;
        cursorY += sin(cursorAngle) * cursorSpeed;
        cursorAngle += distJitt(rng) * angleJitter;
        if (cursorX < 0 || cursorX > width) cursorAngle += M_PI;
        if (cursorY < 0 || cursorY > height) cursorAngle += M_PI;

        // Update and draw rings
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)rings.size(); ++i) {
            rings[i].radius *= growthFactor;
        }

        // Draw polygons
        for (size_t i = 0; i < rings.size(); ++i) {
            // Generate vertices
            vector<Point> pts(sides);
            for (int j = 0; j < sides; ++j) {
                double theta = 2*M_PI * j / sides + rings[i].angle;
                pts[j] = Point(
                    int(rings[i].x + cos(theta) * rings[i].radius),
                    int(rings[i].y + sin(theta) * rings[i].radius)
                );
            }
            polylines(frame, pts, true, Scalar(255,255,255), lineWidth, LINE_AA);
        }

        // Connect ring layers
        for (size_t i = 1; i < rings.size(); ++i) {
            for (int j = 0; j < sides; ++j) {
                double theta = 2*M_PI * j / sides;
                Point p1(
                    int(rings[i].x + cos(theta + rings[i].angle) * rings[i].radius),
                    int(rings[i].y + sin(theta + rings[i].angle) * rings[i].radius)
                );
                Point p2(
                    int(rings[i-1].x + cos(theta + rings[i-1].angle) * rings[i-1].radius),
                    int(rings[i-1].y + sin(theta + rings[i-1].angle) * rings[i-1].radius)
                );
                line(frame, p1, p2, Scalar(255,255,255), 1, LINE_AA);
            }
        }

        // Remove old rings
        while (!rings.empty() && rings.front().radius > max(width, height)) {
            rings.erase(rings.begin());
        }

        // Write and display
        writer.write(frame);
        Mat disp;
        resize(frame, disp, Size(width/2, height/2));
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

