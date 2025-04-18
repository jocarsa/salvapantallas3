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

struct Star {
    double angle;
    double dist;
};

int main() {
    // Window and video parameters
    const int width       = 1920;
    const int height      = 1080;
    const int fps         = 60;
    const int durationMin = 60;
    const int totalFrames = durationMin * 60 * fps;

    // Starfield parameters
    const int   numStars    = 10000;   // number of stars
    const double speedFactor = 0.01;    // radial speed multiplier
    const double trailFade   = 0.9;     // canvas fade per frame (0–1)
    const double starAlpha   = 0.5;     // overlay alpha for new stars
    const double maxDist     = 1200.0;  // spawn/reset radius > viewport
    const double dMin        = 0.1;     // minimum nonzero distance

    // Setup RNG
    mt19937 rng((unsigned)time(nullptr));
    uniform_real_distribution<double> distAngle(0, 2 * M_PI);
    uniform_real_distribution<double> dist01(0.0, 1.0);  // for log‑uniform

    // Helper to sample log‑uniform in [dMin, maxDist]
    auto sampleDist = [&]() {
        double u = dist01(rng);
        return dMin * pow(maxDist / dMin, u);
    };

    // Initialize stars
    vector<Star> stars(numStars);
    for (auto &s : stars) {
        s.angle = distAngle(rng);
        s.dist  = sampleDist();
    }

    // Canvas for drawing
    Mat canvas(height, width, CV_8UC3, Scalar(0, 0, 0));
    Mat newFrame(height, width, CV_8UC3);

    // Video writer
    VideoWriter writer("starfield.mp4",
                       VideoWriter::fourcc('a','v','c','1'),
                       fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Error: cannot open video writer" << endl;
        return -1;
    }

    namedWindow("Framebuffer", WINDOW_NORMAL);
    resizeWindow("Framebuffer", width/2, height/2);

    auto startTime = Clock::now();
    Point center(width/2, height/2);

    for (int frameIdx = 0; frameIdx < totalFrames; ++frameIdx) {
        // 1) Fade old trails
        canvas *= trailFade;

        // 2) Update distances in parallel
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numStars; ++i) {
            stars[i].dist += stars[i].dist * speedFactor;
            if (stars[i].dist > maxDist) {
                stars[i].dist  = sampleDist();
                stars[i].angle = distAngle(rng);
            }
        }

        // 3) Draw each star into newFrame
        newFrame = Scalar(0,0,0);
        for (auto &s : stars) {
            double d1 = s.dist;
            double d2 = s.dist * (1.0 + speedFactor);
            double a1 = s.angle;
            double a2 = s.angle + 0.01;
            Point p1(int(center.x + d1 * cos(a1)), int(center.y + d1 * sin(a1)));
            Point p2(int(center.x + d2 * cos(a1)), int(center.y + d2 * sin(a1)));
            Point p3(int(center.x + d2 * cos(a2)), int(center.y + d2 * sin(a2)));
            Point p4(int(center.x + d1 * cos(a2)), int(center.y + d1 * sin(a2)));
            Point quad[4] = { p1, p2, p3, p4 };
            fillConvexPoly(newFrame, quad, 4, Scalar(255,255,255), LINE_AA);
        }

        // 4) Overlay new stars with motion‐blur alpha
        addWeighted(newFrame, starAlpha, canvas, 1.0, 0, canvas);

        // 5) Write & display
        writer.write(canvas);
        Mat disp;
        resize(canvas, disp, Size(width/2, height/2));
        imshow("Framebuffer", disp);
        if (waitKey(1) == 27) break;  // ESC quits early

        // 6) Stats every 100 frames
        if ((frameIdx % 100) == 0 && frameIdx > 0) {
            auto now     = Clock::now();
            double elapsed = chrono::duration<double>(now - startTime).count();
            double pct     = 100.0 * frameIdx / totalFrames;
            double eta     = elapsed * (totalFrames - frameIdx) / frameIdx;
            cout << "Frame " << frameIdx << "/" << totalFrames
                 << " (" << fixed << setprecision(2) << pct << "%)"
                 << " - Elapsed: " << elapsed << "s"
                 << " - ETA: "     << eta     << "s"
                 << endl;
        }
    }

    writer.release();
    destroyAllWindows();
    return 0;
}

