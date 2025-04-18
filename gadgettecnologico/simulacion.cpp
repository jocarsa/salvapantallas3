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

struct Circulo {
    int t;
    double r, a1, a2, v;
};

int main() {
    // Video and simulation parameters
    const int width = 1920, height = 1080;
    const int fps = 60;
    const int durationMin = 60;
    const int totalFrames = durationMin * 60 * fps;
    const int numCirculos = 20;
    const double circleWidth = 5.0;
    // Overall radius factor (fraction of video height)
    const double baseRadiusFactor = 0.4;
    const double baseRadius = baseRadiusFactor * height;

    // Alpha levels from original JS
    const double alphaStroke = 0.3;
    const double alphaFill   = 0.1;

    // Setup RNGs
    mt19937 baseRng((unsigned)time(nullptr));
    uniform_int_distribution<int> distType(0, 3);
    uniform_real_distribution<double> distInitAngle(0, 2 * M_PI);
    uniform_real_distribution<double> distTurn(-0.5, 0.5);
    vector<mt19937> rngs(numCirculos);
    for (int i = 0; i < numCirculos; ++i) rngs[i].seed(baseRng());

    // Initialize circles with proportional radii
    vector<Circulo> circulos(numCirculos);
    for (int i = 0; i < numCirculos; ++i) {
        circulos[i].t  = distType(baseRng);
        circulos[i].r  = baseRadius * double(i + 1) / numCirculos;
        circulos[i].a1 = distInitAngle(baseRng);
        circulos[i].a2 = circulos[i].a1 + distInitAngle(baseRng);
        circulos[i].v  = (uniform_real_distribution<double>(-0.5, 0.5))(baseRng);
    }

    // Pre-allocated white frame and overlay
    Mat frame(height, width, CV_8UC3), overlay;

    // Video writer
    VideoWriter writer("widget.mp4", VideoWriter::fourcc('a','v','c','1'), fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Error: Cannot open video writer" << endl;
        return -1;
    }

    // Display window
    namedWindow("Framebuffer", WINDOW_NORMAL);
    resizeWindow("Framebuffer", width / 2, height / 2);

    auto startTime = Clock::now();
    double tiempo = 0.0;
    Point center(width / 2, height / 2);

    // Discontinuous segment parameters
    const double segmentLength = 0.03; // radians per segment
    const double segmentGap    = 0.03; // radians between segments

    for (int frameIdx = 1; frameIdx <= totalFrames; ++frameIdx) {
        // Reset to white
        frame.setTo(Scalar(255, 255, 255));
        overlay = frame.clone();

        // Update parameters
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numCirculos; ++i) {
            auto &c = circulos[i];
            auto &rng = rngs[i];
            c.v  += distTurn(rng) * 0.001;
            c.a1 += cos(tiempo * 0.01) * 0.001 + sin(c.v) * 0.01;
            c.a2 += sin(tiempo * 0.01) * 0.001 + sin(c.v) * 0.01;
        }

        // Draw with transparency
        for (auto &c : circulos) {
            double startDeg = c.a1 * 180.0 / M_PI;
            double endDeg   = c.a2 * 180.0 / M_PI;
            double diff = c.a2 - c.a1;
            if (diff < 0) diff += 2 * M_PI;

            switch (c.t) {
                case 0:
                    ellipse(overlay, center, Size(c.r, c.r), 0.0,
                            startDeg, endDeg,
                            Scalar(0,0,0), int(circleWidth + 0.2), LINE_AA);
                    break;

                case 1: {
                    vector<Point> pts;
                    pts.emplace_back(center);
                    int steps = max(2, int(diff / 0.01));
                    for (int k = 0; k <= steps; ++k) {
                        double theta = c.a1 + diff * k / steps;
                        pts.emplace_back(center.x + int(c.r * cos(theta)),
                                          center.y + int(c.r * sin(theta)));
                    }
                    fillConvexPoly(overlay, pts.data(), (int)pts.size(), Scalar(0,0,0), LINE_AA);
                    break;
                }

                case 2:
                    // More dense, sharp-edged discontinuous segments
                    for (double ang = c.a1; ang < c.a2; ang += segmentLength + segmentGap) {
                        double segEnd = min(c.a2, ang + segmentLength);
                        double deg1 = ang * 180.0 / M_PI;
                        double deg2 = segEnd * 180.0 / M_PI;
                        ellipse(overlay, center, Size(c.r, c.r), 0.0,
                                deg1, deg2,
                                Scalar(0,0,0), int(circleWidth + 0.2), LINE_8);
                    }
                    break;

                case 3:
                    ellipse(overlay, center, Size(c.r, c.r), 0.0,
                            startDeg, endDeg,
                            Scalar(0,0,0), 1, LINE_AA);
                    circle(overlay, Point(center.x + int(c.r * cos(c.a1)),
                                           center.y + int(c.r * sin(c.a1))),
                           3, Scalar(0,0,0), FILLED, LINE_AA);
                    circle(overlay, Point(center.x + int(c.r * cos(c.a2)),
                                           center.y + int(c.r * sin(c.a2))),
                           3, Scalar(0,0,0), FILLED, LINE_AA);
                    break;
            }
            double a = (c.t == 1) ? alphaFill : alphaStroke;
            addWeighted(overlay, a, frame, 1 - a, 0, frame);
            overlay = frame.clone();
        }

        // Write and display
        writer.write(frame);
        Mat disp;
        resize(frame, disp, Size(width/2, height/2));
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

        tiempo += 1.0;
    }

    writer.release();
    destroyAllWindows();
    return 0;
}

