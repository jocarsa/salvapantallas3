#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <iostream>

using namespace cv;
using namespace std;
using Clock = chrono::high_resolution_clock;

// HSL to RGB conversion
static Vec3b hslToRgb(double h, double s, double l) {
    s /= 100.0; l /= 100.0;
    double c = (1.0 - fabs(2.0*l - 1.0))*s;
    double x = c * (1.0 - fabs(fmod(h/60.0, 2.0) - 1.0));
    double m = l - c/2.0;
    double r=0, g=0, b=0;
    if (h < 60)      { r = c; g = x; }
    else if (h < 120){ r = x; g = c; }
    else if (h < 180){ g = c; b = x; }
    else if (h < 240){ g = x; b = c; }
    else if (h < 300){ r = x; b = c; }
    else             { r = c; b = x; }
    return Vec3b(
        uchar((b + m) * 255),
        uchar((g + m) * 255),
        uchar((r + m) * 255)
    );
}

// Reset state
void reset(
    vector<double>& angles,
    vector<double>& radii,
    vector<double>& speeds,
    Mat& trace,
    Mat& arms,
    int& lineWidth,
    mt19937& rng
) {
    int N = (int)angles.size();
    uniform_real_distribution<double> distRad(0.0, 360.0);
    uniform_int_distribution<int> distLW(5, 125);
    vector<double> dens = {-8,-7,-6,-5,-4,-3,-2,2,3,4,5,6,7,8};
    uniform_int_distribution<size_t> distDen(0, dens.size()-1);

    for (int i = 0; i < N; ++i) {
        angles[i] = 0.0;
        radii[i] = (i + 1) * (250.0 / N);
        double denom = dens[distDen(rng)];
        speeds[i] = CV_PI / denom / 10.0;
    }
    trace.setTo(Scalar(255,255,255));
    arms.setTo(Scalar(255,255,255));
    lineWidth = distLW(rng);
}

int main() {
    // Video setup
    const int width = 1920;
    const int height = 1080;
    const int fps = 60;
    const double minutes = 1.0;
    const int totalFrames = int(minutes * 60 * fps);

    // Reset tolerance (pixels) and threshold squared
    const double resetTolerance = 400.0;
    const double resetThreshold2 = resetTolerance * resetTolerance;

    // Ensure output directory
    auto t0 = time(nullptr);
    filesystem::create_directories("videos");
    string fname = format("videos/output_%lld.mp4", (long long)t0);
    VideoWriter writer(
        fname,
        VideoWriter::fourcc('M','J','P','G'),
        fps,
        Size(width, height)
    );
    if (!writer.isOpened()) {
        cerr << "Cannot open video writer" << endl;
        return -1;
    }

    // Canvases
    Mat trace = Mat::ones(height, width, CV_8UC3) * 255;
    Mat arms  = Mat::ones(height, width, CV_8UC3) * 255;
    Mat final;

    // Arm parameters
    const int N = 2;
    vector<double> angles(N), radii(N), speeds(N);
    int lineWidth;

    // RNG for main
    mt19937 rng((unsigned)time(nullptr));

    reset(angles, radii, speeds, trace, arms, lineWidth, rng);

    const int initialX = width / 2;
    const int initialY = height / 2;
    int prevX = 0, prevY = 0;
    bool drawing = false;
    int firstX = -1, firstY = -1;

    // Color setup
    uniform_real_distribution<double> distHue(0.0, 360.0);
    uniform_int_distribution<int> distBool(0, 1);
    double hue = distHue(rng);
    bool randomColor = distBool(rng);
    Vec3b traceCol = hslToRgb(hue, 100.0, 50.0);
    const double hueStep = 0.5;

    auto start = Clock::now();
    for (int f = 0; f < totalFrames; ++f) {
        // Clear arms
        arms.setTo(Scalar(255,255,255));

        // Draw arms
        int x = initialX;
        int y = initialY;
        for (int i = 0; i < N; ++i) {
            double a = angles[i];
            double r = radii[i];
            int nx = x + int(cos(a) * r);
            int ny = y + int(sin(a) * r);
            line(arms, Point(x,y), Point(nx,ny), Scalar(0,0,0), 5, LINE_AA);
            circle(arms, Point(x,y), 10, Scalar(0,0,0), FILLED, LINE_AA);
            circle(arms, Point(nx,ny), 10, Scalar(0,0,0), FILLED, LINE_AA);
            x = nx;
            y = ny;
        }
        // Update angles
        for (int i = 0; i < N; ++i)
            angles[i] += speeds[i];

        // Draw trace
        if (drawing) {
            if (randomColor)
                line(trace, Point(prevX, prevY), Point(x, y), Scalar(0,0,0), lineWidth, LINE_AA);
            else
                line(trace, Point(prevX, prevY), Point(x, y), traceCol, lineWidth, LINE_AA);
        } else {
            drawing = true;
        }

        // Capture first point once (unused for reset now)
        if (drawing && firstX < 0 && firstY < 0) {
            firstX = prevX;
            firstY = prevY;
        }
        prevX = x;
        prevY = y;

        // Composite
        multiply(trace, arms, final, 1.0/255.0);
        writer.write(final);

        // Update hue
        hue = fmod(hue + hueStep, 360.0);
        traceCol = hslToRgb(hue, 100.0, 50.0);

        // **Restart** whenever the end effector comes within 40px of the center
        double dx = x - initialX;
        double dy = y - initialY;
        if ((dx*dx + dy*dy) < resetThreshold2) {
            reset(angles, radii, speeds, trace, arms, lineWidth, rng);
            drawing   = false;
            firstX    = firstY = -1;
        }

        // Display (scaled down)
        Mat disp;
        resize(final, disp, Size(width/2, height/2));
        imshow("Framebuffer", disp);
        if (waitKey(1) == 1) break;

        // Progress stats
        if (f % 1000 == 0) {
            auto now = Clock::now();
            double elapsed = chrono::duration<double>(now - start).count();
            double pct = 100.0 * f / totalFrames;
            double eta = elapsed * (totalFrames - f) / f;
            cout << "Frame " << f << "/" << totalFrames
                 << " (" << fixed << setprecision(2) << pct << "% )"
                 << " - Elapsed: " << elapsed << "s"
                 << " - ETA: " << eta << "s" << endl;
        }
    }

    writer.release();
    destroyAllWindows();
    return 0;
}

