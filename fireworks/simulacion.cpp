#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;
using Clock = chrono::high_resolution_clock;

struct Particle {
    double x, y;
    double vx, vy;
    int age;
    int generation;           // 1=rocket, 2=explode1, 3=explode2
    double drag;
    uchar r, g, b;
    double alpha;
    double prevX, prevY;
    double size;
    bool multicolor;          // color flag for explosion
    bool willSecondExplode;   // carries rocket's double-explode flag to fragments
    int secondDelay;          // frames to second explosion
};

int main() {
    const int width = 3840;
    const int height = 2160;
    const int fps = 60;
    const int durationMin = 60*10;
    const int totalFrames = durationMin * 60 * fps;
    const double gravity = 0.2;

    // Lifetime thresholds
    const int launchLifetime      = 80;
    const int explosionLifetime   = 200;

    // Air drag parameters
    const double rocketDrag       = 1.0;
    const double explosionDrag    = 0.98;

    // Depth parameter
    const double depthScale       = 0.2;

    // Trail fade parameter
    const double trailFadeAlpha   = 0.9;
    const double particleFadeRate = 0.005;

    // Launch interval
    const int    launchInterval   = 200;
    const double widthSpread      = 2.0;

    // Glow effect parameters (customizable)
    const bool   enableGlow       = true;      // toggle glow overlay
    const double glowIntensity    = 20;       // strength of the glow blend [0,1]
    const int    glowKernelSize   = 21;        // must be odd: higher = softer glow
    const double glowThreshold    = 0.8;       // pixel brightness threshold [0,1]

    // Initialize RNG
    mt19937 rng((unsigned)time(nullptr));
    uniform_real_distribution<double> distX(0, width);
    uniform_real_distribution<double> distAng(0, 2*M_PI);
    uniform_real_distribution<double> distColorJ(-40, 40);
    uniform_real_distribution<double> distSize(1, 4);
    uniform_real_distribution<double> distLaunchSpeed(20, 70);
    uniform_real_distribution<double> distExplodeSpeed(0, 50);
    uniform_int_distribution<int> distColorBase(0, 255);
    uniform_int_distribution<int> distMulti(1, 10);
    uniform_int_distribution<int> distDouble(1, 5);
    uniform_int_distribution<int> distDelay((int)(0.5*fps), (int)(2.0*fps));

    vector<Particle> particles;
    particles.reserve(10000);

    // Buffers for trails and rendering with separate glow overlay
    Mat trailCanvas(height, width, CV_8UC3, Scalar(0,0,0));
    Mat renderCanvas(height, width, CV_8UC3, Scalar(0,0,0));

    string filename = to_string(time(nullptr)) + string("_fireworks.mp4");
    VideoWriter writer(filename,
                       VideoWriter::fourcc('a','v','c','1'),
                       fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Error opening file " << filename << endl;
        return -1;
    }

    namedWindow("Framebuffer", WINDOW_NORMAL);
    resizeWindow("Framebuffer", width/2, height/2);
    auto startTime = Clock::now();

    for (int frameIdx = 0; frameIdx < totalFrames; ++frameIdx) {
        // Fade existing trails
        trailCanvas *= trailFadeAlpha;

        // Launch rockets
        if (frameIdx % launchInterval == 0) {
            double baseX = distX(rng);
            Particle p;
            p.x = (baseX - width/2.0) * widthSpread + width/2.0;
            p.y = 0;
            double ang = CV_PI/2;
            double speed = distLaunchSpeed(rng);
            p.vx = cos(ang)*speed;
            p.vy = sin(ang)*speed;
            p.age = 0;
            p.generation = 1;
            p.drag = rocketDrag;
            p.r = distColorBase(rng);
            p.g = distColorBase(rng);
            p.b = distColorBase(rng);
            p.alpha = 0.2;
            p.prevX = p.x;
            p.prevY = p.y;
            p.size = distSize(rng);
            p.multicolor = (distMulti(rng) <= 2);
            p.willSecondExplode = (distDouble(rng) == 1);
            p.secondDelay = 0;
            particles.push_back(p);
        }

        int N = particles.size();
        // Physics update
        for (int i = 0; i < N; ++i) {
            auto &p = particles[i];
            p.age++;
            p.vx *= p.drag;
            p.vy *= p.drag;
            p.vy -= gravity;
            p.x += p.vx;
            p.y += p.vy;
        }

        vector<Particle> newParts;
        newParts.reserve(2000);

        // Primary explosions
        for (auto &p : particles) {
            if (p.generation == 1 && p.age == 40) {
                for (int k = 0; k < 100; ++k) {
                    Particle q = {};
                    q.x = p.x; q.y = p.y;
                    double ang = distAng(rng);
                    double sp  = distExplodeSpeed(rng);
                    q.vx = cos(ang)*sp;
                    q.vy = sin(ang)*sp;
                    q.age = 0; q.generation = 2; q.drag = explosionDrag;
                    if (p.multicolor) {
                        q.r = distColorBase(rng);
                        q.g = distColorBase(rng);
                        q.b = distColorBase(rng);
                    } else {
                        q.r = saturate_cast<uchar>(p.r + distColorJ(rng));
                        q.g = saturate_cast<uchar>(p.g + distColorJ(rng));
                        q.b = saturate_cast<uchar>(p.b + distColorJ(rng));
                    }
                    q.alpha = 1.0; q.prevX = q.x; q.prevY = q.y; q.size = distSize(rng);
                    q.willSecondExplode = p.willSecondExplode;
                    q.secondDelay = p.willSecondExplode ? distDelay(rng) : 0;
                    q.multicolor = p.multicolor;
                    newParts.push_back(q);
                }
            }
        }

        // Secondary explosions
        for (auto &p : particles) {
            if (p.generation == 2 && p.willSecondExplode && p.age == p.secondDelay) {
                for (int k = 0; k < 50; ++k) {
                    Particle q = {};
                    q.x = p.x; q.y = p.y;
                    double ang = distAng(rng);
                    double sp  = distExplodeSpeed(rng);
                    q.vx = cos(ang)*sp;
                    q.vy = sin(ang)*sp;
                    q.age = 0; q.generation = 3; q.drag = explosionDrag;
                    q.r = saturate_cast<uchar>(p.r + distColorJ(rng));
                    q.g = saturate_cast<uchar>(p.g + distColorJ(rng));
                    q.b = saturate_cast<uchar>(p.b + distColorJ(rng));
                    q.alpha = 1.0; q.prevX = q.x; q.prevY = q.y; q.size = distSize(rng);
                    q.willSecondExplode = false; q.secondDelay = 0;
                    newParts.push_back(q);
                }
            }
        }

        particles.insert(particles.end(), newParts.begin(), newParts.end());
        particles.erase(
            remove_if(particles.begin(), particles.end(), [&](const Particle &p){
                return (p.generation == 1 && p.age > launchLifetime)
                    || ((p.generation == 2 || p.generation == 3) && p.age > explosionLifetime);
            }), particles.end());

        // Draw particles onto trail canvas
        for (auto &p : particles) {
            double dx1 = p.prevX - width/2.0;
            double dy1 = p.prevY - height/2.0;
            double dx2 = p.x      - width/2.0;
            double dy2 = p.y      - height/2.0;
            if (p.generation >= 2) {
                dx1 *= depthScale;
                dy1 *= depthScale;
                dx2 *= depthScale;
                dy2 *= depthScale;
            }
            Point pt1(int(width/2.0 + dx1), height - int(height/2.0 + dy1));
            Point pt2(int(width/2.0 + dx2), height - int(height/2.0 + dy2));
            Scalar col(p.b*p.alpha, p.g*p.alpha, p.r*p.alpha);
            line(trailCanvas, pt1, pt2, col, int(p.size), LINE_AA);
            if (pt2.inside(Rect(0,0,width,height)))
                trailCanvas.at<Vec3b>(pt2.y, pt2.x) = Vec3b(
                    saturate_cast<uchar>(p.b*p.alpha),
                    saturate_cast<uchar>(p.g*p.alpha),
                    saturate_cast<uchar>(p.r*p.alpha));
            p.prevX = p.x;
            p.prevY = p.y;
            p.alpha -= particleFadeRate;
        }

        // Prepare render canvas from current trails
        renderCanvas = trailCanvas.clone();

        // Glow overlay post-processing
        if (enableGlow) {
            Mat glowImg, gray, mask;
            // Blur the trails to create glow
            GaussianBlur(trailCanvas, glowImg, Size(glowKernelSize, glowKernelSize), 0);
            // Convert to grayscale for thresholding
            cvtColor(glowImg, gray, COLOR_BGR2GRAY);
            // Apply threshold to isolate bright areas
            threshold(gray, mask, glowThreshold*255.0, 255, THRESH_BINARY);
            // Create 3-channel mask
            Mat glowMask;
            cvtColor(mask, glowMask, COLOR_GRAY2BGR);
            // Mask the blurred image
            bitwise_and(glowImg, glowMask, glowImg);
            // Overlay glow onto render canvas
            addWeighted(renderCanvas, 1.0, glowImg, glowIntensity, 0, renderCanvas);
        }

        // Write and display
        writer.write(renderCanvas);
        Mat disp; resize(renderCanvas, disp, Size(width/2, height/2)); imshow("Framebuffer", disp);
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

