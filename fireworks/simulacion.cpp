#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using Clock = chrono::high_resolution_clock;

struct Particle {
    double x, y;
    double vx, vy;
    int age;
    int generation;
    double drag;
    uchar r, g, b;
    double alpha;
    double prevX, prevY;
    double size;
};

int main() {
    const int width = 1920;
    const int height = 1080;
    const int fps = 60;
    const int durationMin = 60;
    const int totalFrames = durationMin * 60 * fps;
    const double gravity = 0.2;

    // Lifetime thresholds
    const int launchLifetime      = 80;
    const int explosionLifetime   = 200;

    // Air drag parameters
    const double rocketDrag       = 1.0;
    const double explosionDrag    = 0.98;

    // Depth parameter: explosion drawn "farther" (scale <1)
    const double depthScale       = 0.2;  // 0.0=horizon, 1.0=front

    // Trail fade parameter (canvas persistence)
    const double trailFadeAlpha   = 0.9;
    // Particle fade-per-frame rate
    const double particleFadeRate = 0.005;

    // Launch interval and spawn width control
    const int    launchInterval   = 200;    // frames between new rockets
    const double widthSpread      = 2.0;  // 0.0=center only, 1.0=full screen

    mt19937 rng((unsigned)time(nullptr));
    uniform_real_distribution<double> distX(0, width);
    uniform_real_distribution<double> distAng(0, 2*M_PI);
    uniform_real_distribution<double> distColorJ(-40, 40);
    uniform_real_distribution<double> distSize(1, 4);
    uniform_real_distribution<double> distLaunchSpeed(20, 70);
    uniform_real_distribution<double> distExplodeSpeed(0, 50);
    uniform_int_distribution<int> distColorBase(0, 255);

    vector<Particle> particles;
    particles.reserve(10000);

    Mat canvas(height, width, CV_8UC3, Scalar(0,0,0));

    VideoWriter writer("fireworks.mp4",
                       VideoWriter::fourcc('a','v','c','1'),
                       fps, Size(width, height));
    if (!writer.isOpened()) return -1;

    namedWindow("Framebuffer", WINDOW_NORMAL);
    resizeWindow("Framebuffer", width/2, height/2);
    auto startTime = Clock::now();

    for (int frameIdx = 0; frameIdx < totalFrames; ++frameIdx) {
        // fade previous trails
        canvas *= trailFadeAlpha;

        // launch new rockets at defined interval
        if ((frameIdx % launchInterval) == 0) {
            double baseX = distX(rng);
            Particle p;
            p.x = (baseX - width/2.0) * widthSpread + width/2.0;
            p.y = 0;
            double ang = CV_PI/2;
            double speed = distLaunchSpeed(rng);
            p.vx = cos(ang) * speed;
            p.vy = sin(ang) * speed;
            p.age        = 0;
            p.generation = 1;
            p.drag       = rocketDrag;
            p.r = distColorBase(rng);
            p.g = distColorBase(rng);
            p.b = distColorBase(rng);
            p.alpha     = 0.2;
            p.prevX     = p.x;
            p.prevY     = p.y;
            p.size      = distSize(rng);
            particles.push_back(p);
        }

        int N = particles.size();
        // update physics
        for (int i = 0; i < N; ++i) {
            auto &p = particles[i];
            p.age++;
            if (p.generation == 2) {
                p.vx *= p.drag;
                p.vy -= gravity;
                p.x += p.vx;
                p.y += p.vy;
            } else {
                p.vy -= gravity;
                p.x += p.vx * p.drag;
                p.y += p.vy * p.drag;
            }
        }

        // spawn explosion particles
        vector<Particle> newParts;
        newParts.reserve(1000);
        for (int i = 0; i < N; ++i) {
            auto &p = particles[i];
            if (p.generation == 1 && p.age == 40) {
                for (int k = 0; k < 100; ++k) {
                    Particle q;
                    q.x = p.x; q.y = p.y;
                    double ang = distAng(rng);
                    double speed = distExplodeSpeed(rng);
                    q.vx = cos(ang) * speed;
                    q.vy = sin(ang) * speed;
                    q.age        = 0;
                    q.generation = 2;
                    q.drag       = explosionDrag;
                    q.r = saturate_cast<uchar>(p.r + distColorJ(rng));
                    q.g = saturate_cast<uchar>(p.g + distColorJ(rng));
                    q.b = saturate_cast<uchar>(p.b + distColorJ(rng));
                    q.alpha     = 1.0;
                    q.prevX     = q.x;
                    q.prevY     = q.y;
                    q.size      = distSize(rng);
                    newParts.push_back(q);
                }
            }
        }
        for (auto &q : newParts) particles.push_back(q);
        particles.erase(
            remove_if(particles.begin(), particles.end(), [&](const Particle &p){
                return p.generation == 1 && p.age > launchLifetime;
            }), particles.end());

        // draw particles
        for (auto &p : particles) {
            double dx1 = p.prevX - width/2.0;
            double dy1 = p.prevY - height/2.0;
            double dx2 = p.x      - width/2.0;
            double dy2 = p.y      - height/2.0;
            if (p.generation == 2) {
                dx1 *= depthScale;
                dy1 *= depthScale;
                dx2 *= depthScale;
                dy2 *= depthScale;
            }
            Point pt1(int(width/2.0 + dx1), height - int(height/2.0 + dy1));
            Point pt2(int(width/2.0 + dx2), height - int(height/2.0 + dy2));
            Scalar col(p.b * p.alpha, p.g * p.alpha, p.r * p.alpha);
            line(canvas, pt1, pt2, col, int(p.size), LINE_AA);
            if (pt2.inside(Rect(0,0,width,height)))
                canvas.at<Vec3b>(pt2.y, pt2.x) = Vec3b(
                    saturate_cast<uchar>(p.b * p.alpha),
                    saturate_cast<uchar>(p.g * p.alpha),
                    saturate_cast<uchar>(p.r * p.alpha));
            p.prevX = p.x; p.prevY = p.y;
            p.alpha -= particleFadeRate;
        }
        particles.erase(
            remove_if(particles.begin(), particles.end(), [&](const Particle &p){
                return p.generation == 2 && p.age > explosionLifetime;
            }), particles.end());

        writer.write(canvas);
        Mat disp;
        resize(canvas, disp, Size(width/2, height/2));
        imshow("Framebuffer", disp);
        if (waitKey(1) == 27) break;

        if ((frameIdx % 100) == 0) {
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

