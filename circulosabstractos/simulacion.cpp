#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using Clock = chrono::high_resolution_clock;

struct Cursor {
    double x, y;
    double angle;
    uchar r, g, b;
    double radius;
};

int main() {
    const int width = 1920;
    const int height = 1080;
    const int fps = 60;
    const int durationMin = 60;
    const int totalFrames = durationMin * 60 * fps;
    const int numCursors = 2000;

    const double alpha = 0.1;
    const double moveSpeed = 0.2;
    const double radiusMultiplier = 3.0;
    
    mt19937 rng((unsigned)time(nullptr));
    uniform_real_distribution<double> distX(0, width);
    uniform_real_distribution<double> distY(0, height);
    uniform_real_distribution<double> distAng(0, 2*M_PI);
    uniform_int_distribution<int> distColor(0, 255);
    uniform_real_distribution<double> distRadius(0, 10);
    uniform_real_distribution<double> distJitter(-0.5, 0.5);

    vector<Cursor> cursors(numCursors);
    for (auto &c : cursors) {
        c.x = distX(rng);
        c.y = distY(rng);
        c.angle = distAng(rng);
        c.r = static_cast<uchar>(distColor(rng));
        c.g = static_cast<uchar>(distColor(rng));
        c.b = static_cast<uchar>(distColor(rng));
        c.radius = distRadius(rng);
    }

    Mat canvas(height, width, CV_8UC3, Scalar(0,0,0));
    Mat rotated;

    VideoWriter writer("abstract_circles.mp4",
                       VideoWriter::fourcc('a','v','c','1'),
                       fps, Size(width, height));
    if (!writer.isOpened()) return -1;

    namedWindow("Framebuffer", WINDOW_NORMAL);
    resizeWindow("Framebuffer", width/2, height/2);

    auto startTime = Clock::now();

    for (int frameIdx = 1; frameIdx <= totalFrames; ++frameIdx) {
        Point2f centerPt(width/2.0f, height/2.0f);
        Mat rotMat = getRotationMatrix2D(centerPt, frameIdx * 0.001, 1.0);
        warpAffine(canvas, rotated, rotMat, canvas.size(),
                   INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
        canvas = rotated;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numCursors; ++i) {
            auto &c = cursors[i];
            c.x += cos(c.angle) * moveSpeed;
            c.y += sin(c.angle) * moveSpeed;
            c.r = saturate_cast<uchar>(c.r + distJitter(rng) * 5);
            c.g = saturate_cast<uchar>(c.g + distJitter(rng) * 5);
            c.b = saturate_cast<uchar>(c.b + distJitter(rng) * 5);
            c.radius += distJitter(rng);
            c.radius = max(0.0, min(10.0, c.radius));
            if (c.x < 0||c.x>width||c.y<0||c.y>height) c.angle += M_PI;
            int cx = int(c.x), cy = int(c.y);
            int rr = int(c.radius * radiusMultiplier);
            if (rr <= 0) continue;
            int rr2 = rr*rr;
            Scalar col(c.b, c.g, c.r);
            for (int dy = -rr; dy <= rr; ++dy) {
                int y = cy + dy;
                if (y < 0 || y >= height) continue;
                for (int dx = -rr; dx <= rr; ++dx) {
                    int x = cx + dx;
                    if (x < 0 || x >= width) continue;
                    if (dx*dx + dy*dy <= rr2) {
                        Vec3b &px = canvas.at<Vec3b>(y, x);
                        px[0] = saturate_cast<uchar>(c.b*alpha + px[0]*(1-alpha));
                        px[1] = saturate_cast<uchar>(c.g*alpha + px[1]*(1-alpha));
                        px[2] = saturate_cast<uchar>(c.r*alpha + px[2]*(1-alpha));
                    }
                }
            }
        }

        writer.write(canvas);
        Mat disp;
        resize(canvas, disp, Size(width/2, height/2));
        imshow("Framebuffer", disp);
        if (waitKey(1)==27) break;

        if (frameIdx % 100 == 0) {
            auto now = Clock::now();
            double elapsed = chrono::duration<double>(now - startTime).count();
            double pct = 100.0 * frameIdx / totalFrames;
            double eta = elapsed * (totalFrames - frameIdx) / frameIdx;
            cout<<"Frame "<<frameIdx<<"/"<<totalFrames
                <<" ("<<pct<<"%) Elapsed:"<<elapsed<<"s ETA:"<<eta<<"s"<<endl;
        }
    }

    writer.release();
    destroyAllWindows();
    return 0;
}

