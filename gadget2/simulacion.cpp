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
    int type;          // 0..3 graphic type
    double r, a1, a2, v;
};

int main() {
    // Video parameters
    const int width = 1920, height = 1080;
    const int fps = 60;
    const int durationMin = 60;
    const int totalFrames = durationMin * 60 * fps;

    // Widget settings
    const int numCirculos = 20;
    const double baseRadiusFactor = 0.4;             // fraction of video height
    const double baseRadius = baseRadiusFactor * height;
    const double ringWidth = 5.0;
    const double textOffset = 15.0;
    // Increased alpha for more vivid colors
    const double alphaGeneral = 0.7;                 // alpha for strokes/fills

    // RNG setup
    mt19937 rng((unsigned)time(nullptr));
    uniform_int_distribution<int> distType(0, 3);
    uniform_real_distribution<double> distInitAng(0, 2 * M_PI);
    uniform_real_distribution<double> distTurn(-0.5, 0.5);

    // Generate circles
    vector<Circulo> circulos(numCirculos);
    for (int i = 0; i < numCirculos; ++i) {
        circulos[i].type = distType(rng);
        circulos[i].r    = baseRadius * double(i + 1) / numCirculos;
        circulos[i].a1   = distInitAng(rng);
        circulos[i].a2   = circulos[i].a1 + distInitAng(rng) * 0.5;
        circulos[i].v    = distTurn(rng);
    }

    // Prepare frame and overlay
    Mat frame(height, width, CV_8UC3), overlay;

    // Setup video writer
    VideoWriter writer("widget_gadget.mp4",
                       VideoWriter::fourcc('a','v','c','1'),
                       fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Error: could not open video writer" << endl;
        return -1;
    }

    // Display window
    namedWindow("Framebuffer", WINDOW_NORMAL);
    resizeWindow("Framebuffer", width/2, height/2);

    // Timing
    auto startTime = Clock::now();
    double t = 0.0;
    Point center(width/2, height/2);

    for (int frameIdx = 1; frameIdx <= totalFrames; ++frameIdx) {
        // black background
        frame.setTo(Scalar(0,0,0));
        overlay = frame.clone();

        // update parameters
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numCirculos; ++i) {
            Circulo &c = circulos[i];
            c.v  += distTurn(rng) * 0.001;
            c.a1 += cos(t * 0.01) * 0.001 + sin(c.v) * 0.01;
            c.a2 += sin(t * 0.01) * 0.001 + sin(c.v) * 0.01;
        }

        // draw each circle
        for (auto &c : circulos) {
            double a1 = c.a1;
            double a2 = c.a2;
            double diff = a2 - a1;
            if (diff < 0) diff += 2*M_PI;
            double startDeg = a1 * 180.0 / M_PI;
            double endDeg   = a2 * 180.0 / M_PI;

            // determine HSLA color: hue proportional to angle span
            double hue = (diff * 180.0 / M_PI) * 0.666;
            Mat hsv(1,1,CV_8UC3, Scalar(hue, 255, 128));
            Mat bgr;
            cvtColor(hsv, bgr, COLOR_HSV2BGR);
            Vec3b col = bgr.at<Vec3b>(0,0);
            Scalar drawColor(col[0], col[1], col[2]);

            // draw on overlay
            switch (c.type) {
                case 0:
                    ellipse(overlay, center, Size(c.r,c.r), 0.0,
                            startDeg, endDeg, drawColor,
                            int(ringWidth+0.2), LINE_AA);
                    break;
                case 1: {
                    vector<Point> pts;
                    pts.emplace_back(center);
                    int steps = max(2, int(diff/0.01));
                    for (int k=0; k<=steps; ++k) {
                        double theta = a1 + diff * k/steps;
                        pts.emplace_back(
                            center.x + int(c.r*cos(theta)),
                            center.y + int(c.r*sin(theta))
                        );
                    }
                    fillConvexPoly(overlay, pts.data(), (int)pts.size(), drawColor, LINE_AA);
                    break;
                }
                case 2: {
                    int segIdx = 0;
                    for (double ang=a1; ang<a2; ang += 0.05) {
                        double segEnd = min(a2, ang+0.02);
                        double dStart = ang*180.0/M_PI;
                        double dEnd   = segEnd*180.0/M_PI;
                        double lw = ringWidth+0.2;
                        if (segIdx%10==0) lw = ringWidth*4+0.2;
                        else if (segIdx%5==0) lw = ringWidth*2+0.2;
                        ellipse(overlay, center, Size(c.r,c.r), 0.0,
                                dStart, dEnd, drawColor, int(lw), LINE_AA);
                        segIdx++;
                    }
                    break;
                }
                case 3:
                    ellipse(overlay, center, Size(c.r,c.r), 0.0,
                            startDeg, endDeg, drawColor, 1, LINE_AA);
                    circle(overlay, Point(
                        center.x+int(c.r*cos(a1)), center.y+int(c.r*sin(a1))
                    ), 3, drawColor, FILLED, LINE_AA);
                    circle(overlay, Point(
                        center.x+int(c.r*cos(a2)), center.y+int(c.r*sin(a2))
                    ), 3, drawColor, FILLED, LINE_AA);
                    break;
            }

            // radial pointer and percentage text
            double px = center.x + cos(a1)*(c.r + textOffset);
            double py = center.y + sin(a1)*(c.r + textOffset);
            line(overlay, center, Point(int(px),int(py)), drawColor, 1, LINE_AA);
            int percent = int((diff*180.0/M_PI)/3.6 + 0.5);
            putText(overlay, to_string(percent) + "%", Point(int(px),int(py)),
                    FONT_HERSHEY_SIMPLEX, 0.7, drawColor, 1, LINE_AA);

            // blend overlay
            addWeighted(overlay, alphaGeneral, frame, 1-alphaGeneral, 0, frame);
            overlay = frame.clone();
        }

        // output frame
        writer.write(frame);
        Mat disp;
        resize(frame, disp, Size(width/2, height/2));
        imshow("Framebuffer", disp);
        if (waitKey(1)==27) break;

        // stats every 100 frames
        if (frameIdx%100==0) {
            auto now = Clock::now(); double elapsed = chrono::duration<double>(now - startTime).count();
            double pct = 100.0*frameIdx/totalFrames;
            double eta = elapsed*(totalFrames-frameIdx)/frameIdx;
            cout<<"Frame "<<frameIdx<<"/"<<totalFrames
                <<" ("<<fixed<<setprecision(2)<<pct<<"%)"
                <<" Elapsed: "<<elapsed<<"s"
                <<" ETA: "<<eta<<"s"<<endl;
        }
        t += 1.0;
    }

    writer.release();
    destroyAllWindows();
    return 0;
}
