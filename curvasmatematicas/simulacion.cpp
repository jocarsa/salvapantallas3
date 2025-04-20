#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <random>
#include <functional>
#include <chrono>
using namespace cv;
using namespace std;

// Curve function signature: takes parameter t, returns (x, y)
typedef function<Point2d(double)> CurveFunc;

// Define various mathematical curve functions and their names
vector<CurveFunc> curveFunctions = {
    // Spiral of Archimedes
    [](double t) {
        double a = 5.0, b = 2.0;
        double r = a + b * t;
        return Point2d(r * cos(t), r * sin(t));
    },
    // Astroid
    [](double t) {
        double a = 200.0;
        return Point2d(a * pow(cos(t), 3), a * pow(sin(t), 3));
    },
    // Cardioid
    [](double t) {
        double a = 150.0;
        return Point2d(a * (1 - cos(t)) * cos(t), a * (1 - cos(t)) * sin(t));
    },
    // Cissoid of Diocles
    [](double t) {
        double a = 150.0;
        double tan_t = tan(t);
        double x = a * tan_t * tan_t / (1 + tan_t * tan_t);
        double y = (tan_t != 0) ? tan_t * x : 0;
        return Point2d(x, y);
    },
    // Deltoid
    [](double t) {
        double R = 100.0;
        return Point2d(2 * R * cos(t) + R * cos(2 * t), 2 * R * sin(t) - R * sin(2 * t));
    },
    // Epicycloid
    [](double t) {
        double R = 120.0, r = 40.0;
        double x = (R + r) * cos(t) - r * cos((R + r) / r * t);
        double y = (R + r) * sin(t) - r * sin((R + r) / r * t);
        return Point2d(x, y);
    },
    // Harmonograph
    [](double t) {
        double A1 = 100, A2 = 120;
        double f1 = 1.5, f2 = 1.2;
        double d1 = 0.02, d2 = 0.03;
        double p1 = 0.5, p2 = 1.0;
        double x = A1 * sin(f1 * t + p1) * exp(-d1 * t);
        double y = A2 * sin(f2 * t + p2) * exp(-d2 * t);
        return Point2d(x, y);
    },
    // Heart curve
    [](double t) {
        double x = 16 * pow(sin(t), 3);
        double y = 13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t);
        return Point2d(x, -y);
    },
    // Lissajous
    [](double t) {
        double A = 200, B = 200;
        double a = 3, b = 2, delta = CV_PI / 2;
        return Point2d(A * sin(a * t + delta), B * sin(b * t));
    },
    // Rose curve (Rhodonea)
    [](double t) {
        double a = 150.0, k = 5.0;
        double r = a * cos(k * t);
        return Point2d(r * cos(t), r * sin(t));
    },
    // Lemniscate of Bernoulli
    [](double t) {
        double a = 200.0;
        double cos2t = cos(2 * t);
        if (cos2t < 0) return Point2d(0, 0);
        double r = a * sqrt(cos2t);
        return Point2d(r * cos(t), r * sin(t));
    },
    // Hypotrochoid
    [](double t) {
        double R = 150, r = 60, d = 80;
        double x = (R - r) * cos(t) + d * cos((R - r) / r * t);
        double y = (R - r) * sin(t) - d * sin((R - r) / r * t);
        return Point2d(x, y);
    },
    // Superellipse
    [](double t) {
        double a = 150, b = 100, n = 2.5;
        double x = a * copysign(pow(fabs(cos(t)), 2.0 / n), cos(t));
        double y = b * copysign(pow(fabs(sin(t)), 2.0 / n), sin(t));
        return Point2d(x, y);
    },
    // Logarithmic spiral
    [](double t) {
        double a = 0.2, b = 0.15;
        double r = a * exp(b * t);
        return Point2d(r * cos(t), r * sin(t));
    },
    // Nephroid
    [](double t) {
        double a = 150.0;
        double x = a * (3 * cos(t) - cos(3 * t));
        double y = a * (3 * sin(t) - sin(3 * t));
        return Point2d(x, y);
    },
    // Viviani's curve
    [](double t) {
        double a = 150.0;
        return Point2d(a * (1 + cos(t)), a * sin(t));
    },
    // Cloverleaf
    [](double t) {
        double n = 4.0, scale = 200.0;
        double x = sin(t) * cos(n * t);
        double y = cos(t) * sin(n * t);
        return Point2d(x * scale, y * scale);
    },
    // Butterfly curve
    [](double t) {
        double scale = 150.0;
        double expr = exp(cos(t)) - 2 * cos(4 * t) - pow(sin(t / 12), 5);
        double x = sin(t) * expr;
        double y = cos(t) * expr;
        return Point2d(x * scale, y * scale);
    },
    // Generalized Rhodonea
    [](double t) {
        double a = 150.0, k = 7.0;
        double r = a * cos(k * t);
        return Point2d(r * cos(t), r * sin(t));
    }
};
vector<string> curveNames = {
    "Spiral of Archimedes: r = a + b t",
    "Astroid: x = a cos^3(t), y = a sin^3(t)",
    "Cardioid: x = a(1 - cos t) cos t, y = a(1 - cos t) sin t",
    "Cissoid of Diocles: x = a tan^2 t/(1+tan^2 t), y = tan t * x",
    "Deltoid: x = 2R cos t + R cos 2t, y = 2R sin t - R sin 2t",
    "Epicycloid: x = (R+r)cos t - r cos((R+r)t/r), y = (R+r)sin t - r sin((R+r)t/r)",
    "Harmonograph: x = A1 sin(f1 t + p1)e^{-d1 t}, y = A2 sin(f2 t + p2)e^{-d2 t}",
    "Heart: x = 16 sin^3 t, y = 13 cos t -5 cos2t -2 cos3t - cos4t",
    "Lissajous: x = A sin(a t+δ), y = B sin(b t)",
    "Rose (Rhodonea): r = a cos(k t)",
    "Lemniscate of Bernoulli: r = a sqrt(cos 2t)",
    "Hypotrochoid: (R-r)cos t + d cos((R-r)t/r), (R-r)sin t - d sin((R-r)t/r)",
    "Superellipse: x = a sign(cos)^(2/n), y = b sign(sin)^(2/n)",
    "Logarithmic spiral: r = a e^{b t}",
    "Nephroid: x = a(3 cos t - cos 3t), y = a(3 sin t - sin 3t)",
    "Viviani's curve: x = a(1+cos t), y = a sin t",
    "Cloverleaf: x = sin t cos(n t), y = cos t sin(n t)",
    "Butterfly: x = sin t(e^{cos t} - 2 cos 4t - sin^5(t/12)), y = cos t(e^{cos t} - 2 cos 4t - sin^5(t/12))",
    "Generalized Rhodonea: r = a cos(k t)"
};

int main() {
    const int width = 1920, height = 1080;
    const int fps = 60;
    const double totalDuration = 3600.0; // 1 hour
    const int totalFrames = int(fps * totalDuration);
    VideoWriter writer("output.mp4", VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height));
    if (!writer.isOpened()) return -1;

    namedWindow("Curve", WINDOW_NORMAL);
    mt19937 rng(random_device{}());
    int n = curveFunctions.size();
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    shuffle(order.begin(), order.end(), rng);
    int idxPos = 0, currentIndex;
    auto nextCurve = [&]() {
        currentIndex = order[idxPos++];
        if (idxPos >= n) { shuffle(order.begin(), order.end(), rng); idxPos = 0; }
    };
    nextCurve();

    int animFrame = 0; auto startChange = chrono::steady_clock::now();
    vector<Point> points; Point startPoint; bool hasStart = false;

    for (int frameNum = 0; frameNum < totalFrames; ++frameNum) {
        double t = animFrame / double(fps);
        auto now = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(now - startChange).count();
        Point2d pt = curveFunctions[currentIndex](t);
        Point pixel(int(width/2 + pt.x), int(height/2 - pt.y));
        if (!hasStart) { startPoint = pixel; hasStart = true; }
        bool closedLoop = animFrame>fps && norm(pixel-startPoint)<5;
        bool oob = pixel.x<0||pixel.x>=width||pixel.y<0||pixel.y>=height;
        if (closedLoop||oob||elapsed>30) { nextCurve(); startChange=now; points.clear(); hasStart=false; animFrame=0; }
        else { points.push_back(pixel); animFrame++; }

        Mat frame(height, width, CV_8UC3, Scalar(0,0,0));
        // Centered grid
        Scalar gridCol(80,80,80);
        int step=100;
        for (int x=width/2; x<width; x+=step) line(frame,Point(x,0),Point(x,height),gridCol,1);
        for (int x=width/2; x>0; x-=step) line(frame,Point(x,0),Point(x,height),gridCol,1);
        for (int y=height/2; y<height; y+=step) line(frame,Point(0,y),Point(width,y),gridCol,1);
        for (int y=height/2; y>0; y-=step) line(frame,Point(0,y),Point(width,y),gridCol,1);
        // Axes
        Scalar axCol(150,150,150);
        line(frame,Point(0,height/2),Point(width,height/2),axCol,2);
        line(frame,Point(width/2,0),Point(width/2,height),axCol,2);
        #pragma omp parallel for
        for (int i=1;i<(int)points.size();++i) {
            double r=(double)i/points.size();
            Scalar col(128+127*sin(2*CV_PI*r),128+127*sin(2*CV_PI*r+2),128+127*sin(2*CV_PI*r+4));
            line(frame,points[i-1],points[i],col,3);
        }
        // Text
        string txt=curveNames[currentIndex];
        int b=0; auto sz=getTextSize(txt,FONT_HERSHEY_SIMPLEX,0.7,2,&b);
        putText(frame,txt,Point(10,height-10),FONT_HERSHEY_SIMPLEX,0.7,Scalar(200,200,200),2);
        imshow("Curve",frame);
        writer.write(frame);
        if(waitKey(1000/fps)=='q') break;
    }
    writer.release();
    destroyAllWindows();
    return 0;
}
