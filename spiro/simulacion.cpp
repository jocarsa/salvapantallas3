#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <random>
#include <ctime>
#include <sstream>
#include <iostream>

// Convert HSL to BGR for OpenCV Scalar
cv::Scalar hslToBgr(double h, double s, double l) {
    s /= 100.0;
    l /= 100.0;
    double c = (1.0 - std::abs(2.0 * l - 1.0)) * s;
    double x = c * (1.0 - std::abs(fmod(h / 60.0, 2.0) - 1.0));
    double m = l - c / 2.0;
    double r = 0, g = 0, b = 0;
    if (h < 60)      { r = c;    g = x;    b = 0; }
    else if (h < 120){ r = x;    g = c;    b = 0; }
    else if (h < 180){ r = 0;    g = c;    b = x; }
    else if (h < 240){ r = 0;    g = x;    b = c; }
    else if (h < 300){ r = x;    g = 0;    b = c; }
    else             { r = c;    g = 0;    b = x; }
    r = (r + m) * 255;
    g = (g + m) * 255;
    b = (b + m) * 255;
    return cv::Scalar(b, g, r);
}

int main() {
    const int width = 1920;
    const int height = 1080;
    const int fps = 60;
    const int64_t totalFrames = 60LL * 60 * fps;  // 1 hour

    // Prepare filename with epoch prefix
    std::time_t now = std::time(nullptr);
    std::ostringstream oss;
    oss << now << "_spirograph.mp4";
    std::string outputFilename = oss.str();

    cv::VideoWriter writer(outputFilename,
        cv::VideoWriter::fourcc('m','p','4','v'), fps,
        cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer: " << outputFilename << std::endl;
        return -1;
    }

    cv::Mat traceCanvas(height, width, CV_8UC3, cv::Scalar(255,255,255));
    cv::Mat armsCanvas(height, width, CV_8UC3, cv::Scalar(255,255,255));
    cv::Mat finalFrame;

    std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_int_distribution<int> distN(2, 4);
    std::uniform_real_distribution<double> distRadius(0.0, height / 3.0);

    std::vector<double> denominators = {-8,-7,-6,-5,-4,-3,-2,2,3,4,5,6,7,8};

    int N;
    std::vector<double> angles;
    std::vector<double> radii;
    std::vector<double> speeds;
    double hue, saturation = 100.0, lightness = 50.0;
    cv::Scalar traceColor;

    auto resetSpiro = [&]() {
        N = distN(rng);
        angles.assign(N, 0.0);
        radii.resize(N);
        speeds.resize(N);
        for (int i = 0; i < N; ++i) {
            radii[i] = distRadius(rng);
            double d = denominators[rng() % denominators.size()];
            speeds[i] = M_PI / d / 10.0;
        }
        hue = std::uniform_real_distribution<double>(0.0, 360.0)(rng);
        traceColor = hslToBgr(hue, saturation, lightness);
        traceCanvas.setTo(cv::Scalar(255,255,255));
    };

    resetSpiro();

    bool drawingStarted = false;
    int prevX = width/2, prevY = height/2;
    int firstX = -1, firstY = -1;

    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Frame", width/2, height/2);

    for (int frame = 0; frame < totalFrames; ++frame) {
        armsCanvas.setTo(cv::Scalar(255,255,255));
        int x = width/2, y = height/2;

        // draw arms
        for (int j = 0; j < N; ++j) {
            double ca = std::cos(angles[j]);
            double sa = std::sin(angles[j]);
            int nx = x + static_cast<int>(ca * radii[j]);
            int ny = y + static_cast<int>(sa * radii[j]);
            cv::line(armsCanvas, cv::Point(x,y), cv::Point(nx,ny), cv::Scalar(0,0,0), 5, cv::LINE_AA);
            cv::circle(armsCanvas, cv::Point(x,y), 10, cv::Scalar(0,0,0), cv::FILLED, cv::LINE_AA);
            cv::circle(armsCanvas, cv::Point(nx,ny), 10, cv::Scalar(0,0,0), cv::FILLED, cv::LINE_AA);
            x = nx; y = ny;
            angles[j] += speeds[j];
        }

        // first point capture
        if (drawingStarted && firstX < 0) {
            firstX = prevX;
            firstY = prevY;
        }

        // draw trace
        if (drawingStarted) {
            cv::line(traceCanvas, cv::Point(prevX,prevY), cv::Point(x,y), traceColor, 5, cv::LINE_AA);
        } else {
            drawingStarted = true;
        }
        prevX = x; prevY = y;

        // merge trace and arms
        cv::Mat tf, af;
        traceCanvas.convertTo(tf, CV_32FC3, 1.0/255.0);
        armsCanvas.convertTo(af, CV_32FC3, 1.0/255.0);
        cv::multiply(tf, af, tf);
        tf.convertTo(finalFrame, CV_8UC3, 255.0);

        writer.write(finalFrame);
        cv::imshow("Frame", finalFrame);

        // update hue
        hue = std::fmod(hue + 0.5, 360.0);
        traceColor = hslToBgr(hue, saturation, lightness);

        // check loop completion
        if (firstX >= 0) {
            double dx = x - firstX;
            double dy = y - firstY;
            if (std::sqrt(dx*dx + dy*dy) <= 2.0) {
                resetSpiro();
                drawingStarted = false;
                firstX = firstY = -1;
                prevX = width/2;
                prevY = height/2;
            }
        }

        if (cv::waitKey(1) == 'q') break;
    }

    writer.release();
    cv::destroyAllWindows();
    std::cout << "Video saved to " << outputFilename << std::endl;
    return 0;
}

