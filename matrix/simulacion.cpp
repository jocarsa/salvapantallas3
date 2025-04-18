#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using Clock = chrono::high_resolution_clock;

int main() {
    // Video parameters
    const int width = 1920;
    const int height = 1080;
    const int fps = 60;
    const int durationMin = 60;
    const int totalFrames = durationMin * 60 * fps;

    // Matrix rain settings
    const int factor = 15;                      // downscale factor for small canvas
    const int smallW = width / factor;
    const int smallH = height / factor;
    const int numStrips = smallW;               // one strip per column
    const double speedMult = 2.0;               // vertical speed multiplier
    const double charChangeProb = 0.02;         // probability letter changes per cell

    // Random generators
    mt19937 rng((unsigned)time(nullptr));
    uniform_real_distribution<double> distVel(0.1, 1.0);
    uniform_int_distribution<int> distX(0, smallW - 1);
    uniform_int_distribution<int> distChar(0, 35); // 26 letters + 10 digits
    uniform_real_distribution<double> distProb(0.0, 1.0);

    // Character set: a-z0-9
    const string charset = "abcdefghijklmnopqrstuvwxyz0123456789";

    // Load strip image
    Mat strip = imread("matrixstrip2.png");
    if (strip.empty()) {
        cerr << "Error: cannot load matrixstrip2.png" << endl;
        return -1;
    }

    // Initialize strip positions and velocities
    vector<int> posX(numStrips), posY(numStrips);
    vector<double> vel(numStrips);
    for (int i = 0; i < numStrips; ++i) {
        posX[i] = i;              // one strip per column
        posY[i] = -strip.rows;    // start above top
        vel[i] = distVel(rng);
    }

    // Initialize character grid
    vector<vector<char>> grid(smallW, vector<char>(smallH));
    for (int x = 0; x < smallW; ++x)
        for (int y = 0; y < smallH; ++y)
            grid[x][y] = charset[distChar(rng)];

    // Create canvases
    Mat smallMat(smallH, smallW, CV_8UC3);
    Mat bigMat(height, width, CV_8UC3);

    // Setup FreeType for custom font
    Ptr<freetype::FreeType2> ft2;
    try {
        ft2 = freetype::createFreeType2();
        ft2->loadFontData("matrix.ttf", 0);
    } catch (...) {
        ft2 = nullptr;
        cerr << "Warning: FreeType2 unavailable, using Hershey." << endl;
    }
    int fontHeight = factor;

    // Video writer
    VideoWriter writer("matrix_rain.mp4",
                       VideoWriter::fourcc('a','v','c','1'),
                       fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "Error: cannot open video writer" << endl;
        return -1;
    }

    namedWindow("MatrixRain", WINDOW_NORMAL);
    resizeWindow("MatrixRain", width/2, height/2);
    auto startTime = Clock::now();

    for (int frame = 0; frame < totalFrames; ++frame) {
        // Clear small canvas
        smallMat = Scalar(0,0,0);

        // Draw strips on small canvas
        for (int i = 0; i < numStrips; ++i) {
            int sx = posX[i];
            int sy = posY[i];
            Rect roi(sx, sy, strip.cols, strip.rows);
            if (sy >= 0 && sy + strip.rows <= smallH) {
                strip.copyTo(smallMat(roi));
            }
            // update position
            posY[i] += int(vel[i] * speedMult);
            if (posY[i] > smallH) {
                posY[i] = -strip.rows;
                vel[i] = distVel(rng);
            }
        }

        // Clear big canvas
        bigMat = Scalar(0,0,0);

        // Map smallMat pixels to characters on bigMat
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < smallH; ++y) {
            for (int x = 0; x < smallW; ++x) {
                Vec3b px = smallMat.at<Vec3b>(y, x);
                int brightness = (px[0] + px[1] + px[2]) / 3;
                if (brightness == 0) continue;
                // Randomly change character
                if (distProb(rng) < charChangeProb)
                    grid[x][y] = charset[distChar(rng)];
                string ch(1, grid[x][y]);
                Point org(x*factor, y*factor + factor);
                Scalar color(px[0], px[1], px[2]);
                if (ft2) {
                    ft2->putText(bigMat, ch, org, fontHeight, color, 1, LINE_AA, false);
                } else {
                    putText(bigMat, ch, org, FONT_HERSHEY_SIMPLEX,
                            double(fontHeight)/30, color, 1, LINE_AA);
                }
            }
        }

        // Write and display
        writer.write(bigMat);
        Mat disp;
        resize(bigMat, disp, Size(width/2, height/2));
        imshow("MatrixRain", disp);
        if (waitKey(1) == 27) break;

        // Stats every 100 frames
        if (frame % 100 == 0) {
            auto now = Clock::now();
            double elapsed = chrono::duration<double>(now - startTime).count();
            double pct = 100.0 * frame / totalFrames;
            double eta = elapsed * (totalFrames - frame) / frame;
            cout << "Frame " << frame << "/" << totalFrames
                 << " (" << fixed << setprecision(2) << pct << "% )"
                 << " - Elapsed: " << elapsed << "s"
                 << " - ETA: " << eta << "s" << endl;
        }
    }

    writer.release();
    destroyAllWindows();
    return 0;
}

