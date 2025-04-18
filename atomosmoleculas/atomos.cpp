#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;
using namespace std::chrono;

// Simulation parameters
const int numLayers       = 6;        // number of depth layers
const int blurIncrement   = 10;       // px blur increase per layer
const int initialAtoms    = 100;
const int durationMinutes = 60;
const int fps             = 60;

const int frameWidth  = 1920;
const int frameHeight = 1080;
const int dispW       = frameWidth  / 2;
const int dispH       = frameHeight / 2;

const float friction             = 1.0f;
const float bondLength           = 50.0f;
const float slotThreshold        = 8.0f;
const float atomRadius           = 20.0f;
const float collisionRestitution = 1.0f;
const int bondThickness          = 6;

struct Atom {
    float x, y;
    float vx, vy;
    float theta;
    float omega;
    vector<float> bondAngles;
    vector<int> partner;
    Scalar color;

    Atom(int w, int h, mt19937 &rng) {
        uniform_real_distribution<float> distX(0, w);
        uniform_real_distribution<float> distY(0, h);
        uniform_real_distribution<float> distDir(0, 2 * M_PI);
        uniform_real_distribution<float> distSpeed(1.0f, 3.0f);
        uniform_real_distribution<float> distOmega(-0.05f, 0.05f);
        int type = rng() % 3;
        if (type == 0) { color = Scalar(0,0,255); bondAngles = {0.0f}; }
        else if (type == 1) { color = Scalar(0,255,0); bondAngles = {0.0f, (float)M_PI}; }
        else           { color = Scalar(255,0,0); bondAngles = {0.0f, float(2*M_PI/3), float(4*M_PI/3)}; }
        partner.assign(bondAngles.size(), -1);
        float angle = distDir(rng);
        float speed = distSpeed(rng);
        x = distX(rng); y = distY(rng);
        vx = cos(angle)*speed; vy = sin(angle)*speed;
        theta = distDir(rng);
        omega = distOmega(rng);
    }
};

int main() {
    mt19937 rng(42);

    // Create independent layers of atoms
    vector<vector<Atom>> layers(numLayers);
    for (int i = 0; i < initialAtoms; ++i) {
        int L = rng() % numLayers;
        layers[L].emplace_back(frameWidth, frameHeight, rng);
    }

    VideoWriter writer("output.mp4",
                       VideoWriter::fourcc('a','v','c','1'),
                       fps,
                       Size(frameWidth, frameHeight));
    namedWindow("Frame", WINDOW_NORMAL);
    resizeWindow("Frame", dispW, dispH);

    int totalFrames = durationMinutes * 60 * fps;
    int frameCount = 0;
    Mat display(frameHeight, frameWidth, CV_8UC3);

    // Start timing
    auto startTime = steady_clock::now();

    while (frameCount < totalFrames) {
        // Render each layer separately
        vector<Mat> layerImgs(numLayers);
        for (int L = 0; L < numLayers; ++L) {
            layerImgs[L] = Mat(frameHeight, frameWidth, CV_8UC3, Scalar(255,255,255));
            auto &atoms = layers[L];
            int n = atoms.size();

            // 1) movement & rotation update
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n; ++i) {
                Atom &a = atoms[i];
                a.vx *= friction;
                a.vy *= friction;
                a.x  += a.vx;
                a.y  += a.vy;
                // wall collisions
                if (a.x < atomRadius) { a.x = atomRadius; a.vx = -a.vx; }
                if (a.x > frameWidth - atomRadius) { a.x = frameWidth - atomRadius; a.vx = -a.vx; }
                if (a.y < atomRadius) { a.y = atomRadius; a.vy = -a.vy; }
                if (a.y > frameHeight - atomRadius) { a.y = frameHeight - atomRadius; a.vy = -a.vy; }
                // rotate only if free
                bool bonded = false;
                for (int p : a.partner) if (p >= 0) { bonded = true; break; }
                if (!bonded) {
                    a.omega *= friction;
                    a.theta += a.omega;
                }
            }

            // 2) collisions within this layer
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    Atom &A = atoms[i], &B = atoms[j];
                    bool bondedAB = false;
                    for (int p : A.partner) if (p == j) { bondedAB = true; break; }
                    if (bondedAB) continue;
                    float dx = B.x - A.x, dy = B.y - A.y;
                    float dist2 = dx*dx + dy*dy;
                    float minD = 2 * atomRadius;
                    if (dist2 < minD * minD) {
                        float dist = sqrt(dist2);
                        if (dist < 1e-4f) continue;
                        float nx = dx / dist, ny = dy / dist;
                        float overlap = minD - dist;
                        A.x -= nx * (overlap * 0.5f);
                        A.y -= ny * (overlap * 0.5f);
                        B.x += nx * (overlap * 0.5f);
                        B.y += ny * (overlap * 0.5f);
                        float rvx = A.vx - B.vx, rvy = A.vy - B.vy;
                        float velN = rvx * nx + rvy * ny;
                        if (velN < 0) {
                            float jimp = -(1.0f + collisionRestitution) * velN * 0.5f;
                            A.vx += jimp * nx; A.vy += jimp * ny;
                            B.vx -= jimp * nx; B.vy -= jimp * ny;
                        }
                    }
                }
            }

            // 3) bond detection within layer
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    Atom &A = atoms[i], &B = atoms[j];
                    for (int si = 0; si < (int)A.bondAngles.size(); ++si) if (A.partner[si] < 0) {
                        float ai = A.theta + A.bondAngles[si];
                        Point2f Pi(A.x + bondLength*cos(ai), A.y + bondLength*sin(ai));
                        for (int sj = 0; sj < (int)B.bondAngles.size(); ++sj) if (B.partner[sj] < 0) {
                            float aj = B.theta + B.bondAngles[sj];
                            Point2f Pj(B.x + bondLength*cos(aj), B.y + bondLength*sin(aj));
                            if (norm(Pi - Pj) < slotThreshold) {
                                A.partner[si] = j; B.partner[sj] = i;
                                goto next_pair;
                            }
                        }
                    }
                    next_pair: ;
                }
            }

            // 4) enforce bonds within layer
            for (int i = 0; i < n; ++i) {
                Atom &A = atoms[i];
                for (int si = 0; si < (int)A.bondAngles.size(); ++si) {
                    int j = A.partner[si]; if (j < 0) continue;
                    Atom &B = atoms[j];
                    Point2f dir(B.x - A.x, B.y - A.y);
                    float d = sqrt(dir.x*dir.x + dir.y*dir.y);
                    if (d > 1e-3f) {
                        Point2f ndir = dir * (1.0f / d);
                        float delta = d - bondLength;
                        A.x += ndir.x * (delta * 0.5f);
                        A.y += ndir.y * (delta * 0.5f);
                        B.x -= ndir.x * (delta * 0.5f);
                        B.y -= ndir.y * (delta * 0.5f);
                    }
                    float angleAB = atan2(B.y - A.y, B.x - A.x);
                    A.theta = angleAB - A.bondAngles[si];
                    int sj = -1;
                    for (int k = 0; k < (int)B.bondAngles.size(); ++k)
                        if (B.partner[k] == i) { sj = k; break; }
                    if (sj >= 0) B.theta = angleAB + CV_PI - B.bondAngles[sj];
                    A.omega = B.omega = 0;
                }
            }

            // 5) draw bonds and atoms onto layer image
            for (auto &a : atoms) {
                for (int si = 0; si < (int)a.bondAngles.size(); ++si) {
                    int j = a.partner[si]; if (j < 0) continue;
                    line(layerImgs[L], Point2f(a.x,a.y), Point2f(atoms[j].x,atoms[j].y), Scalar(0,0,0), bondThickness);
                }
            }
            for (auto &a : atoms) {
                circle(layerImgs[L], Point2f(a.x,a.y), atomRadius, Scalar(0,0,0), FILLED);
                circle(layerImgs[L], Point2f(a.x,a.y), atomRadius * 0.75f, a.color, FILLED);
            }
        }

        // Composite layers in back-to-front order: layer 2, then 1, then 0
        display.setTo(Scalar(255,255,255));
        for (int L = numLayers-1; L >= 0; --L) {
            Mat img = layerImgs[L];
            int k = L * blurIncrement;
            if (k > 0) {
                int ks = (k % 2 == 1) ? k : k+1;                 
                GaussianBlur(img, img, Size(ks, ks), 0);
            }
            Mat mask;
            cvtColor(img, mask, COLOR_BGR2GRAY);
            threshold(mask, mask, 254, 255, THRESH_BINARY_INV);
            img.copyTo(display, mask);
        }

        // 6) spawn a new atom every second into a random layer
        if (frameCount % fps == 0) {
            int L = rng() % numLayers;
            layers[L].emplace_back(frameWidth, frameHeight, rng);
        }

        writer.write(display);
        Mat disp;
        resize(display, disp, Size(dispW, dispH));
        imshow("Frame", disp);
        if (waitKey(1000/fps) == 1) break;

        frameCount++;

        // Logging every 100 frames
        if (frameCount % 100 == 0) {
            auto now = steady_clock::now();
            duration<double> elapsed = now - startTime;
            double elapsedSec = elapsed.count();
            double percent = 100.0 * frameCount / totalFrames;
            double estimatedTotal = elapsedSec / frameCount * totalFrames;
            double remainingSec = estimatedTotal - elapsedSec;
            int elMin = int(elapsedSec / 60);
            int elSec = int(elapsedSec) % 60;
            int remMin = int(remainingSec / 60);
            int remSec = int(remainingSec) % 60;
            cout << elMin << "m" << elSec << "s passed, "
                 << remMin << "m" << remSec << "s remaining ("
                 << fixed << setprecision(2) << percent << "%)" << endl;
        }
    }

    writer.release();
    destroyAllWindows();
    return 0;
}
