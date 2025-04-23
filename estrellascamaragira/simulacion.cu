#include <opencv2/opencv.hpp>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// Basic Perlin noise implementation (2D)
class PerlinNoise {
public:
    PerlinNoise(unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count()) {
        p.resize(512);
        std::iota(p.begin(), p.begin() + 256, 0);
        std::mt19937 gen(seed);
        std::shuffle(p.begin(), p.begin() + 256, gen);
        for (int i = 0; i < 256; ++i) p[256 + i] = p[i];
    }

    double noise(double x, double y) const {
        int xi = static_cast<int>(std::floor(x)) & 255;
        int yi = static_cast<int>(std::floor(y)) & 255;
        double xf = x - std::floor(x);
        double yf = y - std::floor(y);
        double u = fade(xf);
        double v = fade(yf);
        int aa = p[p[xi] + yi];
        int ab = p[p[xi] + yi + 1];
        int ba = p[p[xi + 1] + yi];
        int bb = p[p[xi + 1] + yi + 1];
        double x1 = lerp(u, grad(aa, xf, yf), grad(ba, xf - 1, yf));
        double x2 = lerp(u, grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1));
        return lerp(v, x1, x2);
    }

private:
    std::vector<int> p;

    static double fade(double t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    static double lerp(double t, double a, double b) {
        return a + t * (b - a);
    }
    static double grad(int hash, double x, double y) {
        int h = hash & 7;
        double u = h < 4 ? x : y;
        double v = h < 4 ? y : x;
        return ((h & 1) ? -u : u) + ((h & 2) ? -2.0 * v : 2.0 * v);
    }
};

// Box with center, half-dimensions, and color
struct Box {
    cv::Vec3f center;
    cv::Vec3f dims;
    cv::Vec3b color;
    std::array<cv::Vec3f,8> vertices() const {
        float x = center[0], y = center[1], z = center[2];
        float dx = dims[0], dy = dims[1], dz = dims[2];
        return { cv::Vec3f{x-dx, y-dy, z-dz}, cv::Vec3f{x+dx, y-dy, z-dz},
                 cv::Vec3f{x+dx, y+dy, z-dz}, cv::Vec3f{x-dx, y+dy, z-dz},
                 cv::Vec3f{x-dx, y-dy, z+dz}, cv::Vec3f{x+dx, y-dy, z+dz},
                 cv::Vec3f{x+dx, y+dy, z+dz}, cv::Vec3f{x-dx, y+dy, z+dz} };
    }
};

// Rotation matrices
cv::Matx33f rotationX(float theta) {
    return {1,0,0,  0,cos(theta),-sin(theta),  0,sin(theta),cos(theta)};
}
cv::Matx33f rotationY(float theta) {
    return {cos(theta),0,sin(theta),  0,1,0,  -sin(theta),0,cos(theta)};
}
cv::Matx33f rotationZ(float theta) {
    return {cos(theta),-sin(theta),0,  sin(theta),cos(theta),0,  0,0,1};
}

// Parameters for rendering
const float PERSPECTIVE       = 600.0f;
const float PERLIN_SCALE      = 0.002f;
const float CLUSTER_OFFSET    = 50.0f;
// tweak this between 0.0 (no trails) and 1.0 (infinite trails)
const float TRAIL_PERSISTENCE = 0.3f;
// tweak this between 0.0 (no glow) and 1.0 (full glow)
const float GLOW_INTENSITY    = 1.0f;

cv::Matx33f getCameraRotation(int frameIdx) {
    float ax = 0.2f * std::sin(frameIdx * 0.01f);
    float ay = 0.15f * std::cos(frameIdx * 0.007f);
    float az = 0.1f * std::sin(frameIdx * 0.005f);
    return rotationZ(az) * rotationY(ay) * rotationX(ax);
}

// Draw a box into temp (CPU)
void drawShapeCPU(cv::Mat &temp, const Box &b, const cv::Matx33f &R, int width, int height) {
    auto verts = b.vertices();
    std::vector<cv::Vec3f> rv(8);
    for (int i = 0; i < 8; ++i) rv[i] = R * verts[i];
    std::vector<cv::Point> projected(8);
    for (int i = 0; i < 8; ++i) {
        float y = rv[i][1] + PERSPECTIVE;
        float f = PERSPECTIVE / std::max(y, 1e-3f);
        int X = static_cast<int>(rv[i][0] * f + width/2);
        int Y = static_cast<int>(rv[i][2] * f + height/2);
        projected[i] = {X, Y};
    }
    static const int faceIdx[6][4] = {{0,1,2,3},{4,5,6,7},{0,1,5,4},{2,3,7,6},{1,2,6,5},{0,3,7,4}};
    struct Face {int idx[4]; float depth;};
    std::vector<Face> faces;
    for (int f = 0; f < 6; ++f) {
        Face F;
        for (int k = 0; k < 4; ++k) F.idx[k] = faceIdx[f][k];
        float d = 0;
        for (int k = 0; k < 4; ++k) d += rv[F.idx[k]][1];
        F.depth = d * 0.25f;
        faces.push_back(F);
    }
    std::sort(faces.begin(), faces.end(), [](auto &a, auto &b){ return a.depth > b.depth; });
    for (auto &F : faces) {
        std::vector<cv::Point> pts;
        for (int k = 0; k < 4; ++k) pts.push_back(projected[F.idx[k]]);
        cv::fillConvexPoly(temp, pts, b.color, cv::LINE_AA);
    }
}

int main() {
    const int WIDTH = 960, HEIGHT = 540;
    const int FPS = 60;
    const int DURATION_SEC = 3*60; // 10 minutes
    const int TOTAL_FRAMES = FPS * DURATION_SEC;
    
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distX(-4000.0f, 4000.0f);
    std::uniform_real_distribution<float> distZ(-4000.0f, 4000.0f);
    std::uniform_real_distribution<float> distY(0.0f, 4200.0f);

    PerlinNoise perlin;
    int NUM_CUBES = 100000;
    
    std::vector<Box> shapes;
    shapes.reserve(NUM_CUBES);
    for (int i = 0; i < NUM_CUBES; ++i) {
        float x = distX(rng), z = distZ(rng);
        float y = distY(rng);
        double n = perlin.noise(x * PERLIN_SCALE, z * PERLIN_SCALE);
        double prob = (n + 1.0) / 2.0 * 10.0;
        if (std::uniform_real_distribution<>(0,1)(rng) < prob) {
            float angle = std::uniform_real_distribution<>(0, 2*M_PI)(rng);
            float r = std::uniform_real_distribution<>(0, CLUSTER_OFFSET)(rng);
            x += r * std::cos(angle);
            z += r * std::sin(angle);
        }
        shapes.push_back({{x,y,z}, {1,2,1}, {255,255,255}});
    }

    cv::VideoWriter writer("starfield_screensaver.mp4",
        cv::VideoWriter::fourcc('M','P','4','V'), FPS, cv::Size(WIDTH, HEIGHT));
    if (!writer.isOpened()) {
        std::cerr << "Error opening video writer" << std::endl;
        return -1;
    }

    cv::Mat frame = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    cv::Mat temp = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    cv::Mat blurred, glow;

    auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < TOTAL_FRAMES; ++f) {
        // fade out old frame to create trails
        cv::addWeighted(
            frame,
            TRAIL_PERSISTENCE,
            cv::Mat::zeros(frame.size(), frame.type()),
            1.0f - TRAIL_PERSISTENCE,
            0.0,
            frame
        );

        temp.setTo(cv::Scalar::all(0));
        std::sort(shapes.begin(), shapes.end(), [](const Box &a, const Box &b){ return a.center[1] > b.center[1]; });
        cv::Matx33f R = getCameraRotation(f);

        for (auto &b : shapes) {
            drawShapeCPU(temp, b, R, WIDTH, HEIGHT);
            b.center[1] -= 3.0f;
            if (b.center[1] < -HEIGHT/2) {
                float x = distX(rng), z = distZ(rng);
                float y = std::uniform_real_distribution<float>(2800.0f, 4200.0f)(rng);
                double n = perlin.noise(x * PERLIN_SCALE, z * PERLIN_SCALE);
                if (std::uniform_real_distribution<>(0,1)(rng) < (n + 1.0)/2.0 * 0.7) {
                    float angle = std::uniform_real_distribution<>(0,2*M_PI)(rng);
                    float r = std::uniform_real_distribution<>(0, CLUSTER_OFFSET)(rng);
                    x += r * std::cos(angle);
                    z += r * std::sin(angle);
                }
                b.center = {x,y,z};
            }
        }

        // blur and glow
        cv::GaussianBlur(temp, blurred, cv::Size(21,21), 0);
        cv::addWeighted(temp, 1.0f, blurred, GLOW_INTENSITY, 0.0f, glow);

        // accumulate
        cv::add(frame, glow, frame);

        writer.write(frame);
        if (f % 30 == 0) {
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            double progress = 100.0 * f / TOTAL_FRAMES;
            double eta = elapsed / (f+1) * (TOTAL_FRAMES - f);
            std::cout << "Frame " << f << "/" << TOTAL_FRAMES
                      << " (" << progress << "%), Elapsed=" << elapsed
                      << "s, ETA=" << eta << "s\r" << std::flush;
        }

        cv::imshow("Framebuffer", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    writer.release();
    cv::destroyAllWindows();
    std::cout << std::endl << "Video saved." << std::endl;
    return 0;
}

