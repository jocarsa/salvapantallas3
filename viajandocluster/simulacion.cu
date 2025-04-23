#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>  // for std::clamp, std::sort
#include <iostream>   // for debugging

// ======================
// GENERAL CONSTANTS
// ======================
const int WIDTH = 960;
const int HEIGHT = 540;
const int FPS = 30;
const int DURATION_SEC = 60 * 1; // 3 minutes
const int TOTAL_FRAMES = FPS * DURATION_SEC;
const std::string OUTPUT_FILE = "infinite_gruyere_starfield.mp4";

const float PERSPECTIVE = 600.0f;
const cv::Point2f CENTER_FACTOR(WIDTH / 2.0f, HEIGHT / 2.0f);

// ======================
// CONFIGURABLE PARAMETERS
// ======================
// "Gruyere cheese" parameters
const float MOVE_SPEED = 40.0f;
const int NUM_STARS = 500000;  
const float SPACE_SIZE = 20000.0f; 
const float NOISE_SCALE = 0.0001f; 
const float DENSITY_THRESHOLD = 0.4f; 
const float VOID_SHARPNESS = 3.0f; 
const float STRUCTURE_SCALE = 2.0f; 

// Infinite travel
const float STAR_RECYCLE_DISTANCE = -2000.0f;
const float STAR_PLACEMENT_DISTANCE = 20000.0f;

// ======================
// FOG PARAMETERS
// ======================
bool FOG_ENABLED = true;
cv::Scalar FOG_COLOR(0, 0, 3);
float FOG_START = 0.0f;
float FOG_END = 12000.0f;
float FOG_DENSITY = 0.5f;

// ======================
// RANDOM UTILS
// ======================
std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<float> uni01(0.0f, 1.0f);

// ======================
// Rotation matrices
// ======================
cv::Matx33f rotation_x(float theta) {
    return {1,0,0,
            0,cosf(theta),-sinf(theta),
            0,sinf(theta), cosf(theta)};
}
cv::Matx33f rotation_y(float theta) {
    return { cosf(theta),0,sinf(theta),
             0,           1,0,
            -sinf(theta),0, cosf(theta)};
}
cv::Matx33f rotation_z(float theta) {
    return { cosf(theta),-sinf(theta),0,
             sinf(theta), cosf(theta),0,
             0,           0,          1};
}

cv::Matx33f get_camera_rotation(int frame) {
    float t = frame * 0.001f;
    float x_angle = sinf(t * 0.2f) * 0.1f;
    float y_angle = sinf(t * 0.15f) * 0.15f;
    return rotation_x(x_angle) * rotation_y(y_angle);
}

// ======================
// Projection
// ======================
cv::Point project(const cv::Vec3f& v) {
    float y = std::max(v[1] + PERSPECTIVE, 1e-3f);
    float factor = PERSPECTIVE / y;
    float x = v[0] * factor + CENTER_FACTOR.x;
    float z = v[2] * factor + CENTER_FACTOR.y;
    return cv::Point(cvRound(x), cvRound(z));
}

// ======================
// Improved 3D Noise with fBm
// ======================
class ImprovedNoise {
private:
    std::vector<int> p;

    static float fade(float t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    static float lerp(float t, float a, float b) {
        return a + t * (b - a);
    }
    static float grad(int hash, float x, float y, float z) {
        int h = hash & 15;
        float u = h < 8 ? x : y;
        float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

public:
    ImprovedNoise() {
        p.resize(512);
        std::iota(p.begin(), p.begin() + 256, 0);
        std::shuffle(p.begin(), p.begin() + 256, rng);
        for (int i = 0; i < 256; i++)
            p[256 + i] = p[i];
    }

    // Classic Perlin noise
    float noise(float x, float y, float z) {
        int X = int(std::floor(x)) & 255;
        int Y = int(std::floor(y)) & 255;
        int Z = int(std::floor(z)) & 255;
        x -= std::floor(x);
        y -= std::floor(y);
        z -= std::floor(z);
        float u = fade(x), v = fade(y), w = fade(z);

        int A  = p[X] + Y,    AA = p[A] + Z,    AB = p[A+1] + Z;
        int B  = p[X+1] + Y,  BA = p[B] + Z,    BB = p[B+1] + Z;

        return lerp(w,
                    lerp(v,
                         lerp(u, grad(p[AA],   x,   y,   z),
                                  grad(p[BA],   x-1, y,   z)),
                         lerp(u, grad(p[AB],   x,   y-1, z),
                                  grad(p[BB],   x-1, y-1, z))),
                    lerp(v,
                         lerp(u, grad(p[AA+1], x,   y,   z-1),
                                  grad(p[BA+1], x-1, y,   z-1)),
                         lerp(u, grad(p[AB+1], x,   y-1, z-1),
                                  grad(p[BB+1], x-1, y-1, z-1))));
    }

    // Fractal Brownian Motion helper
    float fBm(float x, float y, float z,
              int octaves = 6,
              float lacunarity = 2.0f,
              float gain = 0.5f) {
        float sum = 0.0f, amp = 1.0f, freq = 1.0f, norm = 0.0f;
        for (int i = 0; i < octaves; ++i) {
            sum  += amp * noise(x * freq, y * freq, z * freq);
            norm += amp;
            amp  *= gain;
            freq *= lacunarity;
        }
        return sum / norm;
    }

    // Gruyere‑style fractal noise
    float getDensity(float x, float y, float z) {
        float nx = x * NOISE_SCALE;
        float ny = y * NOISE_SCALE;
        float nz = z * NOISE_SCALE;

        // base fractal + sharpening
        float baseF = fBm(nx, ny, nz, 6, 2.0f, 0.45f);
        float sharp = powf(baseF, VOID_SHARPNESS);

        // small‑scale fractal detail
        float detail = 0.2f * fBm(nx * 5.0f, ny * 5.0f, nz * 5.0f, 4, 2.0f, 0.5f);

        // warped structure
        float wx = 5.0f * fBm(nx*0.5f + 100.0f, ny*0.5f, nz*0.5f, 4, 2.0f, 0.5f);
        float wy = 5.0f * fBm(nx*0.5f, ny*0.5f + 100.0f, nz*0.5f, 4, 2.0f, 0.5f);
        float wz = 5.0f * fBm(nx*0.5f, ny*0.5f, nz*0.5f + 100.0f, 4, 2.0f, 0.5f);
        float structure = 0.3f * fBm((nx+wx)*STRUCTURE_SCALE,
                                     (ny+wy)*STRUCTURE_SCALE,
                                     (nz+wz)*STRUCTURE_SCALE,
                                     3, 2.0f, 0.5f);

        return (sharp + detail + structure) * STRUCTURE_SCALE + 0.2f;
    }
};

// ======================
// Star color gradient
// ======================
cv::Scalar get_star_color(float r) {
    struct Key { float pos; cv::Scalar col; };
    static std::vector<Key> key_colors = {
        {0.0f, {255,165,0}},
        {0.25f,{255,255,0}},
        {0.5f, {255,255,255}},
        {0.75f,{135,206,255}},
        {1.0f, {255,0,255}}
    };
    Key c0 = key_colors[0], c1 = key_colors.back();
    for (size_t i = 0; i+1 < key_colors.size(); ++i) {
        if (r <= key_colors[i+1].pos) {
            c0 = key_colors[i];
            c1 = key_colors[i+1];
            break;
        }
    }
    float t = (r - c0.pos) / (c1.pos - c0.pos);
    cv::Vec3f cv0(c0.col[0], c0.col[1], c0.col[2]);
    cv::Vec3f cv1(c1.col[0], c1.col[1], c1.col[2]);
    cv::Vec3f colVal = cv0 * (1-t) + cv1 * t;

    cv::Vec3i var(int(rng()%41)-20, int(rng()%41)-20, int(rng()%41)-20);
    cv::Vec3i tmp(int(colVal[0]) + var[0],
                  int(colVal[1]) + var[1],
                  int(colVal[2]) + var[2]);
    tmp[0] = std::clamp(tmp[0], 0, 255);
    tmp[1] = std::clamp(tmp[1], 0, 255);
    tmp[2] = std::clamp(tmp[2], 0, 255);
    return { tmp[0], tmp[1], tmp[2] };
}

// ======================
// Star struct
// ======================
struct Star {
    cv::Vec3f pos;
    float size;
    cv::Scalar color;
    float density;
};

// Place star ahead of camera
void place_star_ahead(Star& star,
                      const cv::Vec3f& camera_pos,
                      const cv::Vec3f& camera_dir,
                      ImprovedNoise& noise,
                      float distance) {
    std::uniform_real_distribution<float> offset_dist(-SPACE_SIZE/2, SPACE_SIZE/2);
    cv::Vec3f up(0,0,1);
    if (std::abs(camera_dir.dot(up)) > 0.9f)
        up = cv::Vec3f(1,0,0);
    cv::Vec3f right = camera_dir.cross(up);
    right /= cv::norm(right);
    up = right.cross(camera_dir);
    up /= cv::norm(up);

    for (int attempts = 0; attempts < 10; ++attempts) {
        float ox = offset_dist(rng), oz = offset_dist(rng);
        cv::Vec3f new_pos = camera_pos + camera_dir*distance + right*ox + up*oz;
        float d = noise.getDensity(new_pos[0], new_pos[1], new_pos[2]);
        if (d > DENSITY_THRESHOLD) {
            float sf = std::min(1.0f, (d - DENSITY_THRESHOLD)/(1.0f - DENSITY_THRESHOLD));
            star.pos = new_pos;
            star.density = d;
            star.size = 0.5f + sf*3.5f;
            return;
        }
    }
    // fallback
    star.pos = camera_pos + camera_dir*distance +
               cv::Vec3f(offset_dist(rng),
                         offset_dist(rng)/2,
                         offset_dist(rng));
    star.density = DENSITY_THRESHOLD;
    star.size = 0.5f;
}

// ======================
// Main
// ======================
int main() {
    cv::VideoWriter writer;
    writer.open(OUTPUT_FILE,
                cv::VideoWriter::fourcc('a','v','c','1'),
                FPS,
                cv::Size(WIDTH, HEIGHT));
    cv::Mat frame(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0,0,0));
    cv::Mat temp(HEIGHT, WIDTH, CV_8UC3);

    ImprovedNoise noise;
    std::vector<Star> stars;
    stars.reserve(NUM_STARS);

    std::cout << "Generating stars..." << std::endl;
    std::uniform_real_distribution<float> posGen(-SPACE_SIZE, SPACE_SIZE);
    int generated = 0;
    while (generated < NUM_STARS) {
        float x = posGen(rng), y = posGen(rng), z = posGen(rng);
        float d = noise.getDensity(x,y,z);
        if (d > DENSITY_THRESHOLD) {
            Star s;
            s.pos = {x,y,z};
            s.density = d;
            float sf = std::min(1.0f,(d-DENSITY_THRESHOLD)/(1.0f - DENSITY_THRESHOLD));
            s.size = 0.5f + sf*3.5f;
            s.color = get_star_color(uni01(rng));
            stars.push_back(s);
            generated++;
            if (generated % 10000 == 0)
                std::cout << "Generated " << generated << " stars...\n";
        }
    }
    std::cout << "Done. Starting render...\n";

    cv::Vec3f camera_pos(0, -SPACE_SIZE*0.5f, 0);
    cv::Vec3f camera_dir(0, 1, 0);

    for (int f = 0; f < TOTAL_FRAMES; ++f) {
        cv::addWeighted(frame, 0.7, cv::Mat::zeros(frame.size(), frame.type()), 0.3, 0, frame);
        temp.setTo(cv::Scalar(0,0,0));

        camera_pos += camera_dir * MOVE_SPEED;
        auto cam_rot = get_camera_rotation(f);

        std::vector<std::tuple<float, cv::Point, int, cv::Scalar>> draw_list;
        int recycled = 0;

        for (auto &s : stars) {
            cv::Vec3f rel = s.pos - camera_pos;
            if (rel.dot(camera_dir) < STAR_RECYCLE_DISTANCE) {
                place_star_ahead(s, camera_pos, camera_dir, noise, STAR_PLACEMENT_DISTANCE);
                s.color = get_star_color(uni01(rng));
                recycled++;
                continue;
            }
            float dist2 = rel.dot(rel);
            if (dist2 > FOG_END*FOG_END) continue;

            cv::Vec3f c = cam_rot * rel;
            float d = c[1] + PERSPECTIVE;
            if (d < 1e-3f) continue;

            cv::Scalar col = s.color;
            if (FOG_ENABLED) {
                float dist = std::sqrt(dist2);
                float t = std::clamp((dist-FOG_START)/(FOG_END-FOG_START), 0.0f, 1.0f);
                col = col*(1-powf(t,FOG_DENSITY)) + FOG_COLOR*powf(t,FOG_DENSITY);
            }

            int sz = std::max(1, int(s.size * (PERSPECTIVE/d)));
            sz = std::max(1, int(sz * (1.0f + s.density*0.5f)));
            draw_list.emplace_back(d, project(c), sz, col);
        }

        // sort farthest first for correct painter's order
        std::sort(draw_list.begin(), draw_list.end(),
                  [](auto &a, auto &b) {
                      return std::get<0>(a) > std::get<0>(b);
                  });

        for (auto &itm : draw_list) {
            auto [d, pt, sz, col] = itm;
            if (pt.x >= 0 && pt.x < WIDTH && pt.y >= 0 && pt.y < HEIGHT)
                cv::circle(temp, pt, sz, col, -1, cv::LINE_AA);
        }

        cv::Mat blurred;
        cv::GaussianBlur(temp, blurred, cv::Size(29,29), 0);
        cv::addWeighted(temp, 1.0, blurred, 2.0, 0, temp);
        cv::add(frame, temp, frame);

        writer.write(frame);
        cv::imshow("Infinite Gruyere Starfield", frame);
        if (cv::waitKey(1) == 27) break;
    }

    writer.release();
    cv::destroyAllWindows();
    return 0;
}

