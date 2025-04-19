#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>  // for std::clamp

// ======================
// GENERAL CONSTANTS
// ======================
const int WIDTH = 1920;
const int HEIGHT = 1080;
const int FPS = 60;
const int DURATION_SEC = 60 * 60; // 60 minutes
const int TOTAL_FRAMES = FPS * DURATION_SEC;
const std::string OUTPUT_FILE = "galaxyfield_screensaver_cpu.mp4";

const float PERSPECTIVE = 600.0f;
const cv::Point2f CENTER_FACTOR(WIDTH / 2.0f, HEIGHT / 2.0f);

// ======================
// CONFIGURABLE PARAMETERS
// ======================
// Number of galaxies
const int NUM_GALAXIES = 1;
// Galaxy creation ranges
const int GALAXY_MIN_ARMS = 2;
const int GALAXY_MAX_ARMS = 4;
const int GALAXY_PARTICLES_PER_ARM = 48000;
const int GALAXY_MIN_SPIRAL_TURNS = 1;
const int GALAXY_MAX_SPIRAL_TURNS = 5;
const float GALAXY_MIN_RADIUS = 1500.0f;
const float GALAXY_MAX_RADIUS = 3200.0f;
const float GALAXY_MOVE_SPEED = 25.0f;

// Loose star parameters
int LOOSE_SPAWN_PER_FRAME = 50;
float LOOSE_SPEED = GALAXY_MOVE_SPEED;

// ======================
// FOG PARAMETERS
// ======================
bool FOG_ENABLED = true;
cv::Scalar FOG_COLOR(1, 1, 1);
float FOG_START = 0.0f;
float FOG_END = 15000.0f;
float FOG_DENSITY = 0.5f;

// ======================
// RANDOM UTILS
// ======================
std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<float> uni01(0.0f, 1.0f);

// Loose star distributions
std::uniform_real_distribution<float> looseX(-9000.0f, 9000.0f);
std::uniform_real_distribution<float> looseY(15000.0f, 20000.0f);
std::uniform_real_distribution<float> looseZ(-9000.0f, 9000.0f);

// ======================
// Rotation matrices
// ======================
cv::Matx33f rotation_x(float theta) { return {1,0,0, 0,cosf(theta),-sinf(theta), 0,sinf(theta),cosf(theta)}; }
cv::Matx33f rotation_y(float theta) { return {cosf(theta),0,sinf(theta), 0,1,0, -sinf(theta),0,cosf(theta)}; }
cv::Matx33f rotation_z(float theta) { return {cosf(theta),-sinf(theta),0, sinf(theta),cosf(theta),0, 0,0,1}; }
cv::Matx33f get_camera_rotation(int) { return cv::Matx33f::eye(); }

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
// Star color gradient
// ======================
cv::Scalar get_star_color(float r) {
    struct Key { float pos; cv::Scalar col; };
    static std::vector<Key> key_colors = {
        {0.0f, {255,165,0}}, {0.25f, {255,255,0}}, {0.5f, {255,255,255}}, {0.75f, {135,206,255}}, {1.0f, {255,0,0}}
    };
    Key c0 = key_colors[0], c1 = key_colors.back();
    for (size_t i = 0; i + 1 < key_colors.size(); ++i) {
        if (r <= key_colors[i+1].pos) { c0 = key_colors[i]; c1 = key_colors[i+1]; break; }
    }
    float t = (r - c0.pos) / (c1.pos - c0.pos);
    cv::Vec3f cv0(c0.col[0], c0.col[1], c0.col[2]);
    cv::Vec3f cv1(c1.col[0], c1.col[1], c1.col[2]);
    cv::Vec3f colVal = (1 - t) * cv0 + t * cv1;
    cv::Vec3i var(int(rng()%61)-30, int(rng()%61)-30, int(rng()%61)-30);
    cv::Vec3i tmp(int(colVal[0])+var[0], int(colVal[1])+var[1], int(colVal[2])+var[2]);
    tmp[0] = std::clamp(tmp[0], 0, 255);
    tmp[1] = std::clamp(tmp[1], 0, 255);
    tmp[2] = std::clamp(tmp[2], 0, 255);
    return {tmp[0], tmp[1], tmp[2]};
}

// ======================
// Star & Galaxy classes
// ======================
struct Star { cv::Vec3f pos; float radius; cv::Scalar color; };
struct LooseStar { cv::Vec3f pos; float size; cv::Scalar color; void update() { pos[1] -= LOOSE_SPEED; } };

struct Galaxy {
    cv::Vec3f center;
    float radius;
    int num_arms, particles_per_arm, spiral_turns;
    float ax, ay, az;
    float rsx, rsy, rsz;
    std::vector<Star> stars;
    Galaxy() {
        center = { float(rng()%2001)-1000, float(GALAXY_MAX_SPIRAL_TURNS*10000), float(rng()%2001)-1000 };
        radius = GALAXY_MIN_RADIUS + uni01(rng)*(GALAXY_MAX_RADIUS - GALAXY_MIN_RADIUS);
        num_arms = GALAXY_MIN_ARMS + int(uni01(rng)*(GALAXY_MAX_ARMS - GALAXY_MIN_ARMS + 1));
        particles_per_arm = GALAXY_PARTICLES_PER_ARM;
        spiral_turns = GALAXY_MIN_SPIRAL_TURNS + int(uni01(rng)*(GALAXY_MAX_SPIRAL_TURNS - GALAXY_MIN_SPIRAL_TURNS + 1));
        ax = uni01(rng)*2*M_PI; ay = uni01(rng)*2*M_PI; az = uni01(rng)*2*M_PI;
        rsx = (uni01(rng)-0.5f)*0.002f; rsy = (uni01(rng)-0.5f)*0.002f; rsz = (uni01(rng)-0.5f)*0.002f;
        generate();
    }
    void generate() {
        stars.clear();
        stars.push_back({{0,0,0}, uni01(rng)*200+100, {255,255,255}});
        float armOff = 2*M_PI/num_arms;
        for(int a=0; a<num_arms; ++a) for(int i=0; i<particles_per_arm; ++i) {
            float f = float(i)/particles_per_arm;
            float r = powf(f, 1.0f);
            float theta = f*2*M_PI*spiral_turns + a*armOff;
            r = std::clamp(r + (uni01(rng)-0.5f)*0.2f, 0.0f, 1.0f);
            float bx = r*cosf(theta)*radius, by = r*sinf(theta)*radius;
            float chaos = 25*(1-r) + 5*r;
            float ox = (uni01(rng)-0.5f)*2*chaos;
            float oy = (uni01(rng)-0.5f)*2*chaos;
            float oz = (uni01(rng)-0.5f)*2*chaos;
            float sr = uni01(rng)*0.1f + 0.1f;
            stars.push_back({{bx+ox, by+oy, oz}, sr, get_star_color(r)});
        }
    }
    void update() { ax+=rsx; ay+=rsy; az+=rsz; }
};

int main() {
    cv::VideoWriter writer;
    writer.open(OUTPUT_FILE, cv::VideoWriter::fourcc('a','v','c','1'), FPS, cv::Size(WIDTH, HEIGHT));
    cv::Mat frame(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0,0,0));
    cv::Mat temp(HEIGHT, WIDTH, CV_8UC3);

    std::vector<Galaxy> galaxies;
    galaxies.reserve(NUM_GALAXIES);
    for(int i=0; i<NUM_GALAXIES; ++i) galaxies.emplace_back();
    std::vector<LooseStar> loose_stars;

    for(int f=0; f<TOTAL_FRAMES; ++f) {
        cv::addWeighted(frame, 0.7, cv::Mat::zeros(frame.size(), frame.type()), 0.3, 0, frame);
        temp.setTo(cv::Scalar(0,0,0));
        auto cam = get_camera_rotation(f);
        std::vector<std::tuple<float, cv::Point, int, cv::Scalar>> draw_list;

        // Galaxies
        for(auto &G : galaxies) {
            G.update(); G.center[1] -= GALAXY_MOVE_SPEED;
            if(G.center[1] < -2200) G = Galaxy();
            for(auto &s : G.stars) {
                cv::Vec3f w = rotation_z(G.az)*(rotation_y(G.ay)*(rotation_x(G.ax)*s.pos)) + G.center;
                cv::Vec3f c = cam * w;
                float d = c[1] + PERSPECTIVE; if(d < 1e-3f) continue;
                cv::Scalar col = s.color;
                if(FOG_ENABLED) {
                    float dist = cv::norm(c);
                    float t = std::clamp((dist - FOG_START)/(FOG_END - FOG_START), 0.0f, 1.0f);
                    col = col*(1-powf(t,FOG_DENSITY)) + FOG_COLOR*powf(t,FOG_DENSITY);
                }
                int sz = std::max(1, int(s.radius * (PERSPECTIVE/d)));
                draw_list.emplace_back(d, project(c), sz, col);
            }
        }

        // Loose Stars - spawn
        for(int i=0; i<LOOSE_SPAWN_PER_FRAME; ++i) {
            LooseStar ls;
            ls.pos = { looseX(rng), looseY(rng), looseZ(rng) };
            ls.size = uni01(rng)*3.0f + 1.0f;
            ls.color = get_star_color(uni01(rng));
            loose_stars.push_back(ls);
        }
        // Loose Stars - update & remove
        for(auto it=loose_stars.begin(); it!=loose_stars.end();) {
            it->update();
            if(it->pos[1] < -2200) it = loose_stars.erase(it);
            else ++it;
        }
        // Loose Stars - draw
        for(auto &ls : loose_stars) {
            cv::Vec3f c = cam * ls.pos;
            float d = c[1] + PERSPECTIVE; if(d < 1e-3f) continue;
            cv::Scalar col = ls.color;
            if(FOG_ENABLED) {
                float dist = cv::norm(c);
                float t = std::clamp((dist - FOG_START)/(FOG_END - FOG_START), 0.0f, 1.0f);
                col = col*(1-powf(t,FOG_DENSITY)) + FOG_COLOR*powf(t,FOG_DENSITY);
            }
            int sz = std::max(1, int(ls.size * (PERSPECTIVE/d)));
            draw_list.emplace_back(d, project(c), sz, col);
        }

        // Render
        std::sort(draw_list.begin(), draw_list.end(), [](auto &a, auto &b){ return std::get<0>(a) < std::get<0>(b); });
        for(auto &item : draw_list) {
            auto [d, pt, sz, col] = item;
            if(pt.x>=0 && pt.x<WIDTH && pt.y>=0 && pt.y<HEIGHT)
                cv::circle(temp, pt, sz, col, -1, cv::LINE_AA);
        }

        // Glow & composite
        cv::Mat blurred;
        cv::GaussianBlur(temp, blurred, cv::Size(49,49), 0);
        cv::addWeighted(temp, 1.0, blurred, 2.0, 0.0, temp);
        cv::add(frame, temp, frame);

        writer.write(frame);
        cv::imshow("Framebuffer", frame);
        if(cv::waitKey(1) == 27) break;
    }
    writer.release();
    cv::destroyAllWindows();
    return 0;
}
