#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iostream>

int main() {
    const int width = 1920;
    const int height = 1080;
    const int fps = 60;
    const int duration_sec = 60*60;
    const int total_frames = fps * duration_sec;
    const float alpha = 0.3f; // semi-transparent blend
    const int num_particles = 10;
    const float speed = 3.0f;

    // Initialize random generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distX(0.0f, width);
    std::uniform_real_distribution<float> distY(0.0f, height);
    std::uniform_real_distribution<float> distAngle(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> distJitter(-0.5f, 0.5f);
    std::uniform_int_distribution<int> distColor(0, 255);

    // Particle state
    std::vector<float> x(num_particles), y(num_particles);
    std::vector<float> d(num_particles), px(num_particles), py(num_particles);
    std::vector<cv::Vec3b> col(num_particles);

    // Initialize particles
    for (int i = 0; i < num_particles; ++i) {
        x[i]  = distX(gen);
        y[i]  = distY(gen);
        d[i]  = distAngle(gen);
        px[i] = x[i];
        py[i] = y[i];
        col[i] = cv::Vec3b((uchar)distColor(gen), (uchar)distColor(gen), (uchar)distColor(gen));
    }

    // Create video writer
    cv::VideoWriter writer("particles.mp4",
                             cv::VideoWriter::fourcc('m','p','4','v'),
                             fps,
                             cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Could not open the output video file for write\n";
        return -1;
    }

    // Create display window at half resolution
    const int dispW = width / 2;
    const int dispH = height / 2;
    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Frame", dispW, dispH);

    // Frame buffer (persistent for trails)
    cv::Mat frame = cv::Mat::zeros(height, width, CV_8UC3);

    // Timing
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int f = 1; f <= total_frames; ++f) {
        // Update particle positions & colors in parallel
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_particles; ++i) {
            if (x[i] < 0 || x[i] > width || y[i] < 0 || y[i] > height) {
                d[i] += static_cast<float>(M_PI);
            }
            x[i] += std::cos(d[i]) * speed;
            y[i] += std::sin(d[i]) * speed;
            d[i] += distJitter(gen) * 3.0f;
            for (int c = 0; c < 3; ++c) {
                int v = col[i][c] + static_cast<int>(std::round(distJitter(gen) * 4.0f));
                col[i][c] = static_cast<uchar>(cv::saturate_cast<uchar>(v));
            }
        }

        // Draw trails and particles
        for (int i = 0; i < num_particles; ++i) {
            cv::Rect roi_rect(cv::Point(static_cast<int>(x[i]), static_cast<int>(y[i])), cv::Size(3,3));
            roi_rect &= cv::Rect(0,0,width,height);
            if (roi_rect.area() > 0) {
                cv::Mat roi = frame(roi_rect);
                cv::Mat color_mat(roi_rect.size(), frame.type(), cv::Scalar(col[i]));
                cv::addWeighted(roi, 1.0f - alpha, color_mat, alpha, 0.0, roi);
            }
            cv::line(frame,
                     cv::Point(static_cast<int>(px[i]), static_cast<int>(py[i])),
                     cv::Point(static_cast<int>(x[i]),  static_cast<int>(y[i])),
                     cv::Scalar(255,255,255));
            px[i] = x[i];
            py[i] = y[i];
        }

        // Write frame to video
        writer.write(frame);

        // Display at half resolution
        cv::Mat disp;
        cv::resize(frame, disp, cv::Size(dispW, dispH));
        cv::imshow("Frame", disp);
        cv::waitKey(1);

        // Every 100 frames, print stats
        if (f % 100 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            double progress = static_cast<double>(f) / total_frames;
            double total_est = elapsed / progress;
            double remaining = total_est - elapsed;
            int percent = static_cast<int>(progress * 100);
            std::cout << "[Frame " << f << "/" << total_frames << "] "
                      << percent << "% complete, "
                      << "elapsed: " << elapsed << "s, "
                      << "remaining: " << remaining << "s\n";
        }
    }

    writer.release();
    cv::destroyAllWindows();
    return 0;
}

