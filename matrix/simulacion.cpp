#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <omp.h>
#include <ctime>
#include <sstream>
#include <iomanip>

// Constants
const int WIDTH = 1920;
const int HEIGHT = 1080;
const int FPS = 60;
const int DURATION_SECONDS = 60 * 60; 
const int FACTOR = 15;
const int SMALL_WIDTH = WIDTH / FACTOR;
const int SMALL_HEIGHT = HEIGHT / FACTOR;
const int NUM_LETTERS = 1550;
const std::string TEXT = "abcdefghijklmnopqrstuvwxyz0123456789";

// Structure to hold a Matrix rain strip
struct MatrixStrip {
    float x;
    float y;
    float velocity;
    int length;  // Length of the trail
    std::vector<float> intensities;  // Brightness of each character in the trail
    std::vector<char> chars;         // Characters in the trail
};

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0, 1);
std::uniform_int_distribution<> dis_text(0, TEXT.length() - 1);

// Get random character from the character set
char getRandomChar() {
    return TEXT[dis_text(gen)];
}

// Get current timestamp as string
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S_") 
       << std::setfill('0') << std::setw(3) << now_ms.count();
    
    return ss.str();
}

// Generate output filename with timestamp
std::string getOutputFilename() {
    std::string timestamp = getCurrentTimestamp();
    return timestamp + "_matrix_rain.mp4";
}

int main() {
    // Get output filename with timestamp
    std::string outputFilename = getOutputFilename();
    std::cout << "Output will be saved as: " << outputFilename << std::endl;
    
    // Initialize video writer
    cv::VideoWriter video(outputFilename, 
                        cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 
                        FPS, 
                        cv::Size(WIDTH, HEIGHT));
    
    if (!video.isOpened()) {
        std::cerr << "Could not open video file for writing" << std::endl;
        return -1;
    }
    
    // Initialize matrix strips
    std::vector<MatrixStrip> strips(NUM_LETTERS);
    for (auto& strip : strips) {
        strip.x = dis(gen) * SMALL_WIDTH;
        strip.y = -dis(gen) * SMALL_HEIGHT * 2;  // Start above screen
        strip.velocity = 0.2 + dis(gen) * 1.8;   // Random speed
        strip.length = 5 + static_cast<int>(dis(gen) * 15);  // Random trail length
        
        // Initialize the trail characters and intensities
        strip.chars.resize(strip.length);
        strip.intensities.resize(strip.length);
        for (int i = 0; i < strip.length; i++) {
            strip.chars[i] = getRandomChar();
            strip.intensities[i] = 1.0f - (static_cast<float>(i) / strip.length);  // Fade out
        }
    }
    
    // Initialize character grid for display
    std::vector<std::vector<char>> letterGrid(SMALL_WIDTH, std::vector<char>(SMALL_HEIGHT));
    std::vector<std::vector<float>> intensityGrid(SMALL_WIDTH, std::vector<float>(SMALL_HEIGHT, 0.0f));
    
    for (int x = 0; x < SMALL_WIDTH; x++) {
        for (int y = 0; y < SMALL_HEIGHT; y++) {
            letterGrid[x][y] = getRandomChar();
        }
    }
    
    // Create window for display
    cv::namedWindow("Matrix Rain", cv::WINDOW_NORMAL);
    cv::resizeWindow("Matrix Rain", WIDTH/2, HEIGHT/2);
    
    // Try to load a custom font if available
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = FACTOR/30.0;
    int fontThickness = 1;
    
    // Main animation loop
    int totalFrames = FPS * DURATION_SECONDS;
    for (int frame = 0; frame < totalFrames; frame++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Clear intensity grid
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < SMALL_HEIGHT; y++) {
            for (int x = 0; x < SMALL_WIDTH; x++) {
                intensityGrid[x][y] = 0.0f;
                
                // Random chance to change letter
                if (dis(gen) < 0.02) {
                    letterGrid[x][y] = getRandomChar();
                }
            }
        }
        
        // Update strip positions and update intensity grid
        #pragma omp parallel for
        for (auto& strip : strips) {
            strip.y += strip.velocity;
            
            // Reset if off screen
            if (strip.y - strip.length > SMALL_HEIGHT) {
                strip.x = dis(gen) * SMALL_WIDTH;
                strip.y = -dis(gen) * 50;  // Random position above screen
                strip.velocity = 0.2 + dis(gen) * 1.8;
                strip.length = 5 + static_cast<int>(dis(gen) * 15);
                
                // Reset characters
                strip.chars.resize(strip.length);
                strip.intensities.resize(strip.length);
                for (int i = 0; i < strip.length; i++) {
                    strip.chars[i] = getRandomChar();
                    strip.intensities[i] = 1.0f - (static_cast<float>(i) / strip.length);
                }
            }
            
            // Update intensity grid
            int x = static_cast<int>(strip.x);
            if (x >= 0 && x < SMALL_WIDTH) {
                for (int i = 0; i < strip.length; i++) {
                    int y = static_cast<int>(strip.y) - i;
                    if (y >= 0 && y < SMALL_HEIGHT) {
                        // The first character is brightest, tail fades out
                        float intensity = strip.intensities[i];
                        
                        // Update the grid with the highest intensity value
                        if (intensity > intensityGrid[x][y]) {
                            intensityGrid[x][y] = intensity;
                            letterGrid[x][y] = strip.chars[i];
                            
                            // Occasionally update the character in the trail
                            if (i == 0 && dis(gen) < 0.3) {
                                strip.chars[i] = getRandomChar();
                            }
                        }
                    }
                }
            }
        }
        
        // Create the large canvas
        cv::Mat largeCanvas(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        
        // Draw characters with proper scaling and intensity
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < SMALL_HEIGHT; y++) {
            for (int x = 0; x < SMALL_WIDTH; x++) {
                float intensity = intensityGrid[x][y];
                if (intensity > 0.0f) {
                    // Create color based on intensity (bright green for head, darker for tail)
                    int brightness = static_cast<int>(intensity * 255);
                    
                    // First character is white/bright green, rest are green with decreasing intensity
                    cv::Scalar color;
                    if (intensity > 0.9f) {
                        // First character is brighter and more white
                        color = cv::Scalar(brightness/2, brightness, brightness/2);
                    } else {
                        // Rest of trail is green with fading intensity
                        color = cv::Scalar(0, brightness, 0);
                    }
                    
                    // Scale coordinates for large canvas
                    cv::Point position(x * FACTOR + FACTOR/2, y * FACTOR + FACTOR);
                    
                    // Draw text with OpenCV's font
                    cv::putText(largeCanvas, std::string(1, letterGrid[x][y]), 
                              position, fontFace, fontScale, color, fontThickness, cv::LINE_AA);
                }
            }
        }
        
        // Apply post-processing effects
        cv::Mat blurCanvas;
        cv::GaussianBlur(largeCanvas, blurCanvas, cv::Size(3, 3), 0);
        
        // Apply brightness and contrast adjustments
        cv::Mat finalCanvas;
        cv::addWeighted(blurCanvas, 1.9, blurCanvas, 0, -20, finalCanvas);
        
        // Add glow effect
        cv::Mat glowCanvas;
        cv::GaussianBlur(finalCanvas, glowCanvas, cv::Size(21, 21), 5);
        cv::addWeighted(finalCanvas, 1.0, glowCanvas, 0.3, 0, finalCanvas);
                
        // Write frame to video
        video.write(finalCanvas);
        
        // Display current frame
        cv::imshow("Matrix Rain", finalCanvas);
        
        // Process any key presses (exit on ESC)
        int key = cv::waitKey(1);
        if (key == 1) // ESC key
            break;
        
        // Calculate how long to wait to maintain desired FPS
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        int wait_time = std::max(1, static_cast<int>(1000/FPS - duration));
        
        // Print progress every 5 seconds
        if (frame % (FPS * 5) == 0) {
            int seconds_completed = frame / FPS;
            std::cout << "Progress: " << seconds_completed << " seconds / " 
                      << DURATION_SECONDS << " seconds (" 
                      << (seconds_completed * 100 / DURATION_SECONDS) << "%)" << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
    }
    
    // Release video writer and close window
    video.release();
    cv::destroyAllWindows();
    
    std::cout << "Video saved as " << outputFilename << std::endl;
    return 0;
}
