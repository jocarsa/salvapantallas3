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
#include <fstream>  // For font file loading

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

// Struct to hold the font data
struct MatrixFont {
    cv::Mat glyphImages[128];  // Store pre-rendered images for ASCII characters
    int charWidth;
    int charHeight;
    
    MatrixFont(const std::string& fontPath, int size) {
        charWidth = FACTOR;
        charHeight = FACTOR;
        
        // Load the TTF font file
        std::ifstream fontFile(fontPath, std::ios::binary);
        if (!fontFile.is_open()) {
            std::cerr << "Failed to open font file: " << fontPath << std::endl;
            return;
        }
        
        // We're not using a font library, so we'll pre-render characters as images
        // For simplicity, we'll create basic green glyphs with a matrix-like appearance
        
        // Initialize all characters with empty images
        for (int i = 0; i < 128; i++) {
            glyphImages[i] = cv::Mat(charHeight, charWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        }
        
        // Generate simple matrix-like characters
        // This is a very simplified approach without actual TTF parsing
        for (char c : TEXT) {
            int index = static_cast<int>(c);
            if (index < 0 || index >= 128) continue;
            
            // Create a basic "matrix-like" representation of each character
            cv::Mat glyph(charHeight, charWidth, CV_8UC3, cv::Scalar(0, 0, 0));
            
            // Draw a simple representation - for real use, you'd need a proper TTF parser
            if (c >= '0' && c <= '9') {
                // Numbers
                cv::rectangle(glyph, cv::Point(charWidth/5, charHeight/5), 
                            cv::Point(4*charWidth/5, 4*charHeight/5), 
                            cv::Scalar(0, 255, 0), 1);
                
                // Add a distinctive mark for each number
                int num = c - '0';
                for (int i = 0; i < num; i++) {
                    int x = charWidth/3 + (i % 3) * charWidth/9;
                    int y = charHeight/3 + (i / 3) * charHeight/9;
                    cv::circle(glyph, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), -1);
                }
            } else {
                // Letters
                // Draw a simple representation based on the letter
                int letterIndex = (c >= 'a') ? (c - 'a') : (c - 'A');
                
                // Horizontal lines
                int numLines = 1 + (letterIndex % 3);
                for (int i = 0; i < numLines; i++) {
                    int y = charHeight/4 + i * charHeight/4;
                    cv::line(glyph, cv::Point(charWidth/5, y), 
                           cv::Point(4*charWidth/5, y), cv::Scalar(0, 255, 0), 1);
                }
                
                // Vertical lines
                int numVLines = 1 + ((letterIndex / 3) % 2);
                for (int i = 0; i < numVLines; i++) {
                    int x = charWidth/3 + i * charWidth/3;
                    cv::line(glyph, cv::Point(x, charHeight/5), 
                           cv::Point(x, 4*charHeight/5), cv::Scalar(0, 255, 0), 1);
                }
            }
            
            glyphImages[index] = glyph.clone();
        }
        
        fontFile.close();
        std::cout << "Font initialized with " << TEXT.length() << " characters" << std::endl;
    }
    
    // Draw a character from the font onto the canvas
    void drawChar(cv::Mat& canvas, char c, int x, int y, const cv::Scalar& color) {
        int index = static_cast<int>(c);
        if (index < 0 || index >= 128) return;
        
        // Get the glyph image
        cv::Mat glyph = glyphImages[index].clone();
        
        // Apply the color
        for (int j = 0; j < glyph.rows; j++) {
            for (int i = 0; i < glyph.cols; i++) {
                cv::Vec3b& pixel = glyph.at<cv::Vec3b>(j, i);
                if (pixel[1] > 0) {  // If the pixel is part of the character (green channel > 0)
                    pixel[0] = static_cast<uchar>(color[0]);
                    pixel[1] = static_cast<uchar>(color[1]);
                    pixel[2] = static_cast<uchar>(color[2]);
                }
            }
        }
        
        // Find the region of interest in the canvas
        cv::Rect roi(x, y, charWidth, charHeight);
        
        // Adjust ROI if it goes outside canvas boundaries
        if (x < 0 || y < 0 || x + charWidth > canvas.cols || y + charHeight > canvas.rows) {
            int x1 = std::max(0, x);
            int y1 = std::max(0, y);
            int x2 = std::min(canvas.cols, x + charWidth);
            int y2 = std::min(canvas.rows, y + charHeight);
            
            // Skip if ROI is completely outside
            if (x1 >= x2 || y1 >= y2) return;
            
            cv::Rect canvasRoi(x1, y1, x2 - x1, y2 - y1);
            cv::Rect glyphRoi(x1 - x, y1 - y, x2 - x1, y2 - y1);
            
            // Blend the glyph onto the canvas
            cv::Mat glyphPart = glyph(glyphRoi);
            cv::Mat canvasPart = canvas(canvasRoi);
            
            // Alpha blending (simple max value)
            for (int j = 0; j < glyphPart.rows; j++) {
                for (int i = 0; i < glyphPart.cols; i++) {
                    cv::Vec3b& canvasPixel = canvasPart.at<cv::Vec3b>(j, i);
                    cv::Vec3b& glyphPixel = glyphPart.at<cv::Vec3b>(j, i);
                    
                    // Simple blending - take the maximum value
                    canvasPixel[0] = std::max(canvasPixel[0], glyphPixel[0]);
                    canvasPixel[1] = std::max(canvasPixel[1], glyphPixel[1]);
                    canvasPixel[2] = std::max(canvasPixel[2], glyphPixel[2]);
                }
            }
        } else {
            // If ROI is completely inside canvas, use direct blending
            cv::Mat canvasPart = canvas(roi);
            
            // Alpha blending (simple max value)
            for (int j = 0; j < glyph.rows; j++) {
                for (int i = 0; i < glyph.cols; i++) {
                    cv::Vec3b& canvasPixel = canvasPart.at<cv::Vec3b>(j, i);
                    cv::Vec3b& glyphPixel = glyph.at<cv::Vec3b>(j, i);
                    
                    // Simple blending - take the maximum value
                    canvasPixel[0] = std::max(canvasPixel[0], glyphPixel[0]);
                    canvasPixel[1] = std::max(canvasPixel[1], glyphPixel[1]);
                    canvasPixel[2] = std::max(canvasPixel[2], glyphPixel[2]);
                }
            }
        }
    }
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
    // Load custom matrix font
    std::string fontPath = "matrix.ttf";
    MatrixFont matrixFont(fontPath, FACTOR);
    
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
                    int posX = x * FACTOR;
                    int posY = y * FACTOR;
                    
                    // Draw the character using our custom font renderer
                    matrixFont.drawChar(largeCanvas, letterGrid[x][y], posX, posY, color);
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
        if (key == 27) // ESC key
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
