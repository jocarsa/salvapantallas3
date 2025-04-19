#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <iostream>

// Parameters
const int WIDTH = 3840;
const int HEIGHT = 2160;
const int FPS = 60;
const int DURATION = 60*60; // 10 seconds for demonstration
const int CELL_SIZE = 1; // Cell size in pixels

// Function to initialize the grid
std::vector<std::vector<bool>> initializeGrid(int rows, int cols) {
    std::vector<std::vector<bool>> grid(rows, std::vector<bool>(cols, false));
    // Initialize grid with some pattern
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            grid[i][j] = rand() % 2;
        }
    }
    return grid;
}

// Function to update the grid based on the Game of Life rules
void updateGrid(const std::vector<std::vector<bool>>& currentGrid, std::vector<std::vector<bool>>& newGrid) {
    int rows = currentGrid.size();
    int cols = currentGrid[0].size();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int liveNeighbors = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    if (di == 0 && dj == 0) continue;
                    int ni = (i + di + rows) % rows;
                    int nj = (j + dj + cols) % cols;
                    liveNeighbors += currentGrid[ni][nj];
                }
            }
            if (currentGrid[i][j] && (liveNeighbors < 2 || liveNeighbors > 3)) newGrid[i][j] = false;
            else if (!currentGrid[i][j] && liveNeighbors == 3) newGrid[i][j] = true;
            else newGrid[i][j] = currentGrid[i][j];
        }
    }
}

// Function to render the grid to a frame
cv::Mat renderGrid(const std::vector<std::vector<bool>>& grid) {
    int rows = grid.size();
    int cols = grid[0].size();
    cv::Mat frame(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (grid[i][j]) {
                cv::rectangle(frame, cv::Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE), cv::Scalar(255, 255, 255), -1);
            }
        }
    }
    return frame;
}

int main() {
    int rows = HEIGHT / CELL_SIZE;
    int cols = WIDTH / CELL_SIZE;

    std::vector<std::vector<bool>> grid = initializeGrid(rows, cols);
    std::vector<std::vector<bool>> newGrid = grid;

    std::string video_path = "game_of_life.mp4";
    cv::VideoWriter video_writer(
        video_path,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        FPS,
        cv::Size(WIDTH, HEIGHT)
    );

    if (!video_writer.isOpened()) {
        std::cerr << "ERROR: Could not open video writer." << std::endl;
        return -1;
    }

    cv::namedWindow("Game of Life", cv::WINDOW_NORMAL);

    for (int frame = 0; frame < FPS * DURATION; ++frame) {
        cv::Mat frameImage = renderGrid(grid);
        video_writer.write(frameImage);
        cv::imshow("Game of Life", frameImage);
        if (cv::waitKey(1) >= 0) break; // Break the loop if a key is pressed
        updateGrid(grid, newGrid);
        std::swap(grid, newGrid);
    }

    video_writer.release();
    cv::destroyAllWindows();
    return 0;
}

