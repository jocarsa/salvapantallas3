#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <omp.h>
#include <vector>
#include <iostream>

// Parameters
const int WIDTH = 3840;
const int HEIGHT = 2160;
const int FPS = 60;
const int DURATION = 60 * 60; // 1 hour
const int CELL_SIZE = 4; // Cell size in pixels

// CUDA Kernel for updating the grid
__global__ void updateGridKernel(const uint8_t* currentGrid, uint8_t* newGrid, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        int liveNeighbors = 0;
        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                if (di == 0 && dj == 0) continue;
                int ni = (i + di + rows) % rows;
                int nj = (j + dj + cols) % cols;
                liveNeighbors += currentGrid[ni * cols + nj];
            }
        }
        if (currentGrid[i * cols + j] && (liveNeighbors < 2 || liveNeighbors > 3)) newGrid[i * cols + j] = 0;
        else if (!currentGrid[i * cols + j] && liveNeighbors == 3) newGrid[i * cols + j] = 1;
        else newGrid[i * cols + j] = currentGrid[i * cols + j];
    }
}

// Function to initialize the grid
std::vector<uint8_t> initializeGrid(int rows, int cols) {
    std::vector<uint8_t> grid(rows * cols, 0);
    // Initialize grid with some pattern
    for (int i = 0; i < rows * cols; ++i) {
        grid[i] = rand() % 2;
    }
    return grid;
}

// Function to render the grid to a frame
cv::Mat renderGrid(const uint8_t* grid, int rows, int cols) {
    cv::Mat frame(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (grid[i * cols + j]) {
                cv::rectangle(frame, cv::Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE), cv::Scalar(255, 255, 255), -1);
            }
        }
    }
    return frame;
}

int main() {
    int rows = HEIGHT / CELL_SIZE;
    int cols = WIDTH / CELL_SIZE;

    std::vector<uint8_t> grid = initializeGrid(rows, cols);
    std::vector<uint8_t> newGrid = grid;

    // Allocate memory on the GPU
    uint8_t *d_currentGrid, *d_newGrid;
    cudaMalloc(&d_currentGrid, rows * cols * sizeof(uint8_t));
    cudaMalloc(&d_newGrid, rows * cols * sizeof(uint8_t));

    // Copy initial grid to the GPU
    cudaMemcpy(d_currentGrid, grid.data(), rows * cols * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_newGrid, newGrid.data(), rows * cols * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Initialize OpenCV video writer with FourCC code directly
    int codec = cv::VideoWriter_fourcc('H', '2', '6', '4'); // or use 'M', 'P', '4', 'V' for MPEG-4 codec
    cv::VideoWriter videoWriter("game_of_life.mp4", codec, FPS, cv::Size(WIDTH, HEIGHT));

    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open video writer." << std::endl;
        return -1;
    }

    cv::namedWindow("Game of Life", cv::WINDOW_NORMAL);

    for (int frame = 0; frame < FPS * DURATION; ++frame) {
        // Update the grid on the GPU
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
        updateGridKernel<<<numBlocks, threadsPerBlock>>>(d_currentGrid, d_newGrid, rows, cols);

        // Copy the updated grid back to the host
        cudaMemcpy(newGrid.data(), d_newGrid, rows * cols * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        // Render the grid to a frame
        cv::Mat frameImage = renderGrid(newGrid.data(), rows, cols);

        // Write the frame to the video file
        videoWriter.write(frameImage);

        cv::imshow("Game of Life", frameImage);
        if (cv::waitKey(1) >= 0) break; // Break the loop if a key is pressed

        // Swap the grids
        std::swap(d_currentGrid, d_newGrid);
        std::swap(grid, newGrid);
    }

    // Clean up
    cudaFree(d_currentGrid);
    cudaFree(d_newGrid);
    videoWriter.release();
    cv::destroyAllWindows();
    return 0;
}

