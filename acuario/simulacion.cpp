#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <filesystem>

// Random number generator
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

// Helper functions
double random_uniform(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

int random_int(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

double angle_difference(double beta, double alpha) {
    double difference = alpha - beta;
    while (difference > M_PI) {
        difference -= 2 * M_PI;
    }
    while (difference < -M_PI) {
        difference += 2 * M_PI;
    }
    return difference;
}

double angle_in_radians(double x1, double y1, double x2, double y2) {
    return std::atan2(y2 - y1, x2 - x1);
}

// Rectangle class for quadtree
class Rectangle {
public:
    double x, y; // Center coordinates
    double w, h; // Half width and height

    Rectangle(double x, double y, double w, double h) 
        : x(x), y(y), w(w), h(h) {}

    template<typename Entity>
    bool contains(const Entity& entity) const {
        return (x - w <= entity.x && entity.x <= x + w &&
                y - h <= entity.y && entity.y <= y + h);
    }

    bool intersects(const Rectangle& range) const {
        return !(range.x - range.w > x + w ||
                 range.x + range.w < x - w ||
                 range.y - range.h > y + h ||
                 range.y + range.h < y - h);
    }
};

// Forward declarations for Quadtree
class Pez;
class Comida;

// Quadtree class
template<typename T>
class Quadtree {
public:
    Rectangle boundary;
    int capacity;
    std::vector<T*> entities;
    bool divided;
    
    std::unique_ptr<Quadtree<T>> northeast;
    std::unique_ptr<Quadtree<T>> northwest;
    std::unique_ptr<Quadtree<T>> southeast;
    std::unique_ptr<Quadtree<T>> southwest;

    Quadtree(const Rectangle& boundary, int capacity) 
        : boundary(boundary), capacity(capacity), divided(false) {}

    void subdivide() {
        double x = boundary.x;
        double y = boundary.y;
        double w = boundary.w / 2;
        double h = boundary.h / 2;

        Rectangle ne(x + w, y - h, w, h);
        northeast = std::make_unique<Quadtree<T>>(ne, capacity);
        
        Rectangle nw(x - w, y - h, w, h);
        northwest = std::make_unique<Quadtree<T>>(nw, capacity);
        
        Rectangle se(x + w, y + h, w, h);
        southeast = std::make_unique<Quadtree<T>>(se, capacity);
        
        Rectangle sw(x - w, y + h, w, h);
        southwest = std::make_unique<Quadtree<T>>(sw, capacity);
        
        divided = true;
    }

    bool insert(T* entity) {
        if (!boundary.contains(*entity)) {
            return false;
        }

        if (entities.size() < capacity) {
            entities.push_back(entity);
            return true;
        } else {
            if (!divided) {
                subdivide();
            }

            if (northeast->insert(entity)) return true;
            if (northwest->insert(entity)) return true;
            if (southeast->insert(entity)) return true;
            if (southwest->insert(entity)) return true;
        }

        return false;
    }

    void query(const Rectangle& range, std::vector<T*>& found) {
        if (!boundary.intersects(range)) {
            return;
        } else {
            for (auto* entity : entities) {
                if (range.contains(*entity)) {
                    found.push_back(entity);
                }
            }

            if (divided) {
                northwest->query(range, found);
                northeast->query(range, found);
                southwest->query(range, found);
                southeast->query(range, found);
            }
        }
    }

    void clear() {
        entities.clear();
        divided = false;
        northeast.reset();
        northwest.reset();
        southeast.reset();
        southwest.reset();
    }
};

// Food class
class Comida {
public:
    double x, y;
    double radio;
    double a; // angle
    double v; // velocity
    bool visible;
    int vida;
    double transparencia;

    Comida(double x = -1, double y = -1, double radius = -1, double angle = -1) {
        this->x = (x >= 0) ? x : random_uniform(0, 3840);
        this->y = (y >= 0) ? y : random_uniform(0, 2160);
        this->radio = (radius >= 0) ? radius : random_uniform(5, 15);
        this->a = (angle >= 0) ? angle : random_uniform(0, 2 * M_PI);
        this->v = random_uniform(0, 0.25);
        this->visible = true;
        this->vida = 0;
        this->transparencia = 1.0;
    }

    void dibuja(cv::Mat& frame) {
        if (visible) {
            cv::Scalar color(255, 255, 255);
            int radius = std::max(static_cast<int>(radio), 1);
            cv::circle(frame, cv::Point(static_cast<int>(x), static_cast<int>(y)), radius, color, -1, cv::LINE_AA);
        }
    }

    void vive(cv::Mat& frame, std::vector<Comida>& comidas) {
        // Wandering logic
        if (random_uniform(0, 1) < 0.1) {
            a += (random_uniform(0, 1) - 0.5) * 0.2;
        }

        // Move based on direction and speed
        x += std::cos(a) * v;
        y += std::sin(a) * v;

        // Boundary conditions
        if (x < 0) {
            x = 0;
            a = -a;
        } else if (x > frame.cols) {
            x = frame.cols;
            a = -a;
        }
        if (y < 0) {
            y = 0;
            a = M_PI - a;
        } else if (y > frame.rows) {
            y = frame.rows;
            a = M_PI - a;
        }

        // Handle life cycle and division
        vida++;
        if (vida % 60 == 0 && radio >= 2) { // Divide every second if radius is >= 2
            divide(comidas);
        }

        // Remove particle if too small
        if (radio < 1) {
            visible = false;
        }

        dibuja(frame);
    }

    void divide(std::vector<Comida>& comidas) {
        // Create two new particles with half the radius and opposite directions
        double angle_offset = M_PI; // 180 degrees
        double child_radius = radio / 1.4;

        if (child_radius >= 1) {
            // Create two new particles in opposite directions
            comidas.emplace_back(x, y, child_radius, a);
            comidas.emplace_back(x, y, child_radius, std::fmod(a + angle_offset, 2 * M_PI));
        }

        // Mark this particle as invisible to "remove" it
        visible = false;
    }
};

// Fish class
class Pez {
public:
    double x, y;
    double a; // angle
    double edad;
    double tiempo;
    double avancevida;
    int sexo;
    cv::Scalar color;
    double energia;
    int direcciongiro;
    int numeroelementos;
    int numeroelementoscola;
    std::vector<int> colorr, colorg, colorb;
    double anguloanterior;
    double giro;
    double max_turn_rate;
    double target_angle;
    double flapping_frequency;
    double max_thrust;
    double drag_coefficient;
    double flapping_phase;
    double base_flapping_frequency;
    double base_max_thrust;
    double speed;
    bool is_chasing_food;
    bool is_avoiding_collision;
    std::vector<std::pair<double, double>> previous_positions;
    int stuck_threshold;
    int stuck_counter;
    bool is_stuck;

    Pez() {
        x = random_uniform(0, 3840);
        y = random_uniform(0, 2160);
        a = random_uniform(0, 2 * M_PI);
        edad = random_uniform(1, 2);
        tiempo = random_uniform(0, 1);
        avancevida = random_uniform(0.05, 0.1);
        sexo = random_int(0, 1);

        // Random color independent of sex
        color = cv::Scalar(random_int(0, 255), random_int(0, 255), random_int(0, 255));

        energia = random_uniform(0, 1);
        direcciongiro = random_int(-1, 1);
        numeroelementos = 10;
        numeroelementoscola = 5;
        
        // Initialize color vectors
        colorr.resize(numeroelementos + 1);
        colorg.resize(numeroelementos + 1);
        colorb.resize(numeroelementos + 1);
        
        for (int i = -1; i < numeroelementos; i++) {
            colorr[i+1] = std::max(0, std::min(static_cast<int>(color[0]) + random_int(-50, 50), 255));
            colorg[i+1] = std::max(0, std::min(static_cast<int>(color[1]) + random_int(-50, 50), 255));
            colorb[i+1] = std::max(0, std::min(static_cast<int>(color[2]) + random_int(-50, 50), 255));
        }
        
        anguloanterior = 0;
        giro = 0;
        max_turn_rate = random_uniform(0.005, 0.02);
        target_angle = a;

        flapping_frequency = random_uniform(0.5, 1.0);
        max_thrust = random_uniform(0.01, 0.03);
        drag_coefficient = random_uniform(0.02, 0.04);
        flapping_phase = 0.0;
        base_flapping_frequency = flapping_frequency;
        base_max_thrust = max_thrust;
        speed = random_uniform(0.5, 1.0);
        is_chasing_food = false;
        is_avoiding_collision = false;

        stuck_threshold = 5;
        stuck_counter = 0;
        is_stuck = false;
    }

    void dibuja(cv::Mat& frame) {
        // Set the main color based on energy level
        cv::Scalar color_main = energia > 0 ? color : cv::Scalar(128, 128, 128);

        // Mouth with breathing effect tied to flapping_phase
        int mouth_radius = std::max(static_cast<int>(sin(2 * M_PI * flapping_phase * 2) * 2 + 3), 1);
        int x_mouth = static_cast<int>(x + cos(a) * 5 * edad);
        int y_mouth = static_cast<int>(y + sin(a) * 5 * edad);
        
        // Oscillation perpendicular to direction
        double mouth_oscillation = sin(2 * M_PI * flapping_phase) * 2;
        x_mouth += static_cast<int>(cos(a + M_PI / 2) * mouth_oscillation);
        y_mouth += static_cast<int>(sin(a + M_PI / 2) * mouth_oscillation);
        
        cv::circle(frame, cv::Point(x_mouth, y_mouth), mouth_radius, color_main, -1, cv::LINE_AA);

        // Eyes
        for (int i = -1; i < numeroelementos; i++) {
            if (i == 1) {
                for (int sign : {-1, 1}) {
                    int x_eye = static_cast<int>(x + sign * cos(a + M_PI / 2) * 4 * edad - i * cos(a) * edad);
                    int y_eye = static_cast<int>(y + sign * sin(a + M_PI / 2) * 4 * edad - i * sin(a) * edad);

                    // Oscillation perpendicular to direction
                    double eye_oscillation = sin((i / 5.0) - 2 * M_PI * flapping_phase) * 4;
                    x_eye += static_cast<int>(cos(a + M_PI / 2) * eye_oscillation);
                    y_eye += static_cast<int>(sin(a + M_PI / 2) * eye_oscillation);
                    
                    int radius_eye = std::max(static_cast<int>((edad * 0.4 * (numeroelementos - i) + 1) / 3), 1);
                    cv::circle(frame, cv::Point(x_eye, y_eye), radius_eye, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
                }
            }
        }

        // Fins
        for (int i = -1; i < numeroelementos; i++) {
            if (i == numeroelementos / 2 || i == static_cast<int>(numeroelementos / 1.1)) {
                for (int sign : {-1, 1}) {
                    int x_fin = static_cast<int>(x + sign * cos(a + M_PI / 2) * 0.3 * edad - i * cos(a) * edad);
                    int y_fin = static_cast<int>(y + sign * sin(a + M_PI / 2) * 0.3 * edad - i * sin(a) * edad);

                    // Oscillation perpendicular to direction
                    double fin_oscillation = sin((i / 5.0) - 2 * M_PI * flapping_phase) * 4;
                    x_fin += static_cast<int>(cos(a + M_PI / 2) * fin_oscillation);
                    y_fin += static_cast<int>(sin(a + M_PI / 2) * fin_oscillation);
                    
                    cv::Size axes(
                        std::max(static_cast<int>((edad * 0.4 * (numeroelementos - i) + 1) * 2), 1),
                        std::max(static_cast<int>((edad * 0.4 * (numeroelementos - i) + 1)), 1)
                    );
                    double angle = (a + M_PI / 2 - cos(2 * M_PI * flapping_phase * 2) * sign) * 180.0 / M_PI;
                    cv::ellipse(frame, cv::Point(x_fin, y_fin), axes, angle, 0, 360, color_main, -1, cv::LINE_AA);
                }
            }
        }

        // Body
        for (int i = -1; i < numeroelementos; i++) {
            int x_body = static_cast<int>(x - i * cos(a) * 2 * edad);
            int y_body = static_cast<int>(y - i * sin(a) * 2 * edad);
            
            // Oscillation perpendicular to direction
            double body_oscillation = sin((i / 5.0) - 2 * M_PI * flapping_phase) * 4;
            x_body += static_cast<int>(cos(a + M_PI / 2) * body_oscillation);
            y_body += static_cast<int>(sin(a + M_PI / 2) * body_oscillation);
            
            int radius_body = std::max(static_cast<int>((edad * 0.4 * (numeroelementos - i) + 1) / 1), 1);
            cv::Scalar color_body(colorb[i+1], colorg[i+1], colorr[i+1]); // BGR in OpenCV
            cv::circle(frame, cv::Point(x_body, y_body), radius_body, color_body, -1, cv::LINE_AA);
        }

        // Tail
        for (int i = numeroelementos; i < numeroelementos + numeroelementoscola; i++) {
            int x_tail = static_cast<int>(x - (i - 3) * cos(a) * 2 * edad);
            int y_tail = static_cast<int>(y - (i - 3) * sin(a) * 2 * edad);
            
            // Oscillation perpendicular to direction
            double tail_oscillation = sin((i / 5.0) - 2 * M_PI * flapping_phase) * 4;
            x_tail += static_cast<int>(cos(a + M_PI / 2) * tail_oscillation);
            y_tail += static_cast<int>(sin(a + M_PI / 2) * tail_oscillation);
            
            int radius_tail = std::max(static_cast<int>(-edad * 0.4 * (numeroelementos - i) * 2 + 1), 1);
            cv::circle(frame, cv::Point(x_tail, y_tail), radius_tail, color_main, -1, cv::LINE_AA);
        }
    }

    void vive(cv::Mat& frame, Quadtree<Pez>& quadtree) {
        if (random_uniform(0, 1) < 0.002) {
            direcciongiro = -direcciongiro;
        }
        if (energia > 0) {
            tiempo += avancevida;
            mueve(quadtree);
        }
        energia -= 0.00003;
        edad += 0.00001;
        if (edad > 3) {
            energia = 0;
        }
        if (energia > 0) {
            dibuja(frame);
        }
    }

    // Modify the Pez::mueve method to improve collision avoidance
void mueve(Quadtree<Pez>& quadtree) {
    is_avoiding_collision = false; // Reset collision avoidance flag

    // Avoidance logic using quadtree
    double perception_radius = 50; // Adjust based on desired interaction range
    Rectangle perception_range(x, y, perception_radius, perception_radius);

    // Query the quadtree for nearby fishes
    std::vector<Pez*> nearby_fishes;
    quadtree.query(perception_range, nearby_fishes);

    // Remove self from the list if present
    nearby_fishes.erase(std::remove_if(nearby_fishes.begin(), nearby_fishes.end(),
                                     [this](Pez* fish) { return fish == this; }),
                      nearby_fishes.end());

    // ---- IMPROVED COLLISION AVOIDANCE ----
    // Calculate the potential collision point with nearby fish
    for (auto* other_fish : nearby_fishes) {
        // Skip dead fish
        if (other_fish->energia <= 0) continue;
        
        double dist = std::hypot(x - other_fish->x, y - other_fish->y);
        
        // Only consider fish that are "close enough" to potentially collide
        if (dist < perception_radius) {
            // Calculate the relative position and velocity vectors
            double rel_x = other_fish->x - x;
            double rel_y = other_fish->y - y;
            
            // Calculate our velocity vector
            double vel_x = speed * std::cos(a);
            double vel_y = speed * std::sin(a);
            
            // Calculate other fish's velocity vector
            double other_vel_x = other_fish->speed * std::cos(other_fish->a);
            double other_vel_y = other_fish->speed * std::sin(other_fish->a);
            
            // Calculate relative velocity vector
            double rel_vel_x = other_vel_x - vel_x;
            double rel_vel_y = other_vel_y - vel_y;
            
            // Calculate the dot product of relative position and velocity
            double dot_product = rel_x * rel_vel_x + rel_y * rel_vel_y;
            
            // If dot product is positive, the fish are moving away from each other
            // If negative, they're moving towards each other
            if (dot_product < 0) {
                // Calculate the square of the minimum distance
                double rel_vel_squared = rel_vel_x * rel_vel_x + rel_vel_y * rel_vel_y;
                double min_dist_squared = (rel_x * rel_vel_y - rel_y * rel_vel_x) * 
                                          (rel_x * rel_vel_y - rel_y * rel_vel_x) / rel_vel_squared;
                
                // If the minimum distance is small enough, they might collide
                if (min_dist_squared < (edad * 5)*(edad * 5)) {  // Using fish size for collision radius
                    // Calculate the time to reach the minimum distance point
                    double t = -dot_product / rel_vel_squared;
                    
                    // If this time is in the near future, take avoidance action
                    if (t > 0 && t < 1.0) {  // Adjust time threshold as needed
                        is_avoiding_collision = true;
                        
                        // Calculate avoidance direction (perpendicular to approaching direction)
                        double approach_angle = std::atan2(rel_y, rel_x);
                        double perp_angle = approach_angle + M_PI/2;
                        
                        // Choose direction based on which side we're on
                        double cross_product = rel_x * vel_y - rel_y * vel_x;
                        if (cross_product < 0) {
                            perp_angle = approach_angle - M_PI/2;
                        }
                        
                        // Set target angle to avoid collision
                        target_angle = perp_angle;
                        
                        // Increase turn rate for faster reaction
                        max_turn_rate *= 1.5;
                        break;  // Only handle the most imminent collision
                    }
                }
            }
        }
    }

    // ---- PHYSICAL COLLISION RESOLUTION ----
    // Check for actual collisions and resolve them
    for (auto* other_fish : nearby_fishes) {
        // Skip dead fish
        if (other_fish->energia <= 0) continue;
        
        double fish_radius = edad * 5;  // Approximate fish body size
        double other_radius = other_fish->edad * 5;
        double min_distance = fish_radius + other_radius;
        
        double dist = std::hypot(x - other_fish->x, y - other_fish->y);
        
        // If there's a collision (fish overlap)
        if (dist < min_distance && dist > 0) {  // dist > 0 avoids division by zero
            // Calculate normal vector (direction of collision)
            double nx = (other_fish->x - x) / dist;
            double ny = (other_fish->y - y) / dist;
            
            // Calculate overlap
            double overlap = min_distance - dist;
            
            // Move both fish apart based on their relative masses (using fish size as mass)
            double total_mass = fish_radius + other_radius;
            double ratio1 = other_radius / total_mass;
            double ratio2 = fish_radius / total_mass;
            
            // Move this fish
            x -= nx * overlap * ratio1;
            y -= ny * overlap * ratio1;
            
            // Move other fish (using atomic operations for thread safety)
            #pragma omp atomic update
            other_fish->x += nx * overlap * ratio2;
            #pragma omp atomic update
            other_fish->y += ny * overlap * ratio2;
            
            // Adjust velocity (like a soft elastic collision)
            // Calculate relative velocity along normal
            double vel_x1 = speed * std::cos(a);
            double vel_y1 = speed * std::sin(a);
            double vel_x2 = other_fish->speed * std::cos(other_fish->a);
            double vel_y2 = other_fish->speed * std::sin(other_fish->a);
            
            double relVelX = vel_x2 - vel_x1;
            double relVelY = vel_y2 - vel_y1;
            
            double relVelDotNormal = relVelX * nx + relVelY * ny;
            
            // If objects moving toward each other
            if (relVelDotNormal > 0) {
                // Coefficient of restitution (bounciness)
                double e = 0.5;
                
                // Impulse scalar
                double impulse = (1 + e) * relVelDotNormal / total_mass;
                
                // Apply impulse to velocities
                double impulse_x = impulse * nx;
                double impulse_y = impulse * ny;
                
                // Update this fish's velocity
                vel_x1 += impulse_x * ratio1;
                vel_y1 += impulse_y * ratio1;
                
                // Update other fish's velocity (using atomic operations)
                #pragma omp critical
                {
                    double new_vel_x2 = vel_x2 - impulse_x * ratio2;
                    double new_vel_y2 = vel_y2 - impulse_y * ratio2;
                    
                    // Update other fish's speed and angle
                    other_fish->speed = std::hypot(new_vel_x2, new_vel_y2);
                    other_fish->a = std::atan2(new_vel_y2, new_vel_x2);
                }
                
                // Update this fish's speed and angle
                speed = std::hypot(vel_x1, vel_y1);
                a = std::atan2(vel_y1, vel_x1);
            }
        }
    }

    // Update flapping phase
    flapping_phase += flapping_frequency * avancevida;

    // Compute thrust (only positive values)
    double thrust = max_thrust * std::max(std::sin(2 * M_PI * flapping_phase), 0.0);

    // Compute drag
    double drag = drag_coefficient * speed;

    // Update speed
    speed += thrust - drag;
    speed = std::max(speed, 0.0);

    // Cap the speed to prevent excessive speeds
    double max_speed = 2.0; // Adjust as needed to match original average speed
    speed = std::min(speed, max_speed);

    // Regular movement logic
    double angle_diff = angle_difference(a, target_angle);
    if (std::abs(angle_diff) > max_turn_rate) {
        double angle_change = angle_diff > 0 ? max_turn_rate : -max_turn_rate;
        a += angle_change;
    } else {
        a = target_angle;
    }

    a = std::fmod(a + M_PI, 2 * M_PI) - M_PI;

    // Move the fish
    x += std::cos(a) * speed * edad * 5;
    y += std::sin(a) * speed * edad * 5;
    colisiona();
}

    void colisiona() {
        if (x < 0 || x > 3840 || y < 0 || y > 2160) {
            // If out of bounds, set a target angle to turn back
            target_angle = std::fmod(a + M_PI, 2 * M_PI);
            is_avoiding_collision = true; // Increase speed to avoid boundary
        }
    }
};

int main() {
    // Video settings
    int width = 3840;
    int height = 2160;
    int fps = 60;
    int duration = 60 * 60; // 60 minutes
    int total_frames = fps * duration;

    // Prepare video directory
    std::filesystem::path video_dir("videos");
    if (!std::filesystem::exists(video_dir)) {
        std::filesystem::create_directory(video_dir);
    }

    // Define video file path with timestamp
    time_t epoch_time = std::time(nullptr);
    std::string video_path = (video_dir / ("fish_simulation_" + std::to_string(epoch_time) + ".mp4")).string();

    // Initialize video writer
    cv::VideoWriter video_writer(
        video_path,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(width, height)
    );

    if (!video_writer.isOpened()) {
        std::cerr << "Could not open the output video file for write" << std::endl;
        return -1;
    }

    // Initialize fishes and food
    int numeropeces = random_int(200, 2000); // Adjusted for testing; increase as needed
    std::vector<Pez> peces(numeropeces);
    std::vector<Comida> comidas;
    
    // Start with some food particles
    for (int i = 0; i < 10; i++) {
        comidas.emplace_back();
    }

    // Create window
    cv::namedWindow("Fish Simulation", cv::WINDOW_NORMAL);
    cv::resizeWindow("Fish Simulation", 1920, 1080); // Scaled down for display

    // Main loop
    for (int frame_count = 0; frame_count < total_frames; ++frame_count) {
        cv::Mat frame = cv::Mat::zeros(height, width, CV_8UC3);
        
        // Possibly add new food
        if (random_uniform(0, 1) < 0.00002 * numeropeces) {
            comidas.emplace_back();
        }

        // Process food first
        std::vector<Comida> new_comidas;
        for (auto& comida : comidas) {
            if (comida.visible) {
                comida.vive(frame, new_comidas);
            }
        }
        
        // Add newly created food particles
        comidas.insert(comidas.end(), new_comidas.begin(), new_comidas.end());
        
        // Remove invisible food
        comidas.erase(
            std::remove_if(comidas.begin(), comidas.end(), 
                         [](const Comida& c) { return !c.visible; }),
            comidas.end()
        );

        // Initialize quadtree for fishes
        Rectangle boundary(width / 2, height / 2, width / 2, height / 2);
        Quadtree<Pez> quadtree(boundary, 4);
        
        // Insert all fish into quadtree
        for (auto& pez : peces) {
            quadtree.insert(&pez);
        }

        // Initialize quadtree for food
        Quadtree<Comida> food_quadtree(boundary, 4);
        
        // Insert all visible food into quadtree
        for (auto& comida : comidas) {
            if (comida.visible) {
                food_quadtree.insert(&comida);
            }
        }

        // Process fish behavior in parallel with OpenMP
        #pragma omp parallel for
        for (size_t i = 0; i < peces.size(); ++i) {
            Pez& pez = peces[i];
            
            // Reset chasing food flag
            pez.is_chasing_food = false;

            // Fish perception radius for food
            double food_perception_radius = 300;
            Rectangle perception_range(pez.x, pez.y, food_perception_radius, food_perception_radius);

            // Query the food quadtree
            std::vector<Comida*> food_in_radius;
            #pragma omp critical
            {
                food_quadtree.query(perception_range, food_in_radius);
            }

            // Filter visible food
            food_in_radius.erase(
                std::remove_if(food_in_radius.begin(), food_in_radius.end(),
                            [](Comida* comida) { return !comida->visible; }),
                food_in_radius.end()
            );

            if (!food_in_radius.empty()) {
                // Find closest food
                Comida* closest_food = nullptr;
                double min_dist = std::numeric_limits<double>::infinity();
                
                for (auto* comida : food_in_radius) {
                    double dist = std::hypot(pez.x - comida->x, pez.y - comida->y);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_food = comida;
                    }
                }
                
                if (closest_food) {
                    double angleRadians = angle_in_radians(pez.x, pez.y, closest_food->x, closest_food->y);

                    // Only pursue food if not avoiding collision or stuck
                    if (!pez.is_avoiding_collision && !pez.is_stuck) {
                        pez.target_angle = angleRadians;
                        pez.is_chasing_food = true; // Fish is chasing food
                    }

                    if (std::hypot(pez.x - closest_food->x, pez.y - closest_food->y) < 10) {
                        #pragma omp critical
                        {
                            closest_food->visible = false;
                        }
                        pez.energia += closest_food->radio / 10;
                    }
                }
            } else {

if (random_uniform(0, 1) < 0.05 && !pez.is_avoiding_collision && !pez.is_stuck) {
                    pez.target_angle += (random_uniform(0, 1) - 0.5) * 0.05;
                }
            }
        }

        // Update each fish's state
        std::vector<int> dead_fish_indices;
        for (size_t i = 0; i < peces.size(); ++i) {
            peces[i].vive(frame, quadtree);
            if (peces[i].energia <= 0) {
                dead_fish_indices.push_back(i);
            }
        }

        // Replace dead fish with new ones
        for (auto idx : dead_fish_indices) {
            peces[idx] = Pez();
        }

        // Display progress
        if (frame_count % fps == 0) {
            std::cout << "Progress: " << frame_count / fps << "/" << duration << " seconds" << std::endl;
        }

        // Write frame to video
        video_writer.write(frame);
        
        // Show frame
        cv::imshow("Fish Simulation", frame);
        
        // Check for user input
        int key = cv::waitKey(1);
        if (key == 1) { // ESC key
            break;
        }
    }

    // Clean up
    video_writer.release();
    cv::destroyAllWindows();
    std::cout << "Video saved as " << video_path << std::endl;

    return 0;
}
