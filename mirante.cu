#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <map>

__global__ void distanceCalculationKernel(
    const float *crime_spots,
    const float *edges,
    int crime_count,
    int edge_count,
    float *distances,
    float *closest_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < crime_count) {
        float min_dist = INFINITY;
        float closest_node_x = 0; // Inicializar con 0
        float closest_node_y = 0; // Inicializar con 0
        float crime_x = crime_spots[idx * 2];
        float crime_y = crime_spots[idx * 2 + 1];

        // imprimir coordenadas del punto de crimen
        // printf("Crime Spot %d: (%f, %f)\n", idx, crime_x, crime_y);

        for (int j = 0; j < edge_count; j++) { // recorre los edges
            // Edge points
            float x1 = edges[j * 4];
            float y1 = edges[j * 4 + 1];
            float x2 = edges[j * 4 + 2];
            float y2 = edges[j * 4 + 3];

            // imprimir coordenadas de los edges
            // printf("  Edge %d: (%f, %f) to (%f, %f)\n", j, x1, y1, x2, y2);

            // Vector desde la linea al punto
            // B->A
            float line_vec_x = x2 - x1;
            float line_vec_y = y2 - y1;

            // Punto->A
            float point_vec_x = crime_x - x1;
            float point_vec_y = crime_y - y1;

            
            float line_len = sqrtf(line_vec_x * line_vec_x + line_vec_y * line_vec_y);

            // Evitar división por cero
            if (line_len == 0)
                continue;

            float line_unit_x = line_vec_x / line_len;
            float line_unit_y = line_vec_y / line_len;

            // Proyectar el punto
            float proj = point_vec_x * line_unit_x + point_vec_y * line_unit_y;
            proj = fmaxf(0, fminf(proj, line_len));

            // Punto proyectado
            float proj_x = x1 + proj * line_unit_x;
            float proj_y = y1 + proj * line_unit_y;

            // Calcular distancia
            float dist_x = crime_x - proj_x;
            float dist_y = crime_y - proj_y;
            float dist = sqrtf(dist_x * dist_x + dist_y * dist_y);

            // imprimir información de proyección
            // printf("  Projection: dist = %f, proj_x = %f, proj_y = %f\n", dist, proj_x, proj_y);

            // Actualizar la distancia minima
            if (dist < min_dist)
            {
                min_dist = dist;
                // Escoger el nodo mas cercano
                float start_dist = sqrtf((proj_x - x1) * (proj_x - x1) + (proj_y - y1) * (proj_y - y1));
                float end_dist = sqrtf((proj_x - x2) * (proj_x - x2) + (proj_y - y2) * (proj_y - y2));

                // Actualizar correctamente los nodos más cercanos
                closest_node_x = (start_dist < end_dist) ? x1 : x2;
                closest_node_y = (start_dist < end_dist) ? y1 : y2;
            }
        }

        // Guardar resultados
        distances[idx] = min_dist;
        closest_nodes[idx * 2] = closest_node_x;
        closest_nodes[idx * 2 + 1] = closest_node_y;
    }
}

class CrimeSpotAnalyzer {
private:
    std::vector<float> crime_spots;
    std::vector<float> edges;
    std::vector<int> node_weights;

    thrust::device_vector<float> d_crime_spots;
    thrust::device_vector<float> d_edges;
    thrust::device_vector<float> d_distances;
    thrust::device_vector<float> d_closest_nodes;

public:
    void loadCrimeSpots(const std::string &filename) {
        crime_spots.clear();
        std::ifstream file(filename);
        std::string line, value;

        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            float lon = 0, lat = 0;

            std::getline(ss, value, ',');
            lon = std::stof(value);
            std::getline(ss, value);
            lat = std::stof(value);

            // std::cout << "Lon=" << lon << ", Lat=" << lat << std::endl;

            crime_spots.push_back(lon);
            crime_spots.push_back(lat);
        }
    }

    void loadEdges(const std::string &filename) {
        edges.clear();
        std::ifstream file(filename);
        std::string line, value;

        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            float x1, y1, x2, y2;

            std::getline(ss, value, ',');
            x1 = std::stof(value);
            std::getline(ss, value, ',');
            y1 = std::stof(value);
            std::getline(ss, value, ',');
            x2 = std::stof(value);
            std::getline(ss, value);
            y2 = std::stof(value);

            // std::cout << "Edge: (" << x1 << "," << y1 << ") / (" << x2 << "," << y2 << ")" << std::endl;

            edges.push_back(x1);
            edges.push_back(y1);
            edges.push_back(x2);
            edges.push_back(y2);
        }
    }

    void processWithCUDA() {
        d_crime_spots = crime_spots;
        d_edges = edges;

        d_distances.resize(crime_spots.size() / 2);
        d_closest_nodes.resize(crime_spots.size());

        int blockSize = 256;
        int numBlocks = (crime_spots.size() / 2 + blockSize - 1) / blockSize;

        distanceCalculationKernel<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_crime_spots.data()),
            thrust::raw_pointer_cast(d_edges.data()),
            crime_spots.size() / 2,
            edges.size() / 4,
            thrust::raw_pointer_cast(d_distances.data()),
            thrust::raw_pointer_cast(d_closest_nodes.data()));

        std::vector<float> h_distances(d_distances.begin(), d_distances.end());
        std::vector<float> h_closest_nodes(d_closest_nodes.begin(), d_closest_nodes.end());

        // contar el peso de los nodos
        std::map<std::pair<float, float>, int> node_weight_map;
        for (size_t i = 0; i < h_closest_nodes.size() / 2; ++i) {
            std::pair<float, float> node_key = {
                h_closest_nodes[i * 2],
                h_closest_nodes[i * 2 + 1]};
            node_weight_map[node_key]++;
        }

        // Save to CSV
        std::ofstream output_file("pesos_nodos.csv");
        output_file << "y,x,weight\n";

        for (const auto &pair : node_weight_map) {
            output_file << pair.first.first << ","  // coordenada x
                        << pair.first.second << "," // coordenada y
                        << pair.second << "\n";     // peso
        }

        output_file.close();
        std::cout << "pesos_nodos.csv creado" << std::endl;

        // node_weights.clear();
        // for (const auto& pair : node_weight_map) {
        //     std::cout << "Node (" << pair.first.first
        //             << ", " << pair.first.second
        //             << "): " << pair.second << " crimes\n";
        //     node_weights.push_back(pair.second);
        // }

        // // Imprimir resultados detallados
        // // for (size_t i = 0; i < h_closest_nodes.size() / 2; ++i) {
        // //     std::cout << "Crime Spot " << i
        // //             << ": Closest Node ("
        // //             << h_closest_nodes[i*2] << ", "
        // //             << h_closest_nodes[i*2 + 1]
        // //             << ") Distance: " << h_distances[i] << std::endl;
        // // }
    }

    void printResults() {
        std::cout << "Node Weights:\n";
        for (size_t i = 0; i < node_weights.size(); ++i) {
            std::cout << "Node " << i << ": " << node_weights[i] << " crimes\n";
        }
    }

    int main() {
        try {
            loadCrimeSpots("crimeEvents.csv");
            loadEdges("coordinated_pairs.csv");
            processWithCUDA();
            // printResults();
        }
        catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        return 0;
    }
};

int main() {
    CrimeSpotAnalyzer analyzer;
    return analyzer.main();
}