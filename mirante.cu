#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>

// Estructura para pares clave-valor
struct KeyValue {
    float key_x, key_y;  // Coordenadas del nodo (clave)
    int value;           // Contador (valor)
    
    __host__ __device__
    KeyValue(float x = 0, float y = 0, int v = 1) : key_x(x), key_y(y), value(v) {}
};

// Comparador para ordenar por clave
struct KeyComparator {
    __host__ __device__
    bool operator()(const KeyValue& a, const KeyValue& b) const {
        if (a.key_x != b.key_x) return a.key_x < b.key_x;
        return a.key_y < b.key_y;
    }
};

// Predicado para detectar claves iguales
struct KeyEqual {
    __host__ __device__
    bool operator()(const KeyValue& a, const KeyValue& b) const {
        const float epsilon = 1e-6f;
        return (fabsf(a.key_x - b.key_x) < epsilon) && 
               (fabsf(a.key_y - b.key_y) < epsilon);
    }
};

// Operador de suma para la fase REDUCE
struct SumValues {
    __host__ __device__
    KeyValue operator()(const KeyValue& a, const KeyValue& b) const {
        return KeyValue(a.key_x, a.key_y, a.value + b.value);
    }
};

// ============= FASE MAP =============
__global__ void mapKernel(
    const float *crime_spots,    // Input: coordenadas de crímenes
    const float *edges,          // Input: segmentos de calles
    int crime_count,
    int edge_count,
    KeyValue *map_output) {      // Output: pares (nodo, 1)
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < crime_count) {
        float min_dist = INFINITY;
        float closest_node_x = 0;
        float closest_node_y = 0;
        float crime_x = crime_spots[idx * 2];
        float crime_y = crime_spots[idx * 2 + 1];
        
        // Buscar el nodo más cercano
        for (int j = 0; j < edge_count; j++) {
            // Coordenadas del edge
            float x1 = edges[j * 4];
            float y1 = edges[j * 4 + 1];
            float x2 = edges[j * 4 + 2];
            float y2 = edges[j * 4 + 3];
            
            // Vector de la línea
            float line_vec_x = x2 - x1;
            float line_vec_y = y2 - y1;
            
            // Vector del punto al inicio de la línea
            float point_vec_x = crime_x - x1;
            float point_vec_y = crime_y - y1;
            
            // Longitud de la línea
            float line_len = sqrtf(line_vec_x * line_vec_x + line_vec_y * line_vec_y);
            
            if (line_len == 0) continue;
            
            // Vector unitario de la línea
            float line_unit_x = line_vec_x / line_len;
            float line_unit_y = line_vec_y / line_len;
            
            // Proyección del punto sobre la línea
            float proj = point_vec_x * line_unit_x + point_vec_y * line_unit_y;
            proj = fmaxf(0, fminf(proj, line_len));
            
            // Punto proyectado
            float proj_x = x1 + proj * line_unit_x;
            float proj_y = y1 + proj * line_unit_y;
            
            // Distancia del crimen al punto proyectado
            float dist_x = crime_x - proj_x;
            float dist_y = crime_y - proj_y;
            float dist = sqrtf(dist_x * dist_x + dist_y * dist_y);
            
            // Actualizar si es la distancia mínima
            if (dist < min_dist) {
                min_dist = dist;
                
                // Elegir el nodo más cercano (x1,y1) o (x2,y2)
                float start_dist = sqrtf((proj_x - x1) * (proj_x - x1) + (proj_y - y1) * (proj_y - y1));
                float end_dist = sqrtf((proj_x - x2) * (proj_x - x2) + (proj_y - y2) * (proj_y - y2));
                
                if (start_dist < end_dist) {
                    closest_node_x = x1;
                    closest_node_y = y1;
                } else {
                    closest_node_x = x2;
                    closest_node_y = y2;
                }
            }
        }
        
        // EMIT: (nodo_más_cercano, 1)
        map_output[idx] = KeyValue(closest_node_x, closest_node_y, 1);
    }
}

class MapReduceCrimeAnalyzer {
private:
    std::vector<float> crime_spots;
    std::vector<float> edges;
    
public:
    void loadCrimeSpots(const std::string &filename) {
        crime_spots.clear();
        std::ifstream file(filename);
        std::string line;
        
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open " << filename << std::endl;
            return;
        }
        
        // Saltar header
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string lat_str, lon_str;
            
            if (std::getline(ss, lat_str, ',') && std::getline(ss, lon_str)) {
                try {
                    float lat = std::stof(lat_str);
                    float lon = std::stof(lon_str);
                    
                    // Almacenar como (x,y) = (lon,lat)
                    crime_spots.push_back(lon);
                    crime_spots.push_back(lat);
                    
                    std::cout << "Crime: (" << lon << "," << lat << ")" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing: " << line << std::endl;
                }
            }
        }
        
        std::cout << "Loaded " << crime_spots.size()/2 << " crime spots" << std::endl;
    }
    
    void loadEdges(const std::string &filename) {
        edges.clear();
        std::ifstream file(filename);
        std::string line;
        
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open " << filename << std::endl;
            return;
        }
        
        // Saltar header
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string y1_str, x1_str, y2_str, x2_str;
            
            if (std::getline(ss, y1_str, ',') && 
                std::getline(ss, x1_str, ',') && 
                std::getline(ss, y2_str, ',') && 
                std::getline(ss, x2_str)) {
                try {
                    float y1 = std::stof(y1_str);
                    float x1 = std::stof(x1_str);
                    float y2 = std::stof(y2_str);
                    float x2 = std::stof(x2_str);
                    
                    // Almacenar como x1,y1,x2,y2
                    edges.push_back(x1);
                    edges.push_back(y1);
                    edges.push_back(x2);
                    edges.push_back(y2);
                    
                    std::cout << "Edge: (" << x1 << "," << y1 << ") to (" << x2 << "," << y2 << ")" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing: " << line << std::endl;
                }
            }
        }
        
        std::cout << "Loaded " << edges.size()/4 << " edges" << std::endl;
    }
    
    void processMapReduce() {
        if (crime_spots.empty() || edges.empty()) {
            std::cerr << "Error: No data loaded" << std::endl;
            return;
        }
        
        int crime_count = crime_spots.size() / 2;
        int edge_count = edges.size() / 4;
        
        std::cout << "\n=== INICIANDO MAP-REDUCE ===" << std::endl;
        
        // ============= FASE MAP =============
        std::cout << "Fase MAP: Mapeando " << crime_count << " crímenes..." << std::endl;
        
        // Copiar datos a GPU
        thrust::device_vector<float> d_crime_spots(crime_spots);
        thrust::device_vector<float> d_edges(edges);
        thrust::device_vector<KeyValue> d_map_output(crime_count);
        
        // Configurar grid y bloques
        int blockSize = 256;
        int numBlocks = (crime_count + blockSize - 1) / blockSize;
        
        // Ejecutar MAP kernel
        mapKernel<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_crime_spots.data()),
            thrust::raw_pointer_cast(d_edges.data()),
            crime_count,
            edge_count,
            thrust::raw_pointer_cast(d_map_output.data())
        );
        
        // Verificar errores
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error en MAP: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        cudaDeviceSynchronize();
        std::cout << "MAP completado. Generados " << d_map_output.size() << " pares clave-valor." << std::endl;
        
        // ============= FASE SHUFFLE =============
        std::cout << "Fase SHUFFLE: Ordenando por clave..." << std::endl;
        
        // Ordenar por clave para agrupar elementos iguales
        thrust::sort(d_map_output.begin(), d_map_output.end(), KeyComparator());
        
        std::cout << "SHUFFLE completado." << std::endl;
        
        // ============= FASE REDUCE =============
        std::cout << "Fase REDUCE: Agregando valores por clave..." << std::endl;
        
        // Reducir sumando valores de claves iguales
        thrust::device_vector<KeyValue> d_reduced_output(crime_count);
        
        auto new_end = thrust::reduce_by_key(
            d_map_output.begin(), d_map_output.end(),
            d_map_output.begin(),  // Usar mismos elementos como valores
            d_reduced_output.begin(),
            d_reduced_output.begin(),
            KeyEqual(),
            SumValues()
        );
        
        // Redimensionar al tamaño real
        int unique_count = new_end.first - d_reduced_output.begin();
        d_reduced_output.resize(unique_count);
        
        std::cout << "REDUCE completado. " << unique_count << " nodos únicos encontrados." << std::endl;
        
        // ============= ESCRIBIR RESULTADOS =============
        std::cout << "Escribiendo resultados..." << std::endl;
        
        // Copiar resultados de vuelta a CPU
        std::vector<KeyValue> h_results(d_reduced_output.begin(), d_reduced_output.end());
        
        // Escribir archivo CSV (formato estándar: lat,lon,weight)
        std::ofstream output_file("pesos_nodos_mapreduce.csv");
        output_file << "lat,lon,weight\n";
        
        for (const auto& result : h_results) {
            output_file << result.key_y << ","  // Latitud (Y) primero
                       << result.key_x << ","  // Longitud (X) segundo  
                       << result.value << "\n";
        }
        
        output_file.close();
        
        // Mostrar estadísticas
        std::cout << "\n=== ESTADÍSTICAS FINALES ===" << std::endl;
        std::cout << "Total de crímenes procesados: " << crime_count << std::endl;
        std::cout << "Total de nodos únicos: " << unique_count << std::endl;
        std::cout << "Archivo generado: pesos_nodos_mapreduce.csv" << std::endl;
        
        // Mostrar los 10 nodos con más crímenes (formato estándar: lat,lon)
        std::cout << "\nTOP 10 nodos con más crímenes:" << std::endl;
        std::sort(h_results.begin(), h_results.end(), 
                  [](const KeyValue& a, const KeyValue& b) {
                      return a.value > b.value;
                  });
        
        for (int i = 0; i < std::min(10, (int)h_results.size()); i++) {
            std::cout << i+1 << ". Nodo (" << h_results[i].key_y << "," // Latitud primero
                     << h_results[i].key_x << "): " << h_results[i].value // Longitud segundo
                     << " crímenes" << std::endl;
        }
    }
    
    int run() {
        try {
            loadCrimeSpots("crimeEvents.csv");
            loadEdges("coordinated_pairs.csv");
            processMapReduce();
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        return 0;
    }
};

int main() {
    MapReduceCrimeAnalyzer analyzer;
    return analyzer.run();
}