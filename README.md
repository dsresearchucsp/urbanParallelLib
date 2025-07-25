# CUDA MapReduce Crime Analysis

Analiza datos de criminalidad usando MapReduce en GPU para identificar los nodos de red con mayor incidencia criminal.

## Funcionamiento

El programa implementa MapReduce en CUDA con tres fases:

1. **MAP**: Cada crimen se asigna al nodo de calle más cercano
2. **SHUFFLE**: Los datos se ordenan y agrupan por nodo  
3. **REDUCE**: Se cuenta el total de crímenes por nodo

## Archivos Requeridos

**crime_locations.csv**
**edges_with_nodes.csv**
## Uso

### Compilar
nvcc -o crime_analyzer main.cu

### Ejecutar
./crime_analyzer
