# CUDA MapReduce Crime Analysis

Analiza datos de criminalidad usando MapReduce en GPU para identificar los nodos de red con mayor incidencia criminal.

## Cómo Funciona

El programa implementa MapReduce en CUDA con tres fases:

1. **MAP**: Cada crimen se asigna al nodo de calle más cercano
2. **SHUFFLE**: Los datos se ordenan y agrupan por nodo  
3. **REDUCE**: Se cuenta el total de crímenes por nodo

## Archivos Requeridos

**crime_locations.csv**

Latitude,Longitude
40.7589,-73.9851
40.7614,-73.9776
