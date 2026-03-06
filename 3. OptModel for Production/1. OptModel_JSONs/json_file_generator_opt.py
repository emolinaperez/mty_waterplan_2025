# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 11:56:14 2025

@author: guill
"""

import json
import os

# Definir el archivo JSON base
archivo_base = 'basefile.json'

# Listas de reemplazos de escenarios de demanda y oferta
reemplazos_demanda = [
    "ANC02_CLIMIN_ECOACE", "ANC02_CLIMIN_ECODIN", "ANC02_CLIMIN_ECOHIS",
    "ANC02_CLIMRCP45_ECOACE", "ANC02_CLIMRCP45_ECODIN", "ANC02_CLIMRCP45_ECOHIS",
    "ANC02_CLIMRCP85_ECOACE", "ANC02_CLIMRCP85_ECODIN", "ANC02_CLIMRCP85_ECOHIS",
    "ANC22_CLIMIN_ECOACE", "ANC22_CLIMIN_ECODIN", "ANC22_CLIMIN_ECOHIS",
    "ANC22_CLIMRCP45_ECOACE", "ANC22_CLIMRCP45_ECODIN", "ANC22_CLIMRCP45_ECOHIS",
    "ANC22_CLIMRCP85_ECOACE", "ANC22_CLIMRCP85_ECODIN", "ANC22_CLIMRCP85_ECOHIS",
    "ANCTEND_CLIMIN_ECOACE", "ANCTEND_CLIMIN_ECODIN", "ANCTEND_CLIMIN_ECOHIS",
    "ANCTEND_CLIMRCP45_ECOACE", "ANCTEND_CLIMRCP45_ECODIN", "ANCTEND_CLIMRCP45_ECOHIS",
    "ANCTEND_CLIMRCP85_ECOACE", "ANCTEND_CLIMRCP85_ECODIN", "ANCTEND_CLIMRCP85_ECOHIS"
]

reemplazos_oferta = [
    "1A, 0.90", "1A, 0.95", "1A, 0.97", "2A, 0.90", "2A, 0.95", "2A, 0.97",
    "3A, 0.90", "3A, 0.95", "3A, 0.97", "3B, 0.90", "3B, 0.95", "3B, 0.97",
    "4A, 0.90", "4A, 0.95", "4A, 0.97", "4B, 0.90", "4B, 0.95", "4B, 0.97",
    "5A, 0.90", "5A, 0.95", "5A, 0.97", "5B, 0.90", "5B, 0.95", "5B, 0.97",
    "6A, 0.90", "6A, 0.95", "6A, 0.97", "6B, 0.90", "6B, 0.95", "6B, 0.97",
    "6C, 0.90", "6C, 0.95", "6C, 0.97", "7A, 0.90", "7A, 0.95", "7A, 0.97"
]

# Cargar el JSON base
with open(archivo_base, 'r') as f:
    data = json.load(f)

# Crear combinaciones de archivos JSON
copias_generadas = []
for reemplazo_demanda in reemplazos_demanda:
    for reemplazo_oferta in reemplazos_oferta:
        # Reemplazar los textos en el JSON
        data_modificada = json.dumps(data).replace("ANC02_CLIMIN_ECOACE", reemplazo_demanda).replace("1A, 0.90", reemplazo_oferta)
        data_modificada = json.loads(data_modificada)  # Convertir de nuevo a JSON
        # Añadir a la lista de copias generadas
        copias_generadas.append((data_modificada, f"{reemplazo_demanda}_{reemplazo_oferta.replace(', ', '_')}.json"))

# Crear carpetas y distribuir los archivos JSON en ellas
num_archivos_por_carpeta = 36
num_carpetas = 27
directorio_base = 'copias_generadas'

for i in range(num_carpetas):
    # Crear nombre y directorio para cada carpeta
    nombre_carpeta = f"{directorio_base}_{i + 1}"
    os.makedirs(nombre_carpeta, exist_ok=True)
    
    # Determinar el rango de archivos para esta carpeta
    inicio = i * num_archivos_por_carpeta
    fin = inicio + num_archivos_por_carpeta
    # if i == num_carpetas - 1:  # Última carpeta con 8 archivos restantes
    #     fin = inicio + 8
    
    # Guardar los archivos JSON en la carpeta correspondiente
    for data_modificada, nombre_archivo in copias_generadas[inicio:fin]:
        ruta_archivo = os.path.join(nombre_carpeta, nombre_archivo)
        with open(ruta_archivo, 'w') as f:
            json.dump(data_modificada, f, indent=4)

        print(f"Copia creada en {ruta_archivo}")