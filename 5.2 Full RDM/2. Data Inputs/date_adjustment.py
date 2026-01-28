# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 22:15:29 2025

@author: guill
"""

import pandas as pd

# Cargar el archivo CSV, especificando que el delimitador es ';'
supply_ts = pd.read_csv("supply.csv", delimiter=";")
demand_ts = pd.read_csv("demand.csv", delimiter=";")
anc_ts = pd.read_csv("anc.csv", delimiter=";")

# Revisar cómo se interpretan las fechas antes de modificarlas
# print(supply_ts.head())  # Asegúrate de que la columna 'Date' esté en el formato correcto.
# print(demand_ts.head())  # Asegúrate de que la columna 'Date' esté en el formato correcto.

# Asegurarse de que la columna 'Date' esté en formato de fecha, detectando si el día está primero
supply_ts['Date'] = pd.to_datetime(supply_ts['Date'], format='%d/%m/%Y', dayfirst=True)
demand_ts['Date'] = pd.to_datetime(demand_ts['Date'], format='%d/%m/%Y', dayfirst=True)
anc_ts['Date'] = pd.to_datetime(anc_ts['Date'], format='%d/%m/%Y', dayfirst=True)

# Guardar de nuevo el archivo CSV con el formato correcto
supply_ts.to_csv("supply_corrected.csv", index=False)
demand_ts.to_csv("demand_corrected.csv", index=False)
anc_ts.to_csv("anc_corrected.csv", index=False)