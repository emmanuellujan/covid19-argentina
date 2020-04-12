################################################################################
# Generates an HTML file with information about COVID-19 in Argentina
#    - Plots the number of infected and dead people
#    - Calculates regressions to predict traces behaviour
# Author: Emmanuel Lujan, Ph.D., elujan@dc.uba.ar
################################################################################


################################################################################
# Imports
################################################################################

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import csv
from datetime import datetime
from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly as plotly
from sir_model import gen_sir_div
from fit import gen_fit_div
from fit import calc_avg_estimate


################################################################################
# Generate HTML web page
################################################################################

def generate_html(html_divs):
    html_str =  '<!DOCTYPE html>' + \
                '<html>' + \
                '<head>' + \
                '  <meta charset="utf-8"/>' + \
                ' <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' + \
                '</head>' + \
                '<body>' + \
                '    <h1 align="center">COVID-19, Argentina</h1>' + \
                '    <h5 align="center">web en construcción</h4>' + \
                '    <div style="text-align: center;">' + \
                '    <div style="display: inline-block;">' + \
                '    <div style="width:800px;text-align:justify;"> En esta página web se presenta información de la evolución del <a href="https://www.argentina.gob.ar/salud/coronavirus-COVID-19">COVID-19</a> en Argentina. Se proporciona información sobre el número de infecciones y muertes por día, basada en datos del <a href="https://www.argentina.gob.ar/coronavirus/informe-diario">Ministerio de Salud</a> de Argentina. Además, se presentan estimaciones básicas de la progresión de las variables mencionadas. <b>Las estimaciones no están aún validadas.</b> El código fuente para la generación de este sitio web es open-source y puede descargarse desde el siguiente <a href="https://github.com/emmanuellujan/covid19-argentina">enlace</a>.</div><br>' 

#    html_str += '    <h2 align="left">Estimaciones a mediano plazo</h2>' + \
#                '    <div style="width:800px;text-align:justify;"> Las estimaciones se calculan en base al modelo epidemiológico <a href="https://es.wikipedia.org/wiki/Modelo_SIR">SIR</a>, ajustando los parámetros del mismo en base a los datos provistos por el Ministerio de Salud. Se asume que la cantidad de pacientes fallecidos es 3.4% de r(t). </div><br>' + \
#                    ''.join(str(div_str) for div_str in html_divs[0])

    html_str += '    <h2 align="left">Estimaciones a corto plazo</h2>' + \
                '    <div style="width:800px;text-align:justify;"> Las estimaciones se calculan en base al ajuste diario de los parámetros <i>a</i> y <i>b</i> de la función<br> <i>f(x) = a * power(x,b)</i>, a partir de datos de los últimos días.</div><br>' + \
                    ''.join(str(div_str) for div_str in html_divs[1:3]) + \
                '    <h4 align="left">Cantidad de infectados desagregada</h4>' + \
                    ''.join(str(div_str) for div_str in html_divs[3:]) + \
                '    </div>' + \
                '</body>' + \
                '</html>'

    f = open("index.html", "w")
    f.write(html_str)
    f.close()


################################################################################
# Main program
################################################################################

# Read CSV and extract data to variables
n = 0
index = []
labels = []
days = []
infected = []
dead = []
n_provinces = 24
provinces = []
for i in range(n_provinces):
    provinces.append([])
with open('covid19.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    labels = next(csv_reader)
    for idx,row in enumerate(csv_reader):
        days.append(row[0])
        infected.append(int(row[1]))
        dead.append(int(row[2]))
        for j in range(n_provinces):
            if row[3+j] == 'N/A':
                val = 0
            else:
                val = int(row[3+j])
            provinces[j].append(val)
    days = np.array(days)
    infected = np.array(infected)
    dead = np.array(dead)

# Calculate HTML DIVs with the plots

html_divs = []
n_short_term = 3
n_middle_term = 160
samples = [14,7]
province_estimates = False

html_divs.append(gen_sir_div(days, infected, dead, n_middle_term))

avg_estimates = calc_avg_estimate(days, infected, n_short_term, samples,
                                    "Infectados totales")
div = gen_fit_div(days, infected, avg_estimates, n_short_term, samples, 
                  'Cantidad total de infectados en Argentina', True)
html_divs.append(div)

avg_estimates = calc_avg_estimate(days, dead, n_short_term, samples,
                                  "Fallecidos totales")
div = gen_fit_div(days, dead, avg_estimates, n_short_term, samples, 
                  'Cantidad total de fallecidos en Argentina', True)
html_divs.append(div)

for i in range(n_provinces):
    accum = [provinces[i][0]]
    for j in range(1, len(provinces[i])):
        accum.append(provinces[i][j] + accum[j-1])
    accum = np.array(accum)
    if province_estimates:
        avg_estimates = calc_avg_estimate(days, accum, n_short_term,
                                          samples, labels[3+i])
    div = gen_fit_div(days, accum, avg_estimates, n_short_term, samples, 
                      'Cantidad total de infectados en '+labels[3+i],
                      province_estimates)
    html_divs.append(div)

# Generate HTML web page
generate_html(html_divs)





