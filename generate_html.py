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
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import csv
from datetime import datetime
from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly as plotly


################################################################################
# Functions: 
################################################################################

# Exponential function, used in regressions
def exponential(x,a,b,c,d):
    #return a * np.exp(b * x) 
    #return a * np.power(b, x)
    return a * np.power(x,b)


# Calculate maximum-error estimation
def calc_avg_estimation(days, y, n_pred, samples,label):
    n = len(days)
    index = np.array(list(range(n)))
    max_sample = max(samples)
    n_samples = len(samples)
    errors = np.zeros(n)
    y_0 = np.zeros(n)
    s_0 = np.zeros(n) + 0.001

    for d in range(max_sample,n):
        for s in samples:
            # Calculate regression data
            fit_successful = True
            try:
                popt, pcov = curve_fit(exponential, index[d-s:d], y[d-s:d], maxfev=30000)
            except RuntimeError:
                print("Warning - a curve fit failed for: " + label)
                fit_successful = False

            # Update max-error estimations
            if fit_successful:
                if d + n_pred < n:
                    m = d + n_pred + 1
                else:
                    m = n - d

                for p in range(d+1,m):
                    index_1 = np.array([p])
                    aux_0 = np.around(exponential(index_1, *popt))
                    y_0[p] += aux_0
                    s_0[p] += 1

    return np.around(y_0 / s_0)


# Plot graphic
def generate_graphic(days_0 , y_0, y_1, n_pred, samples, label_0, estimate):
    fig = go.Figure()

    if estimate:
        n = len(days_0)
        index_0 = np.array(list(range(n)))
        colors = ['#f15e5e','#ef8e8e','#e8afaf','#f15e5e','#ef8e8e','#e8afaf']
        # Calculate a regression based on the number days in samples
        for idx,j in enumerate(samples):
            # Calculate regression data

            fit_successful = True
            try:
                popt, pcov = curve_fit(exponential, index_0[n-j:n], y_0[n-j:n], maxfev=30000)
            except RuntimeError:
                print("Warning - a curve fit failed for: " + label_0)
                fit_successful = False

            if fit_successful:
                index_1 = np.array(list(range(n,n+n_pred)))
                datetime_object = datetime.strptime(days[n-1], '%Y-%m-%d %H:%M:%S')
                days_1 = []
                for i in range(n_pred):
                    days_1.append( datetime_object + timedelta(days=i+1) )
                y_2 =  np.around(exponential(index_1, *popt))

                all_positives = True
                for i in range(n_pred):
                    all_positives = all_positives and y_2[i] >= 0

                if all_positives:
                    # Add regression trace to the subplot panel
                    fig.add_trace(
                        go.Bar(
                            x=days_1,
                            y=y_2,
                            #mode='lines+markers',
                            marker_color=colors[idx],
                            name= 'Estimación en base a datos de los últimos '+str(j)+' días.',
                        ),
                    )

        # Add worst estimation data to the subplot panel
        fig.add_trace(
            go.Scatter(
                x=days_0[16:],
                y=y_1[16:],
                mode='markers',
                marker_color='orange',
                name= 'Estimación promedio pasada.', 
            )
        )

    # Add actual data to the subplot panel
    fig.add_trace(
        go.Bar(
            x=days_0,
            y=y_0,
            #mode='lines+markers',
            marker_color='#6396c1',
            name= 'Datos del Ministerio de Salud.', 
        )
    )

    # Update subplot panel layout
    layout = go.Layout(
        title= label_0,
        titlefont=dict(size=20),
        yaxis=dict(
            title='Cantidad de personas',
            titlefont=dict(size=18),
            side='right',
            overlaying = 'y',
        ),
        font=dict( size=18 ),
        width=800,
        showlegend=True,
        legend=dict(
            x=0,
            y=1.0,
            bgcolor="rgba(0, 0, 0, 0.05)",
            #orientation="h"
        ),
        barmode='overlay',
        #yaxis_type="log",
    )
    fig.update_layout(layout)

    return plotly.offline.plot(fig, output_type='div', include_plotlyjs=False);


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
                '    <div style="width:800px;text-align:justify;"> En esta página web se presenta información de la evolución del <a href="https://www.argentina.gob.ar/salud/coronavirus-COVID-19">COVID-19</a> en Argentina. Se proporciona información sobre el número de infecciones y muertes por día, basada en datos del <a href="https://www.argentina.gob.ar/coronavirus/informe-diario">Ministerio de Salud</a> de Argentina. Además, se presentan estimaciones básicas de la progresión de las variables mencionadas. El código fuente para la generación de este sitio web es open-source y puede descargarse desde el siguiente <a href="https://github.com/emmanuellujan/covid19-argentina">enlace</a>.</div><br>' + \
                '    <h2 align="left">Cantidad total de infectados y fallecidos</h2>' + \
                '    <div style="width:800px;text-align:justify;"> Las estimaciones se calculan en base al ajuste diario de los parámetros <i>a</i> y <i>b</i> de la función<br> <i>f(x) = a * power(x,b)</i>, a partir de datos de los últimos días.</div><br>' + \
                    ''.join(str(div_str) for div_str in html_divs[:2]) + \
                '    <h2 align="left">Cantidad de infectados desagregada</h2>' + \
                    ''.join(str(div_str) for div_str in html_divs[2:]) + \
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
#    print(provinces[0])
    days = np.array(days)
    infected = np.array(infected)
    dead = np.array(dead)

# Generate html
html_divs = []

n_pred = 3
samples = [14,7]
province_estimations = False

error_estimations = calc_avg_estimation(days, infected, n_pred, samples, "Infectados totales")
html_divs.append( generate_graphic(days, infected, error_estimations, n_pred, samples, 
                  'Cantidad total de infectados en Argentina', True) )

error_estimations = calc_avg_estimation(days, dead, n_pred, samples, "Fallecidos totales")
html_divs.append( generate_graphic(days, dead, error_estimations, n_pred, samples, 
                  'Cantidad total de fallecidos en Argentina', True) )


for i in range(n_provinces):
    accum = [provinces[i][0]]
    for j in range(1,len(provinces[i])):
        accum.append( provinces[i][j] + accum[j-1] )
    accum = np.array(accum)
    if province_estimations:
        error_estimations = calc_avg_estimation(days, accum,  n_pred, samples, labels[3+i])
    html_divs.append( generate_graphic(days, accum, error_estimations, n_pred, samples, 
                      'Cantidad total de infectados en '+labels[3+i], province_estimations) )

generate_html(html_divs)





