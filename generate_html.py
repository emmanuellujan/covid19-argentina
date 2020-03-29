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
def exponential(x,a,b,c):
    return a * np.exp(b * x) + c

# Plot graphic
def generate_graphic(index_0, days_0 , y_0, label_0):
    fig = go.Figure()
    n = len(index_0)
    no_samples = [14,7,3]
    colors = ['#f15e5e','#ef8e8e','#e8afaf']
    no_predictions = 3
    # Calculate a regression based on the number days in no_samples
    for idx,j in enumerate(no_samples):
        # Calculate regression data

        fit_successful = True
        try:
            popt, pcov = curve_fit(exponential, index_0[n-j:n], y_0[n-j:n], maxfev=20000)
        except RuntimeError:
            print("Warning - a curve fit failed for: " + label_0)
            fit_successful = False

        if fit_successful:
            index_1 = np.array(list(range(n-1,n+no_predictions)))
            datetime_object = datetime.strptime(days[n-1], '%Y-%m-%d %H:%M:%S')
            days_1 = [datetime_object]
            for i in range(no_predictions):
                days_1.append( datetime_object + timedelta(days=i+1) )
            y_1 =  np.around(exponential(index_1, *popt))

            all_positives = True
            for i in range(no_predictions):
                all_positives = all_positives and y_1[i] >= 0

            if all_positives:
                # Add regression trace to the subplot panel
                fig.add_trace(
                    go.Scatter(
                        x=days_1,
                        y=y_1,
                        mode='lines+markers',
                        marker_color=colors[idx],
                        name= 'Estimación en base a datos de los últimos '+str(j)+' días.',
                    ),
                )
    # Add actual data to the subplot panel
    fig.add_trace(
        go.Scatter(
            x=days_0,
            y=y_0,
            mode='lines+markers',
            marker_color='#6396c1',
            name= 'Datos del Ministerio de Salud.', 
        )
    )

    # Update subplot panel layout
    s = 12
    layout = go.Layout(
        title= label_0,
        titlefont=dict(size=20),
        yaxis=dict(
            title='Cantidad de personas',
            titlefont=dict(size=s),
            side='right'
        ),
        font=dict( size=s ),
        width=900,
        legend=dict(
            x=0,
            y=1.0,
        ),
    )
    fig.update_layout(layout)

    #fig.show()
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
                '    <h5 align="center">web en construcción</h5>' + \
                '    <div style="text-align: center;">' + \
                '    <div style="display: inline-block;">' + \
                '    <div style="width:900px;text-align:justify;"> En esta web se realiza un seguimiento de la evolución del <a href="https://www.argentina.gob.ar/salud/coronavirus-COVID-19">COVID-19</a> en Argentina, concretamente de la cantidad total de infectados y de la cantidad total de decesos por día. Las curvas azules muestran datos suministrados por el <a href="https://www.argentina.gob.ar/coronavirus/informe-diario">Ministerio de Salud</a> argentino, y las curvas rojas muestran estimaciones básicas de la progresión de dichas variables. Dichas estimaciones se realizan en base al ajuste de funciones exponenciales calculadas a partir de los datos de los últimos 3, 7 y 14 días. El código fuente para la generación de esta web es Open Source y puede descargarse desde el siguiente <a href="https://github.com/emmanuellujan/covid19-argentina">enlace</a>.</div>' + \
                    ''.join(str(div_str) for div_str in html_divs) + \
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
    n = len(days)
    index = np.array(list(range(n)))
    days = np.array(days)
    infected = np.array(infected)
    dead = np.array(dead)

# Generate html
html_divs = []
html_divs.append( generate_graphic(index, days, infected, 'Cantidad total de infectados en Argentina') )
html_divs.append( generate_graphic(index, days, dead, 'Cantidad total de fallecidos en Argentina') )

for i in range(n_provinces):
    accum = [provinces[i][0]]
    for j in range(1,len(provinces[i])):
        accum.append( provinces[i][j] + accum[j-1] )
    accum = np.array(accum)
    html_divs.append( generate_graphic(index, days, accum, 'Cantidad total de infectados en '+labels[3+i]) )

generate_html(html_divs)





