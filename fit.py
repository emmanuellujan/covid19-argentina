import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from datetime import datetime
from datetime import timedelta
import plotly as plotly

# Exponential function, used in regressions
def exponential(x, a, b, c, d):
    #return a * np.exp(b * x) 
    #return a * np.power(b, x)
    return a * np.power(x,b)


# Calculate average estimates
def calc_avg_estimate(days, y, n_pred, samples, label):
    n = len(days)
    index = np.array(list(range(n)))
    max_sample = max(samples)
    n_samples = len(samples)
    errors = np.zeros(n)
    y_0 = np.zeros(n)
    s_0 = np.zeros(n) + 0.001

    for d in range(max_sample, n):
        for s in samples:
            # Calculate regression data
            fit_successful = True
            try:
                popt, pcov = curve_fit(exponential, index[d-s:d], y[d-s:d], maxfev=30000)
            except RuntimeError:
                print("Warning - a curve fit failed for: " + label)
                fit_successful = False

            # Update max-error estimates
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

# Generate function fit DIV
def gen_fit_div(days_0 , y_0, y_1, n_pred, samples, label_0, estimate):
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
                index_1 = np.array(list(range(n, n+n_pred)))
                datetime_object = datetime.strptime(days_0[n-1], '%Y-%m-%d %H:%M:%S')
                days_1 = []
                for i in range(n_pred):
                    days_1.append(datetime_object + timedelta(days=i+1))
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

        # Add worst estimate data to the subplot panel
        fig.add_trace(
            go.Scatter(
                x=days_0[16:],
                y=y_1[16:],
                mode='markers',
                marker_color='orange',
                name= 'Estimación promedio anterior.', 
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


