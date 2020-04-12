import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly as plotly
from datetime import datetime
from datetime import timedelta

def calc_sir(n, beta, gamma, nt, i0, r0):
    s = np.zeros(n)
    i = np.zeros(n)
    r = np.zeros(n)
    s[0] = nt - i0 - r0
    i[0] = i0
    r[0] = r0
    for j in range(n-1):
        s[j+1] = s[j] - beta * s[j] * i[j] / nt
        i[j+1] = i[j] + beta * s[j] * i[j] / nt - gamma * i[j]
        r[j+1] = r[j] + gamma * i[j] # recovered = cured + dead
    return s, i, r

def estimate_beta(nt, gamma0, i_val, r_val, m):
    beta = float("inf")
    count = 0
    n = len(i_val)
    nn = 1000
    e_i_min = float("inf")
    e_r_min = float("inf")
    for i0 in range(nn):
        beta0 = i0 * 1./nn

        s, i, r = calc_sir(m, beta0, gamma0, nt, i_val[n-m], r_val[n-m])

        e_i = 0
        for k in range(1):
            e_i += abs( i_val[n-k-1] - i[m-k-1] )

        e_r = 0
        for k in range(1):
            e_r += abs( r_val[n-k-1] - r[m-k-1] )

        if e_i < e_i_min and e_r < e_r_min:
            e_i_min = e_i
            e_r_min = e_r
            beta = beta0

        count += 1 
        #print(round(count/nn*100,2), "%", end="\r")

    return beta

def generate_sir_html(days, i_val, r_val, days_ext, i, r, beta, gamma, nt, m):

    #print("Simulation. beta:", beta, ", gamma:",gamma,", nt:", nt, ", n:", len(days_ext))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=days,
            y=i_val,
            #mode='lines+markers',
            marker_color='#6396c1',
            name= 'Datos del Ministerio de Salud.', 
        )
    )

    fig.add_trace(
        go.Bar(
            x=days_ext,
            y=i[m:],
            #mode='lines+markers',
            marker_color='#f15e5e',
            name= 'Estimación Modelo SIR.', 
        )
    )

    # Update subplot panel layout
    layout = go.Layout(
        title= 'Cantidad total de infectados en Argentina',
        titlefont=dict(size=20),
        yaxis=dict(
            title='Cantidad de personas',
            titlefont=dict(size=18),
            side='right'
        ),
        font=dict( size=18 ),
        width=800,
        showlegend=True,
        legend=dict(
            x=0.52,
            y=1.0,
        ),
        yaxis_type="log",
    )
    fig.update_layout(layout)

    return plotly.offline.plot(fig, output_type='div', include_plotlyjs=False);


def gen_sir_div(days, i_val, d_val, n_middle_term):
    n = len(days)
    m = n
    index = np.array(list(range(n)))
    nt = 44270000

    beta =  0
    gamma = 1./2.2
    r_val = 100./3.4 * d_val

    # SIR model estimation, n days extended
    beta = estimate_beta(nt, gamma, i_val, r_val, m)

    n_extended = m + n_middle_term
    datetime_object = datetime.strptime(days[n-1], '%Y-%m-%d %H:%M:%S')
    days_ext = []
    for i in range(n_middle_term):
        days_ext.append(datetime_object + timedelta(days=i+1))

    s, i, r = calc_sir(n_extended, beta, gamma, nt, i_val[n-m], r_val[n-m])

    return generate_sir_html(days, i_val, r_val, days_ext, i, r, beta, gamma, nt, m)






