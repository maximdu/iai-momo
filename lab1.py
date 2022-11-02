# Вычисления
import numpy as np
import pandas as pd
import scipy
# Визуализация
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


VARS_ANALYZED = [
    'osm_transport_stop_closest_dist',
    'total_square',
    'per_square_meter_price',
]

    
def get_nbins(var):
    nbins = 5 * np.log10(var.size)
    return int(nbins)

def get_main_quantiles(var, quantile_list=[0.10, 0.25, 0.50, 0.75, 0.90]):
    """Основные квантили распределения"""
    return {
        q: var.quantile(q) 
        for q in quantile_list
    }
    
def cut_tails(var):
    """Обрезает хвосты распределения"""
    box_quantiles = var.quantile([0.25, 0.75])
    box_len = box_quantiles.diff().dropna().item()

    bounds = box_quantiles + 1.5 * np.array([-box_len, box_len])
    min_bound, max_bound = bounds.values.round(5)
    
    filt = ((var > min_bound) & (var < max_bound))
    return var[filt]

# ----------------------------------------------------------------------------------
# Оптимизация
# ----------------------------------------------------------------------------------

def least_squares_method(var, theory_distr, x0, k_quantiles=50):
    uniform_quantiles = np.linspace(0, 1, k_quantiles+1)[1:-1]
    
    def least_squares_loss(params):
        return np.square(
            # Фактические значения
            var.quantile(uniform_quantiles).values -
            # Модельные значения
            theory_distr(*params).ppf(uniform_quantiles)
        ).sum()
    
    result = scipy.optimize.minimize(
        fun=least_squares_loss,
        x0=x0,
        bounds=np.sort(
            np.array([x0*0.5, x0*1.5]).T, 
            axis=1
        )
    )
    
    return result



def max_likelihood_method(var, theory_distr, x0, k_quantiles=50):
    uniform_quantiles = np.linspace(0, 1, k_quantiles+1)[1:-1]
    
    def log_likelihood_function(params):
        return -1 * np.sum(np.log( 
            theory_distr(*params).pdf(var) 
        ))
    
    result = scipy.optimize.minimize(
        fun=log_likelihood_function,
        x0=x0,
        bounds=np.sort(
            np.array([x0*0.5, x0*1.5]).T, 
            axis=1
        )
    )
    
    return result

# ----------------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------------

def qq_plot(var, theory_distr_with_params, k_quantiles=50):
    uniform_quantiles = np.linspace(0, 1, k_quantiles+1)[1:-1]
    
    fig = px.scatter(
        x=var.quantile(uniform_quantiles).values,
        y=theory_distr_with_params.ppf(uniform_quantiles),
        template='plotly_white'
    )

    fig.add_trace(go.Scatter(
        x=var.quantile(uniform_quantiles).values,
        y=var.quantile(uniform_quantiles).values,
        line_color='lightgrey',
        line_dash='dash'
    ))

    fig.data[0].marker.size = 5
    fig.layout.showlegend=False

    fig.update_layout(
        xaxis_title='Фактические значения', 
        yaxis_title='Теоретические значения'
    )

    fig.show()
    
    
def kde_plot(var, theory_distr_with_params):
    x = np.linspace(var.min(), var.max(), 500)
    sns.kdeplot(var);
    sns.lineplot(x=x, y=theory_distr_with_params.pdf(x));

    
def hist_fitted_plot(var, theory_distr_with_params):
    fig = px.histogram(var, marginal='box', template='plotly_white', histnorm='probability density')
    fig.data[0].marker.color='rgba(0,0,255,0.5)'

    x = np.linspace(var.min(), var.max(), 500)
    fig.add_trace(go.Scatter(
        x=x,
        y=theory_distr_with_params.pdf(x),
        mode='lines',
        marker_color='black'
    ))

    fig.layout.showlegend=False
    fig.show()

def chi_square_bins(var, theory_distr_with_params, nbins):
    f_obs, borders = np.histogram(var, bins=nbins)
    
    f_exp = (
        theory_distr_with_params.cdf( borders[1:] ) - 
        theory_distr_with_params.cdf( borders[:-1] )
    )
    f_exp = (f_exp * f_obs.sum()).round()
    
    # Исправляем ошибку округления
    f_exp[f_exp.argmax()] += f_obs.sum() - f_exp.sum()

    assert f_obs.sum() == f_exp.sum() 
    return f_obs, f_exp



# ----------------------------------------------------------------------------------
# Сэмплирование
# ----------------------------------------------------------------------------------


def clt_generator(size=10_000, n_layers=20, mean=0, std=1):
    np.random.seed(0)
    # Генерируем матрицу случайных чисел
    clt = np.random.uniform(0, 1, (n_layers, 10_000))
    # Суммируем по столбцам
    clt = np.sum(clt, axis=0)
    # Нормализация
    clt = np.sqrt(12/n_layers) * (clt - n_layers/2)
    # Добавляем параметры распределения
    clt = (clt * std) + mean
    return clt


def accept_reject_sampling(theory_distr_with_params, 
                           x_min, x_max, max_pdf, 
                           size=10_000):
    np.random.seed(0)
    rand_x = np.random.uniform(x_min, x_max, size)
    rand_y = np.random.uniform(0, max_pdf, size)
    is_lower = rand_y < theory_distr_with_params.pdf(rand_x)
    return rand_x[is_lower]
















