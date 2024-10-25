import plotly.graph_objects as go
from Classes.Tree import Tree
from Classes.Market import Market
from Classes.Option import Option
from Classes.Metrics import Metrics
from Classes.Greeks import Greeks
import pandas as pd
from Classes.BlackScholes import BlackScholes
import plotly.graph_objects as go
import xlwings as xw
import matplotlib.pyplot as plt
import tempfile
import os

def get_parameters():
    
    wb=xw.Book("TrinomialTreeModel.xlsm")
    ws = wb.sheets("Results")

    # Market parameters
    S0 = ws.range("D4").value
    sigma = ws.range("D5").value
    rate = ws.range("D6").value
    dividend = ws.range("D7").value
    ex_div_date = ws.range("D8").value / 365

    # Option parameters
    K = ws.range("D12").value
    T = int(ws.range("D13").value) / 365
    option_type = (ws.range("D14").value).lower()
    option_style = (ws.range("D15").value).lower()

    # Tree parameters
    N = int(ws.range("D19").value)
    if ws.range("C20").value == "On":
        factor = ws.range("D20").value
    else:
        factor = 0

    if ws.range("C21").value == "On":
        threshold = ws.range("D21").value
    else:
        threshold = 0

    return S0, sigma, rate, dividend, ex_div_date, K, T, option_type, option_style, N, factor, threshold

@xw.func
def Pricing(light_mode):

    wb=xw.Book("TrinomialTreeModel.xlsm")
    ws = wb.sheets("Results")

    S0, sigma, rate, dividend, ex_div_date, K, T, option_type, option_style, N, factor, threshold = get_parameters()

    market = Market(S0, rate, sigma, dividend, ex_div_date)
    option = Option(K, T, option_type, option_style)
    model = Tree(market, option, N)
    BSModel = BlackScholes(market, option)

    start_time = pd.Timestamp.now()
    price = model.get_option_price(light_mode=light_mode, factor=factor, threshold=threshold)
    pricing_time = (pd.Timestamp.now() - start_time).total_seconds()

    ws.range("G13").value = price
    ws.range("G14").value = pricing_time

    BS_price = BSModel.option_price()

    ws.range("G15").value = BS_price - price

    graph = ws.range("D22").value

    if graph == "Yes" and N <= 100:
        fig = model.visualize_tree(show=False)  # Obtenir la figure Plotly sans l'afficher
        ws = wb.sheets("Python - Graph")
        with tempfile.TemporaryDirectory() as tmpdirname:
            image_path = os.path.join(tmpdirname, 'plot.png')
            # Enregistrer la figure Plotly en tant qu'image
            fig.write_image(image_path, width=1600, height=900, scale=2)

            ws.pictures.add(image_path, name='TreePlot', update=True, left=ws.range('A1').left, top=ws.range('A1').top)
        
    greeks = Greeks(model)
    if greeks.tree_model.N >= 1000:
        greeks.tree_model.N = 1000

    delta, gamma, theta, vega, rho = greeks.get_greeks()
    ws.range("L5").value = delta
    ws.range("L6").value = gamma
    ws.range("L7").value = vega
    ws.range("L8").value = theta
    ws.range("L9").value = rho

    delta_bs, gamma_bs, theta_bs, vega_bs, rho_bs = greeks.get_greeks_bs()
    ws.range("M5").value = delta_bs
    ws.range("M6").value = gamma_bs
    ws.range("M7").value = vega_bs
    ws.range("M8").value = theta_bs
    ws.range("M9").value = rho_bs

def add_plot_to_excel(fig, ws, name, cell):

    with tempfile.TemporaryDirectory() as tmpdirname:
        image_path = os.path.join(tmpdirname, f'{name}.png')
        fig.write_image(image_path)
        ws.pictures.add(image_path, name=name, update=True, left=ws.range(cell).left, top=ws.range(cell).top)

@xw.func
def show_metrics(light_mode):
    
    S0, sigma, rate, dividend, ex_div_date, K, T, option_type, option_style, N, factor, threshold = get_parameters()

    market = Market(S0, rate, sigma, dividend, ex_div_date)
    option = Option(K, T, option_type, option_style)
    model = Tree(market, option, N)
    metrics = Metrics(model)

    wb=xw.Book("TrinomialTreeModel.xlsm")
    ws = wb.sheets("Metrics")

    start = int(ws.range("C3").value)
    end = int(ws.range("D3").value)
    step = int(ws.range("E3").value)
    
    fig1, fig2, fig3, fig4, fig5, fig6 = metrics.compute_and_show_metrics(range(start, end, step), light_mode=light_mode, show=False)

    if not light_mode:
        positions = ['B7', 'B23', 'B39', 'B55', 'B71', 'B87']
        text = 'LightMode'
    else:
        positions = ['H7', 'H23', 'H39', 'H55', 'H71', 'H87']
        text = 'FullMode'

    add_plot_to_excel(fig1, ws, f'BuildingTime{text}', positions[0])
    add_plot_to_excel(fig2, ws, f'PricingTime{text}', positions[1])
    add_plot_to_excel(fig3, ws, f'BuildingTimeByNode{text}', positions[2])
    add_plot_to_excel(fig4, ws, f'MemoryUsage{text}', positions[3])
    if not light_mode:
        add_plot_to_excel(fig5, ws, f'PricingTimeByNode{text}', positions[4])
        add_plot_to_excel(fig6, ws, f'NumberOfNodes{text}', positions[5])

@xw.func
def show_greeks(option_type):
    
    S0, sigma, rate, dividend, ex_div_date, K, T, _, option_style, N, factor, threshold = get_parameters()

    market = Market(S0, rate, sigma, dividend, ex_div_date)
    option = Option(K, T, option_type, option_style)
    model = Tree(market, option, N)
    

    wb=xw.Book("TrinomialTreeModel.xlsm")
    ws = wb.sheets("Greeks")
    
    start = int(ws.range("C3").value)
    end = int(ws.range("D3").value)
    step = int(ws.range("E3").value)

    model.N = int(ws.range("I3").value)

    greeks = Greeks(model)
    
    fig1, fig2, fig3, fig4, fig5 = greeks.compute_and_show_greeks(range(start, end, step), show=False)

    if model.option.type == "call":
        positions = ['B7', 'B23', 'B39', 'B55', 'B71']
        text = 'Call'
    else:
        positions = ['H7', 'H23', 'H39', 'H55', 'H71']
        text = 'Put'

    add_plot_to_excel(fig1, ws, f'Delta{text}', positions[0])
    add_plot_to_excel(fig2, ws, f'Gamma{text}', positions[1])
    add_plot_to_excel(fig3, ws, f'Theta{text}', positions[2])
    add_plot_to_excel(fig4, ws, f'Vega{text}', positions[3])
    add_plot_to_excel(fig5, ws, f'Rho{text}', positions[4])


    
    
    
    
    
    
    