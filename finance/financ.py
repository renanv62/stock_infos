import yfinance as yf
from kivymd.uix.button import MDFlatButton
from kivymd.uix.card import MDCard
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.list import OneLineListItem
from yfinance import Ticker
import matplotlib.pyplot as plt
import statsmodels.api as sm
import arch as arch
from kivymd.uix.label import MDLabel
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from matplotlib.figure import Figure
import numpy as np
from kivymd.app import MDApp
from kivy.core.window import Window
Window.maximize()

class financ(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_stock = yf.Ticker("AAPL")

    def build(self):
        self.create_chart()
        self.regression_analysis()
        self.garch_analysis()
        self.arima_analysis()

    ###### INIT THE STOCKS SEARCH ######
    def search_stocks(self,input_text):
        scroll_layout = self.root.ids.scroll_layout  # Gets the reference to the GridLayout inside the ScrollView
        scroll_layout.clear_widgets()
        # LAYOUT & MD CARD
        mdcard = MDCard(size_hint=(.7,.6),padding=4,elevation=3,orientation="vertical")
        mdlabel_title = MDLabel(halign="center")
        mdgrid = MDGridLayout(size_hint=(1,1),cols=1,rows=2,spacing=5)#self.root.ids.layout_grid_card
        onelinelist = OneLineListItem(divider="Full")
        onelinelist2 = OneLineListItem(divider="Full")

        tickers = yf.Tickers(input_text)
        for ticker in tickers.tickers:
            self.get_stock = yf.Ticker(ticker)
            # get the infos from the ticker
            stock_name = self.get_stock.info.get("longName", "N/A")  # Replace "N/A" with a default value
            stock_volume = self.get_stock.info.get("volume", "N/A")  #
            stock_marketCapitalization = self.get_stock.info.get("marketCap", "N/A")  #
            # add the text infos to respective texts widgets
            mdlabel_title.text = stock_name
            onelinelist.text = "Volume: "+str(stock_volume)
            onelinelist2.text = "MarketCap: "+str(stock_marketCapitalization)
            # add the text to the mdgrid
            mdgrid.add_widget(onelinelist)
            mdgrid.add_widget(onelinelist2)
            # add the title and the grid with items to mdcard
            mdcard.add_widget(mdlabel_title)
            mdcard.add_widget(mdgrid)
            mdcard.bind(on_release=lambda instance, ticker=ticker: self.update_default_stock(instance, ticker))
            # add the md card to the overall layout
            scroll_layout.add_widget(mdcard)

    ###### END THE STOCKS SEARCH ######

    def update_default_stock(self,blank_instance,ticker):
        search_stock = yf.Ticker(ticker)
        self.create_chart(search_stock)
        self.regression_analysis()
        self.garch_analysis()
        self.arima_analysis()

    ###### INIT THE MAIN CHART ######
    def create_chart(self,searched_ticker=None):
        if searched_ticker is None:
            historical_data = self.default_stock.history(start="2022-01-01", end="2022-12-31")  # period="1mo")
        else:
            self.default_stock = self.get_stock
            searched_name = self.default_stock.info.get("longName", "N/A")
            historical_data = self.default_stock.history(start="2022-01-01", end="2022-12-31")

        MDCard_chart = self.root.ids.chart_card
        MDCard_chart.clear_widgets()
        # Create a FigureCanvas object
        fig = Figure(facecolor='none')
        ax = fig.add_subplot(111)
        # Plot the closing prices
        ax.plot(historical_data.index, historical_data['Close'])
        # Add labels and title
        stock_symbol = self.default_stock.info['symbol']
        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price")
        ax.set_title(f"{stock_symbol} Closing Prices")

        # Create a FigureCanvasKivy object
        canvas = FigureCanvasKivyAgg(fig)
        MDCard_chart.add_widget(canvas)

    ###### ENDS THE MAIN CHART ######

    ############ REGRESSION ANALYSIS #############
    def regression_analysis(self):
        #boxsummary = self.root.ids.boxRegression_summary
        boxchart = self.root.ids.boxRegression_chart
        boxchart.clear_widgets()

        # Fetch historical data using yfinance
        MS_detail = self.default_stock.history(start="2022-01-01", end="2022-12-31")
        # Perform the regression analysis
        model = sm.OLS(MS_detail["Close"], MS_detail["Volume"])
        results = model.fit()
        # Print the results of the regression analysis
        #summary = results.summary()

        # Create a FigureCanvas object
        fig = Figure(facecolor='none')
        ax = fig.add_subplot(111)
        # Plot the results of the regression analysis
        ax.scatter(MS_detail["Volume"], MS_detail["Close"])
        ax.plot(MS_detail["Volume"], results.fittedvalues, color="red")
        # Add labels and title
        stock_symbol = self.default_stock.info['symbol']
        ax.set_xlabel("Volume")
        ax.set_ylabel("Close")
        ax.set_title(f"Linear Regression of {stock_symbol} Stock")
        # Create a FigureCanvasKivy object
        canvas = FigureCanvasKivyAgg(fig)
        canvas.size_hint = (1,1)
        # Add the FigureCanvasKivy object to the MDBoxLayout
        boxchart.add_widget(canvas)
        print(results.fittedvalues)
    ############ REGRESSION ANALYSIS #############

    ############ GARCH #############
    def garch_analysis(self):
        box_garch = self.root.ids.garch_summary
        box_garch.clear_widgets()

        # Fetch historical data using yfinance
        historical_data = self.default_stock.history(start="2022-01-01", end="2022-12-31")

        # Calculate daily returns from the 'Close' prices
        daily_returns = historical_data["Close"].pct_change().dropna()

        # Fit the GARCH model
        model = arch.arch_model(daily_returns, p=1, q=1)
        results = model.fit()
        # Evaluate the GARCH model
        garch_summary = str(results.summary())

        garch_label = MDLabel(text=garch_summary,size_hint_y=None)
        box_garch.add_widget(garch_label)
    ############ GARCH #############

    ############ ARIMA #############
    def arima_analysis(self):
        box_arima = self.root.ids.arima_summary
        box_arima.clear_widgets()

        # Fetch historical data using yfinance
        historical_data = self.default_stock.history(start="2022-01-01", end="2022-12-31")

        # Calculate daily returns from the 'Close' prices
        daily_returns = historical_data["Close"].pct_change().dropna()

        # Fit the ARIMA model
        model = sm.tsa.arima.ARIMA(daily_returns, order=(5,1,1))
        model_fit = model.fit()
        # Fazer previsões usando o modelo ajustado
        forecast_steps = 30
        forecast = model_fit.forecast(steps=forecast_steps)

        stock_symbol = self.default_stock.info['symbol']
        # Criar e configurar o gráfico
        fig, ax = plt.subplots()
        ax.plot(daily_returns, label="Observations")
        ax.plot(np.arange(len(daily_returns), len(daily_returns) + forecast_steps), forecast, label="Forecast")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.set_title(f"ARIMA Forecast for {stock_symbol}")

        # Criar o FigureCanvasKivyAgg e adicionar ao MDBoxLayout
        canvas = FigureCanvasKivyAgg(fig)
        canvas.size_hint = (1, 1)
        box_arima.add_widget(canvas)

    ############ ARIMA #############

if __name__ == '__main__':
    financ().run()