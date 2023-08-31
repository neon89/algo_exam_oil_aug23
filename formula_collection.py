    

import matplotlib.pyplot as plt

def myhistplot(data):
    headers = list(data)
    rows = int(len(data.columns) / 2)
    cols = int(len(data.columns) / rows)
    plt.rcParams.update({'font.size': 30})
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(40, 25))
    
    for c, ax in zip(headers, axes.ravel()):
        ax.hist(data[c], bins=30, color='r')
        ax.grid(which='major')
        plt.tight_layout()
        ax.set_title(c)
        
        
def gain(value):
    if value < 0:
        return 0
    else:
        return value

def loss(value):
    if value > 0:
        return 0
    else:
        return abs(value)
        
def specs(df,window):
    import numpy as np
    
    #based on returns calculate vola, momentum, smoothing average, min, max 
    df['return'] = np.log(df.c/df.c.shift(1))
    df['vola'] = df['return'].rolling(window).std()
    df['mom'] = np.sign(df['return'].rolling(window).mean())
    df['sma'] = df['return'].rolling(window).mean()
    df['min'] = df['return'].rolling(window).min()
    df['max'] = df['return'].rolling(window).max()
    return df


def bollinger_band(df, std1, std2, window):
    import numpy as np
    # Calculate Bollinger Bands and signals based on input parameters
    df['ma_window'] = df['c'].rolling(window).mean()
    df['upper_bol'] = df['ma_window'] + (std1 * df['c'].rolling(window).std())
    df['lower_bol'] = df['ma_window'] - (std2 * df['c'].rolling(window).std())
    
    # Generate buy/sell signals
    df['signal_bol'] = np.where(df['c'] < df['lower_bol'], 1, np.nan)
    df['signal_bol'] = np.where(df['c'] > df['upper_bol'], -1, df['signal_bol'])
    df['signal_bol'] = df['signal_bol'].ffill().fillna(0)
    
    # Calculate strategy returns and cumulative metrics
    df['strategy_bol'] = df['signal_bol'].shift(1) * df['return']
    df['cum_strategy_bol'] = df['strategy_bol'].dropna().cumsum().apply(np.exp)
    df['cum_return_bol'] = df['return'].dropna().cumsum().apply(np.exp)
    df['cum_max_bol'] = df['cum_strategy_bol'].cummax()
    
    return df


def rsi_indicator(df,window):
    import numpy as np
    
    #Calculate price delta [RSI]
    df['delta'] = df.c.diff()
    
    #Classify delta into gain & loss [RSI]
    df['gain'] = df['delta'].apply(lambda x:gain(x))
    df['loss'] = df['delta'].apply(lambda x:loss(x))
    
    #Calculate ema [RSI]
    df['ema_gain'] = df['gain'].ewm(window).mean()
    df['ema_loss'] = df['loss'].ewm(window).mean()
    
    #Calculate RSI
    df['rs'] = df['ema_gain']/df['ema_loss']
    df['rsi'] = df['rs'].apply(lambda x: 100 - (100/(x+1)))
    df['signal_rsi'] = np.where((df['rsi'] < 30),1, np.nan)
    
    #sell signal rsi
    df['signal_rsi'] = np.where((df['rsi'] > 70),-1, df['signal_rsi'])
    df['signal_rsi'].fillna(0,inplace=True)
    df['signal_rsi'] = df['signal_rsi'].ffill().fillna(0)
    df['strategy_rsi'] = df['signal_rsi'].shift(1)*df['return']
    df['cum_strategy_rsi'] = df['strategy_rsi'].dropna().cumsum().apply(np.exp)
    df['cum_return_rsi'] = df['return'].dropna().cumsum().apply(np.exp)
    df['cum_max_rsi'] = df['cum_strategy_rsi'].cummax()
    return df

def mean_reversion(df,window,threshold):
    import numpy as np
    # Mean Reversion 
    df['sma_window'] = df.c.rolling(window).mean()
    df['distance'] = df.c - df['sma_window']
    df['neg_threshold'] = -threshold
    df['pos_threshold'] = threshold
    df.dropna(inplace=True)
    
    # sell signals
    df['signal_MR'] = np.where(df['distance'] > threshold,-1, np.nan)
    # buy signals
    
    df['signal_MR'] = np.where(df['distance'] < -threshold,1, df['signal_MR'])
    
    # crossing of current price and SMA (zero distance)
    df['signal_MR'] = np.where(df['distance'] *
                                df['distance'].shift(1) < 0,
                                0, df['signal_MR'])
    df['signal_MR'] = df['signal_MR'].ffill().fillna(0)
    df['strategy_MR'] = df['signal_MR'].shift(1)*df['return']
    
    # strategy cumulated / returns cumulated / max value cumulated
    df['cum_strategy_MR'] = df['strategy_MR'].dropna().cumsum().apply(np.exp)
    df['cum_return_MR'] = df['return'].dropna().cumsum().apply(np.exp)
    df['cum_max_MR'] = df['cum_strategy_MR'].cummax()
    return df
        
def my_ohlcv_plot(data):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    # Create figure with secondary y-axis
    # Create subplots and mention plot grid size
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'),row_width=[0.2, 0.7])
    
    # Plot OHLC on 1st row
    fig.add_trace(go.Candlestick(x=data.index, open=data["o"], high=data["h"],low=data["l"], close=data["c"], name="OHLC"), row=1, col=1)
    
    # Bar trace for volumes on 2nd row without legend
    fig.add_trace(go.Bar(x=data.index, y=data['volume'], showlegend=False), row=2, col=1)
    
    # Do not show OHLC's rangeslider plot 
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()
    
def set_seeds(seed=100):
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seeds()

# Model import  
def create_rnn_model(hu=100, lags=5, layer='SimpleRNN',
                           features=1, algorithm='estimation'):
    from keras.models import Sequential
    from keras.layers import SimpleRNN, LSTM, Dense
    # create model on sequences time series 
    model = Sequential()
    
    # conditions and if else structure 
    if layer == 'SimpleRNN':
        model.add(SimpleRNN(hu, activation='relu',
                            input_shape=(lags, features)))
    else:
        model.add(LSTM(hu, activation='relu',
                       input_shape=(lags, features)))
    if algorithm == 'estimation':
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model

def subtractions_of_combinations(df):
    from itertools import permutations
    import itertools
    import pandas as pd
    # Extract all the combinations
    combinations = list(itertools.combinations(df.columns, 2)) #  or combinations or permutations
    
    # Calculate the two possible subtractions for each combination
    new_df = pd.DataFrame()
    for a, b in combinations:
        new_df[f'{a}__{b}'] = df[a] - df[b]
    return new_df


def normalize(x, mu, std):
    return (x - mu) / std




    
    
def build_dnn(name, neurons=64, hidden_layers=1, drop_pct=0):
    # Reset random seeds to reproduce results
    reset_random_seeds()
    
    # Name the model
    model = Sequential(name=name)

    # Hidden layer which specifies input shape
    model.add(Dense(neurons, activation='relu',
                    kernel_initializer=random_normal(seed=seed),
                    input_shape=(len(cols),)))

    # If specified, dropout layer to reduce overfitting
    if drop_pct > 0:
        model.add(Dropout(rate=drop_pct, seed=seed))

    # If specified, additional hidden layers added
    if hidden_layers > 1:
        for _ in range(1, hidden_layers):
            model.add(Dense(neurons, activation='relu',
                            kernel_initializer=random_normal(seed=seed)))
            
            # If specified, dropout layer to reduce overfitting
            if drop_pct > 0:
                model.add(Dropout(rate=drop_pct, seed=seed))

    # Final layer with 1 neuron uses sigmoid to output a scalar between 0 and 1
    model.add(Dense(1, activation='sigmoid',
                    kernel_initializer=random_normal(seed=seed)))
    # Model is compiled
    model.compile(optimizer='Adagrad',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_results(models, results):
    import math
    import matplotlib.pyplot as plt    # Calculate the number of rows and columns for subplots
    num_models = len(models)
    num_rows = math.ceil(num_models / 2)
    num_cols = min(2, num_models)

    # Create figure and axes
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 10), sharex=True, sharey=True)
    fig.tight_layout()

    # Iterate through each model parameterisation
    for i, model in enumerate(models):
        row = i // num_cols
        col = i % num_cols

        # Extract the relevant column names for the model
        col_names = [
            f'{model.name}_mean_val_loss',
            f'{model.name}_mean_val_acc',
            f'{model.name}_mean_loss',
            f'{model.name}_mean_acc'
        ]

        # Plot the data for the model on the corresponding subplot
        results[col_names].plot(ax=axes[row, col], fontsize=10, title=f'{model.name} fit results', style=['--', '--', '-', '-'])

        # Add a smaller legend
        axes[row, col].legend(fontsize=8)
        axes[row, col].grid(True)

    # Adjust spacing and layout
    fig.tight_layout(pad=1.5)

    # Display the plot
    plt.show()
