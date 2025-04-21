import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import pickle
import os
import glob
import plotly.colors

SAVES_DIR = "0"

# Load all saved pricing results
save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "saves", SAVES_DIR)
all_files = glob.glob(os.path.join(save_dir, "*.pkl"))

# Extract sampler and pricer names from filenames
samplers = set()
sampler_pricer_data = {}

for file in all_files:
    filename = os.path.basename(file)
    parts = filename.split('_')
    sampler_name = parts[0]
    pricer_name = parts[1]
    
    samplers.add(sampler_name)
    
    with open(file, 'rb') as f:
        train_prices, test_prices, cnt_trajectories = pickle.load(f)
    
    if sampler_name not in sampler_pricer_data:
        sampler_pricer_data[sampler_name] = {}
    
    sampler_pricer_data[sampler_name][pricer_name] = {
        'train_prices': train_prices,
        'test_prices': test_prices,
        'cnt_trajectories': cnt_trajectories
    }

# Create Dash app
app = dash.Dash(__name__)

# Generate distinct colors for each pricer
if sampler_pricer_data:
    pricer_names = list(next(iter(sampler_pricer_data.values())).keys())
    colors = plotly.colors.qualitative.Plotly[:len(pricer_names)]
    pricer_colors = dict(zip(pricer_names, colors))
else:
    pricer_colors = {}

def rgba_color(hex_color, alpha):
    """Convert hex color to rgba string with given alpha"""
    rgb = plotly.colors.hex_to_rgb(hex_color)
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"

app.layout = html.Div([
    html.H1("Pricing Methods Comparison Dashboard"),
    
    html.Div([
        html.Label("Select Sampler:"),
        dcc.Dropdown(
            id='sampler-dropdown',
            options=[{'label': s, 'value': s} for s in sorted(samplers)],
            value=list(samplers)[0] if samplers else None
        ),
    ], style={'width': '30%', 'margin': '20px'}),
    
    html.Div([
        html.Label("Y-axis Scale:"),
        dcc.RadioItems(
            id='yaxis-scale',
            options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'Logarithmic', 'value': 'log'}
            ],
            value='linear'
        ),
    ], style={'margin': '20px'}),
    
    dcc.Graph(id='pricing-graph'),
    dcc.Graph(id='std-graph')
])

@app.callback(
    [Output('pricing-graph', 'figure'),
     Output('std-graph', 'figure')],
    [Input('sampler-dropdown', 'value'),
     Input('yaxis-scale', 'value')]
)
def update_graphs(selected_sampler, yaxis_scale):
    price_fig = go.Figure()
    std_fig = go.Figure()
    
    if not selected_sampler or selected_sampler not in sampler_pricer_data:
        return price_fig, std_fig
    
    pricers_data = sampler_pricer_data[selected_sampler]
    
    for pricer_name, data in pricers_data.items():
        cnt_trajectories = data['cnt_trajectories']
        color = pricer_colors.get(pricer_name, '#636EFA')  # default Plotly blue
        
        # Add train data to price figure
        train_prices = data['train_prices']
        train_mean = np.mean(train_prices, axis=1)
        train_std = np.std(train_prices, axis=1)
        
        # Main train line
        price_fig.add_trace(go.Scatter(
            x=cnt_trajectories,
            y=train_mean,
            mode='lines',
            name=f'{pricer_name} (train)',
            line=dict(width=2, color=color),
            legendgroup=f'{pricer_name}_train',
        ))
        
        # Train confidence interval
        price_fig.add_trace(go.Scatter(
            x=np.concatenate([cnt_trajectories, cnt_trajectories[::-1]]),
            y=np.concatenate([train_mean - train_std, (train_mean + train_std)[::-1]]),
            fill='toself',
            fillcolor=rgba_color(color, 0.2),
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=f'{pricer_name}_train',
        ))
        
        # Train std line
        std_fig.add_trace(go.Scatter(
            x=cnt_trajectories,
            y=train_std,
            mode='lines',
            name=f'{pricer_name} (train std)',
            line=dict(width=2, color=color),
            legendgroup=f'{pricer_name}_train_std',
        ))
        
        # Add test data if available
        if data['test_prices'] is not None:
            test_prices = data['test_prices']
            test_mean = np.mean(test_prices, axis=1)
            test_std = np.std(test_prices, axis=1)
            
            # Main test line (dashed)
            price_fig.add_trace(go.Scatter(
                x=cnt_trajectories,
                y=test_mean,
                mode='lines',
                name=f'{pricer_name} (test)',
                line=dict(width=2, dash='dot', color=color),
                legendgroup=f'{pricer_name}_test',
            ))
            
            # Test confidence interval
            price_fig.add_trace(go.Scatter(
                x=np.concatenate([cnt_trajectories, cnt_trajectories[::-1]]),
                y=np.concatenate([test_mean - test_std, (test_mean + test_std)[::-1]]),
                fill='toself',
                fillcolor=rgba_color(color, 0.1),
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f'{pricer_name}_test',
            ))
            
            # Test std line (dashed)
            std_fig.add_trace(go.Scatter(
                x=cnt_trajectories,
                y=test_std,
                mode='lines',
                name=f'{pricer_name} (test std)',
                line=dict(width=2, dash='dot', color=color),
                legendgroup=f'{pricer_name}_test_std',
            ))
    
    price_fig.update_layout(
        title=f'Pricing Results for {selected_sampler}',
        xaxis_title='Number of Trajectories',
        yaxis_title='Price',
        yaxis_type=yaxis_scale,
        hovermode='x unified',
        showlegend=True
    )
    
    std_fig.update_layout(
        title=f'Standard Deviation for {selected_sampler}',
        xaxis_title='Number of Trajectories',
        yaxis_title='Standard Deviation',
        yaxis_type=yaxis_scale,
        hovermode='x unified',
        showlegend=True
    )
    
    return price_fig, std_fig

if __name__ == '__main__':
    app.run_server(debug=False, host="127.0.0.1", port=7777)