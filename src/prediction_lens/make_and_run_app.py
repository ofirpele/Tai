from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

import pandas as pd
import numpy as np

import copy
import math

from TaiClassifier import TaiClassifier

from typing import Any

def make_and_run_app(
        clf : TaiClassifier, 
        y_vis : int, 
        proba_y_vis_name : str, 
        features_vis_order : list[int],
        features_vis_name : list[str],
        features_vis_x_axis_min : list[float],
        features_vis_x_axis_max : list[float],
        features_init_value : list[float],
        features_is_only_min_and_max : list[bool],
        num_update_delta_proba_figures_steps : int = 100,
        max_slider_tics : int = 50,
        run_params : dict[str,Any] = {'debug' : True}
        ):
    
    APP_NAME = 'Tai Prediction Lens'

    assert y_vis in [0, 1]
    # TODO FUTURE: assert on inputs (lengths, include all range)
    
    delta_proba_name = f'\u0394{proba_y_vis_name}'

    BACKGROUND_COLOR = '#121212'  # Dark
    TEXT_COLOR = '#FFFFFF'  # White 
    PURPLE_COLOR = '#BB86FC'  
    TEAL_COLOR = '#03DAC6'  
    FONT_FAMILY = "'Roboto', 'Segoe UI', 'Helvetica Neue', sans-serif"
    
    NEGATIVE_COLOR = '#FF5252'  # Red
    NEUTRAL_COLOR = '#BB86FC'   # Purple
    POSITIVE_COLOR = '#4CAF50'  # Green
    #########################################################################################

    #########################################################################################
    # Create features_infos
    #########################################################################################
    features_infos = []
    for f_name, f_min, f_max, f_init_value, f_is_only_min_and_max in zip(
            clf.active_features_names,
            clf.linear_clf_for_monotone_constraints.scaler.data_min_,
            clf.linear_clf_for_monotone_constraints.scaler.data_max_,
            features_init_value,
            features_is_only_min_and_max
    ):
        step = (f_max - f_min) / max_slider_tics
        if f_is_only_min_and_max:
            step = (f_max - f_min)
        features_infos.append({
            'name': f_name,
            'min': f_min,
            'max': f_max,
            'default': f_init_value,
            'step': step
        })
    #########################################################################################
    
    #########################################################################################
    # Create and group sliders into rows of 3
    #########################################################################################
    rows = []
    total_features = len(features_infos)
    num_rows = math.ceil(total_features / 3)
    
    for i in range(num_rows):
        row_items = []
        for j in range(3):
            idx = i * 3 + j
            if idx < total_features:
                idx_orig = features_vis_order[idx]
                feature = features_infos[idx_orig]
                m = clf.monotone_constraints[idx_orig] * (+1 if y_vis==1 else -1) # monotonicity is in respect to y==1
                if m == -1:
                    m_str = '\u25BC'
                    feature_name_style_color = NEGATIVE_COLOR
                elif m == 0:
                    m_str = '0'
                    feature_name_style_color = NEUTRAL_COLOR
                elif m == +1:
                    m_str = '\u25B2'
                    feature_name_style_color = POSITIVE_COLOR
                feature_name_style = {
                    'textAlign': 'center',
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'color': feature_name_style_color,
                    'marginBottom': '8px',
                    'whiteSpace': 'nowrap',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'
                }
                row_items.append(
                    html.Div([
                        html.H3(f'{features_vis_name[idx_orig]} {m_str}', style=feature_name_style),
                        dcc.Graph(
                            id=f'{delta_proba_name}{feature['name']}',
                            config={'displayModeBar': False},
                            style={
                                'borderRadius': '5px',
                                'overflow': 'hidden',
                                'marginBottom': '8px',
                                'height': '100px'  
                            }
                        ),
                        html.Div(
                            dcc.Slider(
                                min=feature['min'],
                                max=feature['max'],
                                step=feature['step'],
                                value=feature['default'],
                                id=feature['name'],
                                tooltip={'placement': 'bottom', 'always_visible': True, 'transform': 'toFixedTwo'}, # toFixedTwo is a JS function in assets/tooltip
                                marks=None,
                            ),
                            style={'margin': '0 5px', 'padding': '3px 0', 'width': '355px'}
                        ),
                    ], style={
                        'width': 'calc(33.333% - 10px)',
                        'padding': '0 5px',
                        'marginBottom': '15px',
                        'boxSizing': 'border-box',
                        'minWidth': '200px',
                        'maxWidth': '350px',
                        'flexGrow': '1',
                        'flexShrink': '1',
                        'flexBasis': 'calc(33.333% - 10px)'
                    })
                )
        rows.append(html.Div(row_items, style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'margin': '0 -5px',
            'justifyContent': 'flex-start',
            'alignItems': 'stretch'
        }))
    #########################################################################################

    #########################################################################################
    app = Dash(__name__)
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>''' + APP_NAME + '''</title>
            {%favicon%}
            {%css%}
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
            <style>
                body {
                    font-family: ''' + FONT_FAMILY + ''';
                    background-color: ''' + BACKGROUND_COLOR + ''';
                    color: ''' + TEXT_COLOR + ''';
                    margin: 0;
                    padding: 20px;
                }
                .rc-slider-track {
                    background-color: ''' + PURPLE_COLOR + ''';
                }
                .rc-slider-handle {
                    border-color: ''' + TEAL_COLOR + ''';
                    background-color: ''' + TEAL_COLOR + ''';
                }
                .rc-slider-rail {
                    background-color: #333333;
                }
            </style>
            {%scripts%}
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    #########################################################################################

    #########################################################################################
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1(APP_NAME, style={
                'fontSize': '16px',
                'fontWeight': '500',
                'color': PURPLE_COLOR,
                'margin': '0 0 0 0',
                'textAlign': 'left'
            })
        ], style={
            'marginBottom': '4px',
            'textAlign': 'center',
            'padding': '20px',
            'backgroundColor': 'rgba(30, 30, 30, 0.7)',
            'borderRadius': '12px',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.3)',
        }),
        
        # Main content
        html.Div([
            # Left side: Sliders
            html.Div([
                html.Div(rows, style={
                    'marginBottom': '20px',
                    'backgroundColor': '#1E1E1E',
                    'borderRadius': '12px',
                    'padding': '15px',
                    'boxShadow': '0 8px 16px rgba(0, 0, 0, 0.4)'
                })
            ], style={
                'flex': '1',
                'marginRight': '15px',
                'minWidth': '0',
                'maxWidth': '100%',
                'overflowX': 'hidden'
            }),
            
            # Right side: Probability bar
            html.Div([
                html.Div([
                    # html.H2(proba_y_vis_name, style={
                    #     'fontSize': '14px',
                    #     'fontWeight': '500',
                    #     'color': TEAL_COLOR,
                    #     'margin': '0 0 0 0',
                    #     'textAlign': 'center'
                    # }),
                    dcc.Graph(
                        id=proba_y_vis_name,
                        config={'displayModeBar': False},
                        style={
                            'height': '300px',
                            'borderRadius': '8px',
                            'overflow': 'hidden'
                        }
                    )
                ], style={
                    'backgroundColor': '#1E1E1E',
                    'borderRadius': '12px',
                    'padding': '25px',
                    'height': '100%',
                    'boxShadow': '0 8px 16px rgba(0, 0, 0, 0.4)'
                })
            ], style={
                'width': '300px',
                'marginLeft': '15px',
                'position': 'sticky',
                'top': '20px', 
                'alignSelf': 'flex-start', 
                'zIndex': '1000', 
                'height': 'fit-content' 
            })
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap'
        }),

        # Footer
        html.Div([
            html.H3('© 2025 Ofir Pele', style={
                'fontSize': '16px',
                'fontWeight': '500',
                'color': '#777B7E',
                'margin': '0 0 0 0',
                'textAlign': 'left'
            }),
            # dcc.Link("© 2025 Ofir Pele", href="https://www.linkedin.com/in/ofir-pele", style={
            #     'fontSize': '16px',
            #     'fontWeight': '500',
            #     'color': '#777B7E',
            #     'margin': '0 0 0 0',
            #     'textAlign': 'left'
            # }), 
        ], style={
            'marginBottom': '4px',
            'textAlign': 'center',
            'padding': '20px',
            'backgroundColor': 'rgba(30, 30, 30, 0.7)',
            'borderRadius': '12px',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.3)',
        }),

    ], style={
        'fontFamily': FONT_FAMILY,
        'backgroundColor': BACKGROUND_COLOR,
        'color': TEXT_COLOR,
        'margin': '0',
        'padding': '20px',
        'maxWidth': '1400px',
        'marginLeft': 'auto',
        'marginRight': 'auto',
        'marginBottom': 'auto'
    })
    #########################################################################################

    ##############################################################################################################################
    # Callbacks for dynamic functionality
    ##############################################################################################################################
    def create_X_from_args(num_rows, *args):
        num_all_features = len(clf.features_transformer.cols_to_include()) + len(clf.features_transformer.cols_to_remove())
        X = np.empty((num_rows, num_all_features))
        ci = 0
        for val in args:
            X[:, clf.features_transformer.cols_to_include()[ci]].fill(val)
            ci += 1
        return X
    
    def curr_proba_val(*args):
        X = create_X_from_args(1, *args)
        return (clf.predict_proba(X).squeeze()[y_vis]).item()

    callback_inputs = [Input(f_name, 'value') for f_name in clf.active_features_names]

    @app.callback([Output(f'{delta_proba_name}{f_name}', 'figure') for f_name in clf.active_features_names], callback_inputs)
    def update_delta_proba_figures(*args):
        X = create_X_from_args(num_update_delta_proba_figures_steps, *args)
        
        res = []
        for col, f_min, f_max, f_vis_x_axis_min, f_vis_x_axis_max, f_is_only_min_and_max, f_vis_name in zip(
            clf.features_transformer.cols_to_include(),
            clf.linear_clf_for_monotone_constraints.scaler.data_min_,
            clf.linear_clf_for_monotone_constraints.scaler.data_max_,
            features_vis_x_axis_min,
            features_vis_x_axis_max,
            features_is_only_min_and_max,
            features_vis_name
        ):
            X_col_orig = copy.deepcopy(X[:, col])

            X[:, col] = np.linspace(f_min, f_max, num=num_update_delta_proba_figures_steps)
            
            func_vals = clf.predict_proba(X)[:, y_vis]
            
            df_mat = np.empty((num_update_delta_proba_figures_steps, 2))
            df_mat[:, 0] = X[:, col]
            df_mat[:, 1] = func_vals - curr_proba_val(*args)
            xy = ['', delta_proba_name]
            df_res = pd.DataFrame(df_mat, columns=xy)
            
            fig = go.Figure()
            
            i_vals = [0, len(df_res)-1] if f_is_only_min_and_max else range(len(df_res))
            for i in i_vals:
                
                x_curr = df_res[xy[0]][i]
                y_curr = df_res[xy[1]][i]

                zero_thresh = 0.0001
                if y_curr < -zero_thresh:  
                    line_color = NEGATIVE_COLOR
                elif y_curr > zero_thresh: 
                    line_color = POSITIVE_COLOR
                else:  
                    line_color = NEUTRAL_COLOR
                
                scatter_shared_params = {
                    'x' : [x_curr],
                    'y' : [y_curr],
                    'mode' : 'markers',
                    'marker_color' : line_color,
                    'showlegend' : False,
                    'hovertemplate' : 
                        '\u0394' + proba_y_vis_name + ':%{y:.2f}' +
                        '<br>' + 
                        f_vis_name + ': %{x:.2f}' + 
                        '<extra></extra>',
                }
                if f_is_only_min_and_max:                  
                    fig.add_trace(go.Scatter(
                        marker_size = 7,
                        **scatter_shared_params                      
                ))
                else:
                    fig.add_trace(go.Scatter(                        
                        marker_size=4,
                        marker_symbol='square',
                        **scatter_shared_params
                    ))

            xaxis = {
                'gridcolor': '#333333',
                'zerolinecolor': '#444444',
                'range': [f_vis_x_axis_min, f_vis_x_axis_max]
            }
            if f_is_only_min_and_max:
               xaxis['tickmode'] = 'array'
               xaxis['tickvals'] = [f_min, f_max]
               
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor=BACKGROUND_COLOR,
                plot_bgcolor='#1E1E1E',
                font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                yaxis=dict(
                    gridcolor='#333333',
                    zerolinecolor='#444444',
                    range=[-1, 1]
                ),
                xaxis=xaxis,                
            )
            
            res.append(fig)
            
            X[:, col] = X_col_orig
        return res

    @app.callback(Output(proba_y_vis_name, 'figure'), callback_inputs)
    def update_proba_figure(*args):
        current_prob = curr_proba_val(*args)
        current_prob = max(0, min(current_prob, 1))

        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[''],
            y=[current_prob],
            marker=dict(
                color=[
                    f'rgba({51}, {196}, {173}, {current_prob:.3f})',  # Base on TEAL_COLOR with transparency
                ],                
                line=dict(width=0)
            ),
            width=0.3,
            hoverinfo='skip'
        ))
        
        # Add current probability text annotation
        fig.add_annotation(
            x=0,
            y=current_prob + 0.05,
            text=f'{proba_y_vis_name}={current_prob:.3f}',
            showarrow=False,
            font=dict(
                family=FONT_FAMILY,
                size=16,
                color=TEAL_COLOR
            ),
            yshift=10
        )
        
        # Update layout
        fig.update_layout(
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor='#1E1E1E',
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            yaxis=dict(
                range=[0, 1.1],
                gridcolor='#333333',
                zerolinecolor='#444444',
            ),
            font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
        )
        
        return fig
    
    print('Click here: http://127.0.0.1:8050/')
    app.run(**run_params)
    return app