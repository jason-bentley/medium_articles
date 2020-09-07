# Standard
import pandas as pd
import numpy as np
import os

# Dash components
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# For plotting risk indicator and for creating waterfall plot
import plotly.graph_objs as go
import shap

# To import pkl file model objects
import joblib

# Load model and pipeline
current_folder = os.path.dirname(__file__)
hd_model_obj = joblib.load(os.path.join(current_folder, 'heart_disease_prediction_model_Jul2020.pkl'))

# normally we would want the pipeline object as well, but in this example transformation is minimal so we will just
# construct the require format on the fly from data entry. Also means we don't need to rely on PyCaret here
# object has 2 slots, first is data pipeline, second is the model object
hdpred_model = hd_model_obj[1]
hd_pipeline = []

# Start Dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div([
    html.Div([html.H2('Example Heart Disease Risk Prediction Tool',
                      style={'marginLeft': 20, 'color': 'white'})],
             style={'borderBottom': 'thin black solid',
                    'backgroundColor': '#24a0ed',
                    'padding': '10px 5px'}),
    dbc.Row([
        dbc.Col([html.Div("Patient information",
                          style={'font-weight': 'bold', 'font-size': 20}),
            dbc.Row([html.Div("Patient demographics",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Patient Age (years): '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='55',
                        id='age'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('Sex: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Female', 'value': '0'},
                            {'label': 'Male', 'value': '1'}
                        ],
                        value='0',
                        id='sex_male'
                    )
                ]), width={"size": 3}),
            ], style={'padding': '10px 25px'}),
            dbc.Row([html.Div("Patient health",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Blood pressure (mmHg): '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='132',
                        id='resting_bp'
                    )
                ]), width={"size": 3}, style={'padding': '10px 10px'}),
                dbc.Col(html.Div([
                    html.Label('Maximum heart rate (bpm): '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='151',
                        id='maximum_hr'
                    )
                ]), width={"size": 3}, style={'padding': '10px 10px'}),
                dbc.Col(html.Div([
                    html.Label('Serum cholesterol (mg/L): '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='247',
                        id='serum_cholesterol'
                    )
                ]), width={"size": 3}, style={'padding': '10px 10px'}),
                dbc.Col(html.Div([
                    html.Label('High fasting blood sugar: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'No', 'value': '1'},
                            {'label': 'Yes', 'value': '0'}
                        ],
                        value='1',
                        id='high_fasting_blood_sugar_no'
                    )
                ]), width={"size": 3}, style={'padding': '10px 10px'}),
            ], style={'padding': '10px 25px'}),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Type of chest pain: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Asymptomatic', 'value': '0'},
                            {'label': 'Angina', 'value': '1'},
                            {'label': 'Non-anginal', 'value': '2'}
                        ],
                        value='0',
                        id='chest_pain_type'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('Exercise induced angina: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'No', 'value': '0'},
                            {'label': 'Yes', 'value': '1'}
                        ],
                        value='0',
                        id='exercise_induced_angina_yes'
                    )
                ]), width={"size": 3}),
            ], style={'padding': '10px 25px'}),
            dbc.Row([html.Div("ECG results",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Resting ECG: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Normal', 'value': '0'},
                            {'label': 'Not normal', 'value': '1'}
                        ],
                        value='0',
                        id='resting_ecg_not_normal'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('ST depression: '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='1',
                        id='ST_depression_exercise_vs_rest'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('Peak ST slope: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Upsloping', 'value': '1'},
                            {'label': 'Flat or downsloping', 'value': '0'}
                        ],
                        value='1',
                        id='peak_exercise_ST_segment_slope_upsloping'
                    )
                ]), width={"size": 3}),
            ], style={'padding': '10px 25px'}),
            dbc.Row([html.Div("Thallium stress test results",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Blood flow: '),
                    dcc.Dropdown(
                        options=[
                            {'label': 'Normal', 'value': '1'},
                            {'label': 'Defect', 'value': '0'}
                        ],
                        value='1',
                        id='thallium_stress_test_bf_normal'
                    )
                ]), width={"size": 3}),
                dbc.Col(html.Div([
                    html.Label('Affected vessels: '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='1',
                        id='num_affected_major_vessels'
                    )
                ]), width={"size": 3}),
            ], style={'padding': '10px 25px'}),
        ], style={'padding': '10px 25px'}
        ),

        # Right hand column containing the summary information for predicted heart disease risk
        dbc.Col([html.Div("Predicted heart disease risk",
                          style={'font-weight': 'bold', 'font-size': 20}),
            dbc.Row(dcc.Graph(
                id='Metric_1',
                style={'width': '100%', 'height': 80},
                config={'displayModeBar': False}
            ), style={'marginLeft': 15}),
            dbc.Row([html.Div(id='main_text', style={'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([html.Div("Factors contributing to predicted likelihood of heart disease",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([html.Div(["The figure below indicates the impact (magnitude of increase or decrease in "
                               "log-odds) of factors on the model prediction of the patient's heart disease likelihood."
                               " Blue bars indicate a decrease and red bars indicate an increase in heart disease "
                               "likelihood. The final risk value at the top of the figure is equal to log(p/(1-p)) "
                               " where p is the predicted likelihood reported above."],
                              style={'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row(dcc.Graph(
                id='Metric_2',
                config={'displayModeBar': False}
            ), style={'marginLeft': 15}),
            dbc.Row([html.Div(id='action_header',
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row(
                dbc.Col([html.Div(id='recommended_action')], width={"size": 11},
                        style={'font-size': 16, 'padding': '10px 25px',
                               'backgroundColor': '#E2E2E2', 'marginLeft': 25})),
            ],
            style={'padding': '10px 25px'}
        ),
    ]),
    dbc.Row(
        html.Div(
            [
                dbc.Button(
                    "Predictive model information",
                    id="collapse-button",
                    className="mb-3",
                    color="primary",
                ),
                dbc.Collapse(
                    dbc.Card(dbc.Row([
                        dbc.Col([
                            html.Div('Predictive model information',
                                     style={'font-weight': 'bold', 'font-size': 20, 'padding': '0px 0px 20px 0px'}),
                            html.Div('Data source',
                                     style={'font-weight': 'bold', 'font-size': 14}),
                            html.Div(['A cohort of 303 patients at the Cleveland Clinic were assessed on multiple '
                                      'characteristics and whether they had heart disease was also recorded. '
                                      'In total 45% (n=139) of patients had heart disease.'],
                                     style={'font-size': 14, 'padding': '0px 0px 20px 0px'}),
                            html.Div('Model features and cohort summary',
                                     style={'font-weight': 'bold', 'font-size': 14}),
                            html.Div(['The characteristics/features of the study cohort used to develop the predictive '
                                      'model supporting this tool are shown in Table 1 to the right.'],
                                     style={'font-size': 14, 'padding': '0px 0px 20px 0px'}),
                            html.Div('Model Training',
                                     style={'font-weight': 'bold', 'font-size': 14}),
                            html.Div(['The data was split into a training set (65%, n=196) and a test set (35%, '
                                      'n=107). Using 10-fold cross validation with the training set an initial model '
                                      'was developed and further refined with hyper-parameter tuning. The final model '
                                      'achieved an average AUC of 0.91 (+/- 0.11) in the training set and 0.88 in the '
                                      'test set. Figure 1 to the far right indicates what the model identified as '
                                      'importance of predictors of heart disease. The more important features are '
                                      'towards the top of the figure, which includes characteristics like affected '
                                      'major vessels, asymptomatic chest pain and if the thallium stress test '
                                      'indicates normal blood flow.'],
                                     style={'font-size': 14, 'padding': '0px 0px 20px 0px'}),
                            html.Div(['The information provided in this dashboard should not replace the advice or '
                                      'instruction of your Doctor or Health Care Professional.'],
                                     style={'font-weight': 'bold', 'font-size': 14}),
                        ]),
                        dbc.Col([
                            html.Div('Table 1. Cohort Table',
                                     style={'font-weight': 'bold', 'font-size': 20, 'textAlign': 'middle'}),
                            html.Div(className='container',
                                children=[html.Img(src=app.get_asset_url('Cohort_table.png'),
                                                   style={'height': '100%', 'width': '100%'})])]),
                        dbc.Col([
                            html.Div('Figure 1. SHAP feature importance',
                                     style={'font-weight': 'bold', 'font-size': 20}),
                            html.Div(className='container',
                                     children=[html.Img(src=app.get_asset_url('SHAP_importance.png'),
                                                        style={'height': '90%', 'width': '90%'})])])
                        ], style={'padding': '20px 20px'})),
                    id="collapse",
                ),
            ]
        ),
        style={'padding': '10px 25px',
               'position': 'fixed',
               'bottom': '0'},
    ),
    html.Div(id='data_patient', style={'display': 'none'}),
    ]
)


# Responsive elements: toggle button for viewing model information
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# Responsive element: create X matrix for input to model estimation
@app.callback(
    Output('data_patient', 'children'),
    [Input('age', 'value'),
     Input('resting_bp', 'value'),
     Input('serum_cholesterol', 'value'),
     Input('maximum_hr', 'value'),
     Input('ST_depression_exercise_vs_rest', 'value'),
     Input('num_affected_major_vessels', 'value'),
     Input('sex_male', 'value'),
     Input('chest_pain_type', 'value'),
     Input('high_fasting_blood_sugar_no', 'value'),
     Input('resting_ecg_not_normal', 'value'),
     Input('exercise_induced_angina_yes', 'value'),
     Input('peak_exercise_ST_segment_slope_upsloping', 'value'),
     Input('thallium_stress_test_bf_normal', 'value')
     ]
)
def generate_feature_matrix(age, resting_bp, serum_cholesterol, maximum_hr, ST_depression_exercise_vs_rest,
                            num_affected_major_vessels, sex_male, chest_pain_type, high_fasting_blood_sugar_no,
                            resting_ecg_not_normal, exercise_induced_angina_yes,
                            peak_exercise_ST_segment_slope_upsloping, thallium_stress_test_bf_normal):

    # generate a new X_matrix for use in the predictive models
    column_names = ['age', 'resting_bp', 'serum_cholesterol', 'maximum_hr', 'ST_depression_exercise_vs_rest',
                    'num_affected_major_vessels', 'sex_male', 'chest_pain_anginal_pain', 'chest_pain_asymptomatic',
                    'chest_pain_non_anginal_pain', 'high_fasting_blood_sugar_no', 'resting_ecg_not_normal',
                    'exercise_induced_angina_yes', 'peak_exercise_ST_segment_slope_upsloping',
                    'thallium_stress_test_bf_normal']

    # only input that requires additional processing is the chest_pain input
    chest_pain_anginal_pain = 0
    chest_pain_asymptomatic = 0
    chest_pain_non_anginal_pain = 0
    if chest_pain_type == 0:
        chest_pain_asymptomatic = 1
    elif chest_pain_type == 1:
        chest_pain_anginal_pain = 1
    elif chest_pain_type == 2:
        chest_pain_non_anginal_pain = 1

    values = [age, resting_bp, serum_cholesterol, maximum_hr, ST_depression_exercise_vs_rest,
              num_affected_major_vessels, sex_male, chest_pain_anginal_pain, chest_pain_asymptomatic,
              chest_pain_non_anginal_pain, high_fasting_blood_sugar_no, resting_ecg_not_normal,
              exercise_induced_angina_yes, peak_exercise_ST_segment_slope_upsloping, thallium_stress_test_bf_normal]

    x_patient = pd.DataFrame(data=[values],
                             columns=column_names,
                             index=[0])

    return x_patient.to_json()


@app.callback(
    [Output('Metric_1', 'figure'),
     Output('main_text', 'children'),
     Output('action_header', 'children'),
     Output('recommended_action', 'children'),
     Output('Metric_2', 'figure')],
    [Input('data_patient', 'children')]
)
def predict_hd_summary(data_patient):

    # read in data and predict likelihood of heart disease
    x_new = pd.read_json(data_patient)
    y_val = hdpred_model.predict_proba(x_new)[:, 1]*100
    text_val = str(np.round(y_val[0], 1)) + "%"

    # assign a risk group
    if y_val/100 <= 0.275685:
        risk_grp = 'low risk'
    elif y_val/100 <= 0.795583:
        risk_grp = 'medium risk'
    else:
        risk_grp = 'high risk'

    # assign an action related to the risk group
    rg_actions = {'low risk': ['Discuss with patient any single large risk factors they may have, and otherwise '
                               'continue supporting healthy lifestyle habits. Follow-up in 12 months'],
                  'medium risk': ['Discuss lifestyle with patient and identify changes to reduce risk. '
                                  'Schedule follow-up with patient in 3 months on how changes are progressing. '
                                  'Recommend performing simple tests to assess positive impact of changes.'],
                  'high risk': ['Immediate follow-up with patient to discuss next steps including additional '
                                'follow-up tests, lifestyle changes and medications.']}

    next_action = rg_actions[risk_grp][0]

    # create a single bar plot showing likelihood of heart disease
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=[''],
        x=y_val,
        marker_color='rgb(112, 128, 144)',
        orientation='h',
        width=1,
        text=text_val,
        textposition='auto',
        hoverinfo='skip'
    ))

    # add blocks for risk groups
    bot_val = 0.5
    top_val = 1

    fig1.add_shape(
        type="rect",
        x0=0,
        y0=bot_val,
        x1=0.275686 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="green"
    )
    fig1.add_shape(
        type="rect",
        x0=0.275686 * 100,
        y0=bot_val,
        x1=0.795584 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="orange"
    )
    fig1.add_shape(
        type="rect",
        x0=0.795584 * 100,
        y0=bot_val,
        x1=1 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="red"
    )
    fig1.add_annotation(
        x=0.275686 / 2 * 100,
        y=0.75,
        text="Low risk",
        showarrow=False,
        font=dict(color="black", size=14)
    )
    fig1.add_annotation(
        x=0.53 * 100,
        y=0.75,
        text="Medium risk",
        showarrow=False,
        font=dict(color="black", size=14)
    )
    fig1.add_annotation(
        x=0.9 * 100,
        y=0.75,
        text="High risk",
        showarrow=False,
        font=dict(color="black", size=14)
    )
    fig1.update_layout(margin=dict(l=0, r=50, t=10, b=10), xaxis={'range': [0, 100]})

    # do shap value calculations for basic waterfall plot
    explainer_patient = shap.TreeExplainer(hdpred_model)
    shap_values_patient = explainer_patient.shap_values(x_new)
    updated_fnames = x_new.T.reset_index()
    updated_fnames.columns = ['feature', 'value']
    updated_fnames['shap_original'] = pd.Series(shap_values_patient[0])
    updated_fnames['shap_abs'] = updated_fnames['shap_original'].abs()
    updated_fnames = updated_fnames.sort_values(by=['shap_abs'], ascending=True)

    # need to collapse those after first 9, so plot always shows 10 bars
    show_features = 9
    num_other_features = updated_fnames.shape[0] - show_features
    col_other_name = f"{num_other_features} other features"
    f_group = pd.DataFrame(updated_fnames.head(num_other_features).sum()).T
    f_group['feature'] = col_other_name
    plot_data = pd.concat([f_group, updated_fnames.tail(show_features)])

    # additional things for plotting
    plot_range = plot_data['shap_original'].cumsum().max() - plot_data['shap_original'].cumsum().min()
    plot_data['text_pos'] = np.where(plot_data['shap_original'].abs() > (1/9)*plot_range, "inside", "outside")
    plot_data['text_col'] = "white"
    plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] < 0), 'text_col'] = "#3283FE"
    plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] > 0), 'text_col'] = "#F6222E"

    fig2 = go.Figure(go.Waterfall(
        name="",
        orientation="h",
        measure=['absolute'] + ['relative']*show_features,
        base=explainer_patient.expected_value,
        textposition=plot_data['text_pos'],
        text=plot_data['shap_original'],
        textfont={"color": plot_data['text_col']},
        texttemplate='%{text:+.2f}',
        y=plot_data['feature'],
        x=plot_data['shap_original'],
        connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(102, 102, 102)", "dash": "dot"}},
        decreasing={"marker": {"color": "#3283FE"}},
        increasing={"marker": {"color": "#F6222E"}},
        hoverinfo="skip"
    ))
    fig2.update_layout(
        waterfallgap=0.2,
        autosize=False,
        width=800,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            gridcolor='lightgray'
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
            linecolor='black',
            tickcolor='black',
            ticks='outside',
            ticklen=5
        ),
        margin={'t': 25, 'b': 50},
        shapes=[
            dict(
                type='line',
                yref='paper', y0=0, y1=1.02,
                xref='x', x0=plot_data['shap_original'].sum()+explainer_patient.expected_value,
                x1=plot_data['shap_original'].sum()+explainer_patient.expected_value,
                layer="below",
                line=dict(
                    color="black",
                    width=1,
                    dash="dot")
            )
        ]
    )
    fig2.update_yaxes(automargin=True)
    fig2.add_annotation(
        yref='paper',
        xref='x',
        x=explainer_patient.expected_value,
        y=-0.12,
        text="E[f(x)] = {:.2f}".format(explainer_patient.expected_value),
        showarrow=False,
        font=dict(color="black", size=14)
    )
    fig2.add_annotation(
        yref='paper',
        xref='x',
        x=plot_data['shap_original'].sum()+explainer_patient.expected_value,
        y=1.075,
        text="f(x) = {:.2f}".format(plot_data['shap_original'].sum()+explainer_patient.expected_value),
        showarrow=False,
        font=dict(color="black", size=14)
    )

    return fig1,\
        f"Based on the patient's profile, the predicted likelihood of heart disease is {text_val}. " \
        f"This patient is in the {risk_grp} group.",\
        f"Recommended action(s) for a patient in the {risk_grp} group",\
        next_action, \
        fig2


# Start the dashboard with defined host and port.
if __name__ == '__main__':
    app.run_server(debug=False,
                   host='127.0.0.1',
                   port=8000)
