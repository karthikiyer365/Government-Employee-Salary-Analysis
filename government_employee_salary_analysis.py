# -*- coding: utf-8 -*-
"""Government Employee Salary Analysis"""

# Employee Salary Data - Louisville

"""
In the domain of public administration, understanding government employee compensation holds significance in understanding the efficiency of the socio-economic and political standpoints of the government.

We are looking at the real-time data available over a period of the last 5 years to understand the compensation structure for government employees of Louisville.
"""
# %%
## Setup
#Importing Libraries and functions necessary for our initial data scrapping and analysis

import numpy as np
import pandas as pd
import re
import requests
import json
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib as mpl

## Data Scrapping/Collection
"""
Defining our API url where we are going to fetch data from.
Upon going through documentation, we know that the API is open-source and doesnt have a seperate API-Key for accessing the basic employee salary compensation data

Documentation can be found at: https://data.louisvilleky.gov/datasets/LOJIC::louisville-metro-ky-employee-salary-data/api
Data can be found at: https://data.louisvilleky.gov/datasets/LOJIC::louisville-metro-ky-employee-salary-data/explore?showTable=true
API for fetcing json data at: https://services1.arcgis.com/79kfd2K6fskCAkyg/arcgis/rest/services/SalaryData/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json

By going through documentation we can see that we can fetch data as json type by mentioning queryParam `f=json`
"""

url = 'https://services1.arcgis.com/79kfd2K6fskCAkyg/arcgis/rest/services/SalaryData/FeatureServer/0/query?where=calYear=2018&outFields=*&outSR=4326&f=json'
response = requests.get(url)
data = response.json()
print(data)

"""
By parsing data we observe that the necessary data is available inside of the features dictionary
"""

print(data['features'])

"""
We convert this list of multiple dictionary to a dataframe
"""

pd.json_normalize(data['features'])

"""
Hence we have our dataset.
Furthermore by looking at documentation we observe that we can fetch data based of queryParams:
    `calYear` - for years from 2018-2022
    `resultOffset` - for pagination

We use these parameters and fetch remaining data.
This is packaged inside of functions `fetch_json` & `fetch_dataset`
"""

def fetch_json(year,offset):
    url = f'https://services1.arcgis.com/79kfd2K6fskCAkyg/arcgis/rest/services/SalaryData/FeatureServer/0/query?where=calYear={year}&outFields=*&outSR=4326&f=json&resultOffset={(offset*1000)}'
    response = requests.get(url)
    data = response.json()
    return(pd.json_normalize(data['features']))


def fetch_dataset():
    frame = []
    for year in range(2015,2024):
        for offset in range(15):
            fetched_data = fetch_json(year,offset)
            frame.append(fetched_data)

    data = pd.DataFrame()
    data = pd.concat(frame,ignore_index=True)
    return(data)

df = fetch_dataset()

"""
Now that we have our finalized dataset we perform basic data reorganization such as column renaming and save the data to a csv file using function `df.to_csv`
"""

colname = {}
for i in list(df.columns):
    colname[i] = i[11:]
df.rename(columns=colname,inplace=True)

print(df.head())

## Storing Data
"""
We mount our drive and save our dataset to drive as `Salaries_df`.
We can perform further exploration and analysis by recalling this `Salaries_df`
"""

from google.colab import drive
drive.mount('/content/drive')
folder_loc = '/content/drive/MyDrive/Project Files/'
df.to_csv(folder_loc+'Salaries_df.csv', index=False)


## Fetching Stored Data
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv(folder_loc+'Salaries_df.csv')
df.head()

df.describe()

df.isnull().sum()

df.drop(columns='Other', inplace=True,errors='ignore')
df.dropna(inplace=True)
df.nunique()

df['Emp_ID'] = df.groupby('Employee_Name').grouper.group_info[0]
df.drop(columns='ObjectId',inplace=True, errors='ignore')
df.nunique()

df['Calculated_Total'] =df['Regular_Rate'] + df['Incentive_Allowance'] +df['Overtime_Rate']
df

# Assuming 'df' is your DataFrame with the employee data
# Plotting the line trend for each employee
plt.figure(figsize=(10, 6))
i = 0

for employee_id, employee_data in df.groupby('Emp_ID'):
    i = i+1
    plt.plot(employee_data['CalYear'], employee_data['Annual_Rate'], label=f'annual {employee_id}')
    plt.plot(employee_data['CalYear'], employee_data['Calculated_Total'], label=f'calculated {employee_id}')
    if i >=5:
        break

plt.title('Annual Rate Trend for Each Employee')
plt.xlabel('Calendar Year')
plt.ylabel('Annual Rate')
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

salaries_df = df
years = salaries_df['CalYear'].unique()

mean_annual_dict = {}
mean_regular_dict = {}
mean_YTD_dict = {}
yearly_dfs = {}
for year in years:
    yearly_dfs[year] = salaries_df[salaries_df['CalYear'] == year].copy()

for year,df in yearly_dfs.items():
    mean_annual_dict[year] = df.Annual_Rate.mean()
    mean_regular_dict[year] = df.Regular_Rate.mean()
    mean_YTD_dict[year] = df.YTD_Total.mean()


plt.title('Change over time',fontsize=20)
plt.xlabel('Year',fontsize=14)
plt.ylabel('Amount',fontsize=14)

dates = list(mean_YTD_dict.keys())
vals = list(mean_YTD_dict.values())
plt.plot(dates, vals, '-',color='red')

dates = list(mean_annual_dict.keys())
vals = list(mean_annual_dict.values())
plt.plot(dates, vals, '-',color='Blue')

dates = list(mean_regular_dict.keys())
vals = list(mean_regular_dict.values())
plt.plot(dates, vals, '-',color='Green')

plt.figure(figsize=(12, 6))
sns.boxplot(x='CalYear', y='YTD_Total', data=salaries_df)
plt.show()

px.histogram(salaries_df['Annual_Rate'], nbins=50)

px.histogram(salaries_df['Regular_Rate'], nbins=50)

px.histogram(salaries_df['YTD_Total'], nbins=50)

salaries_df.plot(kind='scatter', x='Annual_Rate', y='Regular_Rate', s=32, alpha=.8)
plt.show()

sns.heatmap(salaries_df.corr(),annot=True)

grouped_summary = salaries_df.groupby('Department')['Annual_Rate'].mean()
#grouped_summary
px.scatter(grouped_summary.sort_values())

departments = list(salaries_df['Department'].unique())
#departments
for dept in departments:
    df_dept = salaries_df[salaries_df['Department'] == dept]
    px.scatter(df_dept.groupby('jobTitle')['Annual_Rate'].mean().sort_values())

# Create a dictionary to store DataFrames for each department
department_dfs = {}
for department in departments:
    department_dfs[department] = salaries_df[salaries_df['Department'] == department].copy()

#department_dfs

import matplotlib.pyplot as plt
import pandas as pd


for year, df in yearly_dfs.items():
    department_counts = df['Department'].value_counts()

    plt.figure(figsize=(5, 5))
    plt.pie(department_counts, labels=department_counts.index, autopct = '%1.1f%%',startangle=-90)
    plt.title(f'Distribution of Departments - {year}')
    plt.show()


plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(data=salaries_df, x='Regular_Rate', y='Calculated_Total', hue='Department', s=32, alpha=.8)
plt.title('Scatter Plot of Calculated Total vs. Regular Rate')
plt.xlabel('Calculated_Total')
plt.ylabel('YTD_Total')
plt.legend(title='Department', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(data=salaries_df, x='Annual_Rate', y='YTD_Total', hue='Department', s=32, alpha=.8)
plt.title('Scatter Plot of Annual Rate vs. YTD Total')
plt.xlabel('Annual Rate')
plt.ylabel('YTD_Total')
plt.legend(title='Department', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()





## Dashboard for Government Employee Salary Analysis
import dash
from dash import html, dcc, Input, Output, State
from sklearn.preprocessing import LabelEncoder

records = [record['attributes'] for record in data['features']]
numeric_columns = ['Annual_Rate', 'Regular_Rate', 'Incentive_Allowance', 'Overtime_Rate', 'YTD_Total']
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

df_original_department = df.copy()

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()

df['Department'] = label_encoder.fit_transform(df['Department'])
df['jobTitle'] = label_encoder.fit_transform(df['jobTitle'])

df_original_department['Department_Encoded'] = df_original_department['Department'] + ' (' + df['Department'].astype(str) + ')'

# Initialize Dash app
app = dash.Dash(_name_)

# Define the layout of the dashboard
app.layout = html.Div(style={'backgroundColor': '#f9f9f9', 'padding': '20px'}, children=[
    html.H1("Government Employee Salary Analysis Dashboard", style={'color': '#0066cc'}),

    # Dropdown for selecting department before and after label encoding
    html.Label("Select Department (Before and After Label Encoding):", style={'margin': '20px', 'fontSize': '18px'}),
    dcc.Dropdown(
        id='department-dropdown',
        options=[{'label': department, 'value': department} for department in df_original_department['Department_Encoded'].unique()],
        value=df_original_department['Department_Encoded'].unique()[0],
        style={'width': '50%', 'margin': '20px'}
    ),

    # Bar chart showing average salary by position
    dcc.Graph(id='position-salary-bar-chart'),

    # Scatter plot showing Annual Rate vs. Regular Rate
    dcc.Graph(id='annual-vs-regular-scatter'),

    # Histogram for Annual Rate
    dcc.Graph(id='annual-rate-histogram'),

    # Histogram for YTD Total
    dcc.Graph(id='ytd-total-histogram'),

    # User input for dynamic chart selection
    html.Div([
        html.Label("Select X-axis Attribute:"),
        dcc.Dropdown(
            id='x-axis-dropdown',
            options=[{'label': column, 'value': column} for column in df.columns],
            value='Annual_Rate',
            style={'width': '50%', 'margin': '10px'}
        ),
        html.Label("Select Y-axis Attribute:"),
        dcc.Dropdown(
            id='y-axis-dropdown',
            options=[{'label': column, 'value': column} for column in df.columns],
            value='YTD_Total',
            style={'width': '50%', 'margin': '10px'}
        ),
        html.Label("Select Chart Type:"),
        dcc.RadioItems(
            id='chart-type-radio',
            options=[
                {'label': 'Histogram', 'value': 'histogram'},
                {'label': 'Scatter Plot', 'value': 'scatter'}
            ],
            value='histogram',
            style={'margin': '10px'}
        ),
        dcc.Graph(id='dynamic-chart')
    ], style={'backgroundColor': '#ffffff', 'border': '1px solid #ddd', 'padding': '20px', 'borderRadius': '10px'}),
])

# Define callback to update charts based on dropdown selection
@app.callback(
    [Output('position-salary-bar-chart', 'figure'),
     Output('annual-vs-regular-scatter', 'figure'),
     Output('annual-rate-histogram', 'figure'),
     Output('ytd-total-histogram', 'figure')],
    [Input('department-dropdown', 'value')]
)
def update_charts(selected_department):
    # Extract the original department name from the combined value
    selected_original_department = selected_department.split(' (')[0]

    # Filter data based on selected department
    filtered_df = df_original_department[df_original_department['Department'] == selected_original_department]

    # Bar chart showing average salary by position
    fig1 = px.bar(
        filtered_df,
        x='jobTitle',
        y='Annual_Rate',
        labels={'Annual_Rate': 'Average Annual Rate'},
        title=f'Average Annual Rate by Position in {selected_original_department}',
    )

    # Scatter plot showing Annual Rate vs. Regular Rate
    fig2 = px.scatter(
        filtered_df,
        x='Annual_Rate',
        y='Regular_Rate',
        labels={'Annual_Rate': 'Annual Rate', 'Regular_Rate': 'Regular Rate'},
        title=f'Annual Rate vs. Regular Rate in {selected_original_department}',
    )

    fig3 = px.histogram(filtered_df, x='Annual_Rate', nbins=50, title=f'Annual Rate Distribution in {selected_original_department}')

    # Histogram for YTD Total
    fig4 = px.histogram(filtered_df, x='YTD_Total', nbins=50, title=f'YTD Total Distribution in {selected_original_department}')

    return fig1, fig2, fig3, fig4

# Callback to update dynamic chart based on user input
@app.callback(
    Output('dynamic-chart', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('chart-type-radio', 'value')]
)
def update_dynamic_chart(x_axis, y_axis, chart_type):
    if chart_type == 'histogram':
        fig = px.histogram(df, x=x_axis, y=y_axis, title=f'{y_axis} Distribution')
    else:
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f'{y_axis} vs. {x_axis}')

    return fig

# Run the app
if _name_ == '_main_':
    app.run_server(debug=True, port=8050)