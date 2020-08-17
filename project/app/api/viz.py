from fastapi import APIRouter, HTTPException
import pandas as pd
import plotly.express as px
from app.helpers import *
from app.user import User

router = APIRouter()


@router.get('/viz/{statecode}')
async def viz(statecode: str):
    """
    Visualize state unemployment rate from [Federal Reserve Economic Data](https://fred.stlouisfed.org/) 📈
    
    ### Path Parameter
    `statecode`: The [USPS 2 letter abbreviation](https://en.wikipedia.org/wiki/List_of_U.S._state_and_territory_abbreviations#Table) 
    (case insensitive) for any of the 50 states or the District of Columbia.

    ### Response
    JSON string to render with [react-plotly.js](https://plotly.com/javascript/react/) 
    """

    # Validate the state code
    statecodes = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 
        'DE': 'Delaware', 'DC': 'District of Columbia', 'FL': 'Florida', 
        'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 
        'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 
        'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 
        'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 
        'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 
        'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 
        'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 
        'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 
        'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    statecode = statecode.upper()
    if statecode not in statecodes:
        raise HTTPException(status_code=404, detail=f'State code {statecode} not found')

    # Get the state's unemployment rate data from FRED
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={statecode}UR'
    df = pd.read_csv(url, parse_dates=['DATE'])
    df.columns = ['Date', 'Percent']

    # Make Plotly figure
    statename = statecodes[statecode]
    fig = px.line(df, x='Date', y='Percent', title=f'{statename} Unemployment Rate')

    # Return Plotly figure as JSON string
    return fig.to_json()

@router.get('/{user_id}/moneyflow')
async def moneyflow(user_id: str):
    """
    Visualize a user's money flow
    
    ### Path Parameter
    `user_id`: The unique plaid_account_id of a user

    ### Response
    JSON string to render with [react-plotly.js](https://plotly.com/javascript/react/) 
    """

    transactions = clean_data()
    
    unique_users = set(transactions['plaid_account_id'].unique())
    
    # Validate the user
    if user_id not in unique_users:
        raise HTTPException(status_code=404, detail=f'User {user_id} not found')

    user = User(user_id, transactions)
    return user.money_flow()

@router.get('/{user_id}/spending/{graph_type}')
async def spending(user_id: str, graph_type: str):
    """
    Visualize a user's spending history by category
    
    ### Path Parameter
    `user_id`: The unique plaid_account_id of a user
    
    `graph_type`: pie or bar

    ### Response
    JSON string to render with [react-plotly.js](https://plotly.com/javascript/react/) 
    """ 
    transactions = clean_data()
    
    unique_users = set(transactions['plaid_account_id'].unique())
    
    # Validate the user
    if user_id not in unique_users:
        raise HTTPException(status_code=404, detail=f'User {user_id} not found')
    
    user = User(user_id, transactions)
    if graph_type == 'pie':
        return user.categorical_spending()
    if graph_type == 'bar':
        return user.bar_viz()