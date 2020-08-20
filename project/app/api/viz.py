import logging
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from fastapi import APIRouter, HTTPException
from app.helpers import *
from app.user import User
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""
    user_ID: str = Field(..., example='1635ob1dkQIz1QMjLmBpt0E36VyM96ImeyrgZ')
    graph_type: str = Field(..., example='pie')
    time_period: str = Field(..., example='week')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    def to_dict(self):
        """Convert pydantic object to python dictionary."""
        return dict(self)

    @validator('user_ID')
    def user_ID_must_exist(cls, value):
        """Validate that user_id is a valid ID."""
        # load sample data and create a set of the user ID's
        users = set(clean_data()['plaid_account_id'])
        assert value in users, f'the user_ID {value} is invalid'
        return value
    
class MoneyFlow(BaseModel):
    """Use this data model to parse the request body JSON."""
    user_ID: str = Field(..., example='1635ob1dkQIz1QMjLmBpt0E36VyM96ImeyrgZ')
    time_period: str = Field(..., example='week')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    def to_dict(self):
        """Convert pydantic object to python dictionary."""
        return dict(self)

    @validator('user_ID')
    def user_ID_must_exist(cls, value):
        """Validate that user_id is a valid ID."""
        # load sample data and create a set of the user ID's
        users = set(clean_data()['plaid_account_id'])
        assert value in users, f'the user_ID {value} is invalid'
        return value



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
        raise HTTPException(
            status_code=404, detail=f'State code {statecode} not found')

    # Get the state's unemployment rate data from FRED
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={statecode}UR'
    df = pd.read_csv(url, parse_dates=['DATE'])
    df.columns = ['Date', 'Percent']

    # Make Plotly figure
    statename = statecodes[statecode]
    fig = px.line(df, x='Date', y='Percent',
                  title=f'{statename} Unemployment Rate')

    # Return Plotly figure as JSON string
    return fig.to_json()

@router.post('/moneyflow')
async def moneyflow(moneyflow: MoneyFlow):
    """
    Visualize a user's money flow 📈
    ### Request Body
    - `User_ID`: str
    - `time_period`: str
    ### Response
    - `plotly object`:
    visualizing the user's money flow over the specified time period.
    """
    # Get the JSON object from the POST request body and cast it to a python dictionary
    input_dict = moneyflow.to_dict()
    user_id = input_dict['user_ID']
    time_period = input_dict['time_period']
    
    transactions = clean_data()
    unique_users = set(transactions['plaid_account_id'].unique())

    # Validate the user
    if user_id not in unique_users:
        raise HTTPException(
            status_code=404, detail=f'User {user_id} not found')

    user = User(user_id, transactions)
    return user.money_flow(time_period=time_period)



@router.post('/spending')
async def spending(item: Item):
    """
    Make visualizations based on past spending 📊
    ### Request Body
    - `User_ID`: str
    - `graph_type`: str
    - `time_period`: str
    ### Response
    - `plotly object`:
    visualizing the user's spending habits in the form of the selected graph
    type.
    """
    # Get the JSON object from the POST request body and cast it to a python dictionary
    input_dict = item.to_dict()
    user_id = input_dict['user_ID']
    graph_type = input_dict['graph_type']
    time_period = input_dict['time_period']
    # Everything below is copy and pasted code from the spending() function in viz.py
    transactions = clean_data()
    unique_users = set(transactions['plaid_account_id'].unique())

    # Validate the user
    if user_id not in unique_users:
        raise HTTPException(
            status_code=404, detail=f'User {user_id} not found')

    user = User(user_id, transactions)

    if graph_type == 'pie':
        return user.categorical_spending(time_period=time_period)

    if graph_type == 'bar':
        return user.bar_viz(time_period=time_period)
