import logging
import random

from fastapi import APIRouter
import pandas as pd
import numpy as np
# import operator
from app.helpers import *
from app.user import User
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()



class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    x1: float = Field(..., example=3.14)
    x2: int = Field(..., example=-42)
    x3: str = Field(..., example='banjo')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @validator('x1')
    def x1_must_be_positive(cls, value):
        """Validate that x1 is a positive number."""
        assert value > 0, f'x1 == {value}, must be > 0'
        return value


class Budget(BaseModel):
    """Use this data model to parse the request body JSON."""

    user_id: str = Field(..., example='1635ob1dkQIz1QMjLmBpt0E36VyM96ImeyrgZ')
    monthly_savings_goal: int = Field(..., example=50)
    placeholder: str = Field(..., example='banjo')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    def to_dict(self):
        """Convert pydantic object to python dictionary."""
        return dict(self)

    @validator('user_id')
    def user_ID_must_exist(cls, value):
        """Validate that user_id is a valid ID."""
        # load sample data and create a set of the user ID's
        users = set(clean_data()['plaid_account_id'])
        assert value in users, f'the user_ID {value} is invalid'
        return value


@router.post('/predict')
async def predict(item: Item):
    """
    Make random baseline predictions for classification problem 🔮

    ### Request Body
    - `x1`: positive float
    - `x2`: integer
    - `x3`: string

    ### Response
    - `prediction`: boolean, at random
    - `predict_proba`: float between 0.5 and 1.0, 
    representing the predicted class's probability

    Replace the placeholder docstring and fake predictions with your own model.
    """

    X_new = item.to_df()
    log.info(X_new)
    y_pred = random.choice([True, False])
    y_pred_proba = random.random() / 2 + 0.5
    return {
        'prediction': y_pred,
        'probability': y_pred_proba
    }


@router.post('/future_budget')
async def future_budget(budget: Budget):
    """
    Suggest a budget for a specified user.

    ### Request Body
    - `user_id`: str
    - `monthly_savings_goal`: integer
    - `placeholder`: string

    ### Response
    - `category`: grandparent category name
    - `budgeted_amount`: integer suggesting the maximum the user should spend 
    in that catgory next month

    """

    # Get the JSON object from the POST request body and cast it to a python dictionary
    input_dict = budget.to_dict()
    user_id = input_dict['user_id']
    monthly_savings_goal = input_dict['monthly_savings_goal']

    transactions = clean_data()
    unique_users = set(transactions['plaid_account_id'].unique())

    # Validate the user
    if user_id not in unique_users:
        raise HTTPException(
            status_code=404, detail=f'User {user_id} not found')

    # instantiate the user
    user = User(user_id, transactions)

    return user.future_budget(monthly_savings_goal=monthly_savings_goal)


@router.get('/current_month_spending')
async def current_month_spending(user_id: str):
    """
    Visualize state unemployment rate from [Federal Reserve Economic Data](https://fred.stlouisfed.org/) 📈

    ### Path Parameter
    `statecode`: The [USPS 2 letter abbreviation](https://en.wikipedia.org/wiki/List_of_U.S._state_and_territory_abbreviations#Table) 
    (case insensitive) for any of the 50 states or the District of Columbia.

    ### Response
    JSON string to render with [react-plotly.js](https://plotly.com/javascript/react/) 
    """

    users = set(clean_data()['plaid_account_id'])
    transactions = clean_data()
    unique_users = set(transactions['plaid_account_id'])

    if user_id not in unique_users:
        raise HTTPException(
            status_code=404, detail=f"User {user_id} doesn't exist")

    user = User(user_id, transactions)

    return user.current_month_spending()
