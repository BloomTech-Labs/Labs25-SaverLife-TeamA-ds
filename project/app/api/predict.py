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


def weighted_avg(data):
    """
    Given a dataframe
    Return the datafram with a new column of weighted averages
    """
    categories = data.index
    # get number of timepoints/observations and create weights
    N = data.shape[1]
    weights = [i for i in range(N ,0,-1)]
    averages = []
    # for each categorie, calculate the weighted average and 
    # store in the averages list
    for cat in categories:
        cat_data = list(data.loc[cat])
        # replace nan vaalues with 0 
        for i in range(len(cat_data)):
            if str(cat_data[i]) == 'nan':
                cat_data[i] = 0
        avg = np.average(cat_data, weights=weights)
        averages.append(avg)

    data['mean'] = averages
    data = data.round()
    return data


def monthly_avg_spending(user_expenses_df, num_months=6, category='grandparent_category_name', weighted=True):
    ticker = 0
    cur_month = user_expenses_df['date'].max().month
    cur_year = user_expenses_df['date'].max().year

    while ticker < num_months:
        # on first iteration
        if ticker == 0:

            if cur_month == 1:
                prev_month = 12
                user_exp_prev = user_expenses_df[(user_expenses_df['date'].dt.month == (
                    prev_month)) & (user_expenses_df['date'].dt.year == (cur_year - 1))]
                prev = user_exp_prev.groupby([category]).sum()
                cur_month = prev_month
                cur_year -= 1

            else:
                prev_month = cur_month - 1
                user_exp_prev = user_expenses_df[(user_expenses_df['date'].dt.month == (
                    prev_month)) & (user_expenses_df['date'].dt.year == cur_year)]
                prev = user_exp_prev.groupby([category]).sum()
                cur_month -= 1

            datestring = f"{prev_month}/{str(cur_year)[2:]}"
            prev.rename(columns={'amount_dollars': datestring}, inplace=True)
            ticker += 1

        else:
            if cur_month == 1:
                prev_month = 12
                other = user_expenses_df[(user_expenses_df['date'].dt.month == (
                    prev_month)) & (user_expenses_df['date'].dt.year == (cur_year - 1))]
                other = other.groupby([category]).sum()
                prev = pd.concat([prev, other], axis=1, sort=True)
                cur_month = prev_month
                cur_year -= 1
            else:
                prev_month = cur_month - 1
                user_exp_prev = user_expenses_df[(user_expenses_df['date'].dt.month == (
                    prev_month)) & (user_expenses_df['date'].dt.year == cur_year)]
                other = user_exp_prev.groupby([category]).sum()
                prev = pd.concat([prev, other], axis=1, sort=True)
                cur_month -= 1

            datestring = f"{prev_month}/{str(cur_year)[2:]}"
            prev.rename(columns={'amount_dollars': datestring}, inplace=True)
            ticker += 1

    prev = prev.drop(columns=['amount_cents'])
    
    if weighted:
        prev = weighted_avg(prev)
    else:
        prev['mean'] = round(prev.mean(axis=1))

    return prev



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

    # get dataframe of average spending per category over last 6 months
    avg_spending_by_month_df = monthly_avg_spending(
        user.expenses, num_months=6)
    
    

    # turn into dictionary where key is category and value is average spending
    # . for that category
    avg_cat_spending_dict = dict(avg_spending_by_month_df['mean'])

    # label disctionary columns
    discretionary = ['Food', 'Recreation', 'Shopping', 'Other']

    # add column to df where its True if category is discretionary and False
    # . otherwise
    avg_spending_by_month_df['disc'] = [
        True if x in discretionary else False for x in avg_spending_by_month_df.index.tolist()]

    # get a dictionary of just the discretionary columns and how much was spent
    disc_dict = dict(
        avg_spending_by_month_df[avg_spending_by_month_df['disc'] == True]['mean'])

    # rerverse dictionary so key is amount spent and value is category
    disc_dict_reversed = {}
    for k, v in disc_dict.items():
        disc_dict_reversed[v] = k

    # find the key:value pair that shows which discretionary category the user
    # . spent the most money in
    max_cat = max(disc_dict_reversed.items())

    # subtract the monthly savings goal from that category
    avg_cat_spending_dict[max_cat[1]] -= monthly_savings_goal

    return avg_cat_spending_dict


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
    cur_year = user.expenses['date'].max().year
    cur_month = user.expenses['date'].max().month
    user_exp = user.expenses.copy()
    cur_month_expenses = user_exp[(user_exp['date'].dt.month == cur_month) &
                                  (user_exp['date'].dt.year == cur_year)]
    grouped_expenses = cur_month_expenses.groupby(
        ['grandparent_category_name']).sum()
    grouped_expenses = grouped_expenses.round({'amount_dollars': 2})

    grouped_dict = dict(grouped_expenses['amount_dollars'])

    # get dataframe of average spending per category over last 6 months
    avg_spending_by_month_df = monthly_avg_spending(
        user.expenses, num_months=6)

    for cat in avg_spending_by_month_df.index:
        if cat not in grouped_dict:
            grouped_dict[cat] = 0

    return grouped_dict
