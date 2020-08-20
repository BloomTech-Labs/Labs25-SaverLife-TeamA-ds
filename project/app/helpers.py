import pandas as pd
import os
from sqlalchemy import create_engine
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# variables loaded in from .env file to create the DB connection
USERNAME = os.environ.get("USERNAME")
PASSWORD = os.environ.get("PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")


def convert_to_datetime(df, columns=[]):
    """
    Takes in a dataframe and a list of columns, and converts those columns to
    datetime format.

    Parameters:
            df (dataframe): a dataframe with time columns not in datetime
                            format
            columns (list): list of columns to be converted     
    """
    for col in columns:
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)


def sql_table_to_df(table):
    # Create the connection
    engine = create_engine('postgres://' + USERNAME +
                           ':' + PASSWORD + '@' + DB_HOST + '/' + DB_NAME)

    query = f"select * from {table}"
    df = pd.read_sql(query, engine)
    return df


def clean_data():
    """
    Usage:
        transactions = clean_data()
    Returns:
        dataframe that is cleaned.
    """

    dfPFA = sql_table_to_df('pfa')
    dfBA = sql_table_to_df('ba')
    dfPMT = sql_table_to_df('pmt')
    category_lookups = sql_table_to_df('category_lookups')

    dfPFA = dfPFA.dropna(thresh=600, axis=1)
    dfBA_delete = ['official_name', 'last_balance_update_at',
                   'atlas_id', 'atlas_parent_id', 'rewards_basis']
    dfBA = dfBA.drop(columns=dfBA_delete)
    dfPMT = dfPMT.dropna(thresh=600, axis=1)

    transactions_categorized = pd.merge(dfPMT, category_lookups,  how='left', left_on=[
                                        'category_id'], right_on=['plaid_category_id'])

    transactions_categorized = transactions_categorized[[
        'plaid_account_id', 'category_name', 'parent_category_name', 'grandparent_category_name', 'amount_cents', 'date', 'created_at']]
    transactions_categorized['amount_dollars'] = transactions_categorized['amount_cents'] / 100

    # transactions_categorized.to_csv('transactions_categorized.csv')

    return transactions_categorized
