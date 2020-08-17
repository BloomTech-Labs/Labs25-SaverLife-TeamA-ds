import pandas as pd

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
    
def clean_data():
    """
    Usage:
        transactions = clean_data()
    Returns:
        dataframe that is cleaned.
    """

    dfPFA = pd.read_csv('https://raw.githubusercontent.com/KyleTy1er/dsaverlife_restricted/master/PFA_BA_PMT_Table_examples1.csv?token=AM3U74TIWUFMHLGT5PUOX427HVQOO')
    dfBA = pd.read_csv('https://raw.githubusercontent.com/KyleTy1er/dsaverlife_restricted/master/PFA_BA_PMT_Table_examples2.csv?token=AM3U74RJKDWGSDIVHOXZDQ27HVQ72')
    dfPMT = pd.read_csv('https://raw.githubusercontent.com/KyleTy1er/dsaverlife_restricted/master/PFA_BA_PMT_Table_examples3.csv?token=AM3U74R4QWFKNPAWDQFJMWS7HVRGM')
    category_lookups = pd.read_csv('https://raw.githubusercontent.com/KyleTy1er/dsaverlife_restricted/master/Category%20Lookups.csv?token=AM3U74VVG6EPHFXCKR4XEKS7HMJPM')

    PFA_BA_cols = ['created_at', 'updated_at']
    PMT_cols = ['date', 'created_at', 'updated_at']

    convert_to_datetime(dfPFA, PFA_BA_cols)
    convert_to_datetime(dfBA, PFA_BA_cols)
    convert_to_datetime(dfPMT, PMT_cols)

    dfPFA = dfPFA.dropna(thresh=600 , axis=1)
    dfBA_delete = ['official_name', 'last_balance_update_at', 'atlas_id', 'atlas_parent_id', 'rewards_basis']
    dfBA = dfBA.drop(columns=dfBA_delete)
    dfPMT = dfPMT.dropna(thresh=600 , axis=1)

    transactions_categorized = pd.merge(dfPMT, category_lookups,  how='left', left_on=['category_id'], right_on = ['plaid_category_id'])

    transactions_categorized = transactions_categorized[['plaid_account_id','category_name','parent_category_name','grandparent_category_name', 'amount_cents', 'date', 'created_at']]
    transactions_categorized['amount_dollars'] = transactions_categorized['amount_cents'] / 100

    # transactions_categorized.to_csv('transactions_categorized.csv')

    return transactions_categorized



