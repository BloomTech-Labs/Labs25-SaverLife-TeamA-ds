import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import timedelta


def get_last_time_period(transaction_df, time_period='week'):
    """
    Given a dataframe of transactions + dates and a desired timeframe,
    return the dataframe containing only transactions within that time frame.
    If time_period is set to 'all', return the dataframe sorted by date
    """

    # make a copy of the user's expenses dataframe
    transaction_df = transaction_df.copy()
    # sort by date from earlist to latest
    transaction_df = transaction_df.sort_values(by=['date'])
    # grab the lastest date recorded
    latest_time = transaction_df['date'].iloc[-1]
    """
    based on the time period, establish a cutoff in order to subset
    the data
    """
    if time_period == 'day':
        cutoff = latest_time - timedelta(days=1)
    elif time_period == 'month':
        cutoff = latest_time - timedelta(days=30)
    elif time_period == 'week':
        cutoff = latest_time - timedelta(days=7)
    elif time_period == 'year':
        cutoff = latest_time - timedelta(days=365)
    elif time_period == 'all':
        return transaction_df
    else:
        raise ValueError(
            f"time_period must be one of 'day, week, month, year, or all'. Got {time_period} instead.")
    # subset the data based on the time frame
    subset = transaction_df[transaction_df['date'] >= cutoff]
    # return the subsetted data

    return subset


def weighted_avg(data):
    """
    Given a dataframe
    Return the datafram with a new column of weighted averages
    """
    categories = data.index
    # get number of timepoints/observations and create weights
    N = data.shape[1]
    weights = [i for i in range(N, 0, -1)]
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


class User():
    def __init__(self, id, transactions, name=None, show=False, hole=0):
        """
        Constructor for the User class.

        Parameters:
              id (str): user_id
              transactions (dataframe): dataframe consisting of sample transactions
              name (str): user's name. Default is to have no name.
              show (bool): set to True to display graphs after they are generated. Defaults to False
        """
        self.id = id
        self.name = name
        if not self.name:
            self.name = self.id
        self.transactions = transactions
        self.data = self.transactions[self.transactions['plaid_account_id'] == self.id]
        self.expenses = self.data[(self.data['grandparent_category_name'] != 'Transfers') & (
            self.data['amount_cents'] > 0)]
        self.show = show
        self.past_months = 6
        self.hole = hole

    def get_user_data(self):
        """
        Returns all the user's transactional data in the form of a dataframe.
        """
        return self.data

    def categorical_spending(self, time_period='week', category='grandparent_category_name'):
        """
        Returns jsonified plotly object which is a pie chart of recent
        transactions for the User.

        Parameters:
              time_period (str): time frame used to define "recent" transactions
              category (str): category type to return
                              (grandparent_category_name (default),
                              parent_category_name,
                              category_name)

        Returns:
              Plotly object of a pie chart in json
        """

        category='grandparent_category_name'
        user_expenses = self.expenses.copy()
        
        # get a list of categories
        cat_count = self.expenses[category].value_counts(normalize=True)

        # filter down to recent transactions
        user_expenses = get_last_time_period(user_expenses, time_period)

        # for categories that fall under 5% of transactions, group them into the "Other" category
        fig = go.Figure(data=[go.Pie(labels=user_expenses['grandparent_category_name'],
                                     values=user_expenses['amount_dollars'],
                                     hole = self.hole)])
        
        fig.update_traces(textinfo= "percent")
        fig.update_layout(width=1000, height=600,)

        # Add title
        if time_period == 'all':     
            fig.update_layout(title={"text" : f"Spending by Category", "x":0.5,
                                     "y":0.9},
                              font_size=16,)

        else:
            fig.update_layout(title={"text" : f"Spending by Category for the Last {time_period.capitalize()}", "x":0.5, "y":0.9},
                              font_size=16)

        fig.update_traces(textposition='inside', textfont_size=14)

        fig.update_layout(
            legend=dict(
                x=.43,
                y=.5,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,)
                ))
        
        if self.show:
            fig.show()
        
        return fig.to_json()

    def money_flow(self, time_period='week'):
        """
        Returns jsonified plotly object which is a line chart depicting
        transactions over time for the User.

        Parameters:
              time_period (str): time frame used to define "recent" transactions
        """
        # subset data down to date and transaction amount
        user_transaction_subset = self.data[["date", "amount_dollars"]]

        # filter down to desired timeframe
        user_transaction_subset = get_last_time_period(
            user_transaction_subset, time_period)

        # prepare data for plotting
        user_transaction_subset.set_index("date", inplace=True)
        user_transaction_subset = user_transaction_subset.sort_index()
        total_each_day = pd.DataFrame(
            user_transaction_subset['amount_dollars'].resample('D').sum())

        total_each_day['amount_flipped'] = total_each_day['amount_dollars'] * -1

        fig = go.Figure(data=go.Scatter(x=total_each_day.index,
                                        y=total_each_day['amount_flipped'],
                                        marker=dict(
                                            color='rgb(192,16,137)',
                                            size=10,
                                            line=dict(
                                                color='Black',
                                                width=2
                                            )
                                        ))
                        )

        fig.update_layout(width=1000, height=500,)

        if time_period == 'all':
            fig.update_layout(
                title={"text": f"Money Flow", "x": 0.5, "y": 0.9},
                xaxis_title='Date',
                yaxis_title='Net Income ($)',
                font_size=16,
                template='presentation')

        else:
            fig.update_layout(
                title={"text": f"Money Flow for the Last {time_period.capitalize()}", "x": 0.5, "y": 0.9},
                xaxis_title='Date',
                yaxis_title='Net Income ($)',
                font_size=16,
                template='presentation')

        if self.show:
            fig.show()

        return fig.to_json()

    def bar_viz(self, time_period='week', category="grandparent_category_name"):
        """
        Uses plotly express
        Returns jsonified plotly object which is a bar chart of recent transactions for the User.

        Parameters:
            time_period (str): time frame used to define "recent" transactions
            category (str): category type to return
                            (grandparent_category_name (default),
                            parent_category_name,
                            category_name)
        Returns:
            Plotly object of a bar chart in json
        """
        # subset the data using the get_last_time_period method
        subset = get_last_time_period(self.expenses, time_period)

        subset[category] = subset[category].astype(str)

        # group the sum of a categorie's purchases by each day rather than by each transaction
        subset = subset.groupby([category, 'date']).agg(
            {'amount_dollars': 'sum'})
        subset = subset.reset_index()

        # rename columns for cleaner visualization
        subset.rename({category: 'Category', 'date': 'Date',
                       'amount_dollars': 'Spending ($)'}, axis=1, inplace=True)

        # generate bar chart figure

      
        fig = px.bar(
            subset,
            x='Date',
            y='Spending ($)',
            color='Category',
            opacity=0.9,
            width=1200,
            height=500,
            template='simple_white'
        )

        if time_period == 'all':
            fig.update_layout(title={'text': "Daily Spending by Category "})
        else:
            fig.update_layout(title={'text': f"Daily Spending by Category for the Last {time_period.capitalize()}"})

        fig.update_layout(legend=dict(
            yanchor="top",
            y=1,
            xanchor='left',
            x=1
        ))

        fig.update_layout(font_size=15)
        fig.update_layout(barmode='relative',)
        fig.update(layout=dict(title=dict(x=0.45)))

        if self.show:
            fig.show()

        return fig.to_json()

    def future_budget(self, monthly_savings_goal=50, num_months=6, weighted=True):

        # get dataframe of average spending per category over last 6 months
        avg_spending_by_month_df = monthly_avg_spending(
            self.expenses, num_months=num_months, weighted=weighted)
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

        self.past_months = num_months

        return avg_cat_spending_dict

    def current_month_spending(self):

        cur_year = self.expenses['date'].max().year
        cur_month = self.expenses['date'].max().month
        user_exp = self.expenses.copy()
        cur_month_expenses = user_exp[(user_exp['date'].dt.month == cur_month) &
                                      (user_exp['date'].dt.year == cur_year)]
        grouped_expenses = cur_month_expenses.groupby(
            ['grandparent_category_name']).sum()
        grouped_expenses = grouped_expenses.round({'amount_dollars': 2})

        grouped_dict = dict(grouped_expenses['amount_dollars'])

        # get dataframe of average spending per category over last 6 months
        avg_spending_by_month_df = monthly_avg_spending(
            self.expenses, num_months=self.past_months)

        for cat in avg_spending_by_month_df.index:
            if cat not in grouped_dict:
                grouped_dict[cat] = 0

        return grouped_dict
