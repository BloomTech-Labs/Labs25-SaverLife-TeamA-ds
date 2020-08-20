import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
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


class User():
    def __init__(self, id, transactions, name=None, show=False):
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

    def get_user_data(self):
        """
        Returns all the user's transactional data in the form of a dataframe.
        """
        return self.data

    def categorical_spending(self, time_period='week', category='grandparent_category_name'):
        """
        Returns jsonified plotly object which is a pie chart of recent transactions for the User.

        Parameters:
              time_period (str): time frame used to define "recent" transactions
              category (str): category type to return
                              (grandparent_category_name (default),
                              parent_category_name,
                              category_name)

        Returns:
              Plotly object of a pie chart in json
        """
        user_expenses = self.expenses.copy()

        # get a list of categories
        cat_count = self.expenses[category].value_counts(normalize=True)

        # filter down to recent transactions
        user_expenses = get_last_time_period(user_expenses, time_period)

        # for categories that fall under 5% of transactions, group them into the "Other" category
        for i in range(len(cat_count.index)):
            if cat_count[i] <= .05:
                bad_cat = cat_count.index[i]
                user_expenses = user_expenses.replace([bad_cat], ['Other'])

        # Create plotly graph object
        fig = go.Figure(data=[go.Pie(
            labels=user_expenses[category], values=user_expenses['amount_dollars'])])

        # Add title
        fig.update_layout(title=f"{self.name}'s Spending Habits",
                          font_size=16,)

        if self.show:
            fig.show()
        return fig.to_json()

    def money_flow(self, time_period='week'):
        """
        Returns jsonified plotly object which is a line chart depicting transactions over time for the User.

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

        # generate plot
        fig = go.Figure(data=go.Scatter(x=total_each_day.index,
                                        y=total_each_day['amount_flipped']))

        fig.update_layout(title=f"{self.name}'s Money Flow for the Last {time_period.capitalize()}",
                          xaxis_title='Date',
                          yaxis_title='Dollar Amount',
                          font_size=16,)

        if self.show:
            fig.show()

        return fig.to_json()

    def bar_viz(self, time_period='week', category='grandparent_category_name'):
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

        # generate bar chart figure
        fig = px.bar(subset, x='date', y='amount_dollars', color=category,
                     title=f"{self.name}'s Spending Habits", width=1000)

        if self.show:
            fig.show()

        return fig.to_json()

    def bar_viz_go(self, time_period='week', category='grandparent_category_name'):
        """
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

        # Create bar chart
        fig = go.Figure(data=go.Bar(
            x=subset['date'], y=subset['amount_dollars']))
        fig.update_layout(barmode='stack')

        # Add title
        fig.update_layout(title=f"{self.name}'s Spending Habits",
                          font_size=16)

        if self.show:
            fig.show()

        return fig.to_json()
