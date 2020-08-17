import plotly.express as px
import pandas as pd


class User():
    def __init__(self, id, transactions, name=None):
        """
        Constructor for the User class.

        Parameters:
              id (str): user_id
              transactions (dataframe): dataframe consisting of sample transactions
              name (str): user's name. Default is to have no name.
        """
        self.id = id
        self.name = name
        self.transactions = transactions
        self.data = self.transactions[self.transactions['plaid_account_id'] == self.id]
        self.expenses = self.data[(self.data['grandparent_category_name'] != 'Transfers') & (
            self.data['amount_cents'] > 0)]

    def get_user_data(self):
        """
        Returns all the user's transactional data in the form of a dataframe.
        """
        return self.data

    def categorical_spending(self, category='grandparent_category_name'):
        """
        Reutrns jsonified plotly object which is a pie chart.

        Parameters:
              category (str): category type to return
                              (grandparent_category_name (default),
                              parent_category_name,
                              category_name)
        Returns:
              Plotly object of a pie chart in json
        """
        cat_count = self.expenses[category].value_counts(normalize=True)
        user_expenses = self.expenses.copy()

        for i in range(len(cat_count.index)):
            if cat_count[i] <= .05:
                bad_cat = cat_count.index[i]
                user_expenses = user_expenses.replace([bad_cat], ['Other'])

        if self.name:
            fig = px.pie(user_expenses, values='amount_dollars',
                         names=category, title=f"{self.name}'s Spending Habits")
            return fig.to_json()
        else:
            fig = px.pie(user_expenses, values='amount_dollars',
                         names=category, title=f"{self.id}'s Spending Habits")
            return fig.to_json()

    def money_flow(self):
        """
        Reutrns jsonified plotly object which is a line chart.
        """
        user_transaction_subset = self.data[["created_at", "amount_dollars"]]
        user_transaction_subset.set_index("created_at", inplace=True)
        user_transaction_subset = user_transaction_subset.sort_index()
        total_each_day = pd.DataFrame(
            user_transaction_subset['amount_dollars'].resample('D').sum())
        if self.name:
            fig = px.line(total_each_day, x=total_each_day.index,
                          y="amount_dollars", title=f"{self.name}'s Money Flow")
            return fig.to_json()
        else:
            fig = px.line(total_each_day, x=total_each_day.index,
                          y="amount_dollars", title=f"{self.id}'s Money Flow")
            return fig.to_json()

    def bar_viz(self, category='grandparent_category_name'):
        """
        Reutrns jsonified plotly object which is a bar chart.

        Parameters:
              category (str): category type to return
                              (grandparent_category_name (default),
                              parent_category_name,
                              category_name)
        Returns:
              Plotly object of a bar chart in json
        """
        fig = px.bar(self.data, x='created_at', y='amount_dollars',
                     color=category, title=f"{self.name}'s Spending Habits", width=1000)
        return fig.to_json()
