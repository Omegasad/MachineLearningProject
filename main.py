# Imports
# To use tensorflow Libraries
import tensorflow as tf

# To manipulate datas
import numpy as np
import pandas as pd

# data = pd.read_csv("BTCtoUSD.csv",header=0,usecols=['Date','Price'],index_col=['Price'])
# Read CSV
#data = pd.read_csv("./csv/BTCtoUSD.csv")
data = pd.read_csv("./csv/ETHtoUSD.csv")
#data = pd.read_csv("./csv/LTCtoUSD.csv")

# Drop data
data = data.drop(['Date','Change %'],1)
#print(data.head(1795))

# Data.shape[0] = Columns
# Data.shape[1] = Rows
n = data.shape[0] - 3
p = data.shape[1]

# Reverse order of chart so it goes lower to highest
reversed_df = data.iloc[n::-1]
#print(reversed_df)

#Make Data to numpy array
data = reversed_df.values
print(data)

# Use a regression with the data
# Open : the price at the start of the day
# Price : The price at the end of the day
# High : Highest price at the given day
# Low : Lowest price at the given day