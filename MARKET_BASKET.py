#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sarvandani
"""
## data of this code can be downloaded from the following link:
#https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset/code?datasetId=877335&sortBy=voteCount    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
# from pyfim import eclat



########################@
#reading data
dataset = pd.read_csv("Groceries_dataset.csv")
#Member_number is unique for each customer. Date is date of the transaction,
#itemDescription is the product bought for this date.
# print(dataset.head())
# print(dataset.shape)
#################################
#checking missing values
nan_values = dataset.isna().sum()
# print(nan_values)
################################
##Basket analysis: we can see the bought products of clients for every day
client_basket=dataset.groupby(['Member_number','Date'])['itemDescription'].apply(sum)
# print(client_basket)
## we can see the bought products of clients for every day withot client number
clinet_basket2 = [a[1]['itemDescription'].tolist() for a in list(dataset.groupby(['Member_number','Date']))]
# print(clinet_basket2[0:10])
##################################
#Converting Date into datetime type
Date=dataset.set_index(['Date'])
Date.index=pd.to_datetime(Date.index, infer_datetime_format= True)
###############################
#Items Sold 
fig1 = plt.figure("Figure 1")
ax = plt.axes()
ax.set_facecolor('silver')
Date.resample("D")['itemDescription'].count().plot(figsize=(12,5), grid=True,
color='black').set(xlabel="Date", ylabel="Total Number of Items Sold")
Date.resample("W")['itemDescription'].count().plot(figsize=(12,5), grid=True,
color='blue').set(xlabel="Date", ylabel="Total Number of Items Sold")
Date.resample("M")['itemDescription'].count().plot(figsize=(12,5), grid=True,
color='red',title="Items Sold per month in red, per week in blue and per day in black").set(xlabel="Date", ylabel="Total Number of Items Sold")
#############################
#Number of customers
fig2 = plt.figure("Figure 2")
ax = plt.axes()
ax.set_facecolor('silver')
Date.resample('D')['Member_number'].nunique().plot(figsize=(12,5), grid=True,
color='black').set(xlabel="Date", ylabel="Number of customers")
Date.resample('W')['Member_number'].nunique().plot(figsize=(12,5), grid=True,
color='blue').set(xlabel="Date", ylabel="Number of customers")
Date.resample('M')['Member_number'].nunique().plot(figsize=(12,5), grid=True,
color='red',title="Number of customers per month in red, per week in blue and per day in black").set(xlabel="Date", ylabel="Total Number of Items Sold")

###########################
#sale per customer
fig3 = plt.figure("Figure 3")
ax = plt.axes()
day_ratio = Date.resample("D")['itemDescription'].count()/Date.resample('D')['Member_number'].nunique()
day_ratio.plot(figsize=(12,5), grid=True,
color='black').set(xlabel="Date", ylabel="Sale per customer")
week_ratio = Date.resample("W")['itemDescription'].count()/Date.resample('W')['Member_number'].nunique()
week_ratio.plot(figsize=(12,5), grid=True,
color='blue').set(xlabel="Date", ylabel="Sale per customer")
month_ratio = Date.resample("M")['itemDescription'].count()/Date.resample('M')['Member_number'].nunique()
month_ratio.plot(figsize=(12,5), grid=True,
color='red', title = "Sale per customers per month in red, per week in blue and per day in black").set(xlabel="Date", ylabel="Sale per customer")

#################################
#5 best seller items 
Item_distr = dataset.groupby(by = 'itemDescription').size().reset_index(name='Frequency').sort_values(by = 'Frequency',ascending=False).head(5)
bars = Item_distr["itemDescription"]
height = Item_distr["Frequency"]
x_pos = np.arange(len(bars))
plt.figure(figsize=(16,9))
plt.bar(x_pos, height, color = 'blue')
plt.title("Top 5 Sold Items")
plt.ylabel("Number of sold items")
plt.xticks(x_pos, bars)
plt.show()
#################################@
##Data preaparation and modeling
## before modeling, transcastion must be one-hot
Transactions = dataset.groupby(['Member_number', 'itemDescription'])['itemDescription'].count().unstack().fillna(0).reset_index()
Transactions.head()
def one_hot_encoder(k):
    if k <= 0:
        return 0
    if k >= 1:
        return 1
Transactions = Transactions.iloc[:, 1:Transactions.shape[1]].applymap(one_hot_encoder)
# Transactions.head()
##associate learning 1: apriori
frequent_items1 = apriori(Transactions, min_support=0.027, use_colnames=True, max_len=3).sort_values(by='support')
frequent_items1.head(10)
results1 = association_rules(frequent_items1, metric="lift", min_threshold=1).sort_values('lift', ascending=False)
results1 = results1[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
results1.head(10)
##############
##associate learning 2: fpgrowth
frequent_items2=fpgrowth(Transactions, min_support=0.027, use_colnames=True, max_len=3).sort_values(by='support')
frequent_items2.head(10)
results2 = association_rules(frequent_items2, metric="lift", min_threshold=1).sort_values('lift', ascending=False)
results2 = results2[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
results2.head(10)
########################
# frequent_items3=eclat(Transactions, min_support=0.027, use_colnames=True, max_len=3).sort_values(by='support')
# frequent_items3.head(10)
# results3 = association_rules(frequent_items2, metric="lift", min_threshold=1).sort_values('lift', ascending=False)
# results3 = results2[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
# results3.head(10)
