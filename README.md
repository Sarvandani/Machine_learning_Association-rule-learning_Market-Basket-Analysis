# Market Basket Analysis using Association rule learning

# What is Association rule learning?

Association rule learning is a machine learning technique used to identify patterns, associations, and relationships between variables in large datasets. It is often applied in market basket analysis, where the goal is to discover which products are frequently bought together by customers.

# 

Two approaches of Asscoaite rule learning, known as appriori and FP growth, have been used to do a market basket analysis.
My code is very simple to understand and can be extended by more analysis.

# Data analysis

The items sold per month, week, and day have been plotted to see the variations and the average level of sale.
<img src="items.png" width="800" height="300">

The number of customers per month, week, and day has been plotted to see the variations and the average number of customers.
<img src="clients.png" width="800" height="300">

The number of sale per customers for every day, every week and each month:
<img src="sale_per_client.png" width="800" height="300">

best selling items:

<img src="best_sale.png" width="800" height="400">

# What is Apriori Algorithm?

The most well-known algorithm for association rule learning is the Apriori algorithm, which works by identifying frequent itemsets (i.e., combinations of items that occur together frequently) and then generating association rules from those itemsets. An association rule is a statement that indicates the likelihood of one item being purchased given that another item has been purchased.

For example, if customers who buy milk and bread also tend to buy eggs, then an association rule could be "milk and bread imply the purchase of eggs". These rules can be used to provide insights into consumer behavior, to inform marketing strategies, and to make recommendations for complementary products.

Results of apriori:

<img src="apriori.png" width="800" height="400">


Results of FP growth:

<img src="fp.png" width="800" height="400">

`DISCLAIMER`:  I don't warrant this code in any way whatsoever. This code is provided "as-is" to be used at your own risk.
