# Market Basket Analysis using Association rule learning

# What is Association rule learning?

Association rule learning is a machine learning technique used to identify patterns, associations, and relationships between variables in large datasets. It is often applied in market basket analysis, where the goal is to discover which products are frequently bought together by customers.
In market basket analysis, confidence, support, and lift are used to measure the strength of association between items purchased by customers.

1. Confidence: Confidence in market basket analysis measures the conditional **probability** of a product Y being purchased given that product X has already been purchased. A high confidence value indicates that customers who bought item X are likely to also buy item Y. 

2. Support: Support measures the **frequency of occurrence** of an item or a set of items in all transactions. A high support value indicates that the item or set of items is frequently bought together by customers.

3. Lift: Lift measures the strength of association between two items by comparing the observed frequency of co-occurrence of both items with the expected frequency of co-occurrence under the assumption that the items are independent. A lift value greater than 1 indicates that the items are positively correlated, a lift value of 1 indicates no correlation, and a lift value less than 1 indicates that they are negatively correlated. A high lift value indicates that the two items are likely to be bought together, and the lift value can be used to identify strong association rules between items.

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

# What is Apriori Algorithm in market basket analysis?

The most well-known algorithm for association rule learning is the Apriori algorithm, which works by identifying frequent itemsets (i.e., combinations of items that occur together frequently) and then generating association rules from those itemsets. An association rule is a statement that indicates the likelihood of one item being purchased given that another item has been purchased.

For example, if customers who buy milk and bread also tend to buy eggs, then an association rule could be "milk and bread imply the purchase of eggs". These rules can be used to provide insights into consumer behavior, to inform marketing strategies, and to make recommendations for complementary products.

Results of apriori:

<img src="apriori.png" width="800" height="400">

# What is Frequent Pattern (FP) growth?

Frequent Pattern (FP) growth is a popular algorithm used for mining frequent itemsets in a transactional dataset. It is an efficient algorithm for finding frequent patterns without generating candidate sets, which is a bottleneck in many other algorithms such as Apriori. The FP-growth algorithm builds a compact data structure, called a frequent pattern tree (FP-tree), to represent the transactional dataset. This data structure enables efficient counting of frequent itemsets by compressing the original dataset into a set of conditional databases. The algorithm recursively constructs the FP-tree by repeatedly finding frequent items and creating conditional databases, which are then recursively processed until no more frequent itemsets can be found.

Results of FP growth:

<img src="fp.png" width="800" height="400">

`DISCLAIMER`:  I don't warrant this code in any way whatsoever. This code is provided "as-is" to be used at your own risk.
