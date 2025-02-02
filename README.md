# Predictive analytics for enhanced customer engagement using Machine Learning

<p align="center">
  <img src="https://github.com/riyouuyt/Predict-Customer-Personality-to-Boost-Marketing-Campaign-using-Machine-Learning/assets/122600889/a2f3b4bf-9bef-41e7-8826-cdeafbf96c12" alt="Image" />
</p>


This project aims to revolutionize marketing campaigns by harnessing the power of machine learning to predict and understand customer personalities. By analyzing various data sources and employing advanced algorithms, we will develop a model that categorizes customers into distinct personality types, enabling businesses to tailor their marketing strategies to individual preferences and behavior. The project will not only increase the effectiveness of marketing efforts but also enhance customer engagement, satisfaction, and ultimately, business success.



## **Business Overview**

Our company's rapid growth is intricately tied to our deep understanding of customer personalities. We use historical marketing campaign data to optimize performance and precisely target potential loyal customers, driving transactions on our platform. Our key strategy involves developing a predictive clustering model, enabling data-driven decisions. By clustering customers based on behavior and personality, we provide tailored services and personalized marketing, fostering customer loyalty. Our goal is to set industry standards in customer-centric operations and sustainable growth through continuous data-driven refinement.

## **Objective 🌟**
Our primary objective is to optimize the marketing campaign by leveraging customer segmentation and data analytics. We aim to enhance customer engagement, increase conversion rates, and boost revenue while ensuring a seamless and personalized experience for our customers.

## **Goals 🎯**
Segment-Specific Targeting: Implement targeted marketing strategies for each customer segment, focusing on their specific characteristics and preferences.
Conversion Rate Optimization: Continuously refine and improve our conversion funnels, using data-driven insights to enhance the customer journey and boost conversion rates.
Customer Engagement Enhancement: Elevate the website experience and content to engage customers effectively. Develop loyalty programs and incentives to create lasting relationships.

Note: This is not a real company. The names "ShopSavvy Emporium" provided earlier are fictional names created for the purpose of this project. Please be aware that these names are entirely fictitious and not associated with any real businesses.

## **Library for The Project**

* **Pandas**
* **Numpy**
* **Scipy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

## Data Understanding

Project Data Column Information: 

1. `Unnamed: 0`: An unnamed index or identifier column.
2. `ID`: Customer identification number or code.
3. `Year_Birth`: Year of birth of the customer.
4. `Education`: The level of education attained by the customer.
5. `Marital_Status`: Marital status of the customer.
6. `Income`: Customer's income.
7. `Kidhome`: Number of children in the household.
8. `Teenhome`: Number of teenagers in the household.
9. `Dt_Customer`: Date when the customer became a client.
10. `Recency`: Number of days since the last purchase.
11. `MntCoke`: Amount spent on Coke products.
12. `MntFruits`: Amount spent on fruit products.
13. `MntMeatProducts`: Amount spent on meat products.
14. `MntFishProducts`: Amount spent on fish products.
15. `MntSweetProducts`: Amount spent on sweet products.
16. `MntGoldProds`: Amount spent on gold products.
17. `NumDealsPurchases`: Number of purchases made with deals or discounts.
18. `NumWebPurchases`: Number of purchases made through the web.
19. `NumCatalogPurchases`: Number of purchases made from catalogs.
20. `NumStorePurchases`: Number of purchases made in physical stores.
21. `NumWebVisitsMonth`: Number of web visits per month.
22. `AcceptedCmp3`: Whether the customer accepted Campaign 3 (binary, likely a marketing campaign).
23. `AcceptedCmp4`: Whether the customer accepted Campaign 4 (binary, likely a marketing campaign).
24. `AcceptedCmp5`: Whether the customer accepted Campaign 5 (binary, likely a marketing campaign).
25. `AcceptedCmp1`: Whether the customer accepted Campaign 1 (binary, likely a marketing campaign).
26. `AcceptedCmp2`: Whether the customer accepted Campaign 2 (binary, likely a marketing campaign).
27. `Complain`: Whether the customer has registered a complaint (binary).
28. `Z_CostContact`: Cost of contacting the customer.
29. `Z_Revenue`: Revenue generated from the customer.
30. `Response`: Customer response to a marketing campaign (binary, likely indicating whether they responded positively to a campaign).


## **🚀Feature Engineering🚀**

📊 Introduction to Feature Engineering

In my quest to enhance the success of our marketing campaign, i embarked on a journey through data. Feature engineering was my guiding star, allowing me to uncover deeper insights into customer behavior and boost conversions. 💡

1. **Creating the Conversion Rate📈**
I commenced by calculating the conversion rate, a pivotal metric that measures the percentage of website visitors who responded to our campaign. This served as the foundation for comprehending customer behavior. 🧮 The calculate conversion rate are from:
**Total Responses / Total web visit**

2.   📆**Customer Age Insights**
I segmented our customers into five distinct age groups. This segmentation allowed us to gain insights into the preferences and behaviors of different age cohorts, spanning from children to senior adults. 

3.   💰 **Income Labeling**
We didn't stop at numerical data; we also created a meaningful "Income Level" feature. By categorizing income into four distinct labels—Low Income, Moderate Income, High Income, and Very High Income—we gained insights into spending behavior and purchasing power. 

4.  🕰️ **Unlocking Recency Insights**
Recency is a crucial factor in customer engagement. We segmented customers based on their recency of interaction with our brand. This allowed us to tailor our marketing efforts to customers who were recently active and those who might need a gentle nudge to re-engage.

5.  🛒 **Total Transactions Analysis**
Total transactions give us a comprehensive view of customer engagement. We calculated the total number of transactions for each customer, shedding light on their loyalty and engagement with our products and services. 

6.  👨‍👩‍👧‍👦 **Understanding Family Size**
Family size can influence purchasing decisions. We engineered the "Family_Size" feature, providing insights into the composition of our customers' households. This information is invaluable for crafting family-centric marketing campaigns. 

7.  📆 **Recency Grouping**
To further refine our strategies, we grouped customers by recency into distinct segments. This allowed us to tailor our communication and offers based on how recently customers interacted with our brand, ensuring relevance and engagement.

## **Exploratory Data Analysis**

for this step we gonna investigate more about our data pattern from the distribution numerical and bar graph for categorical data.

* categorical columns = 'ID', 'Education', 'Marital_Status', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 'Response'

* numeric columns = 'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntCoke', 'MntFruits', 'MntMeatProducts',  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Z_CostContact', 'Z_Revenue'

### **Univariate EDA for Numeric**

1. **Checking the Data Distribution**

| Column Name         | Skewness | Kurtosis | Type of Distribution               |
|---------------------|----------|----------|-----------------------------------|
| Year_Birth          | -0.350   | 0.717    | Approximately Symmetrical (Nearly Normal) |
| Income              | 6.763    | 159.637  | Highly Positively Skewed          |
| Kidhome             | 0.635    | -0.780   | Bimodal Distribution               |
| Teenhome            | 0.407    | -0.986   | Bimodal Distribution               |
| Recency             | -0.002   | -1.202   | Approximately Symmetrical (Nearly Normal) |
| MntCoke             | 1.176    | 0.599    | Highly Positively Skewed          |
| MntFruits           | 2.102    | 4.051    | Highly Positively Skewed          |
| MntMeatProducts     | 2.083    | 5.517    | Highly Positively Skewed          |
| MntFishProducts     | 1.920    | 3.096    | Highly Positively Skewed          |
| MntSweetProducts    | 2.136    | 4.377    | Highly Positively Skewed          |
| MntGoldProds        | 1.886    | 3.552    | Highly Positively Skewed          |
| NumDealsPurchases   | 2.419    | 8.937    | Highly Positively Skewed          |
| NumWebPurchases     | 1.383    | 5.703    | Highly Positively Skewed          |
| NumCatalogPurchases | 1.881    | 8.047    | Highly Positively Skewed          |
| NumStorePurchases   | 0.702    | -0.622   | Moderately Positively Skewed     |
| NumWebVisitsMonth   | 0.208    | 1.822    | Approximately Symmetrical (Nearly Normal) |
| Z_CostContact       | 0.000    | 0.000    | Uniform Distribution              |
| Z_Revenue           | 0.000    | 0.000    | Uniform Distribution              |

  * **Histogram Visualization**

![image](https://github.com/riyouuyt/Predict-Customer-Personality-to-Boost-Marketing-Campaign-using-Machine-Learning/assets/122600889/05ecd5a8-e42b-4e67-8ad2-2a3dd54f7ad8)


  * 📊 **Summary of Distribution Characteristics**

    1.  Skewness: Measures the asymmetry in data distribution.Most columns are highly positively skewed (skewness > 1), indicating a longer tail on the right side.
        Columns with skewness values between -0.5 and 0.5 are approximately symmetrical.`Kidhome` and `Teenhome` have a bimodal distribution, suggesting two distinct modes.

    2.  Kurtosis: Measures tailedness or peakedness compared to a normal distribution. Several columns have high positive kurtosis, implying heavier tails and peaks.
        Notably, `NumDealsPurchases`, `NumWebPurchases`, and `NumCatalogPurchases` exhibit this characteristic.

    3.  Type of Distribution: Describes the distribution based on skewness values. Most columns are highly positively skewed. `Year_Birth`, `Recency`, and `NumWebVisitsMonth` are approximately symmetrical.
        `Kidhome` and `Teenhome` exhibit a bimodal distribution. `Z_CostContact` and `Z_Revenue` have a uniform distribution with constant values.


2. **Checking Outlier**
   

|  Column Name         | Lower Bound  | Upper Bound  | Number of Outliers | Percentage of Outliers |
|---------------------|--------------|--------------|---------------------|------------------------|
| Year_Birth          | 1932.0       | 2004.0       | 3                   | 0.133929               |
| Income              | -14525500.0  | 118350500.0  | 8                   | 0.361011               |
| Kidhome             | -1.5         | 2.5          | 0                   | 0.000000               |
| Teenhome            | -1.5         | 2.5          | 0                   | 0.000000               |
| Recency             | -51.0        | 149.0        | 0                   | 0.000000               |
| MntCoke             | -697000.0    | 1225000.0    | 35                  | 1.562500               |
| MntFruits           | -47000.0     | 81000.0      | 227                 | 10.133929              |
| MntMeatProducts     | -308000.0    | 556000.0     | 175                 | 7.812500               |
| MntFishProducts     | -67500.0     | 120500.0     | 223                 | 9.955357               |
| MntSweetProducts    | -47000.0     | 81000.0      | 248                 | 11.071429              |
| MntGoldProds        | -61500.0     | 126500.0     | 207                 | 9.241071               |
| NumDealsPurchases   | -2.0         | 6.0          | 86                  | 3.839286               |
| NumWebPurchases     | -4.0         | 12.0         | 4                   | 0.178571               |
| NumCatalogPurchases | -6.0         | 10.0         | 23                  | 1.026786               |
| NumStorePurchases   | -4.5         | 15.5         | 0                   | 0.000000               |
| NumWebVisitsMonth   | -3.0         | 13.0         | 8                   | 0.357143               |
| Z_CostContact       | 3.0          | 3.0          | 0                   | 0.000000               |
| Z_Revenue           | 11.0         | 11.0         | 0                   | 0.000000               |

   * **Boxplot Visualization**

![image](https://github.com/riyouuyt/Predict-Customer-Personality-to-Boost-Marketing-Campaign-using-Machine-Learning/assets/122600889/4f826b1f-76aa-4271-9a80-27f919473654)


  * **Summary of Outlier Characteristics**

    -  `Year_Birth`: Detected outliers in birth years, but the impact is minimal (0.13% outliers).
    -  `Income`: Some extreme income values but relatively low impact (0.36% outliers).
    -  `Kidhome` & `Teenhome`: No outliers found in the number of children at home.
    -  `Recency:` No outliers detected in the recency of purchases.
    -  `MntCoke` to `MntGoldProds`: Significant outliers in spending on various product categories (ranging from 1.56% to 11.07% outliers).
    -  `NumDealsPurchases` to `NumCatalogPurchases`: Outliers found in the number of deals and catalog purchases (ranging from 1.03% to 3.84% outliers).
    -  `NumWebPurchases` & `NumStorePurchases`: Minimal outliers in web and store purchases (below 0.18% outliers).
    -  `NumWebVisitsMonth`: Detected outliers in web visits, but the impact is relatively low (0.36% outliers).
    -  `Z_CostContact` & `Z_Revenue`: No outliers in contact cost and revenue.

Overall, the dataset contains outliers in income, spending on various product categories, and some purchase-related variables. These outliers should be further investigated for data quality and potential impact on the analysis.


### **Biavariate EDA for Numeric**

![image](https://github.com/riyouuyt/Predict-Customer-Personality-to-Boost-Marketing-Campaign-using-Machine-Learning/assets/122600889/df3f41ce-5a5e-445d-83da-9ecb61553b16)

   
**Key Correlations for Conversion Rate Analysis**
In our exploration of the data, we've uncovered intriguing correlations between various factors and our Conversion Rate. These insights are crucial for tailoring our marketing strategies.

 Top Positive Correlations: 
  * `Total Responses`: A strong positive correlation of 0.766 suggests that higher total responses lead to a better conversion rate.
  * `Total Accepted`: With a correlation of 0.750, a higher number of total accepted offers positively impacts conversion.
  * `AcceptedCmp5`: This campaign, with a correlation of 0.700, plays a significant role in boosting conversion.
  * `AcceptedCmp1`: Positive correlation of 0.577 highlights its effectiveness in driving conversions.
  * `Response`: Respondents exhibit a positive correlation of 0.486 with conversion.

**Other Positive Correlations:**
`Total Expenses`, `MntCoke`, `MntMeatProducts`, `NumCatalogPurchases`, and more contribute positively to conversion.

**Negative Correlations:**
`Z_CostContact` and `Z_Revenue` exhibit no correlation with Conversion Rate, indicating they don't significantly impact our campaigns.



## **Data Preprocessing** 

I have executed a robust data preprocessing phase, aiming to prepare the dataset for comprehensive analysis. Let's explore the remarkable steps i've accomplished through a series of powerful actions:

* **Data Cleaning** 🧹:
  
  In the dataset, we have a total of 2240 rows and 29 columns. Notably, the `Income` column contains 2216 non-null values, while 24 values are missing. Rather than imputing these missing `Income` values, we       have chosen to remove the rows with missing data. This approach simplifies the dataset and ensures that only complete and available information is used for analysis. Removing the null data helps maintain the    integrity of the dataset, avoiding any potential bias introduced by imputing missing values with a specific measure.

  The other data cleansing process, we executed two key steps to enhance the quality and relevance of our dataset.

  1. **Removing Duplicate Rows**
  We initially conducted a check for duplicate rows within the dataset. Fortunately, no duplicated rows were identified. Duplicate rows, if present, could lead to inaccuracies in our analysis, and their removal   ensures data integrity.
  
  2. **Dropping Unnecessary Columns**
  To streamline our dataset and focus on the most relevant features, we dropped several columns deemed unnecessary for our analysis. These columns, namely `Unnamed: 0`, `ID`, `Dt_Customer`, `Z_CostContact`, and   `Z_Revenue`, were excluded from the cleaned dataset.
  
  By eliminating these columns, we simplify the data structure, making it more manageable and conducive for subsequent analysis and modeling. This step also aids in improving computational efficiency and data     interpretability.

* **Outlier Handling** 🚫:

  The next step data pre-processing journey has uncovered some critical observations regarding outliers in specific columns. Let's break down the key insights:

  **Outlier Identification:** 
  * We've identified outliers in multiple columns, including `MntGoldProds`, `MntGoldProds`, and various spending categories (e.g., `MntCoke`, `MntFruits`).
  * Some of these outliers are exceptionally extreme, such as an `Income` above 600M and `MntCoke` spending above 1.2M.
  
  **Outlier Handling:** 
  * To address these outliers, we applied a log transformation to the data.
  * As a result, we were able to mitigate the impact of extreme values, which is particularly crucial due to our relatively small dataset (2240 rows).
  * This step aligns with best practices for handling outliers, as it avoids the need for data deletion, preserving valuable information.
  
  **Further Steps:** 📜🧹
  * For columns with a large number of remaining outliers, we may consider additional methods or transformations to better understand and manage these data points.
  * Removing outliers based on Interquartile Range (IQR) or Z-score could be a next step to enhance data quality.


* **Standardization** 📏:
  Standardization is the process of transforming our numeric features to have a mean of 0 and a standard deviation of 1. This step is essential for ensuring that all numeric features have the same scale,       
  preventing any single feature from dominating the analysis or modeling process.


* **Feature Encoding** 🛠️:
  Feature encoding allows us to work with categorical data effectively. Here's what i've done:
  * We applied One-Hot Encoding to the `Marital_Status` columns. This technique converts categorical variables into binary vectors, making them suitable for machine learning models.
  * For the `Education` column, we performed Label Encoding. This method assigns unique numerical values to each category, preserving the ordinal relationships in the data.



## **Data Modelling**

In the modeling phase, we delve into understanding the inherent structure of the data through clustering techniques. Clustering helps unveil patterns, segment similar data points, and reveal underlying relationships within the dataset.

**elbow Plot Analysis:**

![image](https://github.com/riyouuyt/Predict-Customer-Personality-to-Boost-Marketing-Campaign-using-Machine-Learning/assets/122600889/efd5c3fc-5b55-4eae-99f9-51a133bbdd36)

* In our quest for the ideal number of clusters, we conducted an elbow plot analysis.
* The plot revealed a distinct "elbow point" at **K=4**, suggesting that four clusters were a suitable choice for our customer segmentation.
* This decision allows us to strike the right balance between granularity in understanding customer behavior and practicality in tailoring marketing campaigns.

**Silhoutte Score Analysis:**

![image](https://github.com/riyouuyt/Predict-Customer-Personality-to-Boost-Marketing-Campaign-using-Machine-Learning/assets/122600889/c8b5095d-a5fe-4871-9a23-74824303a58f)

As we examined the Elbow Method plot, we observed a significant "elbow" forming at 4 clusters. This suggests that 4 clusters is an appropriate choice, balancing complexity and meaningful segmentation. 
Further validating our choice, the Silhouette Score for our clustering was an impressive 0.62, reinforcing the quality and separation of the clusters we've created. 
Let's move forward and explore how these clusters can power our targeted marketing strategies and elevate customer engagement. 

## **Customer Personality Analysis for Marketing Retargeting**

In our data universe, we've ventured deep into the world of customer behavior, exploring their preferences, habits, and interactions. Our quest is to decipher distinct personalities among these customers, akin to characters in a tale, to tailor our marketing strategies effectively.

The code snippet map_cluster is akin to a legend—a guide that translates the abstract language of clusters into relatable personas. Each cluster is assigned a distinct persona, enabling us to understand and connect with these "characters" in our dataset.

Let's unravel the narrative further:

* **"Low Prospect":** This persona signifies customers with lower engagement, a group that requires targeted nurturing.

* **"Risk of Churn":** Here lies a group displaying signs of potential churn, indicating the need for retention strategies.

* **"Mid Prospect":** These customers exhibit moderate engagement—a promising ground for further exploration and targeted engagement.

* **"High Prospect":** This persona embodies our most engaged customers, representing a goldmine for tailored premium offerings and loyalty programs.

By mapping clusters to these personas, we aim to breathe life into our data, transforming raw numbers into relatable characters. This, in turn, will guide our marketing endeavors, enabling us to create bespoke strategies that resonate with each group's unique personality traits.

### **Barplot for each Cluster Total**

![image](https://github.com/riyouuyt/Predict-Customer-Personality-to-Boost-Marketing-Campaign-using-Machine-Learning/assets/122600889/67309bc8-e4e9-4803-93b7-4825d3bb0a1f)

**Analysis of Customer Clusters**

1. Low Prospect
The count plot reveals that the "Low Prospect" cluster stands tall, boasting the highest count among all clusters. These customers, while abundant, demonstrate lower engagement levels, highlighting an area where targeted efforts for reactivation or engagement might be warranted.

2. Mid Prospect
Following the "Low Prospect" cluster, the count plot illustrates the presence of the "Mid Prospect" group. Although not as extensive as the "Low Prospect" cluster, these customers display moderate engagement levels, presenting a promising avenue for further exploration and targeted marketing initiatives.

3. High Prospect
The count plot showcases the "High Prospect" cluster, indicating a segment of highly engaged customers. Though not as numerous as the "Low" and "Mid Prospect" clusters, this group holds immense value, representing a cohort that might respond favorably to exclusive offers or premium services.

4. Risk of Churn
Lastly, the count plot unveils the "Risk of Churn" cluster, depicting customers exhibiting potential churn indicators. While this cluster might have the lowest count, it serves as a crucial focal point, signaling the need for robust retention strategies to prevent customer attrition.

Through these count plots, we gain a panoramic view of our customer landscape, discerning the varying sizes and engagement levels within each cluster. This visualization guides us in crafting targeted approaches, ensuring our strategies resonate with each cluster's unique characteristics and needs.

### **Distribution of Each Cluster**

![image](https://github.com/riyouuyt/Predict-Customer-Personality-to-Boost-Marketing-Campaign-using-Machine-Learning/assets/122600889/c3d896d8-70ad-40f7-bd2b-22359ec6676d)

Cluster Summary Based on The Plot

* **low Prospect** : they were very low at spending and buying something on their money to actually buy one of the product, but yet the had the most total clicked on the website.

* **Risk of Churn** : this cluster is a mid spender and buyer on the event and has second of the most clicked total on the website.

* **Mid Prospect** : this is the trusty prospect to actually interested on spending their money on the campaign even though they had very low clicked total on the website.

* **High Prospect** : this is the most guarantee prospect we had, they had very high expenses and transactions of all of this cluster, even though just like the Mid Prospect they had very low clicked total, but when they do clicked it's a guarantee that they will accepted


## **Business Recomendation🚀**

Based on the clustering analysis of customer segments, we propose the following recommendations for optimizing the marketing campaign:

Segment-Specific Targeting 🎯:
* Tailor marketing efforts to each customer segment's unique characteristics.
* For the "Low Prospect" segment, focus on converting high website engagement into actual purchases by providing compelling incentives. 💰
* The "Risk of Churn" segment should be targeted with personalized offers and retention strategies to prevent them from exploring alternatives. 🛡️
* "Mid Prospect" customers represent a potential growth opportunity. Enhance their website experience and encourage purchases through user-friendly interfaces. 📈
* Prioritize the "High Prospect" group by offering exclusive and high-value products or services, as they have shown strong purchase potential. 🌟

Conversion Rate Optimization 📊:
* Implement data-driven strategies to improve conversion rates for each segment.
* A/B testing and targeted promotions can be used to refine conversion funnels for different clusters.
* For the "Low Prospect" and "Risk of Churn" groups, consider implementing retargeting campaigns to re-engage potential customers who have shown interest. 🔄

Customer Engagement Enhancement 💬:
* Develop loyalty programs and incentives tailored to the "High Prospect" group to increase their loyalty and lifetime value. 🏆
* Use data analytics to identify high-value products or services for this segment.
* By implementing these recommendations, we can build stronger customer relationships, improve conversion rates, and increase revenue. Our success will depend on the ability to adapt strategies in real-time and * create exceptional experiences for all customers. Let's embark on this journey to drive growth and satisfaction across all clusters! 🌟🛍💰

## **Impact for Company Revenue**

Calculating The Impact 💡

The analysis of the financial impact of total expenses by cluster unveils critical insights that can guide our marketing strategy. Here's what we've uncovered:

Mid Prospect Dominance 🌟

* The "Mid Prospect" cluster stands out with a total expense impact of `$285,418,000.00`. Their willingness to spend is a valuable asset, even though they have low website engagement.
* This segment represents a significant revenue opportunity. By enhancing their website experience and offering personalized incentives, we can tap into their potential.

High Prospect Potential 🚀
* The "High Prospect" cluster showcases immense promise with an impact of `$135,285,000.00`. These customers have high expenses and transactions.
* Focusing on retaining their loyalty and providing exclusive, high-value products can further boost revenue.

Risk of Churn Challenge 📢
* The "Risk of Churn" cluster, while not the highest spender, still contributes significantly with an impact of `$243,629,000.00`.
* Their moderate spending and high website engagement suggest that retaining their interest is crucial. Implementing personalized retention strategies can be a game-changer.

Low Prospect Engagement Opportunity 💬
* "Low Prospect" customers, with a total expense impact of `$66,891,000.00`, may not be big spenders, but their extensive website engagement is noteworthy.
* With targeted efforts and enticing incentives, we can turn this high engagement into increased sales.
In summary, this analysis provides a clear roadmap for optimizing our marketing strategy. By segmenting our approach, we can unlock the potential of each customer group, boosting revenue and ensuring long-term success. The financial impact of each segment emphasizes the value they bring to our business and underscores the importance of tailored engagement strategies.


