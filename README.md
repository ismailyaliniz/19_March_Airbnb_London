# 19_March_Airbnb_London
In this project, we conducted a comprehensive data analysis and modeling study using the latest dataset for London from March 19th, available on insideairbnb.com. The steps of our study were as follows:

Data Exploration and Cleaning:

We examined the dataset, identified, and removed outliers.
We found missing values and applied the necessary imputation techniques to address them.
Correlation and VIF Analysis:

We analyzed the correlations between variables to identify closely related features.
We performed VIF (Variance Inflation Factor) analysis to check for multicollinearity.
Modeling:

We used different machine learning algorithms for modeling: Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.
We evaluated the models both before and after applying the best parameters.
Our modeling results are shown in the tables above. The first table displays the performance metrics before applying the best parameters, while the second table shows the performance metrics after applying the best parameters.

Notably, the Decision Tree model performed very well on the training data after applying the best parameters, but its performance significantly dropped on the test data. This indicates that the model experienced overfitting. Similar performance variations were observed in the other models as well.

In conclusion, when evaluating model performance, it is essential to consider both training and test data performances to select the most suitable model.
