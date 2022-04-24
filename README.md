# Module 17 Assignment - Credit Risk Analysis

## Background
We have been tasked with assisting Jill in using machine learning and statistical reasoning to predict the credit risk of an individual based on a variety of variables collected during the loan application process. The credit card credit data was provided by LendingClub, a peer-to-peer lending service company, and includes an indicator for whether or not the loan was provided ('loan_status').

After preparing this data for use (cleaning and sampling), we will perform a variety of machine learning techniques to determine which method would be best to use for predicting future credit risk.


## Credit Risk Analysis
After working with the data, we decided to test the accuracy of two different machine learning models, Logistic Regression and Ensambles Learners, while using a variety of sampling techniques to properly account for the imbalance of data that exists for good and bad creditors (risky loans are largely outnumbered by good loans). In total, we will compare the balance accuracy scores and the precision of six different machine learning model and sampling technique combinations. 

These results will help us determine the effectiveness of the models when compared to each other and help us determine if there is a model that could be relied on to effectively predict whether an individual is a good or bad credit risk.

1. Logistic Regression
	a. Naive Random Oversampling - The naive random oversampling algorithm is used to oversample the high risk data points
		- Balanced Accuracy Score: .6439
		- Precision and Recall: The precision ("pre" column) for the both classes are in the 99th percentile or higher, while the recall ("rec" column) for both classes is average. This model doesn't seem to be overly strong at predicting the risk classes.

		![random_over]("https://github.com/kjminges/Credit_Risk_Analysis/blob/main/random_over.png")


	b. SMOTE Oversampling - The SMOTE oversampling algorithm is used to oversample the high risk data points 
		- Balanced Accuracy Score: .6629		
		- Precision and Recall: Similar to the previous model, the precision for both classes is in the 99th percentile, and the recall for both classes is again average. This oversampling technique is a slightly stronger predictor with an average recall of .64 (compared to .60 with the naive random oversampling).

		![smote](https://github.com/kjminges/Credit_Risk_Analysis/blob/main/smote.png)


	c. Undersampling - The Cluster Centroids algorithm is used to undersample the low risk data points
		- Balanced Accuracy Score: .5903		
		- Precision and Recall: The undersampling technique leads to similar precision results as the undersampling, but the recall is worse with more weight being shifted to the low_risk class, driving the average down to .57.

		![under](https://github.com/kjminges/Credit_Risk_Analysis/blob/main/under.png)


	d. Combination Sampling (SMOTEENN) - A combination over- and under-sampling algorithm is used to simultaneously oversample the high risk data points, while undersample the low risk data points
		- Balanced Accuracy Score: .6376		
		- Precision and Recall: The results for the SMOTEENN are similar to that of the SMOTE oversampling. The precision metrics have stayed the same throughout the logistic regression models, but the recall is maxed out at an average of .64.

		![smoteenn](https://github.com/kjminges/Credit_Risk_Analysis/blob/main/smoteenn.png)


2. Ensemble Learners
	a. Balanced Random Forest Classifier - 
		- Balanced Accuracy Score: .8731		
		- Precision and Recall: Compared to the logistic regression models, the precision for the high risk individuals has decreased slightly but still within an acceptable range at .03 (for predicting the low risk class). As for the recall, this has increased to a more acceptable level with a recall of .70 and .87 for the high and low classes, respectively. The average recall of .87 is very indicative of a strong model.

		![random_forest](https://github.com/kjminges/Credit_Risk_Analysis/blob/main/random_forest.png)


	b. Easy Ensemble Ada Classifier - 
		- Balanced Accuracy Score: .9317		
		- Precision and Recall: The Easy ensemble model shows even stronger recall results than the random forest with an average recall of .94. Unfortunately, this increase in recall is coupled with a worse precision for the high risk class. The precision of .09 is in a range where the model starts to be questioned.

		![easy_ensamble](https://github.com/kjminges/Credit_Risk_Analysis/blob/main/easy_ensamble.png)


## Conclusion
After reviewing the results of the machine learning models, it is clear that while the logistic models provide strong precision results, the accuracy and recall of the models is not within acceptable ranges for the product that we are looking to develop for Jill. The ensemble models boast higher accuracy and recall scores, but this seems to come at the cost of lover precision. While the precision drop for the easy ensemble model is too much to overcome, the naive random forest still has a precision metric that is acceptable. Moving forward, it is clear that the random forest is the best machine learning model for our prediction purposes, but this doesn't necessarily mean that the model is strong enough to be trusted. 

While the random forest does provide what looks to be a strong model, there are things that we need to consider. For one, random forest models are much less intuitive (more of a "black box") and although this tends to lead to stronger models, it may become harder to interpret. One must also consider the possibility that overfitting is occurring. To check this, we ran the feature importance metrics for all the variables in the model to see if there are variables that could be removed to reduce the impact of over-fitting. Over half of the variables had feature importance measurements below .01. This suggests that there are a log of variables that could be removed. Our recommendation is to begin removing variables and re-running the model to see the impact of their removal. Once we have removed an adequate number of variables, we can re-evaluate the model to determine if it is still strong enough to use for our credit risk modeling.
