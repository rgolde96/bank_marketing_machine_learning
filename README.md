# A Comparison of Naive Bayes and Tree Based Ensembles Applied to the UCI Bank Marketing Dataset

## 1.0 Motivation and problem description

Data driven decision making (business intelligence) is becoming more common as data sources and machine capabilities increase. When engaging with clients these often presents binary classification problems, will a client buy the business's product? 

As the majority of customers do not respond to marketing, datasets are highly unbalanced leading models to take a majority vote. Data augmentation techniques such as SMOTE, SMOTEEN, RUS and more recently GAN's have shown success in the literature at fixing class inbalances.

This work seeks to investigate the relationship between performance, resampling techinques and algorithms on a bench marking dataset for new techniques, containing information about bank marketing https://archive.ics.uci.edu/ml/datasets.php.

## 2.0 Training and evaluation methodology
● 80 % of the data was used for training and 20% was set aside for testing. 10 fold cross validation for all training metrics were used.

● Due to the class imbalance present in the data (88.3% vs 11.7 %) over, under and over and under sampling was done on the
training data. To ensure models were not simply performing a majority vote F1 scores were considered for optimisations.

● Categorical data was label encoded rather than one hot encoded to allow for Naive bayes kernel optimisations.

● Data was min/max scaled prior to applying sampling methods as both oversampling methods use KNN and relative positions in
vector space affects results.

● Its noted that neither scaling nor encoding of categorical variables is explicitly required for NB or RF prior to optimisation.

## 3.0 Experimental Summary

### Naive bayes

1. Naive Bayes F1 score was significantly better than RF on an unsampled dataset,
contrary to our assumption that the effect on the priors of a Naive Bayes model would
make it a worse predictor.
2. Counterintuitively, log transformation of the continuous variables decreased the F1
score for the gaussian naïve bayes model by 42%, although post logging the
variables were closer to the normal distribution.
3. Sampling methods: had no positive effect on F1 score although the precision recall
balance changed, as models got worse at predicting majority class.
4. Kernel optimisation of continuous variables improved the F1 score, with a grid search
showing optimal values at width of 0.01.
5. Kernel smoothing type changing from gaussian to box, triangle and epanechnikov
had no effect on the results.

### Random Forest

1. Random forest recall rate of 25.98% before resampling data showed a poor ability to
classifying the minority class.
2. Sampling methods: greatly improved F1 and recall scores, SMOTE was the best
method.
3. Forest size: Increasing from a forest size of 2 trees to 16 showed increase in F1
score, however past this point changes in F1 score were negligible
4. No of splits: had no effect on F1 score
5. Min leaf size: F1 score got worse as min leaf size increased from 1
6. Grid search was performed on SMOTE data to show max no of splits at 7001 and
min leaf size of 1 were the best model.
7. Boosting RUSBoost/AdaBoost/GentleBoost all increased F1 score 

![image](https://user-images.githubusercontent.com/52289894/75583427-776c3b80-5a65-11ea-842d-911ee9ea5587.png)
![image](https://user-images.githubusercontent.com/52289894/75583468-910d8300-5a65-11ea-9047-fe9dff12332a.png)



## 4.0  Analysis 
● In contrast to our hypothesis Naive bayes outperformed Random Forest on unsampled data based on F1 score 40.04% vs 37.20%, this was due to RF taking a more significant majority class vote than naive bayes (RF recall = 25.98%).

● As expected the RF (AUC 0.9) outperformed the decision trees in literature (AUC 0.87), even though a larger number of features were used in literature so a direct comparison would require further work[3].

● Investigating sampling methods across the models showed improvements in RF F1 score but had no significant positive effect on NB. This was contrary to our hypothesis, as we expected rebalancing the dataset and changing the priors of NB would have a significant negative impact on F1.

● Its thought that undersampling increased F1 score of RF by balancing the classes; at the cost of the precision of model. The removal of data from the majority
class is likely the reason the model was worse at classifying this class, evidenced in training and test scores dropping equal amounts.

● Both SMOTE based sampling methods for RF showed improvements in recall and F1 score when compared to random undersampling. Its thought that the retention of the data of the majority class is the reason for this. However precision was still significantly lower than when no sampling method was applied.

● Our initial hypothesis was SMOTENN would outperform SMOTE as the noise generated in oversampling would be removed, this was not the case however for either model [10]. It is thought that the algorithm was unable to correctly assign clusters to the dataset.

● Data level optimisations highlighted the importance of balancing accuracy, precision and recall. Rebalancing data sets changes the priors of the model and will typically lead to lower accuracy as the precision decreases and majority class predictions get worse. However this led to better recall scores as the ability to predict the minority class increased. These considerations would be important in industry where slightly more unconventional optimisations metrics like
recall score could be used if the focus is only on understanding the minority class (if customers respond yes to marketing).

● A final consideration with sampling methods was the size of the data and how it affected run time of the models, with more computationally heavy models like random forest of large size and tree depth there was a factor of 4 difference in run time between the oversampled and undersampled data sets.

### Algorithm Optimisations on resampled SMOTE data:
● Evaluating the improvement in F1 score vs the size of the forest was consistent with literature, averaging more decision trees gives better results, attributed to law of large numbers. [7] A drop off in performance improvements was seen after a forest size of 16 and as this ran 6 times faster than the algorithm default (forest size of 100), 16 was chosen as the hyper parameter for investigation.  Both minimum leaf size and no of split were evaluated as proxies for tree depth. It was shown that deeper trees improved the prediction of F1 score with minimum leaf size being the more significant optimisation parameter.

● Given the non gaussian distribution of continuous features kernel density estimation was applied within naive Bayes, where it improved results. This was in line with our hypothesis that non parametric assumption would be a better representation of the data.

● Kernel density estimation massively outperformed normal Naive bayes but the 600 fold increase in computational time relative to a unoptimised naive bayes decreased the value of these result. This was especially true given a random forest optimised for tree depth out performed the kernel naive bayes in every parameter and ran 283 times faster.

![image](https://user-images.githubusercontent.com/52289894/75583183-0f1d5a00-5a65-11ea-872c-674fa2e8f0d9.png)


## 4.0  Future work/Lessons learned

1. Despite the simplicity of Naive Bayes classifier its out the box performance is competitive with the more sophisticated ensemble methods and it runs much faster (prior to optimisation).
2. Resampling data changes the models priors, it was assumed this would have a large effect on NB F1 scores and not on RF, however the inverse was true. Further working investigating changing the priors while maintaining the data may provide insight on why.
3. Boosting was briefly investigated and showed promising improvements on a random forest, future investigations looking at how tree structures differ in a boosted model vs a traditional random forest potentially could highlight why.
4. SVM is reported within the literature as the most successful classifier for this problem, applying different resampling methods to SVM could further improve the models performance [2].

## 5.0 Reference

1. Chen, Chiang, and Storey, ‘Business Intelligence and Analytics: From Big Data to Big Impact’, MIS Q., vol. 36, no. 4, p. 1165, 2012.
2. S. Moro, P. Cortez, and P. Rita, ‘A data-driven approach to predict the success of bank telemarketing’, Decis. Support Syst., vol. 62, pp. 22–31, Jun. 2014.
3. D. Pavlovic, M. Reljic, and S. Jacimovic, ‘Application of data mining in direct marketing in banking sector’, Industrija, vol. 42, no. 1, pp. 189–201, 2014.
4. H. A.Elsalamony, ‘Bank Direct Marketing Analysis of Data Mining Techniques’, Int. J. Comput. Appl., vol. 85, no. 7, pp. 12–22, Jan. 2014.
5. R. Caruana and A. Niculescu-Mizil, ‘An empirical comparison of supervised learning algorithms’, in Proceedings of the 23rd international conference on Machine learning - ICML ’06, Pittsburgh, Pennsylvania, 2006, pp. 161–168.
6. A. Pérez, P. Larrañaga, and I. Inza, ‘Bayesian classifiers based on kernel density estimation: Flexible classifiers’, Int. J. Approx. Reason., vol. 50, no. 2, pp. 341–362, Feb. 2009.
7. G. James, D. Witten, T. Hastie, and R. Tibshirani, Eds., An introduction to statistical learning: with applications in R. New York: Springer, 2013.
8. M. N. Murty and V. S. Devi, ‘Bayes Classifier’, in Pattern Recognition, vol. 0, London: Springer London, 2011, pp. 86–102.
9. L. Breiman, ‘Random Forests’, Mach. Learn., vol. 45, no. 1, pp. 5–32, 2001.
10.  G. E. A. P. A. Batista, R. C. Prati, and M. C. Monard, ‘A study of the behavior of several methods for balancing machine learning training data’, ACM SIGKDD Explor. Newsl., vol. 6, no. 1, p. 20, Jun. 2004


