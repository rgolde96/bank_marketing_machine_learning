%% importing datasets

%loading test dataset
    dataTest= readtable('/Users/matthewgalloway/Documents/ML_coursework/datasets/min_max/test.csv');
    dataTest= dataTest(:,2:end);
    X_test = dataTest(:,[1:end-1]);
    y_test = dataTest(:,[end]);
    y_test = table2array(y_test);
    
% loading train dataset
% Train dataset is required due to data being resampled
    dataTrain= readtable('/Users/matthewgalloway/Documents/ML_coursework/datasets/min_max/training_smote.csv');
    dataTrain= dataTrain(:,2:end);   
    X_train = dataTrain(:,[1:end-1]);
    y_train = dataTrain(:,end);

%% Random Forest model

%implementing and timing the RF
   tic
   template = templateTree(...
       'MaxNumSplits', 7001,'MinLeafSize',1);%optimised hyperparams
   random_forest = fitcensemble(X_train,y_train,'Method','Bag','NumLearningCycles',16,'Learners', template);

%calculating errors for the model
   error_rf_train = loss(random_forest,X_train,y_train);
   error_rf_test = loss(random_forest,X_test,y_test);
   
%Creating confusion matrix to calculate F1/precision/recall
   [rf_predict,rf_score] = predict(random_forest,X_test);
   rf_cm=confusionmat(y_test,rf_predict); 
   rf_recall=rf_cm(2,2)/sum(rf_cm(2,:));
   rf_precision=rf_cm(2,2)/sum(rf_cm(:,2));
   rf_F1 = 2*rf_recall*rf_precision/(rf_recall+rf_precision);
% calculating AUC
   [rf_fpr,rf_tpr,Trf,AUCrf] = perfcurve(y_test,rf_score(:,2),'1');
   toc
   %final results table Random Forst
   RF_Result=table(1-error_rf_train,1-error_rf_test, rf_F1,rf_precision,rf_recall,AUCrf,'VariableNames',{'Train Accuracy','Test Accuracy','F1','Precision','recall','AUC'})

%% Naive Bayes
% declaring Kernel Distributions
   var_distributions = ['kernel', 'mvnm', 'mvnm', 'mvnm', 'mvnm', 'kernel','mvnm', 'mvnm', 'mvnm','kernel','kernel', 'kernel', 'mvnm','kernel','kernel', 'mnvm'];

%implementing model and timing  
   tic
   naive_bayes = fitcnb(X_train,y_train,'DistributionNames',var_distributions,'width',0.01);
   
%calculating errors for the model
   nb_train = loss(naive_bayes,X_train,y_train);
   nb_test = loss(naive_bayes,X_test,y_test);
   
%Creating confusion matrix to calculate F1/precision/recall
   [nb_predict,nb_score] = predict(naive_bayes,X_test);
   nb_cm=confusionmat(y_test,nb_predict); 
   confusionchart(y_test,nb_predict); 
   nb_recall=nb_cm(2,2)/sum(nb_cm(2,:));
   nb_precision=nb_cm(2,2)/sum(nb_cm(:,2));
   nb_F1 = 2*nb_recall*nb_precision/(nb_recall+nb_precision);
% calculating AUC
   [fpr,tpr,Trf,AUCnb] = perfcurve(y_test,nb_score(:,2),'1');
   toc
%final results table Naive Bayes
    nb_Result=table(1-nb_train,1-nb_test, nb_F1,nb_precision,nb_recall,AUCnb,'VariableNames',{'Train Accuracy','Test Accuracy','F1','Precision','recall','AUC'})


