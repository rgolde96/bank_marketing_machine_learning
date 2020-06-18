    
%% Visualisation: Import data and produce summary of class imbalance
    data = readtable('/Users/matthewgalloway/Documents/ML_coursework/bank-full.csv');
    summary(data); %no missing values
    tabulate(nominal(data.y));% 88% no/11% yes
%% Visualisation: set up variables to allow manipulation of categorical and numerical value
    numeric_cols = [{'age'}, {'balance'}, {'duration'}, {'campaign'},{'pdays'},{'previous'}]
    categorical_cols = [{'job'},{'marital'},{'day'},{'education'},{'default'},{'housing'},{'loan'},{'contact'},{'month'},{'poutcome'},{'y'}]

    % Converting catetgorical columns to matlab categorical data type 
    data = convertvars(data,categorical_cols,'categorical');
%% Visualisation: dealing with outliers for visualisation

    [val, idx] = max(data.previous);
    data.previous(idx) = mean(data.previous);
    
    [val, idx] = max(data.balance);
    data.balance(idx) = mean(data.balance);
    
    [val, idx] = max(data.balance);
    data.balance(idx) = mean(data.balance);
%% Visualisation: normalising numerical variables for visualisation
%  variables remain scaled for the remained of the code

    data.age = (data.age - min(data.age))/(max(data.age)-min(data.age));
    data.balance = (data.balance - min(data.balance))/(max(data.balance)-min(data.balance));
    data.duration = (data.duration - min(data.duration))/(max(data.duration)-min(data.duration));
    data.campaign = (data.campaign - min(data.campaign))/(max(data.campaign)-min(data.campaign));
    data.pdays = (data.pdays - min(data.pdays))/(max(data.pdays)-min(data.pdays));
    data.previous = (data.previous - min(data.previous))/(max(data.previous)-min(data.previous));
%% EDA for numeric variables -box plots
% multiple box plot is custom matlab library taken from file exchange
% https://uk.mathworks.com/matlabcentral/fileexchange/47233-multiple_boxplot-m
    A=data(string(data.y)=="yes", data.Properties.VariableNames(numeric_cols));
    A= table2array(A);

    B=data(string(data.y)=="no", data.Properties.VariableNames(numeric_cols));
    B= table2array(B);
    plot_data=cell(6,2);
    
    for ii=1:size(plot_data,1)
        Ac{ii}=A(:,ii);
        Bc{ii}=B(:,ii);
    end
    
    plot_data=vertcat(Ac,Bc);
    col=[34,139,34, 200;
    255,153,51, 200]
    col=col/255;
    multiple_boxplot(plot_data',numeric_cols,{'No', 'Yes'},col')% custom matlab function require plugin
    title('Normalised Numeric Variables')
    xlabel('Numeric Variables')
    ylabel('Normalised Values')
%% EDA: Correlation of numeric variables
    
    numeric_cor  = table2array(data(:, data.Properties.VariableNames(numeric_cols)));
    
    c= array2table(corrcoef(numeric_cor),...
        'VariableNames',numeric_cols, 'RowNames',numeric_cols )
    
% heatmap for numeric cols 
    clf
    heatmap(corrcoef(numeric_cor),'Colormap',parula)
    ax = gca
    ax.XDisplayLabels = numeric_cols
    ax.YDisplayLabels = numeric_cols
    title('Correlation of numeric variables')
 

%% Data importing: Unsampled
%test data 20% split imported from python
    dataTest= readtable('/Users/matthewgalloway/Documents/ML_coursework/datasets/min_max/test.csv');
    dataTest= dataTest(:,2:end);
    X_test = dataTest(:,[1:end-1]);
    y_test = dataTest(:,[end]);
    y_test = table2array(y_test);    
    
%train data 80% split imported from python   
    dataTrain= readtable('/Users/matthewgalloway/Documents/ML_coursework/datasets/min_max/training.csv');
    dataTrain= dataTrain(:,2:end);   
    X_train = dataTrain(:,[1:end-1]);
    y_train = dataTrain(:,end);
    
    
%% Data importing: optional cell: Smote Oversampling
% this cell is not run for the inital model creation, this is a smote dataset 

    smote_sample = readtable('/Users/matthewgalloway/Documents/ML_coursework/datasets/min_max/training_smote.csv');
%redefine training data (uses the same 80 % split as above, processed using smote in sklearn)    
    dataTrain = smote_sample(:,2:end);
    dataTrain.Properties.VariableNames = dataTest.Properties.VariableNames;
    X_train =  dataTrain(:,[1:end-1]);
    y_train = dataTrain(:,end);
    y_train = table2array(y_train);

%% Data importing: optional cell:random undersampling
% this cell is not run for the inital model creation, this  is importing
% random undersample data from sklearn

    ru_sample = readtable('/Users/matthewgalloway/Documents/ML_coursework/datasets/min_max/training_rus.csv');
% redefininging training data
    dataTrain = ru_sample(:,2:end);
    dataTrain.Properties.VariableNames = dataTest.Properties.VariableNames;
    X_train =  dataTrain(:,[1:end-1]);
    y_train = dataTrain(:,end);
    y_train = table2array(y_train);

%% Data importing: optional cell:SMOTE and cleaning using ENN
% this cell is not run for the inital model creation, this recreates the
% train data with SMOTENN from sklearn

    % from python
    SMOTENN_sample = readtable('/Users/matthewgalloway/Documents/ML_coursework/datasets/min_max/training_smoteenn.csv');
% redefining training data
    dataTrain = SMOTENN_sample(:,2:end);
    dataTrain.Properties.VariableNames = dataTest.Properties.VariableNames;
    X_train =  dataTrain(:,[1:end-1]);
    y_train = dataTrain(:,end);
    y_train = table2array(y_train);

%% Models: Grid search for random forst
   folds = 1:1:10 ;
   
   foldindex=crossvalind('Kfold',size(dataTrain,1),10);
   
    k=1; %initialize the number of combination
    for MaxNumSplit = 1:7000:63873
       for MinLeaf = 1:3500:31937
            for numTrees = 1:10:100
               for i = folds
                   validation = (dataTrain(foldindex==i,:));
                   training = (dataTrain(foldindex~=i,:));
                   template = templateTree(...
                       'MaxNumSplits', MaxNumSplit,'MinLeafSize',MinLeaf);
                   random_forest = fitcensemble(training(:,[1:end-1]),training(:,end),'NumLearningCycles',numTrees,'Method','Bag','Learners', template);
                   error_rf_train(i) = loss(random_forest,training(:,[1:end-1]),training(:,end));
                   error_rf_validation(i) = loss(random_forest,validation(:,[1:end-1]),validation(:,end));
                   error_rf_test(i) = loss(random_forest,X_test,y_test);
                   %confusion matrix

                   [rf_predict,rf_score] = predict(random_forest,X_test);
                   rf_cm=confusionmat(y_test,rf_predict); 
                   rf_recall(i)=rf_cm(2,2)/sum(rf_cm(2,:));
                   rf_precision(i)=rf_cm(2,2)/sum(rf_cm(:,2));
                   rf_F1(i) = 2*rf_recall(i)*rf_precision(i)/(rf_recall(i)+rf_precision(i));
                   [fpr,tpr,Trf,AUCrf(i)] = perfcurve(y_test,rf_score(:,2),'1');
               end
               kfl_rf_test(k)= mean(error_rf_test);
               kfl_rf_train(k) = mean(error_rf_train);
               kfl_rf_val(k) = mean(error_rf_train);
               kfl_rf_recall(k) = mean(rf_recall);
               kfl_rf_precision(k) = mean(rf_precision);
               kfl_rf_F1(k) = mean(rf_F1);
               kfl_rf_AUCrf(k) = mean(AUCrf);
               numTrees_r(k) = numTrees
               MaxNumSplit_r(k) = MaxNumSplit
               MinLeafSize_r(k) = MinLeaf
               k=k+1;
            end
       end
    end
    
    RF_Result=table((1:(k-1))',numTrees_r',MaxNumSplit_r',MinLeafSize_r',(1-kfl_rf_test)',kfl_rf_F1',kfl_rf_precision',kfl_rf_recall',kfl_rf_AUCrf','VariableNames',{'No','MaxNumSplit','MinLeafSize','ForestSize','Accuracy','F1Score','precision','recall','AUC'})

%% Modeling: Optional Cell: looking at tree structures

Tree50 = random_forest.Trained{25}
view(Tree50,'Mode','graph')


%%  Modeling:Naive Bayes Grid search

   	
   folds = 1:1:10 ;
   var_distributions = ['kernel', 'mvnm', 'mvnm', 'mvnm', 'mvnm', 'kernel','mvnm', 'mvnm', 'mvnm','kernel','kernel', 'kernel', 'mvnm','kernel','kernel', 'mnvm'];
  
   foldindex=crossvalind('Kfold',size(dataTrain,1),10);
   widths = 0.005:0.05:0.5
 for width = widths
   for i = folds
       
       validation = (dataTrain(foldindex==i,:));
       training = (dataTrain(foldindex~=i,:));
       
       naive_bayes = fitcnb(training(:,[1:end-1]),training(:,end),'DistributionNames',var_distributions,'width',width);
       error_nb_test(i)= loss(naive_bayes,X_test,y_test);
       error_nb_train(i) = loss(naive_bayes,training(:,[1:end-1]),training(:,end));
       error_nb_validation(i) = loss(naive_bayes,validation(:,[1:end-1]),validation(:,end));
       %confusion matrix
       
       [nb_predict,nb_score] = predict(naive_bayes,X_test);
       nb_cm=confusionmat(y_test,nb_predict); 
       nb_recall(i)=nb_cm(2,2)/sum(nb_cm(2,:));
       nb_precision(i)=nb_cm(2,2)/sum(nb_cm(:,2));
       nb_F1(i) = 2*nb_recall(i)*nb_precision(i)/(nb_recall(i)+nb_precision(i));
       [fpr,tpr,Trf,AUCnb(i)] = perfcurve(y_test,nb_score(:,2),'1');
       [fpr,tpr,Trf,AUCnb2(i)] = perfcurve(y_test,nb_score(:,1),'1');
       
   end
 end 
   kfl_nb_test= mean(error_nb_test);
   kfl_nb_train = mean(error_nb_train)
   kfl_nb_val = mean(error_nb_validation)
   kfl_nb_recall = mean(nb_recall)
   kfl_nb_precision = mean(nb_precision)
   kfl_nb_F1 = mean(nb_F1)
   kfl_nb_AUCnb = mean(AUCnb)
   kfl_nb_AUCnb2 = mean(AUCnb2)

%results table 
    nb_Result=table(1-kfl_nb_test,1-kfl_nb_train,1-kfl_nb_val, kfl_nb_F1,kfl_nb_precision,kfl_nb_recall,kfl_nb_AUCnb,kfl_nb_AUCnb2,'VariableNames',{'Test Accuracy','Train Accuracy','Validation Accuracy','F1','Precision','recall','AUC','AUC2'})

    
%%  Modeling:Naive bayes - kernel

%Placeholder table for evaluation metrics

nb_metrics = array2table(zeros(0,7), 'VariableNames',{'Kernel_Width','Train_Accuracy','Validation_Accuracy','Test_Accuracy',...
'F1','Precision','recall'});

%Assign the distribution used for modelling
Distributions = ["kernel","'box","epanechnikov","triangle"];

%Assign kernel to non-normal numerical variables (day and month are considered categorical)
var_distributions = ['kernel', 'mvnm', 'mvnm', 'mvnm', 'mvnm', 'kernel','mvnm',...
'mvnm', 'mvnm','mvnm','mvnm', 'kernel', 'kernel','mvnm','mvnm', 'mnvm'];

%Create index for segementing the data into validation and training
foldindex = crossvalind('Kfold',size(dataTrain,1),10);

%Search for optimal hyperparemters
for i = length(Distributions)
    for kernel_width = 0.05:0.05:5
        for fold = 1:10
        %Further segement the training data into validation and
        %training data
        validation = (dataTrain(foldindex==i,:));
        training = (dataTrain(foldindex~=i,:));
        xTrain = training(:,1:end-1);
        yTrain = training(:,end);
        xVal = validation(:,1:end-1);
        yVal = validation(:,end);
        % Fit model and calculate error for training, validation and
        % test data
        nb_model = fitcnb(xTrain,yTrain, 'DistributionNames',var_distributions,...
        'Width', kernel_width);
        [nb_predictedY, nb_score] = predict(nb_model,xVal);
        nb_TrainError(fold) = loss(nb_model,xTrain,yTrain);
        nb_ValError(fold) = loss(nb_model,xVal,yVal);
        nb_TestError(fold) = loss(nb_model,xTest,yTest);
        %Confusion matrix and performance stats
        nb_cm = confusionmat(yVal, nb_predictedY);
        nb_recall(fold) = nb_cm(2,2)/sum(nb_cm(2,:))* 100;
        nb_precision(fold) = nb_cm(2,2)/sum(nb_cm(:,2))*100;
        nb_F1(fold) =2*nb_recall(i)*nb_precision(i)/(nb_recall(i)+nb_precision(i));
        nb_result = {kernel_width,mean(nb_TrainError),1-mean(nb_ValError),1-mean(nb_TestError),mean(nb_F1),mean(nb_precision),mean(nb_recall)};
        nb_metrics = [nb_metrics;nb_result];
        end
    end
end

%% Average the Kfold models

kfold_avg_nb_metrics = array2table(zeros(0,7), 'VariableNames',{'Kernel_Width','Train_Accuracy','Validation_Accuracy','Test_Accuracy',...
'F1','Precision','recall'});

count = 1;

nb_metrics = table2array(nb_metrics);
for j = 1:length(nb_metrics)/10
    temp = mean(nb_metrics(count:count+9,:));
    temp = array2table(temp,'VariableNames',{'Kernel_Width','Train_Accuracy','Validation_Accuracy','Test_Accuracy',...
    'F1','Precision','recall'});
    kfold_avg_nb_metrics = [kfold_avg_nb_metrics;temp]
    count = count + 10;
end


