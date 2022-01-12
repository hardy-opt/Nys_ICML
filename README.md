This is a repository to perform Nystrom SVRG and Nystrom SGD experiments on various datasets

Note that the base library of used in this study is available at:
SGDLibrary by Hiriyoki Kasai : https://github.com/hiroyuki-kasai/SGDLibrary

1) Initialization

        Add Nys_Curve repository in the MATLAB path (all folders and all sub-folders)

        Once you have all datasets, seprate it into train and test data as follows

        ".MAT" file is used as input data to a problem

        Create ".MAT" file for a dataset by seperating tain and test data as
        x_train (n x d) for train data
        y_train for train label
        x_test  for test  data
        y_test  for test  lebel

        n : samples, and d : dimension

        For example: To use w8a data set, create

        " W8A.mat " which contains x_train, y_train, x_test and y_test

        To initialize the probem, matlab (".m") file for each data set can be found in " Nys/Datasets/ " folder

2) We use l2-logistic regression problem and we used " /Nys/logistic_regression1.m " to perform the experiments

        " logistic_regression1.m " computes the 

        cost of loss function
        gradient and stochastic gradient 
        Hessian 
        Matrix C,M and Z for Nystrom SVRG and Nystrom SGD


3) Nystrom SVRG and Nystrom SGD

        We use following format to perform Nystrom SVRG and Nystrom SGD

        Nystrom_svrg(problem, options,rho,v)

        Nystrom_sgd(problem, options,rho,v)

        where input of 

        problem is " loss function "

        options contains various hyper-parameters ( see in 4) nys_curve_exp )

        rho is the regularizer for Nystrom  ( Z Z^T + rho I )

        v represent the variant

        v = 1 performs the DP sampling variant NSVRG-D and NSGD-D

        v not equal to 1 perform the NSVRG and NSGD


4) nys_curve_exp

        " nys_curve_exp.m " is the main file to perform all of the experiments together

        It requires various hyper-parameters and initialize conditions as follows

        no. of columns to create C 
        step size/ learning rate
        batch size
        no. of partition for DP sampling
        no. of clusters for DP sampling
        no. of RUN (for corresponing method)
        no. of epoch
        l_2 regularizer of logistic_regression


