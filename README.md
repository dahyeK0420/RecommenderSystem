# Recommender System using ALS and MF Models
This notebook utilises ALS and MF models to recommend the top 10 items to each user based on the user interaction history.

## Introduction 
This notebook creates a recommender system for the users on Flickr. The recommender system predicts the rating of the user to candidate items in the test set and recommends top-15 recommended items from the candidate items. In total, 3456 users and 9004 items are involved in each dataset. 

Model Candidates and Hypothesis 
To build the recommender systems, the algorithms trained and compared are as follow: 

*	Matrix Factorisation model (MF) using torch 
*	Matrix Factorisation model with bias (MFB) using torch 
*	Alternating Least Squares (ALS) model using implicit

![alt text](https://github.com/dahyeK0420/RecommenderSystem/blob/3aec14f52824ce2c5b19e7dce78ef9ff4291533c/Picture%201.png)

All models assumes that the user and items are representable within a low-dimensional space. The matrix factorisation with bias artificially adds training bias in order to trade-off lower variance of the model. ALS model, on the other hand, optimises each parameter in the model one at a time in order to convert the object function of the matrix factorisation model into a least-squares problem. Considering that the train set does not have the rating of all users to all items, and assuming that both items and users are representable in a low-dimensional space, we can assume that ALS should have a better performance than MF or MFB. 

## Train, Validation, and Test Datasets
The dataset utilised for training, validating, and testing three algorithms above are: 

*	***‘flickr_train_data.csv’***: a dataset with three columns – each user’s unique ID, each item’s unique ID, and a binary attribute indicating whether the user pressed liked for the item. All values in the binary attribute are 1s, which means that the dataset is showing which item the user has already interacted with. This is the train set for training the algorithms mentioned above 

*	***‘flickr_valid_data.csv’***: a dataset with exactly the same structure as the train set. However, each user has only one item with the ‘rating’ value of 1. All the rest of the items listed in the data frame are not yet interacted by the users yet. This is the validation set for evaluating each algorithm by calculating the discounted cumulative gain (DCG) scores. 

*	***‘flickr_test_data.csv’***: the dataset with only user IDs and item IDs. The user-item combinations in this dataset are utilised for deriving the top-15 recommended items to each user

The notebook first starts off by inspecting the structure of each dataset. Duplicates detected from both validation and test sets are removed. For the MF and MFB, the user_id, item_id, and rating attributes are all converted into long tensors. Each user_id and item_id is then embedded into a vector length of K, and all users’ IDs and items’ IDs then become user and item matrices respectively. The item matrix has a dimension of K*number of items IDs in the train set, and the user matrix has a dimension of K*number of users’ IDs in the train set. The rating of a user to an item is the product of respective user and item vectors. To enable the multiplication, the user matrix is transposed to the dimension of number of users’ IDs * K. K is one of the hyperparameters to tune for optimising the performance of all three models. For ALS, the train set is converted into a sparse matrix and fitted into an ALS model.
