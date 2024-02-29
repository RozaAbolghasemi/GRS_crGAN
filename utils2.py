import math
import heapq
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import statistics

num_users, num_items = 0, 0


def get_pairwise_train_dataset(path='data/ml1m_train.dat'):
    global num_users, num_items
    print('loading pair-wise data from flie %s...' % path)
    user_input, item_i_input, item_j_input = [], [], []
    with open(path, 'r') as f:
        for line in f:
            arr = line.split(' ')
            u, i, j = int(arr[0]), int(arr[1]), int(arr[2])
            user_input.append(u)
            item_i_input.append(i)
            item_j_input.append(j)
            if u > num_users:
                num_users = u
            if i > num_items or j > num_items:
                num_items = max(i, j)
    return num_users, num_items, user_input, item_i_input, item_j_input


def get_test_data(path):
    print('loading test data from file %s...' % path)
    testRatings = dict()
    testItems = dict()
    with open(path, 'r') as f:
        for u, line in enumerate(f):
            testRatings[u], testItems[u] = list(), list()
            for item_rating in line.strip().split(' '):
                item, rating = int(item_rating.split(':')[0]), int(float(item_rating.split(':')[1]))
                testItems[u].append(item)
                testRatings[u].append(rating)

    return testItems, testRatings


_model = None
_testItems = None
_testRatings = None
_K = None
_item_rating_dict = None

def evaluate_model(model, testItems, testRatings, K):
    global _model
    global _testItems
    global _testRatings
    global _K
    _model = model
    _testItems = testItems
    _testRatings = testRatings
    _K = K

    metrics = np.array([0. for _ in range(6)])
    for user in _testItems.keys():
        predictions, METRICS = eval_one_rating(user)
        metrics += METRICS
    return metrics / len(_testItems) 


def eval_one_rating(user):
    global _item_rating_dict
    ratings = _testRatings[user]
    items = _testItems[user]
    item_rating_dict = dict()
    for item, rating in zip(items, ratings):
        item_rating_dict[item] = rating
    _item_rating_dict = item_rating_dict
    k_largest_items = heapq.nlargest(_K, item_rating_dict, key=item_rating_dict.get)
    
    # Get prediction scores
    users = np.full(len(items), user, dtype='int32')
    predictions = _model.predict([users, np.array(items)],
                                 batch_size=100, verbose=0)    
    item_prediction_dict = dict()
    for item, prediction in zip(items, predictions):
        item_prediction_dict[item] = prediction
    sorted_item = heapq.nlargest(len(item_rating_dict), item_prediction_dict, key=item_prediction_dict.get)
    top_labels = [1 if item in k_largest_items else 0 for item in sorted_item]

    # Evaluate top rank list
    hr = getHitRatio(top_labels[:_K])
    p = getPrecision(top_labels[:_K])
    ndcg_bin = getNDCG_bin(top_labels[:_K])
    auc = getAUC(top_labels, _K)
    map = getMAP(top_labels, _K)
    mrr = getMRR(top_labels, _K)
    METRICS = np.array([hr, p, ndcg_bin, auc, map, mrr])

    
    return predictions, METRICS



def getHitRatio(labels):
    return 1 if 1 in labels else 0


def getPrecision(labels):
    return sum(labels) / len(labels)


def getNDCG_bin(labels):
    dcg, max_dcg = 0, 0
    for i, label in enumerate(labels):
        dcg += label / math.log2(i + 2)
        max_dcg += 1 / math.log2(i + 2)
    return dcg / max_dcg


def getAUC(labels, _K):
#    global _K
    if len(labels) <= _K:
        return 1

    auc = 0
    for i, label in enumerate(labels[::-1]):
        auc += label * (i + 1)

    return (auc - _K * (_K + 1) / 2) / (_K * (len(labels) - _K))


def getMAP(labels, _K):
#    global _K
    MAP = 0
    for i, label in enumerate(labels):
        MAP += label * getPrecision(labels[:i + 1])
    return MAP / _K


def getMRR(labels, _K):
#    global _K
    mrr = 0
    for i, label in enumerate(labels):
        mrr += label * (1 / (i + 1))
    return mrr / _K



#Roza's functions:

def create_random_groups(Clustered_groups, num_users):
    import random
    # Generate 5 positive numbers whose sum is 20
    #group_sizes = random.sample(range(2, 8), 4)  # Generate 4 random numbers between 1 and 19
    #group_sizes.append(20 - sum(group_sizes))  # Calculate the last number to ensure the sum is 20
    #group_sizes = [10,5,2]
    
    
    num_groups = len(Clustered_groups)

    # Shuffle the numbers between
    numbers = list(range(0, num_users-1))
    random.shuffle(numbers)

    # Create the groups
    groups = []
    start_index = 0
    for group_id in range(0, num_groups):
        # We generate random groups the same size as the clustered groups:
        size = len(Clustered_groups[group_id])
        groups.append(numbers[start_index:start_index+size])
        start_index += size

    # Print the groups
#    for i, group in enumerate(groups, 1):
#        print(f"Group {i}: {group}")
    return groups

def Aggregation(groups, Items, Ratings):
    num_groups = len(groups)
    GroupsRatings = np.zeros((num_groups,num_items))
    for groupID in range (num_groups):
        for item in Items[0]:
            for user in groups[groupID]:   
                GroupsRatings[groupID][item-1]+= Ratings[user][item-1]
            GroupsRatings[groupID][item-1] /= len(groups[groupID]) # average ratings of the users in the group.    
    return GroupsRatings    

def predict_user_ratings(user, items):
    users = np.full(len(items), user)
    # Convert the items list to a NumPy array
    items_array = np.array(items)
    x = [users, items_array]
    predictions = _model.predict(x , batch_size=10, verbose=0)
    return predictions


def fairness(Recommended_items, groups, groupID, median_value, ground_truth):
    Sum_Fairness = 0
    for item in Recommended_items: # set of "RECOMMENDED item for that group"  
            Satisfied_User = 0
            for user in groups[groupID]:
                if ground_truth[user][item-1] >= median_value: 
                    Satisfied_User += 1
            num_groupMembers = len(groups[groupID])        
            Fairness = Satisfied_User/ num_groupMembers
            Sum_Fairness += Fairness
    Group_Fairness = Sum_Fairness / len(Recommended_items)   
#    print ("Group Fairness:", Group_Fairness) 
    #SumGroup_Fairness += Group_Fairness 
    return Group_Fairness
    


def eval_groups(groupID, groups, Groups_ActualRatings, Groups_PredictedRatings, items, AllRatings):
    ratings = Groups_ActualRatings[groupID]
    item_rating_dict = dict()
    for item, rating in zip(items, ratings):
        item_rating_dict[item] = rating
        
    ##***** Instead of making evaluation only based on the top-k items, I considered the items with rank higher that mid as best items and made the evaluation based on this assumption.
#    k_largest_items = heapq.nlargest(_K, item_rating_dict, key=item_rating_dict.get)
    median_value = statistics.median(Groups_ActualRatings[groupID])
    Best_items = [item for item, rating in item_rating_dict.items() if rating >= median_value]
    y_true = [1 if rating >= median_value else 0 for item, rating in item_rating_dict.items()]
#    print("Best_items:", Best_items)
#    print("y_true:",y_true)
#    print("k_largest_items:", k_largest_items)
    
    predictions = Groups_PredictedRatings[groupID]  
    item_prediction_dict = dict()
    for item, prediction in zip(items, predictions):
        item_prediction_dict[item] = prediction
    sorted_item = heapq.nlargest(len(item_rating_dict), item_prediction_dict, key=item_prediction_dict.get)
 #   print("sorted_item:", sorted_item)
#    top_labels2 = [1 if item in k_largest_items else 0 for item in sorted_item]
    top_labels = [1 if item in Best_items else 0 for item in sorted_item]
    median_value = statistics.median(Groups_PredictedRatings[groupID])
    y_pred = [1 if rating >= median_value else 0 for item, rating in item_prediction_dict.items()]
    Recommended_items = [item for item, rating in item_prediction_dict.items() if rating >= median_value]
#    print("top_labels:", top_labels)
#    print("y_pred:",y_pred)
#    print("top_labels2:", top_labels2)

    _K = len(Best_items)
    # Evaluate top rank list
    hr = getHitRatio(top_labels[:_K])
    p = getPrecision(top_labels[:_K])
    ndcg_bin = getNDCG_bin(top_labels[:_K])
    auc = getAUC(top_labels, _K)
    map = getMAP(top_labels, _K)
    mrr = getMRR(top_labels, _K)
    #**** New metrics for Group Recommendation ******
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    Fairness = fairness(Recommended_items, groups, groupID, median_value, AllRatings)

    METRICS = np.round(np.array([hr, p, ndcg_bin, auc, map, mrr, accuracy, precision, recall, f1, Fairness]), 2)
#    print("\n \n ***********\n hr, p, ndcg_bin, auc, map, mrr, accuracy, precision, recall, f1")
#    print(METRICS)   
#    print("***********")
    return  METRICS






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

