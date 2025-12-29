import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def cm_calcMetrics(x, labels=None, pos_label=None):
    """ Calculate performance measures and print similar to R

    Input:
        x: 2x2 confusion matrix of integers as Numpy array or DataFrame
        labels: List of 2 elements str or int (optional); default = [0, 1] 
        pos_label: str or int (Optional); default = labels[0]
        
    Output:
        Prints performance measures in R-like format
        Returns a dictionary of the performance measures
        
    """
#-------------------------------------------------------
# Unpack confusion matrix into tp, tn, fp. fn
#-------------------------------------------------------
    
# x = Confusion matrix - Numpy array or DataFrame
# lables = List of labels of outcome - length must be 2
# pos_label = Positive label

#-------------------------------------------------------------------------------
# Determine tp, fp, fn and tn
#-------------------------------------------------------------------------------
    
# Get elements from the confusion matrix
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

# Validate shape
    if x.shape != (2,2):
        raise ValueError("Input must be 2x2 matrix.")

# Handle labels
    if labels is None:
        labels=[0, 1]
    if len(labels) != 2:
        raise ValueError("There must be exactly two labels.")

# Default primary label
    if pos_label is None:
        pos_label = labels[0]
    elif pos_label not in labels:
        raise ValueError(f"pos_label '{pos_label}' not in {labels}")

# Extract matrix elements
    a11, a12, a21, a22 = x[0, 0], x[0, 1], x[1, 0], x[1, 1]

# Assign based on which label is positive
    if pos_label == labels[0]:  # first row = positive
        tp, fp, fn, tn = a11, a21, a12, a22
    else:                       # second row = positive
        tp, fp, fn, tn = a22, a12, a21, a11

#-------------------------------------------------------
# Calculate metrics
#-------------------------------------------------------
    n_pos = tp+fn
    n_neg = tn+fp
    pred_pos = tp+fp
    pred_neg = tn+fn
    
    n = n_pos+n_neg
    
    accuracy = (tp+tn)/n
    misclass = 1-accuracy

    sensitivity = tp/n_pos
    specificty = tn/n_neg
    
    precision = tp/pred_pos
    omission = tn/pred_neg
    
    f1 = (2*precision*sensitivity)/(precision+sensitivity)
    
    false_doscovery_rate = fp/pred_pos
    false_omission_rate = fn/pred_neg
    prevalence = n_pos/n
    detection = tp/n
    detection_prevalace = pred_pos/n
    balanced_accuracy = (sensitivity+specificty)/2

#-------------------------------------------------------
# Print metrics
#-------------------------------------------------------
    print('Confusion Matrix')
    print(f"{'Prediction':>28}")
    print(f"{"Actual":15}{labels[0]:>8} {labels[1]:>10}")
    print(f"{labels[0]:15}{tp:>8}{fp:>10}")
    print(f"{labels[1]:15}{fn:>8}{tn:>10}\n\n")

    
    print('Classification metrics')
    print(f'{"Positive lablel : ":>40} {pos_label}')
    
    print(f'{"Accuracy : ":>40} {accuracy:.4f}')
    print(f'{"Misclassification rate : ":>40} {misclass:.4f}\n')
    
    print(f'{"Sensitivity (Recall) : ":>40} {sensitivity:.4f}')
    print(f'{"Specificity : ":>40} {specificty:.4f}')
    print(f'{"True discovery rate (Precision) : ":>40} {precision:.4f}')
    print(f'{"True omission rate : ":>40} {omission:.4f}\n')
    
    print(f'{"F1-score : ":>40} {f1:.4f}\n')
    
    
    print(f'{"False discovery rate : ":>40} {false_doscovery_rate:.4f}')
    print(f'{"False omission rate : ":>40} {false_omission_rate:.4f}\n')
    
    print(f'{"Prevalence : ":>40} {prevalence:.4f}')
    print(f'{"Detection rate : ":>40} {detection:.4f}')
    print(f'{"Detection prevalence : ":>40} {detection_prevalace:.4f}')
    print(f'{"Balanced accuracy : ":>40} {balanced_accuracy:.4f}')
    
#-------------------------------------------------------
# Assemble and return metrics
#-------------------------------------------------------
    metricsDict = {"Accuracy":accuracy,
                    "Misclassification rate":misclass,
                    
                    "Sensitivity (Recall)":sensitivity,
                    "Specificity":specificty,
                    "True discovery rate (Precision)":precision,
                    "True omission rate":omission,
                    
                    "F1-score":f1,
                    
                    
                    "False discovery rate":false_doscovery_rate,
                    "False omission rate":false_omission_rate,
                    
                    "Prevalence":prevalence,
                    "Detection rate":detection,
                    "Detection prevalence":detection_prevalace,
                    "Balanced accuracy":balanced_accuracy}
    return metricsDict

def dw_gain(y, y_hat, pos_label=None):
    """ Create a decile-wise gains chart

    Input:
        y: Array of actual outcome -- must have only two unique values
        y_hat: Probability for positive outcome
        pos_label = Optional, default = First encountered value in y
        
    Output:
        Decile-wise gains chart
    """
# Validate size of y and y_hat
    if len(y) != len(y_hat):
        raise ValueError("Length of array of actual outcome msut be the same as the length of probability array.")

# Validate two outcomes in y
    if y.nunique() != 2:
        raise ValueError("Number of unique actual outcome must be 2.")

# Validate positive label
    if pos_label is None:
        pos_label = y.unique()[0]  # If None first unique value

    if pos_label not in y.unique():
        raise ValueError("Given pos_label not a valid actual.")

# Convert y into numeric
    y_numeric = pd.Series([1 if x == pos_label else 0 for x in y])
    
# Validate y_hat is floating point
    if not np.issubdtype(y_hat.dtype, np.floating):
        raise ValueError("y_hat must be floating point data type.")
    
    deciles = pd.qcut(y_hat, 10, labels=np.arange(10,0,-1))
    decile_events = y_numeric.groupby(deciles, observed=True).sum() # Sum events in each decile
    decile_gains = decile_events/decile_events.sum()*100
    
    fig, ax = plt.subplots()
    ax.bar(decile_gains.index, decile_gains)
    ax.set_ylabel('Gains %')
    ax.set_xlabel('Decile')
    ax.set_title(r'Decile-wise gains chart')
    ax.set_xticks(range(1,11))
    
    ax.bar_label(ax.containers[0], fmt='%4.0f%%', label_type='edge', padding=1, size=8)
    plt.show()

    
 
def dw_cumulative_gain(y, y_hat, pos_label=None):
    """ Create a decile-wise cumulative gains chart

    Input:
        y: Array of actual outcome -- must have only two unique values
        y_hat: Probability for positive outcome
        pos_label = Optional, default = First encountered value in y
        
    Output:
        Decile-wise cumulative gains chart
        
    """
    
# Validate size of y and y_hat
    if len(y) != len(y_hat):
        raise ValueError("Length of array of actual outcome msut be the same as the length of probability array.")

# Validate two outcomes in y
    if y.nunique() != 2:
        raise ValueError("Number of unique actual outcome must be 2.")

# Validate positive label
    if pos_label is None:
        pos_label = y.unique()[0]  # If None first unique value

    if pos_label not in y.unique():
        raise ValueError("Given pos_label not a valid actual.")

# Convert y into numeric
    y_numeric = pd.Series([1 if x == pos_label else 0 for x in y])
    
# Validate y_hat is floating point
    if not np.issubdtype(y_hat.dtype, np.floating):
        raise ValueError("y_hat must be floating point data type.")


    deciles = pd.qcut(y_hat,10, labels=np.arange(10,0,-1))
    decile_events = y.groupby(deciles, observed=True).sum() # Sum events in each decile
    decile_events.sort_index(ascending=False, inplace=True)
    decile_cum_events = decile_events.cumsum()
    decile_cum_gains = decile_cum_events/decile_events.sum()*100
    
    fig, ax = plt.subplots()
    ax.bar(decile_cum_gains.index, decile_cum_gains)
    ax.set_ylabel('Cunulative gains %')
    ax.set_xlabel('Decile')
    ax.set_title(r'Decile-wise cunulative gains chart')
    ax.set_xticks(range(1,11))
    
    ax.bar_label(ax.containers[0], fmt='%4.0f%%', label_type='edge', padding=1, size=8)
    plt.show()
    

def dw_cumulative_lift(y, y_hat, pos_label=None):
    """ Create a decile-wise cumulative gains chart

    Input:
        y: Array of actual outcome -- must have only two unique values
        y_hat: Probability for positive outcome
        pos_label = Optional, default = First encountered value in y
        
    Output:
        Decile-wise cumulative gains chart
        
    """
    
# Validate two outcomes in y
    if y.nunique() != 2:
        raise ValueError("Number of unique actual outcome must be 2.")

# Validate positive label
    if pos_label is None:
        pos_label = y.unique()[0]  # If None first unique value

    if pos_label not in y.unique():
        raise ValueError("Given pos_label not a valid actual.")

# Convert y into numeric
    y_numeric = pd.Series([1 if x == pos_label else 0 for x in y])
    
# Validate y_hat is floating point
    if not np.issubdtype(y_hat.dtype, np.floating):
        raise ValueError("y_hat must be floating point data type.")


    deciles = pd.qcut(y_hat,10, labels=np.arange(10,0,-1))
    decile_events = y_numeric.groupby(deciles, observed=True).sum() # Sum events in each decile
    decile_events.sort_index(ascending=False, inplace=True)
    decile_cum_events = decile_events.cumsum()

    decile_cases = y.groupby(deciles, observed=True).count() # number of cases
    decile_cases.sort_index(ascending=False, inplace=True)
    decile_cum_cases = decile_cases.cumsum()

#    naive_events_rate = decile_events.sum()/len(y)
    naive_events_rate = y_numeric.mean()
    decile_cum_events_naive = decile_cum_cases * naive_events_rate
    decile_cum_lift = decile_cum_events/decile_cum_events_naive
    
    fig, ax = plt.subplots()
    ax.bar(decile_cum_lift.index, decile_cum_lift)
    ax.set_ylabel('Cunulative lift')
    ax.set_xlabel('Decile')
    ax.set_title(r'Decile-wise cunulative lift')
    ax.set_xticks(range(1,11))
    ax.bar_label(ax.containers[0], fmt='{:.2f}', label_type='edge', padding=1, size=8)
    plt.show()
    

def cum_costbenefit_gain(y, y_hat, fpcost, tpbenefit, pos_label=None):
    """ Create a decile-wise cumulative gains chart

    Input:
        y: Array of actual outcome -- must have only two unique values
        y_hat: Probability for positive outcome

        fpcost = Cost for false-positive
        tpbenefit = Benefit for true-positive

        pos_label = Optional, default = First encountered value in y
        
    Output:
        Cumulative cost/benefit gains chart
        
    """
# Validate size of y and y_hat
    if len(y) != len(y_hat):
        raise ValueError("Length of array of actual outcome msut be the same as the length of probability array.")
    
# Validate two outcomes in y
    if y.nunique() != 2:
        raise ValueError("Number of unique actual outcome must be 2.")

# Validate positive label
    if pos_label is None:
        pos_label = y.unique()[0]  # If None first unique value

    if pos_label not in y.unique():
        raise ValueError("Given pos_label not a valid actual.")

# Validate y_hat is floating point
    if not np.issubdtype(y_hat.dtype, np.floating):
        raise ValueError("y_hat must be floating point data type.")    
    

# Sort y by y_hat
    # Combine y and y_hat into a DataFrame
    # Sort values by y_hat
    # Extract y from the sorted DataFrame
    y_sorted = pd.DataFrame({'y':y, 'y_hat':y_hat}).sort_values(by='y_hat')['y']
    
    cost_benefit = y_sorted.transform(lambda x: tpbenefit if x==pos_label else -fpcost)
    cost_benefit_cumsum = pd.concat([pd.Series([0]), cost_benefit.cumsum()], ignore_index=True)  #Use pd.concat to add 0 as the first value

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(y)+1), cost_benefit_cumsum)  # Line for cumulative cost/benefit; add 1 to length for the zero added as the first value 
    ax.plot([0, len(y)], [0, cost_benefit_cumsum.iloc[len(y)]], color='navy', lw=2, linestyle='--')   # Diagonal line for Naive predictions
    
    ax.set_xlabel('Cases')
    ax.set_ylabel('Cost/Benefit')
    ax.set_title('Cumulative Cost/Benefit Gain Chart')
    
    plt.show()