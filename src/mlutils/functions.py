import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm

def prediction_accuracy_summary(y, y_hat, model='Model'):
    # Date Check: Jan 6, 2026
    # y = True y (Series or Numpy array)
    # y_hat = Predicted y (Series or Numpy array)
    # model =  Label for the model used for y_hat


# Create index
    index_names = ['Mean Error', 'Root Mean Square Error', 
                   'Mean Absolute Error', 'Mean Percentage Error', 
                   'Mean Absolute Percentage Error']
# Assemble to accuracy measures
    accuracy_vals = [(y-y_hat).mean(),
                     skm.root_mean_squared_error(y, y_hat),
                     skm.mean_absolute_error(y, y_hat), 
                     ((y-y_hat)/y).mean()*100, 
                     skm.mean_absolute_percentage_error(y, y_hat)*100]
   
# Set up DataFrame
    accuracy_df = pd.DataFrame(accuracy_vals, index=index_names, columns=[model])
    
# Return the completed DataFrame
    return accuracy_df


def dw_gains_reg(y, y_hat, title='Decile-wise gains chart'):
    """ Create a decile-wise gains chart

    Input:
        y: Pandas Series of actual outcome
        y_hat: Array of predicted y (Pandas Series or Numphy array)
        
    Output:
        Decile-wise gains chart
    """
# Validate size of y and y_hat
    if len(y) != len(y_hat):
        raise ValueError("Length of array of actual outcome msut be the same as the length of probability array.")

    deciles = pd.qcut(y_hat, 10, labels=np.arange(10,0,-1))
    decile_sum = y.groupby(deciles, observed=True).sum() 
    y_sum = y.sum()
    
    decile_gains = decile_sum/y_sum * 100
    
    fig, ax = plt.subplots()
    ax.bar(decile_gains.index, decile_gains)
    ax.set_ylabel('Gains %')
    ax.set_xlabel('Decile')
    ax.set_title(title)
    ax.set_xticks(range(1,11))
    
    ax.bar_label(ax.containers[0], fmt='%4.0f%%', label_type='edge', padding=1, size=8)
    plt.show()
    
def dw_cumulative_gains_reg(y, y_hat, title='Decile-wise gains chart'):
    """ Create a decile-wise cumulative gains chart

    Input:
        y: Pandas Series of actual outcome
        y_hat: Array of predicted y (Pandas Series or Numphy array)
        
    Output:
        Decile-wise cumulative gains chart
    """
# Validate size of y and y_hat
    if len(y) != len(y_hat):
        raise ValueError("Length of array of actual outcome msut be the same as the length of probability array.")

   
    deciles = pd.qcut(y_hat, 10, labels=np.arange(10,0,-1))
    decile_sum = y.groupby(deciles, observed=True).sum() 
    decile_sum.sort_index(ascending=False, inplace=True)
    decile_cumsum = decile_sum.cumsum() 
    
    y_sum = y.sum()
    
    decile_cumgains = decile_cumsum/y_sum * 100

  
    fig, ax = plt.subplots()
    ax.bar(decile_cumgains.index, decile_cumgains)
    ax.set_ylabel('Gains %')
    ax.set_xlabel('Decile')
    ax.set_title(title)
    ax.set_xticks(range(1,11))
    
    ax.bar_label(ax.containers[0], fmt='%4.0f%%', label_type='edge', padding=1, size=8)
    plt.show()
    

def dw_cumulative_lift_reg(y, y_hat, title='Decile-wise cumulative lift chart'):
    """ Create a decile-wise cumulative lift chart

    Input:
        y: Pandas Series of actual outcome
        y_hat: Array of predicted y (Pandas Series or Numphy array)
        
    Output:
        Decile-wise cumulative lift chart
    """

# Validate size of y and y_hat
    if len(y) != len(y_hat):
        raise ValueError("Length of array of actual outcome msut be the same as the length of probability array.")

    deciles = pd.qcut(y_hat, 10, labels=np.arange(10,0,-1))
    decile_sum = y.groupby(deciles, observed=True).sum() 
    decile_count = y.groupby(deciles, observed=True).count() 

    decile_sum.sort_index(ascending=False, inplace=True)
    decile_count.sort_index(ascending=False, inplace=True)
    
    decile_cumsum = decile_sum.cumsum() 
    decile_cumcount = decile_count.cumsum() 

    decile_cummean = decile_cumsum/decile_cumcount
    
    y_mean = y.mean()
    
    decile_cum_lift = decile_cummean/y_mean

    fig, ax = plt.subplots()
    ax.bar(decile_cum_lift.index, decile_cum_lift)
    ax.set_ylabel('Cumulative lift')
    ax.set_xlabel('Decile')
    ax.set_title(title)
    ax.set_xticks(range(1,11))
    ax.bar_label(ax.containers[0], fmt='{:.2f}', label_type='edge', padding=1, size=8)
    plt.show()


def rlike_metrics(x, labels=None, pos_label=None):
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
    print(f"{labels[0]:15}{a11:>8}{a12:>10}")
    print(f"{labels[1]:15}{a21:>8}{a22:>10}\n\n")

    
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

def prf1plot(y, y_hat, labels=None, pos_label=None, title='Precision-Recall-F1 Plot'):
    """ Create a Precision-Recall-F1 Score plot

    Input:
        y: Pandas Series of actual outcome -- must have only two unique values
        y_hat: Array of probability
        pos_label = Optional, default = First encountered value in y
        
    Output:
        Precision-Recall-F1 Score plot
        Returns DataFrame of the Precision, Recall, and F1-Score 
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
#    y_numeric = pd.Series([1 if x == pos_label else 0 for x in y])
    precision_recall_df = pd.DataFrame(skm.precision_recall_curve(y, y_hat, pos_label=pos_label)).T
    precision_recall_df.columns = ['Precision', 'Recall', 'Threshold']
    precision_recall_df.set_index('Threshold', inplace=True)
    precision_recall_df['F1 Score'] = (2*precision_recall_df['Precision']*precision_recall_df['Recall'])/(precision_recall_df['Precision']+precision_recall_df['Recall'])

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(precision_recall_df, label=precision_recall_df.columns)

    ax.set_title(title)
    ax.set_xlabel('Cutoff probability')
    ax.set_ylabel('Metric')
    ax.legend()
    plt.show()
    return precision_recall_df

def misclassification_cost_plot(y, yhat, q1, q2, pos_label=None, title='Misclassification Cost Plot'):

    """ Create a plot of misclassification cost for a range of cutoff probability

    Input:
    y = Pandas Series of actual class  -- must have only two unique values
    yhat = Array of probability
    q1 = Cost of false positive
    q2 = Cost of false negative
    pos_label = String - Label for positive class (Optional)
    title = title for the plot
        
    Output:
        Misclassification cost plot
    """
    
# Validate size of y and y_hat
    if len(y) != len(yhat):
        raise ValueError("Length of array of actual outcome msut be the same as the length of probability array.")

# Validate two outcomes in y
    if y.nunique() != 2:
        raise ValueError("Number of unique actual outcome must be 2.")

# Validate positive label
    ordered_classes = np.sort(y.unique())
    if pos_label is None:
        pos_label = ordered_classes[0]  # If None pos_label = first unique value

    if pos_label not in ordered_classes:
        raise ValueError("Given pos_label not a valid actual.")

# Set up label for negative class
    if pos_label == ordered_classes[0]:
        neg_label = ordered_classes[1]
    else:
        neg_label = ordered_classes[0]
    
# Validate y_hat is floating point
    if not np.issubdtype(yhat.dtype, np.floating):
        raise ValueError("yhat must be floating point data type.")

    miss_cost=pd.Series()   # Initialize Series for misclassification cost
    for cutoff_probability in np.arange(0, 1.01, .05):
        yp = [pos_label if p >= cutoff_probability else neg_label for p in yhat]

        # Generate confusion matrix
        x = skm.confusion_matrix(y, yp)

        # Extract matrix elements
        a11, a12, a21, a22 = x[0, 0], x[0, 1], x[1, 0], x[1, 1]

        # Assign tp, tn, fp, fn based on which label is positive
        if pos_label == ordered_classes[0]:  # first row = positive
            tp, fp, fn, tn = a11, a21, a12, a22
        else:                       # second row = positive
            tp, fp, fn, tn = a22, a12, a21, a11

        # Add cost to the Pandas Series
        miss_cost.loc[cutoff_probability] = (fp*q1+fn*q2)

# Plot the Pandas Series of misclassification costs
    fig, ax = plt.subplots()
    ax.plot(miss_cost.index, miss_cost)
    
    ax.set_xlabel('Cutoff (threshold probability)')
    ax.set_ylabel('Opportunity cost')
    
    ax.set_title(title)
    plt.show()


def dw_gains_class(y, y_hat, pos_label=None, title='Decile-wise gains chart'):
    """ Create a decile-wise gains chart

    Input:
        y: Pandas Series of actual outcome -- must have only two unique values
        y_hat: Array of probability
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
    ax.set_title(title)
    ax.set_xticks(range(1,11))
    
    ax.bar_label(ax.containers[0], fmt='%4.0f%%', label_type='edge', padding=1, size=8)
    plt.show()

    
 
def dw_cumulative_gains_class(y, y_hat, pos_label=None, title='Decile-wise cumulative gains chart'):
    """ Create a decile-wise cumulative gains chart

    Input:
        y: Pandas Series of actual outcome -- must have only two unique values
        y_hat: Array of probability
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
    decile_events = y_numeric.groupby(deciles, observed=True).sum() # Sum events in each decile
    decile_events.sort_index(ascending=False, inplace=True)
    decile_cum_events = decile_events.cumsum()
    decile_cum_gains = decile_cum_events/decile_events.sum()*100
    
    fig, ax = plt.subplots()
    ax.bar(decile_cum_gains.index, decile_cum_gains)
    ax.set_ylabel('Cumulative gains %')
    ax.set_xlabel('Decile')
    ax.set_title(title)
    ax.set_xticks(range(1,11))
    
    ax.bar_label(ax.containers[0], fmt='%4.0f%%', label_type='edge', padding=1, size=8)
    plt.show()
    

def dw_cumulative_lift_class(y, y_hat, pos_label=None, title='Decile-wise cumulative lift chart'):
    """ Create a decile-wise cumulative lift chart

    Input:
        y: Pandas Series of actual outcome -- must have only two unique values
        y_hat: Array of probability
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
    ax.set_ylabel('Cumulative lift')
    ax.set_xlabel('Decile')
    ax.set_title(title)
    ax.set_xticks(range(1,11))
    ax.bar_label(ax.containers[0], fmt='{:.2f}', label_type='edge', padding=1, size=8)
    plt.show()
    

def cum_costbenefit_gains(y, y_hat, fpcost, tpbenefit, pos_label=None, title='Cumulative Cost/Benefit Gains Chart'):
    """ Create a decile-wise cumulative cost/benefit gains chart

    Input:
        y: Pandas Series of actual outcome -- must have only two unique values
        y_hat: Array of probability

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
    y_sorted = pd.DataFrame({'y':y, 'y_hat':y_hat}).sort_values(by='y_hat', ascending=False)['y']
    
    cost_benefit = y_sorted.transform(lambda x: tpbenefit if x==pos_label else -fpcost)
    cost_benefit_cumsum = pd.concat([pd.Series([0]), cost_benefit.cumsum()], ignore_index=True)  #Use pd.concat to add 0 as the first value

# Best performance possible
# Sort y with positive cases on top
    uniques = np.sort(y.unique())
    if pos_label==uniques[0]:
        y_sorted_best = y.sort_values(ascending=True)
    else:
        y_sorted_best = y.sort_values(ascending=False)
    
    cost_benefit_best =  y_sorted_best.transform(lambda x: tpbenefit if x==pos_label else -fpcost)
    cost_benefit_best_cumsum = pd.concat([pd.Series([0]), cost_benefit_best.cumsum()], ignore_index=True)  #Use pd.concat to add 0 as the first value


    fig, ax = plt.subplots()
    ax.plot(np.arange(len(y)+1), cost_benefit_best_cumsum, color='lightgrey')  # Best possible performance
    ax.plot(np.arange(len(y)+1), cost_benefit_cumsum)  # Line for cumulative cost/benefit; add 1 to length for the zero added as the first value 
    ax.plot([0, len(y)], [0, cost_benefit_cumsum.iloc[len(y)]], color='navy', lw=2, linestyle='--')   # Diagonal line for Naive predictions
    
    ax.set_xlabel('Cases')
    ax.set_ylabel('Cost/Benefit')
    ax.set_title(title)
    
    plt.show()
    
    

