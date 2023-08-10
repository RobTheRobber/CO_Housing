import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from itertools import combinations 
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
import math
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None  # default='warn'


def map_values(df,column):
    mapping ={}
    for category in df[column].unique():
        if category not in mapping:
            # Assign a numeric value for each category
            mapping[category] = len(mapping) + 1
    return df[column].map(mapping)

def load_data(file_path):
    # Loads the dataset from file path
    df = pd.read_csv(file_path)
    return df


def remove_small_groupings(df,column,size):
    # Filters out groups based on the specified number of entries
    counts = df.groupby(column)[column].transform('count')
    filtered_df = df[counts > size]
    return filtered_df


def filter_outliers(df,column,std_threshold = 3):
    # Filter outliers using the mean, std, and the threshold to filter out the data
    mean = df[column].mean()
    std = df[column].std()
    std_threshold = 3
    df_filtered = df[(np.abs(df[column] - mean) <= std_threshold * std)]
    return df_filtered


def remove_unwanted_columns(df, columns):
    # Removes unwanted columns from the dataframe
    df = df.drop(columns= columns)
    return df

def remove_empty_data(df, columns = None):
    # Removes empty entries from specified column or all columns if columns left as None
    if columns == None:
        df = df.dropna()

    else:
        df = df.dropna(columns=columns)
    return df

def create_boxplot(df,data,grouping,title,y_ax,x_ax):
    ax = df.boxplot(data, by=grouping, figsize=(16, 16), showfliers=False, vert=False,
                    patch_artist=True, boxprops=dict(facecolor="grey"),
                    medianprops=dict(color="red", linewidth=1.5),
                    whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2))
    plt.suptitle("")
    ax.set_title(title)
    ax.set_ylabel(y_ax)
    ax.set_xlabel(x_ax)
    return ax

def cross_val(X_train, y_train, k, model=LinearRegression()):
    scores = []
    split_k = KFold(n_splits=k, shuffle=True)
    for train_index, test_index in split_k.split(X_train):
        # from chad discussion
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        model = model
        model.fit(X_train_fold, y_train_fold)
        pred = model.predict(X_test_fold)

        rmse_fold = rmse(y_test_fold, pred)
        scores.append(rmse_fold)

        avg_scores = np.mean(scores)
    return avg_scores

def cv(X, y, base_estimator, n_folds, random_seed=154):
    """Estimate the in- and out-of-sample error of a model using cross
    validation.
    
    Parameters
    ----------
    
    X: np.array
      Matrix of predictors.
      
    y: np.array
      Target array.
      
    base_estimator: sklearn model object.
      The estimator to fit.  Must have fit and predict methods.
      
    n_folds: int
      The number of folds in the cross validation.
      
    random_seed: int
      A seed for the random number generator, for repeatability.
    
    Returns
    -------
      
    train_cv_errors, test_cv_errors: tuple of arrays
      The training and testing errors for each fold of cross validation.
    """
    random_seed = 1
    mse_scores = []
    kf = KFold(n_splits=n_folds, shuffle=True,random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X)):
        # Split into train and test
      train = train
      test = test
      # Standardize data, fit on training set, transform training and test.
      scalar  = StandardScaler()
      X_train_fold, X_test_fold = X[train], X[test]
      y_train_fold, y_test_fold = y[train], y[test]
      X_train_scaled= scalar.fit_transform(X_train_fold)
      X_test_scaled= scalar.transform(X_test_fold)
      # Fit ridge regression to training data.
  
      base_estimator.fit(X_train_scaled,y_train_fold)
      # Make predictions.
      pred = base_estimator.predict(X_train_scaled)
      # Calculate MSE.
      # pred = pred.astype(float)
      mse = mean_squared_error(y_train_fold, pred)
      # Record the MSE in a numpy array.
      mse_scores.append(mse)

    return mse_scores

def one_sample(sample_mean, known_mean, std, size, two_sided = True):
    t = (sample_mean - known_mean) / (std/size ** .5)
    ddof = size - 1
    
    
    p =  stats.t.sf(np.abs(t), ddof)
    if two_sided:
        p = p*2
        
    return t, ddof, p

def cross_val_size(reg,X, y,size):
  scores = []
  for i in range(2, size):
    scores.append(np.abs(np.mean(cross_val_score(reg,X, y, cv=i, scoring="neg_root_mean_squared_error"))))
  return scores

def test_pairs(df, grouping_var, value, alpha=.05):
    
    results = pd.DataFrame()
    combos = combinations(pd.unique(df[grouping_var]), 2)
    for grp1, grp2 in combos:
        grp1_values = df[df[grouping_var] == grp1][value]
        grp2_values = df[df[grouping_var] == grp2][value]
        
        ttest_p_value = stats.ttest_ind(grp1_values, grp2_values, alternative='two-sided')[1]
        mw_p_value = stats.mannwhitneyu(grp1_values, grp2_values, alternative='two-sided')[1]
        
        grp1_mean = grp1_values.mean()
        grp2_mean = grp2_values.mean()
        
        diff = grp1_mean-grp2_mean
        is_significant = ttest_p_value < alpha
        
        
        results = results.append({
                'first_group':grp1, 'second_group':grp2, 
                'first_group_mean':grp1_mean, 'second_group_mean':grp2_mean,
                'mean_diff':diff, 'ttest_p_value':ttest_p_value, 'mw_p_value': mw_p_value,
                'is_significant': is_significant},
                ignore_index=True)

    #order logically
    results = results[['first_group', 'second_group', 
                    'first_group_mean', 'second_group_mean', 
                    'mean_diff', 'ttest_p_value', 'mw_p_value', 'is_significant']
                     ].sort_values('first_group_mean')

    return results
if __name__ == "__main__":
    pass
    # filter_outliers_list = "Users"
    # sat = pd.read_excel('data\UCS-Satellite-Database.xlsx')
    # # Create the df and filter out unused data
    # filtered_df = remove_small_groupings(sat,filter_outliers_list,1)
    # t_housing = remove_unwanted_columns(filtered_df,sat.columns[24:-1])

    # # Create pie chart
    # pie_group = t_housing['LandUse'].value_counts().head(5)
    # pie_group.plot(kind='pie',title="Nashville Land Use Percentages", figsize=(20,9),ylabel=" ",  labels=pie_group.index, autopct ="%1.1f%%" )

    # # Create Total Properties per District Bar Chart
    # tax_district= t_housing.groupby('TaxDistrict')["LandUse"].size()
    # ax = tax_district.plot.barh(title="Total Properties Per Tax District", ylabel="")
    # ax.ticklabel_format(style='plain', axis='x')
    # for i, count in enumerate(tax_district):
    #     ax.annotate(str(count), xy=(count, i), va='center')

    # # Create Bar Chart for properties types per district
    # tax_district= t_housing.groupby(["LandUse",'TaxDistrict']).size().sort_values(ascending=False).head(10)
    # tax_district= tax_district.sort_values(ascending=True)
    # ax = tax_district.plot.barh(title="Total Properties Per Tax District", ylabel="",figsize=(16,16))
    # ax.ticklabel_format(style='plain', axis='x')
    # for i, count in enumerate(tax_district):
    #     ax.annotate(str(count), xy=(count, i), va='center')

    # #Create Histogram of Total Cost
    # tax_district= t_housing.groupby(["TaxDistrict","LandUse"])["SalePrice"].mean().to_frame()
    # tax_district=tax_district.sort_values(by=["TaxDistrict","SalePrice"], ascending=False)
    # fig, ax = plt.subplots(figsize=(15,15))
    # tax_district= filter_outliers(tax_district, 'SalePrice')
    # tax_district.plot(kind='barh', stacked=True,ax=ax)
    # ax.ticklabel_format(style='plain', axis='x')
    # ax.set_title("Price Averages Per Land Use Category")
    # ax.set_ylabel("Land Use Groups")
    # ax.set_xlabel("Sale Price")

    # # Create Boxchart
    # average_cost = t_housing.groupby(["LandUse"]).mean().sort_values(by = "SaleDifference", ascending=False)
    # average_cost = average_cost.fillna(0)
    # average_cost_chart = average_cost[["LandValue","SalePrice","BuildingValue","TotalValue","SaleDifference"]].sort_values(by='TotalValue',ascending=True)
    # average_cost_chart = average_cost_chart[average_cost_chart['TotalValue'] != 0]
    # fig, ax = plt.subplots(figsize=(15,15))

    # # Plotting the Total Value as a line plot
    # average_cost_chart['TotalValue'].plot(kind="barh",ax=ax, color='red')
    # # Plotting the mean values of Land Value and Building Value as horizontal stacked bars
    # average_cost_chart[['LandValue', 'BuildingValue']].plot(kind='barh', stacked=True, ax=ax, color=['green','blue'])
    # ax.legend(["Total Value","Land Value", "Building Value"])
    # plt.xlabel('Mean Cost')
    # plt.ylabel('Land Use')
    # plt.title('Mean Cost Breakdown of Land by Categories')

    # th_filtered = filter_outliers(t_housing, 'SaleDifference')
    # ax = create_boxplot(th_filtered,"SaleDifference","LandUse","Price Difference Ranges Per Land Use Category","Land Use Groups","Sale Price")
    # plt.show()
