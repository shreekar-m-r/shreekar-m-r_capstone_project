# Data analysis and visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data preprocessing libraries
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Data splitting library
from sklearn.model_selection import train_test_split

# Model building libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
import xgboost as xgb

# Hyperparameter tuning libraries
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# Evaluation metric libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score

# Jupyter Notebook magic command to display plots inline
%matplotlib inline

# Creating an array of color codes
colors = ["#4178FB", "#4DE0FA", "#7DFFC6"]

# Seaborn configuration to add background to the graphs & Setting custom color palette
sns.set(color_codes=True)
sns.set_palette(sns.color_palette(colors))

# Pandas settings to control the display of rows and columns
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", None)

# Pandas setting to suppress scientific notation for floating-point numbers
pd.set_option("display.float_format", lambda x: "%.2f" % x)


# Load the dataset
file_path = 'D:/Capstone/Dataset/Customer+Churn+Data.xlsx'
data = pd.read_excel(file_path, sheet_name='Data for DSBA')
meta_data = pd.read_excel(file_path, sheet_name='Meta Data')
data_copy = data.copy()  # creating a copy of the dataset

# Inspect the shape of the data
print(data.shape)

# Display a random sample of 10 rows from the data
print(data.sample(10))

# Get a concise summary of the DataFrame including the number of non-null entries and data types
print(data.info())

# Count the number of duplicate rows in the data
print(data.duplicated().sum())

# Count the number of missing values in each column
print(data.isnull().sum())

# Calculate the percentage of missing values in each column
print(data.isnull().sum() / data.isnull().count() * 100)

# Get the number of unique values in each column
print(data.nunique())


# Handling Categorical Variables
# List of all categorical variables
cat_cols = ["Payment", "Gender", "account_segment", "Marital_Status", "Login_device"]

# Printing the number of unique values in each categorical variable
for col in cat_cols:
    print(f"\nUnique values in {col} are:")
    print(data[col].value_counts())
    print("\n" + "-" * 40)

# Fixing data entry errors in categorical variables
data["Gender"] = data["Gender"].replace({"F": "Female", "M": "Male"})
data["account_segment"] = data["account_segment"].replace({"Regular +": "Regular Plus", "Super +": "Super Plus"})
data["Login_device"] = data["Login_device"].replace("&&&&", "Other device")  # Consider whether to treat as 'Unknown' or missing value

# Print unique values after fixing
print("Unique values in Gender are:")
print(data["Gender"].value_counts())
print("\n" + "-" * 40)
print("Unique values in account_segment are:")
print(data["account_segment"].value_counts())
print("\nUnique values in Login_device are:")
print(data["Login_device"].value_counts())

# Handling Numerical Variables
# List of all numerical variables
numerical_columns = [
    "Tenure",
    "City_Tier",
    "CC_Contacted_LY",
    "Service_Score",
    "Account_user_count",
    "CC_Agent_Score",
    "rev_per_month",
    "Complain_ly",
    "rev_growth_yoy",
    "coupon_used_for_payment",
    "Day_Since_CC_connect",
    "cashback"
]


# Printing the number of unique values in each numerical variable
num_cols = data.select_dtypes(include=['number']).columns
for i in num_cols:
    print("\nUnique values in", i, "are :")
    print(data[i].value_counts())
    print("\n")
    print("-" * 40)


# Convert non-numeric values to NaN and handle missing values for numerical columns
for column in numerical_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')
    data[column] = data[column].fillna(data[column].median())  # Remove inplace=True

# Check the cleaned numerical columns
print("\nCleaned numerical columns:")
print(data[numerical_columns].head())

# Handling missing values in categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])  # Remove inplace=True

# Check the cleaned categorical columns
print("\nCleaned categorical columns:")
print(data[categorical_columns].head())


# Replacing invalid data with np.nan
data["Tenure"] = data["Tenure"].replace("#", np.nan)
data["Account_user_count"] = data["Account_user_count"].replace("@", np.nan)
data["rev_per_month"] = data["rev_per_month"].replace("+", np.nan)
data["rev_growth_yoy"] = data["rev_growth_yoy"].replace("$", np.nan)
data["coupon_used_for_payment"] = data["coupon_used_for_payment"].replace(["#", "$", "*"], np.nan)
data["Day_Since_CC_connect"] = data["Day_Since_CC_connect"].replace("$", np.nan)
data["cashback"] = data["cashback"].replace("$", np.nan)

# Checking for the percentage of missing values
missing_percentages = data.isnull().sum() / data.isnull().count() * 100
print("Missing Values Percentage:\n", missing_percentages)

# Checking the statistical summary of the dataset
print("Statistical Summary:\n", data.describe().T)

# Summary of the categorical variables
print("Categorical Variables Summary:\n", data.describe(include="object").T)

# Printing the number of unique values in each numerical variable
for col in num_cols:
    unique_count = data[col].nunique()
    print(f"Column '{col}' has {unique_count} unique values.")


# Function to plot histogram and boxplot combined
def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined.
    feature : dataframe column
    figsize : size of figure (default (12,7))
    kde : whether to show the density curve (default False)
    bins : number of bins (default None / auto)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid = 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots

    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column

    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram

    ax_hist2.axvline(
        data[feature].mean(), color="purple", linestyle="--"
    )  # Add mean to the histogram

    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

# Function to print the percentage of data points on a bar plot
def perc_on_bar(ax, feature):
    """
    Annotate barplot with percentage labels.
    ax : Matplotlib axis object
    feature : Series or list-like, categorical data
    """
    total = len(feature)  # length of the column
    for p in ax.patches:
        count = p.get_height()
        if count == 0:
            continue  
        percentage = "{:.1f}%".format(100 * count / total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(
            percentage,
            (x, y),
            size=12,
            ha='center',  # horizontal alignment
            va='bottom',  # vertical alignment
        )
    plt.show()  # show the plot

# Univariate Analysis

# 1. Churn
plt.figure(figsize=(5, 5))
ax = sns.countplot(data=data, x="Churn")
perc_on_bar(ax, data["Churn"])

# 2. Tenure
histogram_boxplot(data, "Tenure")

# 3. City Tier
histogram_boxplot(data, "City_Tier")

# 4. Customer Care contact in the past year
histogram_boxplot(data, "CC_Contacted_LY")

# 5. Payment
plt.figure(figsize=(9, 5))
ax = sns.countplot(data=data, x="Payment")
perc_on_bar(ax, data["Payment"])

# 6. Gender
plt.figure(figsize=(5, 5))
ax = sns.countplot(data=data, x="Gender")
perc_on_bar(ax, data["Gender"])

# 7. Service Score
histogram_boxplot(data, "Service_Score")

# 8. Account User Count
histogram_boxplot(data, "Account_user_count")

# 9. Account Segment
plt.figure(figsize=(9, 5))
ax = sns.countplot(data=data, x="account_segment")
perc_on_bar(ax, data["account_segment"])

# 10. Customer Care Agent Score
histogram_boxplot(data, "CC_Agent_Score")

# 11. Marital Status
plt.figure(figsize=(6, 5))
ax = sns.countplot(data=data, x="Marital_Status")
perc_on_bar(ax, data["Marital_Status"])

# 12. Average Revenue per Month
histogram_boxplot(data, "rev_per_month")

# 13. Complaints raised in the past year
plt.figure(figsize=(5, 5))
ax = sns.countplot(data=data, x="Complain_ly")
perc_on_bar(ax, data["Complain_ly"])

# 14. Revenue Growth Percentage
histogram_boxplot(data, "rev_growth_yoy")

# 15. Coupon Used for Payment
histogram_boxplot(data, "coupon_used_for_payment")

# 16. Days since Customer Care was contacted
histogram_boxplot(data, "Day_Since_CC_connect")

# 17. Cashback
histogram_boxplot(data, "cashback")

# 18. Login Device
plt.figure(figsize=(6, 5))
ax = sns.countplot(data=data, x="Login_device")
perc_on_bar(ax, data["Login_device"])

# Function to plot stacked bar plot
def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart with percentage labels.
    data : dataframe
    predictor : independent variable
    target : target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    ax = tab.plot(kind="bar", stacked=True, figsize=(count + 3, 5))
    
    # Adding percentage labels to the stacked bar plot
    total = len(data)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        percentage = f'{height * 100:.1f}%'
        ax.text(x + width/2, y + height/2, percentage, ha='center', va='center')

    plt.legend(loc="lower left", frameon=False)
    plt.legend(loc="lower left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=0)
    plt.show()

# Bivariate Analysis

# 1. Target vs Tenure
stacked_barplot(data, "Tenure", "Churn")

# 2. Target vs City_Tier
stacked_barplot(data, "City_Tier", "Churn")

# 3. Target vs Customer Care Contacted (Past year)
stacked_barplot(data, "CC_Contacted_LY", "Churn")

# 4. Target vs Payment
stacked_barplot(data, "Payment", "Churn")

# 5. Target vs Gender
stacked_barplot(data, "Gender", "Churn")

# 6. Target vs Service Score
stacked_barplot(data, "Service_Score", "Churn")

# 7. Target vs Account user count
stacked_barplot(data, "Account_user_count", "Churn")

# 8. Target vs Account segment
stacked_barplot(data, "account_segment", "Churn")

# 9. Target vs Customer Care Agent Score
stacked_barplot(data, "CC_Agent_Score", "Churn")

# 10. Target vs Marital Status
stacked_barplot(data, "Marital_Status", "Churn")

# 11. Target vs Average Revenue per Month
stacked_barplot(data, "rev_per_month", "Churn")

# 12. Target vs Complain_ly
stacked_barplot(data, "Complain_ly", "Churn")

# 13. Target vs rev_growth_yoy
stacked_barplot(data, "rev_growth_yoy", "Churn")

# 14. Target vs coupon_used_for_payment
stacked_barplot(data, "coupon_used_for_payment", "Churn")

# 15. Target vs Day_Since_CC_connect
stacked_barplot(data, "Day_Since_CC_connect", "Churn")

# 16. Target vs cashback
plt.figure(figsize=(7, 7))
sns.barplot(y="cashback", x="Churn", data=data)
plt.show()

# 17. Target vs Login Device
stacked_barplot(data, "Login_device", "Churn")


# Statistical Summary of Customers Who Have Churned
churned_summary = data[data["Churn"] == 1].describe(include="all").T
print(churned_summary)

# Continuous Variables
cont_col = data[
    [
        "Tenure",
        "CC_Contacted_LY",
        "Service_Score",
        "CC_Agent_Score",
        "rev_per_month",
        "Complain_ly",
        "rev_growth_yoy",
        "coupon_used_for_payment",
        "Day_Since_CC_connect",
        "cashback",
    ]
]

plt.figure(figsize=(15, 15))
for i, variable in enumerate(cont_col):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(x=data["Churn"], y=data[variable])  
    plt.tight_layout()
    plt.title(variable)
    plt.ylabel(variable) 
plt.show()


# Removal of Unwanted Variables
data1 = data.copy()  # Making a copy before removing variables
data.drop("AccountID", axis=1, inplace=True)
print("Number of columns after dropping AccountID:", data.shape[1])


# Defining a method to plot distributions with respect to the target variable
def distribution_plot_wrt_target(data, predictor, target):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    target_uniq = data[target].unique()

    if len(target_uniq) != 2:
        raise ValueError(f"Target variable {target} must have exactly 2 unique values.")

    axs[0, 0].set_title(f"Distribution of {predictor} for target = {target_uniq[1]}")
    sns.histplot(
        data=data[data[target] == target_uniq[1]], x=predictor, kde=True, ax=axs[0, 0]
    )

    axs[0, 1].set_title(f"Distribution of {predictor} for target = {target_uniq[0]}")
    sns.histplot(
        data=data[data[target] == target_uniq[0]], x=predictor, kde=True, ax=axs[0, 1]
    )

    axs[1, 0].set_title(f"Boxplot (with outliers) of {predictor} w.r.t {target}")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0])

    axs[1, 1].set_title(f"Boxplot (without outliers) of {predictor} w.r.t {target}")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 1], showfliers=False)

    plt.tight_layout()
    plt.show()

# various predictors and the 'Churn' target variable
distribution_plot_wrt_target(data, "Tenure", "Churn")
distribution_plot_wrt_target(data, "City_Tier", "Churn")
distribution_plot_wrt_target(data, "CC_Contacted_LY", "Churn")
distribution_plot_wrt_target(data, "Service_Score", "Churn")
distribution_plot_wrt_target(data, "Account_user_count", "Churn")
distribution_plot_wrt_target(data, "CC_Agent_Score", "Churn")
distribution_plot_wrt_target(data, "rev_per_month", "Churn")
distribution_plot_wrt_target(data, "Complain_ly", "Churn")
distribution_plot_wrt_target(data, "rev_growth_yoy", "Churn")
distribution_plot_wrt_target(data, "coupon_used_for_payment", "Churn")
distribution_plot_wrt_target(data, "Day_Since_CC_connect", "Churn")
distribution_plot_wrt_target(data, "cashback", "Churn")

# Churn vs Tenure vs Gender
sns.catplot(x="Gender", y="Tenure", col="Churn", data=data, kind="box", height=5, aspect=1.5)
plt.show()

# Churn vs Day_Since_CC_connect vs Gender
sns.catplot(
    x="Gender",
    y="Day_Since_CC_connect",
    col="Churn",
    data=data,
    kind="box",
    showmeans=True,
    height=5,
    aspect=1.5
)
plt.show()

# Churn vs Marital_Status vs Gender
sns.catplot(data=data, x="Marital_Status", hue="Gender", kind="count", col="Churn", height=5, aspect=1.5)
plt.show()

# Churn vs Payment vs Gender
sns.catplot(data=data, x="Payment", hue="Gender", kind="count", col="Churn", height=5, aspect=1.5)
plt.show()

# Churn vs account_segment vs Gender
sns.catplot(data=data, x="account_segment", hue="Gender", kind="count", col="Churn", height=5, aspect=1.5)
plt.show()

# Churn vs Login_device vs Gender
sns.catplot(data=data, x="Login_device", hue="Gender", kind="count", col="Churn", height=5, aspect=1.5)
plt.show()

# Churn vs Account_user_count vs Service_Score
plt.figure(figsize=(10, 5))
sns.boxplot(data=data, x="Account_user_count", y="Service_Score", hue="Churn")
plt.show()            


# Define the account segments you are interested in
segments = ["Regular", "Regular Plus", "Super", "Super Plus", "HNI"]

# Function to calculate and display descriptive statistics for churned customers
def describe_churned_customers(data, segments):
    """
    This function prints the descriptive statistics for churned customers
    in each specified account segment.
    """
    for segment in segments:
        print(f"Descriptive statistics for '{segment}' segment with churn:")
        display(
            data[(data["account_segment"] == segment) & (data["Churn"] == 1)]
            .describe(include="all")
            .T
        )
        print("\n")  # Add a newline for better readability between outputs

# Function to calculate outlier percentage using IQR
def calculate_outlier_percentage(data):
    """
    This function calculates the percentage of outliers in numerical columns using the IQR method.
    """
    # Select only numerical columns
    numerical_data = data.select_dtypes(include=["number"])
    
    # Calculate Q1, Q3, and IQR for numerical data
    Q1 = numerical_data.quantile(0.25)
    Q3 = numerical_data.quantile(0.75)
    IQR = Q3 - Q1

    # Determine the lower and upper bounds for outliers
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Calculate the percentage of outliers in each numerical column
    outlier_percentage = (
        (numerical_data < lower) | (numerical_data > upper)
    ).sum() / len(numerical_data) * 100
    return outlier_percentage

# Function to handle missing values without SimpleImputer
def impute_missing_values(data, numerical_cols, categorical_cols):
    """
    This function imputes missing values in specified numerical and categorical columns.
    """
    # Create a copy of the data to avoid chained assignment issues
    data_copy = data.copy()
    
    # Impute numerical columns using the median
    for col in numerical_cols:
        median_value = data_copy[col].median()
        data_copy[col] = data_copy[col].fillna(median_value)

    # Impute categorical columns using the most frequent value (mode)
    for col in categorical_cols:
        mode_value = data_copy[col].mode()[0]
        data_copy[col] = data_copy[col].fillna(mode_value)

    return data_copy

# Main script
if __name__ == "__main__":
    # Display statistics for each segment
    describe_churned_customers(data, segments)

    # Calculate and print outlier percentages
    outlier_percentage = calculate_outlier_percentage(data)
    print("Percentage of outliers in each numerical column:")
    print(outlier_percentage)

    # Calculate and print missing value percentages
    missing_values_percentage = data.isnull().sum() / len(data) * 100
    print("Percentage of missing values in each column:")
    print(missing_values_percentage)

    # Columns to impute
    numerical_cols = [
        "Tenure", "Service_Score", "rev_per_month", "rev_growth_yoy",
        "Day_Since_CC_connect", "cashback"
    ]
    categorical_cols = [
        "City_Tier", "Gender", "Marital_Status", "account_segment", "Login_device"
    ]

    # Impute missing values
    data = impute_missing_values(data, numerical_cols, categorical_cols)

    # Print data after imputation to verify
    print("Data after imputation:")
    print(data.head())


# Handling Missing Categories and Encoding categorical variables into numerical values to perform KNN imputation

# Define the mappings
payment = {
    "Debit Card": 0,
    "UPI": 1,
    "Credit Card": 2,
    "Cash on Delivery": 3,
    "E wallet": 4,
}
gender = {"Female": 0, "Male": 1}
account_segment = {
    "Super": 0,
    "Regular Plus": 1,
    "Regular": 2,
    "HNI": 3,
    "Super Plus": 4,
}
marital_status = {"Single": 0, "Divorced": 1, "Married": 2}
login_device = {"Mobile": 0, "Computer": 1, "Other device": 2}

# Define a function for mapping and handling unexpected categories
def map_and_handle_unexpected(data, column, mapping, default_value=-1):
    unexpected_categories = data[~data[column].isin(mapping.keys())][column].unique()
    if len(unexpected_categories) > 0:
        print(f"Warning: Some {column} categories are not in the mapping dictionary: {unexpected_categories}")
    data[column] = data[column].map(mapping).fillna(default_value)

# Apply mappings
map_and_handle_unexpected(data, "Payment", payment)
map_and_handle_unexpected(data, "Gender", gender)
map_and_handle_unexpected(data, "account_segment", account_segment)
map_and_handle_unexpected(data, "Marital_Status", marital_status)

# Check the unique values in 'Login_device' column before mapping
print("Unique values in Login_device before mapping:", data["Login_device"].unique())

# Apply mapping for 'Login_device'
map_and_handle_unexpected(data, "Login_device", login_device)

# Check the unique values in 'Login_device' column after mapping
print("Unique values in Login_device after mapping:", data["Login_device"].unique())

# KNN Imputation
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data)
data = pd.DataFrame(data_imputed, columns=data.columns)

# Checking if any null values remain
if data.isnull().sum().any():
    print("Warning: There are still missing values after KNN imputation.")

# Separating the dependent and independent variables
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Splitting data into 2 parts: temporary and test sets
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Splitting the temporary set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=1, stratify=y_temp
)

# Output shapes and class distribution
print("Shape of Training set:", X_train.shape)
print("Shape of Validation set:", X_val.shape)
print("Shape of Testing set:", X_test.shape)
print("\nPercentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("\nPercentage of classes in validation set:")
print(y_val.value_counts(normalize=True))
print("\nPercentage of classes in test set:")
print(y_test.value_counts(normalize=True))


# Check if any missing values exist in the split datasets
print("Missing values in training set:", X_train.isna().sum().sum())
print("Missing values in validation set:", X_val.isna().sum().sum())
print("Missing values in test set:", X_test.isna().sum().sum())
print("-" * 40)

# If there are missing values, identify the columns
cols_to_impute = X_train.columns[X_train.isna().any()].tolist()

if cols_to_impute:
    # Fit and transform the train data
    X_train[cols_to_impute] = imputer.fit_transform(X_train[cols_to_impute])

    # Transform the validation and test data
    X_val[cols_to_impute] = imputer.transform(X_val[cols_to_impute])
    X_test[cols_to_impute] = imputer.transform(X_test[cols_to_impute])

    # Check again for missing values after imputation
    print("Missing values in training set after imputation:", X_train.isna().sum().sum())
    print("Missing values in validation set after imputation:", X_val.isna().sum().sum())
    print("Missing values in test set after imputation:", X_test.isna().sum().sum())
else:
    print("No missing values to impute in the training set.")



# Function to inverse the encoding of categorical variables
def inverse_mapping(x, y):
    inv_dict = {v: k for k, v in x.items()}
    X_train[y] = np.round(X_train[y]).map(inv_dict).astype("category")
    X_val[y] = np.round(X_val[y]).map(inv_dict).astype("category")
    X_test[y] = np.round(X_test[y]).map(inv_dict).astype("category")

# Inverse mapping for categorical columns
inverse_mapping(payment, "Payment")
inverse_mapping(gender, "Gender")
inverse_mapping(account_segment, "account_segment")
inverse_mapping(marital_status, "Marital_Status")
inverse_mapping(login_device, "Login_device")

# Checking inverse mapped categorical values in train data
print("Inverse Mapped Values in Train Data:")
cols = X_train.select_dtypes(include=["object", "category"])
for i in cols.columns:
    print(f"{i} value counts:\n{X_train[i].value_counts()}")
    print("-" * 40)

# Checking inverse mapped categorical values in validation data
print("Inverse Mapped Values in Validation Data:")
cols = X_val.select_dtypes(include=["object", "category"])
for i in cols.columns:
    print(f"{i} value counts:\n{X_val[i].value_counts()}")
    print("-" * 40)

# Checking inverse mapped categorical values in test data
print("Inverse Mapped Values in Test Data:")
cols = X_test.select_dtypes(include=["object", "category"])
for i in cols.columns:
    print(f"{i} value counts:\n{X_test[i].value_counts()}")
    print("-" * 40)


# Interaction Variables
# Marital_Status and Gender
X_train["Married_M"] = ((X_train["Marital_Status"] == "Married") & (X_train["Gender"] == "Male")).astype(int)
X_val["Married_M"] = ((X_val["Marital_Status"] == "Married") & (X_val["Gender"] == "Male")).astype(int)
X_test["Married_M"] = ((X_test["Marital_Status"] == "Married") & (X_test["Gender"] == "Male")).astype(int)

X_train["Single_M"] = ((X_train["Marital_Status"] == "Single") & (X_train["Gender"] == "Male")).astype(int)
X_val["Single_M"] = ((X_val["Marital_Status"] == "Single") & (X_val["Gender"] == "Male")).astype(int)
X_test["Single_M"] = ((X_test["Marital_Status"] == "Single") & (X_test["Gender"] == "Male")).astype(int)

X_train["Divorced_M"] = ((X_train["Marital_Status"] == "Divorced") & (X_train["Gender"] == "Male")).astype(int)
X_val["Divorced_M"] = ((X_val["Marital_Status"] == "Divorced") & (X_val["Gender"] == "Male")).astype(int)
X_test["Divorced_M"] = ((X_test["Marital_Status"] == "Divorced") & (X_test["Gender"] == "Male")).astype(int)

# Payment and Gender
payment_methods = ["Cash on Delivery", "Credit Card", "Debit Card", "E wallet", "UPI"]
for method in payment_methods:
    X_train[f"{method.replace(' ', '_')}_M"] = ((X_train["Payment"] == method) & (X_train["Gender"] == "Male")).astype(int)
    X_val[f"{method.replace(' ', '_')}_M"] = ((X_val["Payment"] == method) & (X_val["Gender"] == "Male")).astype(int)
    X_test[f"{method.replace(' ', '_')}_M"] = ((X_test["Payment"] == method) & (X_test["Gender"] == "Male")).astype(int)

# Login_device and Gender
devices = ["Computer", "Mobile", "Other"]
for device in devices:
    X_train[f"{device}_M"] = ((X_train["Login_device"] == device) & (X_train["Gender"] == "Male")).astype(int)
    X_val[f"{device}_M"] = ((X_val["Login_device"] == device) & (X_val["Gender"] == "Male")).astype(int)
    X_test[f"{device}_M"] = ((X_test["Login_device"] == device) & (X_test["Gender"] == "Male")).astype(int)

# Ratio Variable
X_train["user_count_ss"] = (X_train["Account_user_count"] + 1) / (X_train["Service_Score"] + 1)
X_val["user_count_ss"] = (X_val["Account_user_count"] + 1) / (X_val["Service_Score"] + 1)
X_test["user_count_ss"] = (X_test["Account_user_count"] + 1) / (X_test["Service_Score"] + 1)

print(X_train.shape, X_val.shape, X_test.shape)


# Label Encoding for ordinal features
le = LabelEncoder()

X_train["Payment"] = le.fit_transform(X_train["Payment"])
X_val["Payment"] = le.transform(X_val["Payment"])
X_test["Payment"] = le.transform(X_test["Payment"])

X_train["account_segment"] = le.fit_transform(X_train["account_segment"])
X_val["account_segment"] = le.transform(X_val["account_segment"])
X_test["account_segment"] = le.transform(X_test["account_segment"])

X_train["Login_device"] = le.fit_transform(X_train["Login_device"])
X_val["Login_device"] = le.transform(X_val["Login_device"])
X_test["Login_device"] = le.transform(X_test["Login_device"])

# Creating dummy variables for nominal features
X_train = pd.get_dummies(X_train, drop_first=True)
X_val = pd.get_dummies(X_val, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns across all datasets
# Get the union of all columns
all_columns = X_train.columns.union(X_val.columns).union(X_test.columns)

# Reindex datasets to include all columns with missing values filled as 0
X_train = X_train.reindex(columns=all_columns, fill_value=0)
X_val = X_val.reindex(columns=all_columns, fill_value=0)
X_test = X_test.reindex(columns=all_columns, fill_value=0)

# Checking the shape of data
print(X_train.shape, X_val.shape, X_test.shape)

# Checking if the columns are properly encoded
print(X_train.head())
print(X_val.head())
print(X_test.head())



# Function to compute classification model performance
def model_performance_classification(model, predictors, target):
    """
    Compute performance metrics for a classification model.
    
    Parameters:
    model : classifier object 
        The model to evaluate
    predictors : pandas DataFrame or array-like
        The independent variables
    target : array-like
        The true labels

    Returns:
    pd.DataFrame : 
        A DataFrame containing Accuracy, Recall, Precision, and F1-score.
    """
    # Predicting using the independent variables
    pred = model.predict(predictors)
    acc = accuracy_score(target, pred)  # Accuracy
    recall = recall_score(target, pred, average='binary')  # Recall
    precision = precision_score(target, pred, average='binary')  # Precision
    f1 = f1_score(target, pred, average='binary')  # F1-score

    # Creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1}, index=[0]
    )
    return df_perf

# Function to plot the confusion matrix
def confusion_matrix_sklearn(model, predictors, target):
    """
    Plot the confusion matrix with percentages.
    
    Parameters:
    model : classifier object
        The model to evaluate
    predictors : pandas DataFrame or array-like
        The independent variables
    target : array-like
        The true labels
    """
    # Predicting the target
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)

    # Formatting the labels to show both count and percentage
    labels = np.asarray(
        [
            ["{0:0.0f}\n{1:.2%}".format(item, item / cm.sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    # Plotting the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=True)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.show()

# Model building and evaluation with K-Folds cross-validation

# List to store models
models = [
    ("LR", LogisticRegression(solver="newton-cg", random_state=1)),
    ("Dtree", DecisionTreeClassifier(random_state=1)),
    ("Bagging", BaggingClassifier(random_state=1)),
    ("RandomForest", RandomForestClassifier(random_state=1)),
    ("Adaboost", AdaBoostClassifier(algorithm="SAMME", random_state=1)),
    ("GBM", GradientBoostingClassifier(random_state=1)),
    ("XGBoost", XGBClassifier(random_state=1, eval_metric="logloss")),
]

# List to store cross-validation results and model names
results = []
names = []

# Looping through the models to evaluate performance via cross-validation
print("Cross-Validation Performance:\n")
for name, model in models:
    scoring = "recall"  # Focusing on Recall
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    cv_result = cross_val_score(model, X_train, y_train, scoring=scoring, cv=kfold)
    results.append(cv_result)
    names.append(name)
    print(f"{name} : {cv_result.mean() * 100:.2f}%")

# Validation set performance
print("\nValidation Set Performance:\n")
val_scores = {}  # Store validation recall scores
for name, model in models:
    model.fit(X_train, y_train)
    val_recall = recall_score(y_val, model.predict(X_val)) * 100
    val_scores[name] = val_recall
    print(f"{name} : {val_recall:.2f}%")

#  Plotting boxplot for CV scores of models with custom colors
plt.figure(figsize=(12, 8), facecolor='black')  # Dark background for the figure
ax = plt.gca()  # Get current axes
ax.set_facecolor('black')  # Dark background for the axes

plt.title("Algorithm Comparison", fontsize=16, color='white')  # Title color
plt.xlabel('Algorithms', fontsize=14, color='white')  # X-axis label color
plt.ylabel("Cross-Validation Recall (%)", fontsize=14, color='white')  # Y-axis label color

# Set custom colors
colors = dict(boxes="teal", whiskers="orange", medians="red", caps="purple")

# Plot the boxplot with custom colors
boxplot = plt.boxplot(results, patch_artist=True)

# Customize boxplot colors
for box in boxplot['boxes']:
    box.set(facecolor='skyblue', edgecolor=colors['boxes'])  # Face color and edge color
for whisker in boxplot['whiskers']:
    whisker.set(color=colors['whiskers'], linewidth=2)
for cap in boxplot['caps']:
    cap.set(color=colors['caps'], linewidth=2)
for median in boxplot['medians']:
    median.set(color=colors['medians'], linewidth=2)

# Customizing x-axis labels and y-axis
plt.xticks(ticks=np.arange(1, len(names) + 1), labels=names, rotation=45, fontsize=12, color='white')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7, color='gray')

plt.tight_layout()  # Adjust layout to fit everything
plt.show()


# Decision Tree Classifier
# Fitting the initial Decision Tree model
dtree = DecisionTreeClassifier(random_state=1)
dtree.fit(X_train, y_train)

# Evaluating performance on training data
print("Training Performance :")
dtree_train_perf = model_performance_classification(dtree, X_train, y_train)
print(dtree_train_perf)

# Evaluating performance on validation data
print("\nValidation Performance :")
dtree_val_perf = model_performance_classification(dtree, X_val, y_val)
print(dtree_val_perf, "\n")

# Display confusion matrix for validation data
confusion_matrix_sklearn(dtree, X_val, y_val)

# Hyperparameter Tuning using GridSearchCV for Decision Tree
# Defining the classifier with class weight balancing (important for imbalanced classes)
dtree_tuned = DecisionTreeClassifier(class_weight={0: 0.20, 1: 0.80}, random_state=1)

# Defining the grid of parameters to tune
parameters = {
    "max_depth": np.arange(2, 30),
    "min_samples_leaf": [1, 2, 5, 7, 10],
    "max_leaf_nodes": [2, 3, 5, 10, 15],
}

# Running Grid Search with Recall as the scoring metric
grid_obj = GridSearchCV(dtree_tuned, parameters, scoring="recall", n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Updating the decision tree classifier with the best parameters from Grid Search
dtree_tuned = grid_obj.best_estimator_

# Fitting the best Decision Tree model to the training data
dtree_tuned.fit(X_train, y_train)

# Evaluating the tuned model's performance on training data
print("Training Performance :")
dtree_tuned_train_perf = model_performance_classification(dtree_tuned, X_train, y_train)
print(dtree_tuned_train_perf)

# Evaluating the tuned model's performance on validation data
print("\nValidation Performance :")
dtree_tuned_val_perf = model_performance_classification(dtree_tuned, X_val, y_val)
print(dtree_tuned_val_perf, "\n")

# Display confusion matrix for validation data using the tuned model
confusion_matrix_sklearn(dtree_tuned, X_val, y_val)

# Bagging Classifier
# Fitting Bagging Classifier with default Decision Tree
bagging = BaggingClassifier(random_state=1)
bagging.fit(X_train, y_train)

# Evaluating Bagging model's performance on training data
print("Training Performance :")
bagging_train_perf = model_performance_classification(bagging, X_train, y_train)
print(bagging_train_perf)

# Evaluating Bagging model's performance on validation data
print("\nValidation Performance :")
bagging_val_perf = model_performance_classification(bagging, X_val, y_val)
print(bagging_val_perf, "\n")

# Display confusion matrix for validation data
confusion_matrix_sklearn(bagging, X_val, y_val)

# Bagging Classifier with tuned Decision Tree as base estimator
bagging_tdtree = BaggingClassifier(estimator=dtree_tuned, random_state=1)
bagging_tdtree.fit(X_train, y_train)

# Evaluating performance of Bagging with tuned Decision Tree on training data
print("Training Performance :")
bagging_tdtree_train_perf = model_performance_classification(bagging_tdtree, X_train, y_train)
print(bagging_tdtree_train_perf)

# Evaluating performance of Bagging with tuned Decision Tree on validation data
print("\nValidation Performance :")
bagging_tdtree_val_perf = model_performance_classification(bagging_tdtree, X_val, y_val)
print(bagging_tdtree_val_perf, "\n")

# Display confusion matrix for validation data using Bagging with tuned Decision Tree
confusion_matrix_sklearn(bagging_tdtree, X_val, y_val)


# Bagging Classifier with tuned Decision Tree

# 1. Setup the classifier using the tuned decision tree as base estimator
bagging_tuned_tdtree = BaggingClassifier(estimator=dtree_tuned, random_state=1)

# 2. Define grid of parameters for hyperparameter tuning
parameters_tdtree = {
    "max_samples": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Use standard floats
    "n_estimators": [40, 50, 60],
    "max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Use standard floats
}

# 3. Run GridSearchCV to find the best parameters
grid_obj_tdtree = GridSearchCV(bagging_tuned_tdtree, parameters_tdtree, scoring="recall", cv=5)
grid_obj_tdtree.fit(X_train, y_train)

# 4. Update the classifier with the best found parameters
bagging_tuned_tdtree = grid_obj_tdtree.best_estimator_

# 5. Fit the best algorithm to the data
bagging_tuned_tdtree.fit(X_train, y_train)

# Print the best estimator after tuning
print(bagging_tuned_tdtree)

# 6. Evaluate performance on training and validation sets
print("Training Performance:")
bagging_tuned_tdtree_train_perf = model_performance_classification(bagging_tuned_tdtree, X_train, y_train)
print(bagging_tuned_tdtree_train_perf)

print("\nValidation Performance:")
bagging_tuned_tdtree_val_perf = model_performance_classification(bagging_tuned_tdtree, X_val, y_val)
print(bagging_tuned_tdtree_val_perf)

# 7. Confusion matrix on validation set
confusion_matrix_sklearn(bagging_tuned_tdtree, X_val, y_val)

# Bagging Classifier with untuned Decision Tree

# 1. Setup the classifier using the untuned decision tree as base estimator
bagging_tuned_dt = BaggingClassifier(estimator=dtree, random_state=1)

# 2. Define grid of parameters for hyperparameter tuning
parameters_dt = {
    "max_samples": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Use standard floats
    "n_estimators": [60, 70, 80],
    "max_features": [0.7, 0.8, 0.9, 1],
}

# 3. Run GridSearchCV to find the best parameters
grid_obj_dt = GridSearchCV(bagging_tuned_dt, parameters_dt, scoring="recall", cv=5)
grid_obj_dt.fit(X_train, y_train)

# 4. Update the classifier with the best found parameters
bagging_tuned_dt = grid_obj_dt.best_estimator_

# 5. Fit the best algorithm to the data
bagging_tuned_dt.fit(X_train, y_train)

# Print the best estimator after tuning
print(bagging_tuned_dt)

# 6. Evaluate performance on training and validation sets
print("Training Performance:")
bagging_tuned_dt_train_perf = model_performance_classification(bagging_tuned_dt, X_train, y_train)
print(bagging_tuned_dt_train_perf)

print("\nValidation Performance:")
bagging_tuned_dt_val_perf = model_performance_classification(bagging_tuned_dt, X_val, y_val)
print(bagging_tuned_dt_val_perf)

# 7. Confusion matrix on validation set
confusion_matrix_sklearn(bagging_tuned_dt, X_val, y_val)


# Random Forest Classifier

# 1. Fit the model
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, y_train)

# Print the fitted model
print(rf)

# 2. Evaluate performance on training and validation sets
print("Training Performance:")
rf_train_perf = model_performance_classification(rf, X_train, y_train)
print(rf_train_perf)

print("\nValidation Performance:")
rf_val_perf = model_performance_classification(rf, X_val, y_val)
print(rf_val_perf)

# 3. Confusion matrix on validation set
confusion_matrix_sklearn(rf, X_val, y_val)


# Feature Importance of Random Forest

# Print feature importances
feature_importances = pd.DataFrame(
    rf.feature_importances_, 
    columns=["Importance"], 
    index=X_train.columns
).sort_values(by="Importance", ascending=False)

print(feature_importances)

# Plot feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)
feature_names = list(X_train.columns)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# Random Forest Classifier with Grid Search
rf_tuned = RandomForestClassifier(class_weight={0: 17, 1: 83}, random_state=1)

# Parameters grid for GridSearchCV
parameters = {
    "n_estimators": [206],
    "min_samples_leaf": [5, 6, 7],
    "max_samples": [0.5, 0.6, 0.7],
    "max_features": np.arange(0.2, 0.7, 0.1),
}

# Perform grid search
grid_obj = GridSearchCV(rf_tuned, parameters, scoring="recall", cv=5, n_jobs=-1)
grid_obj.fit(X_train, y_train)

# Update the classifier with the best parameters
rf_tuned = grid_obj.best_estimator_

# Fit the best model to the data
rf_tuned.fit(X_train, y_train)

# Print tuned RandomForestClassifier details
print(rf_tuned)

# Model performance metrics
print("Training Performance:")
rf_tuned_train_perf = model_performance_classification(rf_tuned, X_train, y_train)
print(rf_tuned_train_perf)

print("\nValidation Performance:")
rf_tuned_val_perf = model_performance_classification(rf_tuned, X_val, y_val)
print(rf_tuned_val_perf, "\n")

confusion_matrix_sklearn(rf_tuned, X_val, y_val)

# XGBoost Classifier
xgb = XGBClassifier(random_state=1, eval_metric="logloss")
xgb.fit(X_train, y_train)

# Print XGBoostClassifier details
print(xgb)

# Model performance metrics
print("Training Performance:")
xgb_train_perf = model_performance_classification(xgb, X_train, y_train)
print(xgb_train_perf)

print("\nValidation Performance:")
xgb_val_perf = model_performance_classification(xgb, X_val, y_val)
print(xgb_val_perf, "\n")

confusion_matrix_sklearn(xgb, X_val, y_val)


# Choose the type of classifier.
xgb_tuned = XGBClassifier(random_state=1, eval_metric="logloss")

# Grid of parameters to choose from
parameters = {
    "n_estimators": [90],  
    "scale_pos_weight": [5],  
    "subsample": [1],  
    "learning_rate": [0.2],  
    "gamma": [3],  
    "colsample_bytree": [0.9],  
    "colsample_bylevel": [0.5],  
}

# Run the grid search with verbose
grid_obj = GridSearchCV(xgb_tuned, parameters, scoring="recall", cv=5, n_jobs=-1, verbose=1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
xgb_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data
xgb_tuned.fit(X_train, y_train)

# Function to calculate performance
def model_performance_classification(model, X, y):
    predictions = model.predict(X)
    recall = recall_score(y, predictions)
    precision = precision_score(y, predictions)
    f1 = f1_score(y, predictions)
    accuracy = accuracy_score(y, predictions)
    return {"Recall": recall, "Precision": precision, "F1 Score": f1, "Accuracy": accuracy}

# Training Performance
print("Training Performance :")
xgb_tuned_train_perf = model_performance_classification(xgb_tuned, X_train, y_train)
print(xgb_tuned_train_perf)

# Validation Performance
print("\nValidation Performance :")
xgb_tuned_val_perf = model_performance_classification(xgb_tuned, X_val, y_val)
print(xgb_tuned_val_perf, "\n")

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, xgb_tuned.predict(X_val))
print("Confusion Matrix:\n", conf_matrix)

# Feature Importance of XGBoost (Tuned)
print(
    pd.DataFrame(
        xgb_tuned.feature_importances_, columns=["Importance"], index=X_train.columns
    ).sort_values(by="Importance", ascending=False)
)

# Visualizing Feature Importance
importances = xgb_tuned.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in descending order
feature_names = list(X_train.columns)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances (Tuned XGBoost)")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.gca().invert_yaxis()  # Invert y-axis for descending order
plt.show()

# Cross-validation recall boxplot for the XGBoost model
cv_results = cross_val_score(xgb_tuned, X_train, y_train, cv=5, scoring="recall")

plt.figure(figsize=(8, 6))
plt.gca().set_facecolor('lightgray')  # Set background color of the plot
plt.gcf().set_facecolor('white')  # Set background color of the figure

plt.boxplot(cv_results, vert=False, patch_artist=True, 
            boxprops=dict(facecolor='violet', color='darkviolet'),
            whiskerprops=dict(color='darkviolet'),
            capprops=dict(color='darkviolet'),
            medianprops=dict(color='darkviolet'))
plt.title("Cross-Validation Recall Scores (XGBoost)")
plt.xlabel("Recall")
plt.grid(True, linestyle='--', alpha=0.7)  # Add gridlines for better visibility
plt.show()

# Converting the model performance dictionaries to DataFrames
# Assuming dtree_train_perf, bagging_train_perf, rf_train_perf, and xgb_train_perf already exist

# Convert dictionaries to DataFrames before concatenating them
dtree_train_perf_df = pd.DataFrame(dtree_train_perf, index=[0]).T
dtree_tuned_train_perf_df = pd.DataFrame(dtree_tuned_train_perf, index=[0]).T
bagging_train_perf_df = pd.DataFrame(bagging_train_perf, index=[0]).T
bagging_tdtree_train_perf_df = pd.DataFrame(bagging_tdtree_train_perf, index=[0]).T
bagging_tuned_tdtree_train_perf_df = pd.DataFrame(bagging_tuned_tdtree_train_perf, index=[0]).T
bagging_tuned_dt_train_perf_df = pd.DataFrame(bagging_tuned_dt_train_perf, index=[0]).T
rf_train_perf_df = pd.DataFrame(rf_train_perf, index=[0]).T
rf_tuned_train_perf_df = pd.DataFrame(rf_tuned_train_perf, index=[0]).T
xgb_train_perf_df = pd.DataFrame(xgb_train_perf, index=[0]).T
xgb_tuned_train_perf_df = pd.DataFrame(xgb_tuned_train_perf, index=[0]).T

# Concatenating the DataFrames for training performance
models_train_comp_df = pd.concat(
    [
        dtree_train_perf_df,
        dtree_tuned_train_perf_df,
        bagging_train_perf_df,
        bagging_tdtree_train_perf_df,
        bagging_tuned_tdtree_train_perf_df,
        bagging_tuned_dt_train_perf_df,
        rf_train_perf_df,
        rf_tuned_train_perf_df,
        xgb_train_perf_df,
        xgb_tuned_train_perf_df,
    ],
    axis=1,
)

models_train_comp_df.columns = [
    "Decision Tree",
    "Decision Tree Tuned",
    "Bagging (Default Params)",
    "Bagging (Base DTree Tuned)",
    "Bagging Tuned (Base DTree Tuned)",
    "Bagging Tuned (Base DTree)",
    "Random Forest",
    "Random Forest Tuned",
    "XGBoost",
    "XGBoost Tuned",
]

print("Training performance comparison:")
print(models_train_comp_df)

# Validation performance comparison
# Convert validation performance dictionaries to DataFrames before concatenating
dtree_val_perf_df = pd.DataFrame(dtree_val_perf, index=[0]).T
dtree_tuned_val_perf_df = pd.DataFrame(dtree_tuned_val_perf, index=[0]).T
bagging_val_perf_df = pd.DataFrame(bagging_val_perf, index=[0]).T
bagging_tdtree_val_perf_df = pd.DataFrame(bagging_tdtree_val_perf, index=[0]).T
bagging_tuned_tdtree_val_perf_df = pd.DataFrame(bagging_tuned_tdtree_val_perf, index=[0]).T
bagging_tuned_dt_val_perf_df = pd.DataFrame(bagging_tuned_dt_val_perf, index=[0]).T
rf_val_perf_df = pd.DataFrame(rf_val_perf, index=[0]).T
rf_tuned_val_perf_df = pd.DataFrame(rf_tuned_val_perf, index=[0]).T
xgb_val_perf_df = pd.DataFrame(xgb_val_perf, index=[0]).T
xgb_tuned_val_perf_df = pd.DataFrame(xgb_tuned_val_perf, index=[0]).T

# Concatenating the DataFrames for validation performance
models_val_comp_df = pd.concat(
    [
        dtree_val_perf_df,
        dtree_tuned_val_perf_df,
        bagging_val_perf_df,
        bagging_tdtree_val_perf_df,
        bagging_tuned_tdtree_val_perf_df,
        bagging_tuned_dt_val_perf_df,
        rf_val_perf_df,
        rf_tuned_val_perf_df,
        xgb_val_perf_df,
        xgb_tuned_val_perf_df,
    ],
    axis=1,
)

models_val_comp_df.columns = [
    "Decision Tree",
    "Decision Tree Tuned",
    "Bagging (Default Params)",
    "Bagging (Base DTree Tuned)",
    "Bagging Tuned (Base DTree Tuned)",
    "Bagging Tuned (Base DTree)",
    "Random Forest",
    "Random Forest Tuned",
    "XGBoost",
    "XGBoost Tuned",
]

print("Validation performance comparison:")
print(models_val_comp_df)


# Training performance comparison
# Creating a dataframe of training scores
best_models_train_comp_df = pd.concat(
    [bagging_tdtree_train_perf.T, bagging_tuned_tdtree_train_perf.T], axis=1
)

best_models_train_comp_df.columns = [
    "Bagging with base dtree_tuned",
    "Bagging Tuned with base dtree_tuned",
]

# Validation performance comparison
# Creating a dataframe of validation scores
best_models_val_comp_df = pd.concat(
    [bagging_tdtree_val_perf.T, bagging_tuned_tdtree_val_perf.T], axis=1
)

best_models_val_comp_df.columns = [
    "Bagging with base dtree_tuned",
    "Bagging Tuned with base dtree_tuned",
]

# Display performance comparisons
print("Training performance comparison:")
print(best_models_train_comp_df)

print("\nValidation performance comparison:")
print(best_models_val_comp_df)

# Stacking 1: Building a stacking classifier with the best individual models
estimators1 = [
    ("Bagging with base dtree_tuned", bagging_tdtree),
    ("Bagging Tuned with base dtree_tuned", bagging_tuned_tdtree),
]
final_estimator = xgb_tuned

# Initialize Stacking Classifier
stacking_classifier1 = StackingClassifier(
    estimators=estimators1, final_estimator=final_estimator
)

# Fitting the Stacking Classifier on training data
stacking_classifier1.fit(X_train, y_train)

# Calculate different performance metrics
print("\nTraining Performance:")
stacking1_train_perf = model_performance_classification(stacking_classifier1, X_train, y_train)
print(stacking1_train_perf)

print("\nValidation Performance:")
stacking1_val_perf = model_performance_classification(stacking_classifier1, X_val, y_val)
print(stacking1_val_perf)

# Confusion matrix for the validation set
print("\nConfusion Matrix (Validation):")
confusion_matrix_sklearn(stacking_classifier1, X_val, y_val)




# Define estimators
estimators2 = [
    ("Bagging with base dtree_tuned", BaggingClassifier(
        estimator=DecisionTreeClassifier(
            class_weight={0: 0.2, 1: 0.8},
            max_depth=7,
            max_leaf_nodes=15,
            random_state=1),
        random_state=1)),
    
    ("Bagging with default parameters", BaggingClassifier(random_state=1)),
    
    ("Bagging Tuned with base dtree_tuned", BaggingClassifier(
        estimator=DecisionTreeClassifier(
            class_weight={0: 0.2, 1: 0.8},
            max_depth=7,
            max_leaf_nodes=15,
            random_state=1),
        random_state=1)),
]

# Final estimator with updated parameters
final_estimator = XGBClassifier(
    eval_metric='logloss',
    gamma=3,
    learning_rate=0.2,
    max_depth=6,
    n_estimators=90,
    scale_pos_weight=5,
    subsample=1,
    random_state=1,
    verbosity=None,
    tree_method='hist',  # Use 'hist' for GPU training or other methods if needed
    device='cuda'        # Set device to 'cuda' for GPU usage
)

# Stacking Classifier 2
stacking_classifier2 = StackingClassifier(
    estimators=estimators2, 
    final_estimator=final_estimator
)

# Fit the model
stacking_classifier2.fit(X_train, y_train)

# Calculate Performance Metrics
print("Training Performance :")
stacking2_train_perf = model_performance_classification(
    stacking_classifier2, X_train, y_train
)
print(stacking2_train_perf)

print("\nValidation Performance :")
stacking2_val_perf = model_performance_classification(
    stacking_classifier2, X_val, y_val
)
print(stacking2_val_perf)

# Confusion Matrix
confusion_matrix_sklearn(stacking_classifier2, X_val, y_val)



# Define your base estimators
estimators3 = [
    ("Bagging with base dtree_tuned", bagging_tdtree),
    ("Bagging with default parameters", bagging),
    ("Bagging Tuned with base dtree", bagging_tuned_dt),
]

# Define the final estimator
final_estimator = xgb_tuned

# Create and fit the stacking classifier
stacking_classifier3 = StackingClassifier(
    estimators=estimators3, 
    final_estimator=final_estimator
)
stacking_classifier3.fit(X_train, y_train)

# Define a function to compute model performance metrics
def model_performance_classification(model, X, y):
    from sklearn.metrics import classification_report
    y_pred = model.predict(X)
    return classification_report(y, y_pred, output_dict=True)

# Print training performance
print("Training Performance :")
stacking3_train_perf = model_performance_classification(
    stacking_classifier3, X_train, y_train
)
print(stacking3_train_perf)

# Print validation performance
print("\nValidation Performance :")
stacking3_val_perf = model_performance_classification(
    stacking_classifier3, X_val, y_val
)
print(stacking3_val_perf)

# Print confusion matrix for validation set
print("\nConfusion Matrix :")
conf_matrix = confusion_matrix(y_val, stacking_classifier3.predict(X_val))
print(conf_matrix)



# Define the base estimators and final estimator
estimators4 = [
    ("Bagging with base dtree_tuned", bagging_tdtree),
    ("Bagging with default parameters", bagging),
    ("Bagging Tuned with base dtree", bagging_tuned_dt),
    ("Random Forest", rf),
]

final_estimator = xgb_tuned 

# Initialize the StackingClassifier
stacking_classifier4 = StackingClassifier(
    estimators=estimators4,
    final_estimator=final_estimator
)

# Train the stacking classifier
stacking_classifier4.fit(X_train, y_train)

# Function to calculate performance metrics (ensure it's correctly defined)
print("Training Performance:")
stacking4_train_perf = model_performance_classification(
    stacking_classifier4, X_train, y_train
)
print(stacking4_train_perf)

print("\nValidation Performance:")
stacking4_val_perf = model_performance_classification(
    stacking_classifier4, X_val, y_val
)
print(stacking4_val_perf)

# Print confusion matrix for validation set (ensure this function is defined)
conf_matrix = confusion_matrix(y_val, stacking_classifier4.predict(X_val))
print("Confusion Matrix:")
print(conf_matrix)


# Training performance comparison for all stacking classifiers
stack_models_train_comp_df = pd.DataFrame({
    "Stacking_Classifier1": stacking1_train_perf,
    "Stacking_Classifier2": stacking2_train_perf,
    "Stacking_Classifier3": stacking3_train_perf,
    "Stacking_Classifier4": stacking4_train_perf,
})

print("Stacking Models Training Performance Comparison:")
print(stack_models_train_comp_df)

# Validation performance comparison for all stacking classifiers
stack_models_val_comp_df = pd.DataFrame({
    "Stacking_Classifier1": stacking1_val_perf,
    "Stacking_Classifier2": stacking2_val_perf,
    "Stacking_Classifier3": stacking3_val_perf,
    "Stacking_Classifier4": stacking4_val_perf,
})

print("Stacking Models Validation Performance Comparison:")
print(stack_models_val_comp_df)


# Evaluate Stacking Classifier 4 on Train, Validation, and Test sets
print("Training Performance:")
stacking4_train_perf = model_performance_classification(stacking_classifier4, X_train, y_train)
print(stacking4_train_perf)

print("\nValidation Performance:")
stacking4_val_perf = model_performance_classification(stacking_classifier4, X_val, y_val)
print(stacking4_val_perf)

print("\nTest Performance:")
stacking4_test_perf = model_performance_classification(stacking_classifier4, X_test, y_test)
print(stacking4_test_perf)

# Confusion matrix for the test set
confusion_matrix_sklearn(stacking_classifier4, X_test, y_test)


# Check if the model supports predict_proba
if hasattr(stacking_classifier4, "predict_proba"):
    churn_probs = stacking_classifier4.predict_proba(X_test)[:, 1]
    print("Churn probabilities calculated successfully.")
else:
    print("The model does not support predict_proba.")

# Predict churn probabilities for the test set
churn_probs = stacking_classifier4.predict_proba(X_test)[:, 1]  # Churn class probability

# Segment customers based on churn probability thresholds
high_risk = X_test[churn_probs >= 0.70]  # High-risk customers (70% or higher churn probability)
medium_risk = X_test[(churn_probs >= 0.40) & (churn_probs < 0.70)]  # Medium-risk customers (40% to 69%)
low_risk = X_test[churn_probs < 0.40]  # Low-risk customers (below 40%)

# Get the count of customers in each segment
high_risk_count = len(high_risk)
medium_risk_count = len(medium_risk)
low_risk_count = len(low_risk)

# Print the number of customers in each risk category
print(f"High-risk customers: {high_risk_count}")
print(f"Medium-risk customers: {medium_risk_count}")
print(f"Low-risk customers: {low_risk_count}")

# Define retention offer cost and average revenue per account
retention_offer_cost = 50  # Adjust as needed
average_revenue_per_account = 200  # Adjust as needed

# Use actual counts for high, medium, and low-risk customers
high_risk_customers = high_risk_count
medium_risk_customers = medium_risk_count
low_risk_customers = low_risk_count

# Define retention rates for each risk category
retention_rate_high = 0.70  # Adjust based on past data or expectations
retention_rate_medium = 0.50  # Adjust based on past data or expectations
retention_rate_low = 0.30  # Adjust based on past data or expectations

# Calculate total revenue from retained customers
retained_high = high_risk_customers * retention_rate_high * average_revenue_per_account
retained_medium = medium_risk_customers * retention_rate_medium * average_revenue_per_account
retained_low = low_risk_customers * retention_rate_low * average_revenue_per_account

# Calculate total cost of retention campaigns
total_retention_cost = (high_risk_customers + medium_risk_customers + low_risk_customers) * retention_offer_cost

# Calculate total revenue from retained customers
total_revenue_retained = retained_high + retained_medium + retained_low

# Calculate profit from retention campaigns
profit_from_campaigns = total_revenue_retained - total_retention_cost

# Display the cost-benefit analysis
print(f"Cost of Retention Campaigns: ₹{total_retention_cost:,.2f}")
print(f"Total Revenue from Retained Customers: ₹{total_revenue_retained:,.2f}")
print(f"Profit from Retention Campaigns: ₹{profit_from_campaigns:,.2f}")

