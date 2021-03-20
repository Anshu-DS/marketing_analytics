#!/usr/bin/env python
# coding: utf-8

# # Exploratory & Statistical Data Analysis on Marketing_analytics

# # I am just running my hands on this dataset to find out ideas about EDA on marketing domain and extracting insights and statistical analysis related to marketing topics.

# # Defining columns:

# 1. Year_Birth : Customer's birth year
# 2. Education: Customer's education level
# 3. Marital_Status: Customer's marital status
# 4. Income: Customer's yearly household income
# 5. Kidhome: Number of children in customer's household
# 6. Teenhome: Number of teenagers in customer's household
# 7. Dt_Customer: Date of customer's enrollment with the company
# 8. Recency: Number of days since customer's last purchase
# 9. MntWines: Amount spent on wine in the last 2 years
# (Mnt = amount spent on the product in the last 2 years)
# 10. NumDealsPurchases: Number of purchases made with a discount
# Each Num column represents the number of purchases.
# 11. AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# 12. AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise 
# All Accepted columns represent the accepted offer wrt to the campaign no.
# 13. Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# 14. Complain: 1 if customer complained in the last 2 years, 0 otherwise
# 15. Country: Customer's location
# 16. ID: Customer's unique identifier

# # We have a total of 28 variables. I have segregated in the above list with common names. You can give a read to the above columns.

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[5]:


#turning off warnings for final notebook
import warnings
warnings.filterwarnings('ignore')


# In[2]:


pd.set_option('display.max_columns', None)
sns.set_context('notebook')
sns.set_style('whitegrid')
sns.set_palette('Blues_r')


# In[3]:


#importing the file
df = pd.read_csv('marketing_data.csv')


# In[4]:


df.head()


# In[6]:


df.info()


# In[7]:


df.columns


# In[8]:


df.isnull().sum()


# In[30]:


#lets work with the null values of Income
#We need to plot a feature to identify the best strategy for imputation


# In[33]:


plt.figure(figsize=(8,4))
sns.distplot(df['Income'], kde=False, hist=True)
plt.title('Income distribution', size=16)
plt.ylabel('count');


# In[34]:


df['Income'].plot(kind='box', figsize=(3,4), patch_artist=True)


# In[36]:


#Finding from the above graph:
#most income is distributed between $0-$100,000 with a few outliers


# In[41]:


#adding median values to the null values to remove the outliers 
df['Income'] = df['Income'].fillna(df['Income']).median()

#checking null values incase
df['Income'].isnull().sum()                         


# In[13]:


#cleaning the column space containing whitespaces 
df.columns = df.columns.str.replace(' ', '')


# In[14]:


df.columns    
#we are good to go.


# In[15]:


df.head()


# Now, we need to change the format of the 'Income' column to Float and do a reformatting to remove '$' and ',' from the data

# In[18]:


df['Income'] = df['Income'].str.replace(' ', '')

#removing $ and , from the data and transforming into numerical column
df['Income']=df['Income'].str.replace('$','')
df['Income']=df['Income'].str.replace(',','').astype('float')


# In[19]:


#new dataset
df.head()


# In[29]:


#to check for the changes in the data type
print(df['Income'].dtypes)


# In[42]:


#Identifying features containing outliers
# select columns to plot
df_to_plot = df.drop(columns=['ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain']).select_dtypes(include=np.number)
# subplots
df_to_plot.plot(subplots=True, layout=(4,4), kind='box', figsize=(12,14), patch_artist=True)
plt.subplots_adjust(wspace=0.5);


# Multiple features contain outliers (see boxplots below), 
# but the only that likely indicate data entry errors are Year_Birth <= 1900.
# So, we have to remove them.

# In[44]:


df= df[df['Year_Birth']>1900].reset_index(drop=True)

#checking for the same:
plt.figure(figsize=(3,4))
df['Year_Birth'].plot(kind='box', patch_artist=True)


# In[45]:


df.info()


# We found that the column 'Dt_Customer' should be coverted to Datetime format

# In[47]:


df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

#to check:
print(df['Dt_Customer'].dtypes)

We have certain inputs of the feature names below:
1. The total number of dependents in the home ('Dependents') can be engineered from the sum of 'Kidhome' and 'Teenhome'

2. The year of becoming a customer ('Year_Customer') can be engineered from 'Dt_Customer'

3. The total amount spent ('TotalMnt') can be engineered from the sum of all features containing the keyword 'Mnt'

4. The total purchases ('TotalPurchases') can be engineered from the sum of all features containing the keyword 'Purchases'

5. The total number of campains accepted ('TotalCampaignsAcc') can be engineered from the sum of 
   all features containing the keywords 'Cmp' and 'Response' (the latest campaign)
# In[48]:


list(df.columns)


# In[62]:


# Dependents
df['Dependents']= df['Kidhome']+df['Teenhome']

# Year becoming a Customer
df['Year_Customer'] = pd.DatetimeIndex(df['Dt_Customer']).year

# Total Amount Spent
mnt_cols = [col for col in df.columns if 'Mnt' in col]
df['TotalMnt'] = df[mnt_cols].sum(axis=1)

# Total Purchases
purchases_cols = [col for col in df.columns if 'Purchases' in col]
df['TotalPurchases'] = df[purchases_cols].sum(axis=1)

# Total Campaigns Accepted
campaigns_cols = [col for col in df.columns if 'Cmp' in col] + ['Response'] # 'Response' is for the latest campaign
df['TotalCampaignsAcc'] = df[campaigns_cols].sum(axis=1)

# view new features, by customer ID
df[['ID', 'Dependents', 'Year_Customer', 'TotalMnt', 'TotalPurchases', 'TotalCampaignsAcc']].head()


# To identify patterns, we will first identify feature correlations.
# positive correlations between features appear red, negative correlations appear blue, 
# and no correlation appears grey in the clustered heatmap below.
# 

# In[63]:


# calculate correlation matrix
## using non-parametric test of correlation (kendall), since some features are binary
corrs = df.drop(columns='ID').select_dtypes(include=np.number).corr(method = 'kendall')

# plot clustered heatmap of correlations
sns.heatmap(df.corr(), cmap='coolwarm', center=0)


# In[66]:


#Plot illustrating negative effect of having dependents (kids & teens) on spending:

plt.figure(figsize=(4,4))
sns.boxplot(x='Dependents', y='TotalMnt', data=df);


# In[67]:


#Plot illustrating positive effect of having dependents (kids & teens) on number of deals purchased:

plt.figure(figsize=(4,4))
sns.boxplot(x='Dependents', y='NumDealsPurchases', data=df);


# In[68]:


df.head()


# In[79]:


#Using boxplot to plot correlation between TotalCampaignsAcc and Dependents

plt.figure(figsize=(5.5,4))
sns.boxplot(x='TotalCampaignsAcc', y='Dependents', data=df);


# In[ ]:


Comparing two lmplots: 
    One between Number of Web Visits per month and Number of Web Purchases
    Second between Number of Web Visits per month and Number of Discount Purchases


# In[77]:


sns.lmplot(x='NumWebVisitsMonth', y='NumWebPurchases', data=df)


# In[78]:


sns.lmplot(x='NumWebVisitsMonth', y='NumDealsPurchases', data=df)


# Number of web visits in the last month is not positively correlated with number of web purchases
# Instead, it is positively correlated with the number of deals purchased(as the line is positive), suggesting that deals are an effective way of stimulating purchases on the website

# # STATISTICAL ANALYSIS

# In[80]:


df.head()


# We need to present our statistical analysis by using Linear Regression Model on the column 'NumStorePurchases' as our target variable and then use Machine Learning techniques to get insights about which features predict the number of store purchases.

# In[83]:


#Plotting the target variable:

plt.figure(figsize=(8,3))
sns.distplot(df['NumStorePurchases'], kde=False , hist=True, bins=12)
plt.title('NumStorePurchases Distribution', size=16)
plt.ylabel('Counts');


# Dropping uninformative Features:

# ID is unique to each customer
# 

# Dt_Customer is dropped because we will use the Year_Customer engineered variable.

# We will perform one-hot encoding technique of categorical features

# In[84]:


#dropping uninformative columns
df.drop(columns=['ID', 'Dt_Customer'], inplace=True)


# In[85]:


#one-hot encoding
from sklearn.preprocessing import OneHotEncoder


# In[86]:


#get categorical features and review number of unique values
cat = df.select_dtypes(exclude = np.number)
print('Number of unique values per categorical feature: \n', cat.nunique())


# In[87]:


enc = OneHotEncoder(sparse =False).fit(cat)
cat_encoded = pd.DataFrame(enc.transform(cat))
cat_encoded.columns = enc.get_feature_names(cat.columns)


# In[88]:


#merge with numeric data
num=df.drop(columns=cat.columns)
df2 = pd.concat([cat_encoded, num], axis=1)
df2.head()


# From the above data, we can see that our One-Hot encoding technique is working fine.

# We will use Linear Regression model to our dataset. 70% of data will go into training dataset and 30% of data will go into testing data set.

# We will use RSME on our testing data

# In[90]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[91]:


#performing train_test_split and isolating X and y variables
X= df2.drop(columns='NumStorePurchases')
y= df2['NumStorePurchases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[93]:


#implementing LinearRegression model
model =LinearRegression()
model.fit(X_test, y_test)

#predictions
predictions = model.predict(X_test)


# In[98]:


#evaluate the model using RSME
print('Linear Regression using RSME:'), np.sqrt(mean_squared_error(y_test, predictions))


# In[99]:


#median value of target variable
print('The median of target variable:', y.median())


#  Here, as you can see the RSME is extremely small as compared to the median value, indicating good model predictions.

# Identifying significant features that affect the number of store purchases, using permutation importance:
#     

# In[102]:


import eli5
from eli5.sklearn import PermutationImportance


# In[103]:


perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist(), top=5)


# Significant Features: 'TotalPurchases', 'NumCatalogPurchases', 'NumWebPurchases', 'NumDealsPurchases'

# Explore the directionality of these effects, using SHAP values:

# In[104]:


import shap 

# calculate shap values 
ex = shap.Explainer(model, X_train)
shap_values = ex(X_test)

# plot
plt.title('SHAP summary for NumStorePurchases', size=16)
shap.plots.beeswarm(shap_values, max_display=5);


# From the above shap plot, we can see that:
# 1. The NumStorePurchases increases when there is an increase in TotalPurchases.
# 2. The NumStorePurchases decreases when there is an increase in NumCatalogPurchases, NumWebPurchases, NumDealsPurchases.

# In[108]:


#Plotting total number of purchases by country:

plt.figure(figsize=(5,4))
df.groupby('Country')['TotalPurchases'].sum().sort_values(ascending=False).plot(kind='bar')
plt.title('Total Number of Purchases by Country', size=16)
plt.ylabel('Number of Purchases')


# Inference from the above barchart:
# 1. Spain has the highest number of purchases.
# 2. US is second to the last, therefore US doesnot fare better in terms of total number of purchases compared with the rest of the world.

# In[111]:


#Plotting total amount spent by country:

plt.figure(figsize=(5,4))
df.groupby('Country')['TotalMnt'].sum().sort_values(ascending=False).plot(kind='bar')
plt.title('Total Amount Spent by Country', size=16)
plt.ylabel('Amount Spent')


# Inference from the above barchart:
# 1. Spain has spent the maximum amount on purchases.
# 2. US is second to the last, therefore US doesnot fare better in terms of total amount spent on purchases compared with the rest of the world.

# We can assume a case where people who spent an above average amount on gold in the last 2 years would have more in store purchases. We will check by using lmplot from seaborn of two columns MntGoldProds and NumStorePurchases.
# 

# In[116]:


sns.lmplot(x='MntGoldProds', y='NumStorePurchases', data= df)


# There is a positive relationship but we have to find out whether it is statistically significant.

# MntGoldProds contains outliers so we need to perform Kendall correlation analysis(non-parametric test)

# In[117]:


from scipy.stats import kendalltau

kendall_corr = kendalltau(x=df['MntGoldProds'], y=df['NumStorePurchases'])

# print results
print('Kendall correlation (tau): ', kendall_corr.correlation)
print('Kendall p-value: ', kendall_corr.pvalue)


# Yes, there is a significant positive corelation between MntGoldProds and NumStoreProcedures

# Fish has Omega 3 fatty acids which are good for the brain. Accordingly, do "Married PhD candidates" have a significant relation with amount spent on fish?

# We will compare 'MntFishProducts' between Married PHD candidates and all other candidates.

# In[118]:


# sum the marital status and phd dummy variables - the Married+PhD group will have value of 2
df2['Married_PhD'] = df2['Marital_Status_Married'] + df2['Education_PhD']
df2['Married_PhD'] = df2['Married_PhD'].replace({2:'Married-PhD', 1:'Other', 0:'Other'})


# In[121]:


# plot MntFishProducts between Married-PhD and others
plt.figure(figsize=(2.5,4))
sns.boxplot(x='Married_PhD', y='MntFishProducts', data=df2, palette="Set2");


# Married PhD candidates spent less amount on Fishproducts as compared to other candidates.

# In[122]:


# independent t-test p-value
from scipy.stats import ttest_ind
pval = ttest_ind(df2[df2['Married_PhD'] == 'Married-PhD']['MntFishProducts'], df2[df2['Married_PhD'] == 'Other']['MntFishProducts']).pvalue
print("t-test p-value: ", round(pval, 3))


# In[123]:


# now drop the married-phD column created above, to include only the original variables in the analysis below
df2.drop(columns='Married_PhD', inplace=True)


# Like the NumStorePurchases LinearRegression model performed above, we will create another LinearRegression model on MntFishProducts as our target variable, and then use Machine Learning Algorithms to get insights about which features predict the amount spend on fish.

# In[126]:


plt.figure(figsize=(10,6))
sns.distplot(df['MntFishProducts'], kde=False, hist=True, bins=12)
plt.title('MntFishProducts distribution', size=16)
plt.ylabel('count')


# In[134]:


#we will create the Linearregression model now.

X= df2.drop(columns='MntFishProducts')
y= df2['MntFishProducts']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)


# In[135]:


model = LinearRegression()


# In[136]:


model.fit(X_train, y_train)


# In[138]:


predict = model.predict(X_test)


# In[139]:


print('Linear Regression Model, RMSE:', np.sqrt(mean_squared_error(y_test,predict)))
print('\n')
print('The median value of target variable:', y.median())


# As it is clear that the RSME is much smaller than the target variable, so our model predictions is doing extremely well.

# Identify features that significantly affect the amount spent on fish, using permutation importance

# In[140]:


perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist(), top=7)


# Significant Features: 'TotalMnt', 'MntWines', 'MntMeatProducts', 'MntGoldProds','MntSweetProducts','MntFruits'

# In[141]:


#We will follow the same technique of Shap values as we have used above:


import shap

# calculate shap values 
ex = shap.Explainer(model, X_train)
shap_values = ex(X_test)

# plot
plt.title('SHAP summary for MntFishProducts', size=16)
shap.plots.beeswarm(shap_values, max_display=7);


# Findings:
# 1. Amount spend on fish increases with higher amount spent.
# 2. Amount spend on fish decreases with higher amount spent on Wines, Meat, Gold, Fruits, Sweet products.
# 
# So, the customer who spent more on fish are likely to spent less on other products like Wines, Meat, Gold, Fruits, Sweet products

# # Finding Significant relationship between Geographical Regional and Success of a Campaign:

# In[142]:


# convert country codes to correct nomenclature for choropleth plot
# the dataset doesn't provide information about country codes
#so I'm taking my best guess about the largest nations that make sense given the codes provided

df['Country_code'] = df['Country'].replace({'SP': 'ESP', 'CA': 'CAN', 'US': 'USA', 'SA': 'ZAF', 'ME': 'MEX'})


# In[143]:


# success of campaigns by country code
#using melt functions(melt()-Unpivot a DataFrame from wide to long format, optionally leaving identifiers set)

df_cam = df[['Country_code', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']].melt(
    id_vars='Country_code', var_name='Campaign', value_name='Accepted (%)')


# In[147]:


df_cam.head()


# In[148]:


df_cam = pd.DataFrame(df_cam.groupby(['Country_code', 'Campaign'])['Accepted (%)'].mean()*100).reset_index(drop=False)


# In[149]:


# rename the campaign variables so they're easier to interpret
df_cam['Campaign'] = df_cam['Campaign'].replace({'AcceptedCmp1': '1',
                                                'AcceptedCmp2': '2',
                                                'AcceptedCmp3': '3',
                                                'AcceptedCmp4': '4',
                                                'AcceptedCmp5': '5',
                                                 'Response': 'Most recent'
                                                })


# In[153]:


df_cam['Campaign'].head()


# In[154]:


# choropleth plot
import plotly.express as px

fig = px.choropleth(df_cam, locationmode='ISO-3', color='Accepted (%)', facet_col='Campaign', facet_col_wrap=2,
                    facet_row_spacing=0.05, facet_col_spacing=0.01, width=700,
                    locations='Country_code', projection='natural earth', title='Advertising Campaign Success Rate by Country'
                   )
fig.show()


# 1. We conclude that the Campaign acceptance rates are low overall.
# 2. The campaign with the highest overall acceptance rate is the most recent campaign (column name: Response)
# 3. The country with the highest acceptance rate in any campaign is Mexico

# We need to find if the effect on regions on campaign success is significantly successful or not.

# In[160]:


# calculate logistic regression p-values for campaign acceptance ~ country using generalized linear model
import statsmodels.formula.api as smf
import statsmodels as sm
from scipy import stats


# In[156]:


## get the data of interest for Genralised Linear Model
df_cam_wide = df[['Country', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']]


# In[157]:


df_cam_wide.head()


# In[158]:


## to store statistics results
stat_results = []


# In[163]:


## perform glm
for col in df_cam_wide.drop(columns='Country').columns:
    this_data = df_cam_wide[['Country', col]]
    
    # define formula
    formula = col+'~Country'
    
    model = smf.glm(formula =formula, data =this_data, family=sm.genmod.families.Binomial())
    result = model.fit()


# In[172]:


# get chisquare value for overall model (CampaignAccepted ~ Country) and calculate p-value

chisq = result.pearson_chi2
pval = stats.distributions.chi2.sf(chisq, 7)    #df Model =7 degrees of freedom when we run result.summary()


# In[173]:


# append to stat_results
stat_results.append(pval)


# In[174]:


# print stat summary for entire model
print(result.summary())

## check results
print("\nChisq p-values: ", stat_results)


# Findings: The regional differences in advertising campaign success are statistically significant.

# In[176]:


# plotting
## merge in the original country codes provided in the dataset
countries = df[['Country', 'Country_code']].drop_duplicates().reset_index(drop=True)
df_cam2 = df_cam.merge(countries, how='left', on='Country_code')
df_cam2.head()

## bar graphs
g = sns.FacetGrid(df_cam2, col='Campaign', col_wrap=3)
g.map(sns.barplot, 'Country', 'Accepted (%)')
for ax, pval in zip(g.axes.flat, stat_results):
    ax.text(0, 65, "Chisq p-value: "+str(pval), fontsize=9) #add text;


# # Data Visualization

# We will now plot the marketing campaign with overall acceptance rates:

# In[177]:


df.head()


# In[178]:


# calculate success rate (percent accepted)
cam_success = pd.DataFrame(df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']].mean()*100, 
                           columns=['Percent']).reset_index()


# In[179]:


cam_success.head()


# We have to plot between 'Index' and 'percent'

# In[180]:


# plot
sns.barplot(x='Percent', y='index', data=cam_success.sort_values('Percent'), palette='Blues')
plt.xlabel('Accepted (%)')
plt.ylabel('Campaign')
plt.title('Marketing campaign success rate', size=16)


# We conclude that the most successful campaign is the most recent one. Column: 'Response'

# In[182]:


#Finding the average customer look like for the company:
# list of cols with binary responses
binary_cols = [col for col in df.columns if 'Accepted' in col] + ['Response', 'Complain']
binary_cols


# In[183]:


# list of cols for spending 
mnt_cols = [col for col in df.columns if 'Mnt' in col]
mnt_cols


# In[184]:


# list of cols for channels
channel_cols = [col for col in df.columns if 'Num' in col] + ['TotalPurchases', 'TotalCampaignsAcc']
channel_cols


# In[185]:


# average customer demographics
demographics = pd.DataFrame(round(df.drop(columns=binary_cols+mnt_cols+channel_cols).mean(), 1), columns=['Average']).reindex([
    'Year_Birth', 'Year_Customer', 'Income', 'Dependents', 'Kidhome', 'Teenhome', 'Recency'])
demographics


# We conclude that:
# 1. The average birth year of a customer is 1969
# 2. The average year of becoming a customer is 2013
# 3. The average income is around 52000dollar
# 4. Has 1 dependent (roughly equally split between kids or teens)
# 5. Made a purchase from our company in the last 49 days

# In[188]:


#WHICH PRODUCT ARE PERFORMING THE BEST?

spending = pd.DataFrame(round(df[mnt_cols].mean(), 1), columns=['Average']).sort_values(by='Average').reset_index()
spending.head()


# In[191]:


# plot
ax = sns.barplot(x='Average', y='index', data=spending, palette='Blues')
plt.ylabel('Amount spent on...')

## add text labels for each bar's value
for p,q in zip(ax.patches, spending['Average']):
    ax.text(x=q+40,
            y=p.get_y()+0.5,
            s=q,
            ha="center") ;


# In[190]:


## add text labels for each bar's value
for p,q in zip(ax.patches, spending['Average']):
    ax.text(x=q+40,
            y=p.get_y()+0.5,
            s=q,
            ha="center") ;


# # We conclude that:
# The average customer spent...
# 1. 25-50(dollar) on Fruits, Sweets, Fish, or Gold products
# 2. Over 160Dollar on Meat products
# 3. Over 300Dollar on Wines
# 4. Over 2400Dollar total
# 

# Products performing best:
# Wines
# Followed by meats

# Lets find out which channels are underperforming:

# In[192]:


channels = pd.DataFrame(round(df[channel_cols].mean(),1), columns = ['Average']).sort_values(by='Average').reset_index()


# In[193]:


# plot
ax = sns.barplot(x='Average', y='index', data=channels, palette='Blues')
plt.ylabel('Number of...')

## add text labels for each bar's value
for p,q in zip(ax.patches, channels['Average']):
    ax.text(x=q+0.8,
            y=p.get_y()+0.5,
            s=q,
            ha="center") ;


# # We conclude that:
# 1. Accepted less than 1 advertising campaign
# 2. Made 2 deals purchases, 2 catalog purchases, 4 web purchases, and 5 store purchases
# 3. Averaged 14 total purchases
# 4. Visited the website 5 times

# Underperforming channels:
# 1. Advertising campaigns
# 2. Followed by deals, and catalog

# In[ ]:


Thank you!
I will add a PDF where I will showcase my summary of the whole process of Marketing Analytics

