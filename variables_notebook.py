import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (MinMaxScaler,
                                   StandardScaler,
                                   RobustScaler,
                                   OneHotEncoder)

aplication = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")
df = aplication.merge(credit,how='inner',on=['ID'])

#Data clean functions
def detect_low_variance(dataframe, threshold):
   """
   This function returns a list of column names that have a low variance.
   :dataframe: dataset
   :threshold: needs to be define, recommended 0.001
   :retun: low variance from column
   """
   #defines a threshold to detect variables with low variation
   threshold = threshold
 #instance the normalizer
   scaler = MinMaxScaler()
 #delect numeric variables
   num_cols = dataframe.select_dtypes(include = [int, float])
 #normalizes numeric variable data
   scaled_num_cols = scaler.fit_transform(num_cols)
 #creates a dataframe with normalized variables
   scaled_num_df = pd.DataFrame(scaled_num_cols,
   columns = num_cols.columns)
 #create an empty list
   low_variance_columns = list()
 #creates a loop that will go through all the columns
   for column in scaled_num_df:
 #calculates the variance of the columns
    column_variance = scaled_num_df[column].var()
 #checks if the variance is less than the threshold, if so, it will add in thelow_variance_columns
   if column_variance < threshold:
    low_variance_columns.append(column)
   print(low_variance_columns)

def check_missing(df):
      import pandas
      if isinstance(df, pandas.core.frame.DataFrame):
            return(((df.isnull().sum()/df.shape[0])*100).round(2)).sort_values(ascending = False)
      return None

#Feature Engineering
def get_education_status(x):
      """
      This function split function to NAME_EDUCATION_TYPE
      :NAME_EDYCATION_TYPE:Education leval
      :return:split of NAME_EDUCATION_TYPE column
      """
      if x == 'Secondary / secondary special' :
       x= x.split(' /')[0]       
      return x


def get_family_status(x):
     """
     This function split function to NAME_FAMILY_STATUS
     :NAME_FAMILY_STATUS:Marital status
     :return:split of NAME_FAMILY_STATUS
     """
     if x == 'Single / not married' :
       x= x.split(' /')[0]       
     return x


def get_house_status(x):
     """
    This function split function to NAME_HOUSING_TYPE
    :NAME_HOUSING_TYPE:Way of living
    :return:split of NAME_HOUSING_TYPE
    """
     if x == 'House / apartment' :
       x= x.split(' /')[0]       
     return x


#Target
def labels (x):
     '''
     creating target labels
     :Satatus - 
     0: 1-29 days past due 
     1: 30-59 days past due 
     2: 60-89 days overdue 
     3: 90-119 days overdue 
     4: 120-149 days overdue 
     5: Overdue or bad debts, write-offs for more than 150 days 
     6: paid off that month 
     7: No loan for the month
     :retun: Target where 1 means Risk when 0 means No risc
     '''
     target = ''
     if x in (2,3,4,5) :
       target = 1 
     else:
         target = 0 

     return target  

#Statistics
def plot_histbox(data, column):
     """
     This function returns the plot of a boxplot and a distplot of each column
     :data: numerical variables
     :column: column from dataframe
     :return: boxplot viz
     """
     #setting the size of graphics
     f, ax = plt.subplots ( figsize=(24, 5) )
     #background color
     background_color = '#FFFFFF'
     f.set_facecolor( background_color )
     #color palette
     palette_colors = sns.color_palette( 'Dark2_r', len( data.columns ) * 2)
     #plot on grid / boxplot
     plt.subplot( 1, 2, 1 )
     #title
     plt.title( f'{column}', loc='left', fontsize=14, fontweight=200 )
     #plot
     sns.boxplot( data=data, y=column, showmeans=True, saturation=0.75, linewidth=1, width=0.25, color= palette_colors[ list(df.columns).index(column)])
     #plot on grid / distplot
     plt.subplot( 1, 2, 2 )
     #title
     plt.title( f'{column}', loc='left', fontsize=14, fontweight=200 )
     #plot
     sns.distplot( df[ column ], color= palette_colors[ list(df.columns).index(column)])
     #grid adjust
     plt.subplots_adjust( top=0.95, hspace=0.3 )

 

 
def plot_corr(corr):
    """"
    This function returns the Pearson correlation matrix
    we will cut the top half as it mirrors the bottom half
    """
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, 1)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5,  annot=True, square=True)

#Data Viz
def categorical_plotting(df,col1,hue1,title1,title1_leg,col2,hue2,title2,title2_leg):
  fig, ax = plt.subplots (1,2, figsize = (20,10))
  sns.countplot(data = df, y = col1, hue = hue1, ax=ax[0])
  sns.countplot(data = df, y = col2, hue = hue2, ax=ax[1])
  ax[0].set(ylabel = '', xlabel = 'Client')
  ax[1].set(ylabel = '', xlabel = 'Client')
  ax[0].set_title( title1, pad = 10, fontsize = 15, fontweight = 'bold')
  ax[1].set_title( title2, pad = 10, fontsize = 15, fontweight = 'bold')
  ax[0].legend(title= title1_leg)
  ax[1].legend(title= title2_leg)
  fig.tight_layout(pad=4);

def numerical_plotting(df,col,title,xtitle):
  fig, ax = plt.subplots(figsize = (20, 8))
  df[col].hist(bins=25, grid=False, color='#86bf91', zorder=2, rwidth=0.9, ax=ax)
  ax.set_title( title, pad = 10, fontsize = 15, fontweight = 'bold')
  ax.set(ylabel = 'Frequency', xlabel = xtitle);