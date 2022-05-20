# -*- coding: utf-8 -*-
"""
Created on  Tue May 10 22:13:16 2022
Autor: Leonardo Brozinga Viglino
Data ult versao: 19/05/2022
@author: leovibroz@hotmail.com
"""
import pandas as pd
import os
import numpy as np
import seaborn as sns
import math
from datetime import date,datetime
from dateutil.relativedelta import relativedelta

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='svg'

####### useful functions
def do_kmeans(df,k):
    wcss = []
    silhouette = pd.DataFrame()
    for i in range(2, k):
        print(i)
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        preds = kmeans.fit_predict(df)
        wcss.append(kmeans.inertia_)
        centers = kmeans.cluster_centers_
    
    return wcss        
       
def do_cluster(df,test = False,k = 12,std = False):    
    # df = df_cls_products
    df =df.set_index('ID')
    ## Since percentages are already normalized (values between 0% and 100%) no standardization is required
    if std == True:
        sc = StandardScaler()
        sc.fit(df)
        # Standardized values
        standardized_columns = sc.transform(df)
        # DF With Standardized Values
        df2 = pd.DataFrame(standardized_columns, index=df.index, columns=df.columns)
        # DF With original values
        inversed = sc.inverse_transform(standardized_columns)
        df3 = pd.DataFrame(inversed, index=df.index, columns=df.columns)        
    else:
        # df2 with original values if std == False
        df2 = df
        
    if test == True:
        wcss = do_kmeans(df2,12)
        plt.plot(range(2, 12), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
    else:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        # Fit to Standardized values if std == True or original values otherwise
        kmeans.fit(df2)
        # Predict cluster to Standardized values if std == True or original values otherwise 
        df3['cluster'] = kmeans.predict(df2)
        print(df3['cluster'].value_counts())
        
        categories = df.columns
        # categories = categories.drop('cluster')
        # Understand each cluster behavior based on average pct
        df_plot = df3.groupby('cluster')[categories].mean()            
        
        
        # Plot Radar Charts for visual representation
        for n in range(0,k):
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                  r=df_plot[df_plot.index == n].values[0],
                  theta=categories,
                  fill='toself',
                  name='Cluster {}'.format(n)
            ))
                            
                    
            fig.update_layout(
              polar=dict(
                radialaxis=dict(
                  visible=True,
                  range=[0, 1]
                )),
              showlegend=True,
             title = 'Resultado'
            )
            
            fig.show()        
        
        return df3,df_plot
###############

## import data
os.chdir(r'C:\Users\leovi\Documents\Trabalho\case_ifood')
df_case = pd.read_csv('data/data.csv')


## some fields creation
ls_accept = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5']
 
## has ever accepted any campaign ?
df_case['fl_acc_camp'] =df_case.apply(lambda row: min(row['AcceptedCmp1'] + row['AcceptedCmp2'] + row['AcceptedCmp3'] + row['AcceptedCmp4'] + row['AcceptedCmp5'],1),axis = 1)
## for classification purposes
df_case['fl_acc_camp_4th'] =df_case.apply(lambda row: min(row['AcceptedCmp1'] + row['AcceptedCmp2'] + row['AcceptedCmp3'] + row['AcceptedCmp4'],1),axis = 1)
## how many?
df_case['qt_acc_camp'] = df_case['AcceptedCmp1'] + df_case['AcceptedCmp2'] + df_case['AcceptedCmp3'] + df_case['AcceptedCmp4'] + df_case['AcceptedCmp5']
## for classification purposes
df_case['qt_acc_camp_4th'] = df_case['AcceptedCmp1'] + df_case['AcceptedCmp2'] + df_case['AcceptedCmp3'] + df_case['AcceptedCmp4']


## age (assuming 2020)
df_case['Age'] = 2020 - df_case['Year_Birth']
        
## time since enrollment
df_case['Dt_Customer'] = df_case['Dt_Customer'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
df_case['qt_months']  = df_case['Dt_Customer'].apply(lambda x: np.floor(abs((x - date(2020,1,1)).days)/30))

## total amount spent
df_case['qt_spent'] = df_case['MntWines'] +  df_case['MntFruits'] + df_case['MntMeatProducts'] +  df_case['MntFishProducts'] + df_case['MntSweetProducts']
## pct spent on each type of spent
df_case['pct_spent_wine'] = df_case['MntWines']/df_case['qt_spent']
df_case['pct_spent_fruit'] = df_case['MntFruits']/df_case['qt_spent']
df_case['pct_spent_meat'] = df_case['MntMeatProducts']/df_case['qt_spent']
df_case['pct_spent_fish'] = df_case['MntFishProducts']/df_case['qt_spent']
df_case['pct_spent_sweet'] = df_case['MntSweetProducts']/df_case['qt_spent']

## pct spent on gold
df_case['pct_spent_gold'] = df_case['MntGoldProds']/df_case['qt_spent']

## total number of purchases
df_case['qt_purchases']  =  df_case['NumWebPurchases'] +  df_case['NumCatalogPurchases'] +   df_case['NumStorePurchases']
## pct of purchases on each channel
df_case['pct_purch_web'] = df_case['NumWebPurchases']/df_case['qt_purchases']
df_case['pct_purch_catalog'] = df_case['NumCatalogPurchases']/df_case['qt_purchases']
df_case['pct_purch_store'] = df_case['NumStorePurchases']/df_case['qt_purchases']
## pct of deal purchases
df_case['pct_deal_purchases'] = df_case['NumDealsPurchases']/df_case['qt_purchases']

#df_case.to_csv('data/data_treated.csv',decimal = ',',index = False)

# Clustering product behavior
# df_cls_products = df_case[df_case['qt_spent'] > 100]
df_cls_products = df_case[['ID','pct_spent_wine'  ,'pct_spent_fruit','pct_spent_meat','pct_spent_fish','pct_spent_sweet']]

# df_cls_r = do_cluster(df_cls_products,test=True)
# 4 clusters it is
df_cls_r = do_cluster(df_cls_products,k = 4,std = True)
df_cls_spent_r,df_plot = do_cluster(df_cls_products,k = 4,std = True)
df_cls_spent_r = df_cls_spent_r.reset_index(drop = False)
df_case = pd.merge(df_case,df_cls_spent_r[['ID','cluster']], how = 'left',on=['ID'])

# Clustering by total spent
#df_cls_spent= df_case[['ID','qt_spent']]
#df_cls_spent_r = do_cluster(df_cls_spent,test = True)
#df_cls_spent_r,df_plot = do_cluster(df_cls_spent,k = 4,std = True)

# Clustering by channel behavior
#df_cls_channel= df_case[['ID','pct_purch_web','pct_purch_catalog','pct_purch_store']].fillna(0)
#df_cls_channel_r = do_cluster(df_cls_channel,test = True)
#df_cls_channel_r,df_plot = do_cluster(df_cls_channel,k = 5,std = True)


# Exporting to PowerBI
# df_cls_spent_r.to_csv('data/product_behavior_clusters.csv',index = True,decimal = ',')
# df_case.to_csv('data/data_treated.csv',decimal = ',',index = False)


# Classification Model
## Will be Tranined and Tested based on 1-4th campaign historic and 5th campaign as target
## Will be Validated on Response field (since it's available)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier

columns_to_train = ['ID'
              # ,'Age'
              ,'Income'
              # ,'fl_acc_camp_4th'
              ,'qt_acc_camp_4th'
              # ,'qt_months'
            #  ,'qt_spent'
               # ,'pct_spent_wine'
              # ,'pct_spent_fruit'
              # ,'pct_spent_meat'
              # ,'pct_spent_fish'
              # ,'pct_spent_sweet'
            # Product Behavior
              # ,'cluster'
              # ,'pct_spent_gold'
              # ,'pct_purch_web'
              # ,'pct_purch_catalog'
              # ,'pct_purch_store'
              ,'pct_deal_purchases'
                ,'qt_purchases'         
            ,'AcceptedCmp5' # as target
            ]



df_class = df_case[columns_to_train]
df_class = df_class.fillna(0)

# Fixing some values
for col in  ['pct_deal_purchases']:
# ['pct_spent_gold','pct_purch_web','pct_purch_catalog','pct_purch_store'
            # ,'pct_spent_wine','pct_spent_fruit' ,'pct_spent_meat'
            # ,'pct_spent_fish','pct_spent_sweet']:
    df_class[col] = df_class[col].apply(lambda x: 0 if math.isinf(x) else x  ) 
    df_class[col] = df_class[col].apply(lambda x: 1 if x > 1 else x  )     
    

df_class = df_class.set_index('ID')
x_train,x_test,y_train,y_test = train_test_split(df_class.drop('AcceptedCmp5',axis = 1),df_class['AcceptedCmp5'],test_size = 0.4, random_state = 33)

model = XGBClassifier()
model.fit(x_train, y_train)

# make predictions for trainig data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

y_pred = pd.DataFrame(y_pred, columns = ['pred'],index = x_test.index)
y_pred_prob = model.predict_proba(x_test)
y_pred_prob = pd.DataFrame(y_pred_prob,columns = ['prob0','prob1'],index = x_test.index)
y_data = pd.DataFrame(y_test,index = x_test.index)
y_data['prob'] = y_pred_prob['prob1'].values


# FEATURE IMPORTANCES
df_feat_imp = pd.DataFrame(columns = ['variable','feat_imp'])
for i in range(0, len(columns_to_train[1:-1])):
    df_feat_imp = df_feat_imp.append({'variable':columns_to_train[1:-1][i],
                                      'feat_imp':model.feature_importances_[i]
                                      },ignore_index = True  )
  
print(metrics.classification_report(y_test, y_pred))

accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)

precision, recall, thresholds = metrics.precision_recall_curve(y_data['AcceptedCmp5'], y_data['prob'])
#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')
#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
#display plot
plt.show()



# Validation

columns_to_valid = ['ID'
             # ,'Age'
             ,'Income'
            # ,'fl_acc_camp
             ,'qt_acc_camp'
            # ,'qt_months'
            # ,'qt_spent'
            # ,'pct_spent_wine'
            # ,'pct_spent_fruit'
            # ,'pct_spent_meat'
            # ,'pct_spent_fish'
            # ,'pct_spent_sweet'
            # Product Behavior
            # ,'cluster'
            # ,'pct_spent_gold'
            # ,'pct_purch_web'
            # ,'pct_purch_catalog'
            # ,'pct_purch_store'
             ,'pct_deal_purchases'
             ,'qt_purchases'         
            ,'Response' # as target
            ]
df_valid = df_case[columns_to_valid]
df_valid = df_valid.set_index('ID')
for col in  ['pct_deal_purchases']:
# ['pct_spent_gold','pct_purch_web','pct_purch_catalog','pct_purch_store'
            # ,'pct_spent_wine','pct_spent_fruit' ,'pct_spent_meat'
            # ,'pct_spent_fish','pct_spent_sweet']:
    df_valid[col] = df_valid[col].apply(lambda x: 0 if math.isinf(x) else x  ) 
    df_valid[col] = df_valid[col].apply(lambda x: 1 if x > 1 else x  )     
    
    
x_valid = df_valid.drop('Response',axis = 1)
y_valid = df_valid['Response']

# make predictions for trainig data
y_pred = model.predict(x_valid)

y_pred = pd.DataFrame(y_pred, columns = ['pred'],index = x_valid.index)
y_pred_prob = model.predict_proba(x_valid)
y_pred_prob = pd.DataFrame(y_pred_prob,columns = ['prob0','prob1'],index = x_valid.index)
y_data = pd.DataFrame(y_valid,index = x_valid.index)
y_data['prob'] = y_pred_prob['prob1'].values

print(metrics.classification_report(y_data['Response'], y_pred['pred']))

accuracy = metrics.accuracy_score(y_data['Response'], y_pred['pred'])
recall = metrics.recall_score(y_data['Response'], y_pred['pred'])
precision = metrics.precision_score(y_data['Response'], y_pred['pred'])

precision, recall, thresholds = metrics.precision_recall_curve(y_data['Response'], y_data['prob'])
#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

plt.show()


        
# Selecting Best Threshold based on probability of success
df_valid['prob1'] = y_data['prob']
df_case = df_case.set_index('ID')
df_case['prob'] = df_valid['prob1']

avg_cost= 600
avg_revenue =5508
df_profit = pd.DataFrame(columns = ['threshold','total_cost','total_revenue','profit_loss','qtd_customers'])

# Based on average cost and revenue
# A loop through different threshold values give us a different set of customers
# Based on the average customer revenue that accepts * the probability of accepting we estimate a probable revenue for each customer
# And on a fixed cost of contact per customer
# We have the total cost, total estimated revenue and profit per threshold

for i in range(0,100,5):
    df_case['selected'] = df_case['prob'].apply(lambda x: 1 if x >= i/100 else 0)
    df_case['prob_revenue'] = df_case['prob']*avg_revenue
    qtd_customers = len(df_case[df_case['selected'] == 1])
    total_cost = qtd_customers * avg_cost
    total_revenue = df_case[df_case['selected'] == 1]['prob_revenue'].sum()
    profit_loss = total_revenue-total_cost
    df_profit = df_profit.append({'threshold':i/100,
                                    'total_cost':total_cost,
                                    'total_revenue':total_revenue,
                                    'profit_loss':profit_loss,
                                    'qtd_customers':qtd_customers},ignore_index = True
                                     )
df_profit.to_excel('data/df_profit.xlsx',index = False)

# results are interpretated on Excel file






















