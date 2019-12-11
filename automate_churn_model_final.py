
"""churn_model_automation
Version: 3
Nargis Pauran
Date:07-06-2017
"""
# import all libraries

import pandas as pd
import numpy as np
import time
import psycopg2 as pg
from sklearn.model_selection import train_test_split
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve,auc,accuracy_score
from sklearn.externals import joblib


class AvonLyaltyPedictiveRFM(object):
    
    
        
     
     def __init__(self, config = None, mkt =None):
 
        super(AvonLyaltyPedictiveRFM, self).__init__()
        self.config = config
        self.mkt = mkt
    
    
    
    
     def createConnectionString(self):
            
        df1 = pd.read_csv(self.config)
        df = df1.loc[df1['market']==self.mkt]
        
        self.connection_string = """dbname= '"""+df.iloc[0]['dbname']+"""' user = '"""+df.iloc[0]['user']+"""' host='"""+df.iloc[0]['host']+"""' password='"""+df.iloc[0]['password']+"'"
        
        self.conn = pg.connect(self.connection_string)
        
        self.mrkt_id = df.iloc[0]['market']
        self.X_start= df.iloc[0]['X_start']
        self.X_end = df.iloc[0]['X_end']
        self.Y_start=df.iloc[0]['Y_start']
        self.Y_end = df.iloc[0]['Y_end']
        self.actprd= df.iloc[0]['activity_period']
        self.rmvprd = df.iloc[0]['remove_period']
        
        self.prob_1 = df.iloc[0]['prob_1']
        self.prob_2 = df.iloc[0]['prob_2']
        self.prob_3 = df.iloc[0]['prob_3']
        self.prob_4 = df.iloc[0]['prob_4']
        
       
        self.avo_1 = df.iloc[0]['avo_1']
        self.avo_2 = df.iloc[0]['avo_2']
        self.avo_3 = df.iloc[0]['avo_3']
        self.avo_4 = df.iloc[0]['avo_4']
        
        return(self)         
    
     def createXCampaignPeriodList(self):
            
        self.createConnectionString()
        
        str_sql=""" select distinct cmpgn_perd_id from cdw.v_dim_mrkt_cmpgn where mrkt_id="""
        str_sql=str_sql+str(self.mrkt_id)+""" and cmpgn_perd_id between """+str(self.X_start) + """ and """+str(self.X_end)+"""
                order by cmpgn_perd_id desc"""
        
        
        df=pd.read_sql_query(str_sql,self.conn)
        
        self.summary_time_period=list(df['cmpgn_perd_id'])
        
        
        return(self)

       
    
     def createYCampaignPeriodList(self):
        
        self.createConnectionString()
    
        str_sql2=""" select distinct cmpgn_perd_id from cdw.v_dim_mrkt_cmpgn where mrkt_id="""
        str_sql2=str_sql2+str(self.mrkt_id)+""" and cmpgn_perd_id between """+str(self.Y_start) + """ and """+str(self.Y_end)+"""
                order by cmpgn_perd_id desc"""
        df=pd.read_sql_query(str_sql2,self.conn)
        self.Y_time_period=list(df['cmpgn_perd_id'])
        
        return(self)
    
     
    
     def createList(self,df):
        
        print(df.columns)
        self.active_list = []
        self.not_active_list =[]
        self.remove_list = []
        
        
        for col in list (df.columns.values):
            if col.startswith('actv_rep_ind_lag_'):
                if int(col[-1])< self.actprd:
                    self.active_list.append(col)
            elif col.startswith('rmovl_ind_lag_'):
                if int(col[-1])< self.rmvprd:
                    self.remove_list.append(col) 
            else:
                self.not_active_list.append(col)
      
        
        self.sql="""CREATE TABLE ALL_FEATURES_"""+str(self.mrkt_id)+""" as 
                    SELECT * from EXTRACT_FUll_"""+str(self.mrkt_id)+""" where """
        
        for active in self.active_list:
            if active!=self.active_list[0]:
                if active!= self.active_list[len(self.active_list)-1]:
                    self.sql = self.sql+""" OR """+active+"""=1 """
                else:
                    self.sql = self.sql+""" OR """+active+"""=1)"""
        
            else: 
                self.sql = self.sql+"""("""+active+"""=1 """

        
        self.sql = self.sql + """ AND """

        for remove in self.remove_list:
            if remove!=self.remove_list[0]:
                if remove!= self.remove_list[len(self.remove_list)-1]:
                    self.sql = self.sql+""" OR """+remove+"""=0 """
                else:
                    self.sql = self.sql+""" OR """+remove+"""=0)"""
        
            else: 
                self.sql = self.sql+"""("""+remove+"""=0 """
                
        
        

        return (self.sql)
    
    
     def prepareLagSQL(self,measure_list, lag, period):
        
        self.lagsql="""Create table extract_"""+str(self.mrkt_id)+"""_"""+str(period)+""" as 
                                select acct_key,"""
            
        for meas in measure_list:
            if meas!=measure_list[len(measure_list)-1]:
                if meas.find(' as ')>0:  
                    self.lagsql=self.lagsql+meas+"_lag_"+str(lag)+","
                else:
                    self.lagsql=self.lagsql+meas+" as "+meas+"_lag_"+str(lag)+","
            else:
                if meas.find(' as ')>0:
                    self.lagsql=self.lagsql+meas+"_lag_"+str(lag)
                else:
                    self.lagsql=self.lagsql+meas+" as "+meas+"_lag_"+str(lag)
            
            
        self.lagsql=self.lagsql+" from cdw.v_sum_mrkt_acct_sls "
        self.lagsql=self.lagsql+" where mrkt_id="+str(self.mrkt_id)
        self.lagsql=self.lagsql+" and fld_sls_cmpgn_perd_id ="""+str(period)
           
        return (self)
         
            
           
     
     def calculateLagMeasures(self,input, output_file):
    
        self.createXCampaignPeriodList()
        self.createYCampaignPeriodList()
        
        cur = self.conn.cursor()
        
       
        #-----------------------Use the input text file to create a feature list-------------------#
      
        
        f = open( input, "r")
        g = f.read()  
        measure_list1 = g.splitlines()
        measure_list=[i.replace("'",'') for i in measure_list1]
   
        
        #---------For each previous period, save the lagged values of all the measures into a seperate file---------#
        
        
        
        counter__=1
        self.summary_time_period.sort()
        
        for period in self.summary_time_period:
            start_time = time.time()
            lag=self.summary_time_period.index(self.X_end)-self.summary_time_period.index(period)
            cur.execute("""DROP TABLE IF EXISTS extract_"""+str(self.mrkt_id)+"""_"""+str(period)+""" CASCADE""")
            
            
            self.prepareLagSQL(measure_list,lag, period)
           
            
            message_str=" Starting download process "+str(period)
            print (self.lagsql)
            cur.execute(self.lagsql)  
            
            
            #sql= """select * from extract_"""+str(self.mrkt_id)+"""_"""+str(period)+""" limit 3;"""
            #df=pd.read_sql(sql,self.conn)
            #print (df)
             
                
            message_str_3=" Full data extract for campaign period "+str(period)+" is  %s seconds ---" % (time.time() - start_time)
            
            print (message_str_3)
            
    
    ##### 
    
        cur.execute("""DROP TABLE IF EXISTS extract_keys_"""+str(self.mrkt_id)+""" CASCADE""")
        sqlstr="""CREATE TABLE extract_keys_"""+str(self.mrkt_id)+""" as 
                        select distinct ext.acct_key 
                        from ("""
        counter__=0
        
        for period in self.summary_time_period:
            if counter__<len(self.summary_time_period)-1:
                sqlstr=sqlstr+""" select * from extract_"""+str(self.mrkt_id)+"""_"""+str(period)+""" union all """
            else:
                sqlstr=sqlstr+""" select * from extract_"""+str(self.mrkt_id)+"""_"""+str(period)
            counter__+=1
        sqlstr=sqlstr+") as ext "
        print sqlstr
        cur.execute(sqlstr)
        
        
        
        cur.execute("""DROP TABLE IF EXISTS Y_extract_keys CASCADE""")
        sqlstr="""CREATE TABLE Y_extract_keys as 
                        select acct_key ,max(actv_rep_ind) as target_class
                        from cdw.v_sum_mrkt_acct_sls where mrkt_id="""+str(self.mrkt_id)+""" and 
                        fld_sls_cmpgn_perd_id between """+str(self.Y_start)+""" and """+str(self.Y_end)+""" 
                        group by acct_key """
        
        
        cur.execute(sqlstr)
        
        #sql= """select * from  Y_extract_keys limit 3;"""
        #df=pd.read_sql(sql,self.conn)
        #print (df)
        
        
         
        final_sql_str=""" CREATE TABLE EXTRACT_FUll_"""+str(self.mrkt_id)+""" as 
                          select a.acct_key, case when yy.target_class=0 then 1 else 0 end as target_class,""" 
        
        
        for period in self.summary_time_period:
            lag=self.summary_time_period.index(self.X_end)-self.summary_time_period.index(period)
            for meas in measure_list:
                if meas.find(' as ')>0:
                    meas=meas[meas.find(' as ')+4:]
                
                final_sql_str=final_sql_str+" a_"+str(period)+"."+str(meas)+"_lag_"+str(lag)+","        
        final_sql_str=final_sql_str+"""1 as id from extract_keys_"""+str(self.mrkt_id)+""" a """
        
        
        for period in self.summary_time_period:
            final_sql_str=final_sql_str+""" LEFT JOIN extract_"""+str(self.mrkt_id)+"""_"""+str(period)+""" a_"""+str(period)+""" 
                                        on a.acct_key=a_"""+str(period)+""".acct_key """
        
        final_sql_str=final_sql_str+""" LEFT JOIN Y_extract_keys yy on a.acct_key=yy.acct_key """
        print " Running the final sql joining all tables "
        
        print (final_sql_str)
        
        cur.execute("""DROP TABLE IF EXISTS EXTRACT_FUll_"""+str(self.mrkt_id)+""" CASCADE""")
        
        cur.execute(final_sql_str)
        
        
        
        #sql_2= """select * from EXTRACT_FULL_"""+str(self.mrkt_id)+""" limit 3;"""
        #df=pd.read_sql(sql_2,self.conn)
        #self.createList(df)
        
        
        #cur.execute("""DROP TABLE IF EXISTS ALL_FEATURES_"""+str(self.mrkt_id)+""" CASCADE""")
        #cur.execute(self.sql)
        
        #print (self.active_list)
        #print (self.remove_list)
        #print (len(df.columns))
        #print (len((self.not_active_list)))
        #print (self.sql)
        
       
     
        
        print (" Downloading all the data into the disk ")
        outputquery="COPY  EXTRACT_FULL_"""+str(self.mrkt_id)+" TO STDOUT WITH CSV HEADER"
        with open(str(output_file)+"/extract_features_"+str(self.mrkt_id)+"_"+str(self.X_end)+".csv", 'w') as f:
            cur.copy_expert(outputquery, f)
        
        
        cur.close()
        
        
        print (outputquery)
        
        
        final_message=" Feature generation has been completed for period ending "+str(self.X_end)
        final_message=final_message+". Total time taken is  %s seconds ---" % (time.time()-start_time)
        
        print (final_message)
        return
    
     def printAll(self):
        #self.createXCampaignPeriodList()
        #self.createYCampaignPeriodList()
        self.createConnectionString()
        print (self.connection_string)
        print (self.mrkt_id)
        print (self.X_start)
        print (self.Y_start)
        print (self.Y_end)
        #print (self.summary_time_period)
        #print (self.Y_time_period)
        
    
    #------------------------------------------------------------------------------------#
    #------------------------  STARTED MODEL BUILDING -----------------------------------#
    #--------------WILL ONLY EXECUTE WHEN NO MODEL IS AVAILABLE TO USE-------------------# 
    #------------------------------------------------------------------------------------#
    
     def createModel(self, output_file):
        
        
        self.createConnectionString()
        
        model_data=pd.read_csv(output_file+'extract_features_'+str(self.mrkt_id)+'_'+str(self.X_end)+'.csv')
        
        print (model_data.describe())
        
        model_data=model_data.fillna(0)
        
        self.createTrainTestData(model_data)
        
        print (self.train.head(5))
        print (self.test.head(5))
        
        model=xgboost.XGBClassifier(n_estimators=1,
                                colsample_bytree=0.15,
                                min_child_weight=250,
                                scale_pos_weight=0.01,
                                learning_rate=0.01)
        
        
        self.trained_model=model.fit(self.train[self.feature_cols],self.train['target_class'].map(int))
        
     
        
        return (self)
        
     
     def createTrainTestData(self,df):
            
        self.train, self.test = train_test_split(df, test_size = 0.1)
        self.feature_cols=[x for x in df.columns if '_lag' in x ]
        
        return (self)
    
     def validateOnTrainingData(self, model):
            
        Y_test_class_score = model.predict(self.train[self.feature_cols])
        Y_test_class_prob = model.predict_proba(self.train[self.feature_cols])
        fpr, tpr, _ = roc_curve(self.train['target_class'].map(int), Y_test_class_prob[:,1])
        roc_auc = auc(fpr, tpr)
        acc_score=accuracy_score(self.train['target_class'].map(int),Y_test_class_score)
        
        print (" The Area under the ROC curve on the training data using XGBOOST is "+str(roc_auc))
        print (" The accuracy score is "+str(acc_score))
        
     
     def validateOnTestingData(self, model):
 
        print " Now predicting on the test data using the fitted XGBOOST model"
    
        Y_test_class_score = model.predict(self.test[self.feature_cols])
        Y_test_class_prob = model.predict_proba(self.test[self.feature_cols])
        
        fpr, tpr, _ = roc_curve(self.test['target_class'].map(int), Y_test_class_prob[:,1])
        roc_auc = auc(fpr, tpr)
        acc_score=accuracy_score(self.test['target_class'].map(int),Y_test_class_score)

        print (" The Area under the ROC curve based on a validation set of size 10% of the total data using XGBOOST is "+str(roc_auc))
        print (" The accuracy score is "+str(acc_score))
     
    
    #---------------------------------------------------------------------#
    #------------------------ APPLY A MODEL FOR PREDICTION----------------#
    #---------------------------------------------------------------------#
     
      
     def applyModel(self, filename, output_file):
            
         
        self.createConnectionString()
        
        model_df=pd.read_csv (output_file+'extract_features_'+str(self.mrkt_id)+'_'+str(self.X_end)+'.csv')
        
        
        
        #print (self.model_df.describe())
        #print (len(self.model_df))
        
        model_df= model_df.fillna(0)
        print (model_df.head(2))
        
        self.model_df2 = model_df.sample(frac=0.25)
        
        feature_cols=[x for x in model_df.columns if '_lag' in x ]
        
        #print (len(self.model_df2))
        #print (self.model_df2.columns)
            
        self.XGB_Model = joblib.load(filename)
        
        self.model_df2['predicted_probability']=self.XGB_Model.predict_proba(self.model_df2[feature_cols])
        
        return (self)
    
      
    #---------------------------------------------------------------------#
    #------------------------GEMERATE LOYALTY SEGMENTATION ---------------#
    #---------------------------------------------------------------------#
       
     
        
      
     def createDataFrame(self, data):
     
        self.createConnectionString()
        
        self.deb_data=pd.read_csv (data+'.csv')
        
        return
        
     
     def generateProbBandColumn(self):
        
        def applyProbabilities(n):
        
            if n < self.prob_1:
                return 'Very_Low'
            elif n < self.prob_2:
                return 'Low'
            elif n < self.prob_3:
                return 'Moderate'
            elif n < self.prob_4:
                return 'High'
            else:
                return 'very_High'
         
        
      
        self.deb_data['probability_band'] = self.deb_data['predicted_probability'].apply(lambda x: applyProbabilities(x))
         
         
    
        return(self)
        
      
     def calculateAvgOrderValue(self):
              
         
        feature_cols=[x for x in self.deb_data.columns if '_lag' in x ]
          
        sls_cols=[x for x in feature_cols if 'fld_net_sls' in x and 'ytd' not in x]
            
         
         #self.model_df['Total_orders']=self.model_df[sls_cols].sum(axis=1)
        
        self.deb_data['avg_units_per_order']=self.deb_data[sls_cols].replace(0,np.NAN).mean(axis=1)
         
        return (self)
        
        
      
     def generateAvgOrderBandColumn(self):
        
        def applyAvgOrderBand(n):
            
            if n<self.avo_1:
                return 'Very_Low'
            elif n<self.avo_2:
                return 'Low'
            elif n<self.avo_3:
                return 'Moderate'
            elif n<self.avo_4:
                return 'High'
            else:
                return 'Very_High'
         
        
      
        self.deb_data['avg_order_band'] = self.deb_data['avg_units_per_order'].apply(lambda n: applyAvgOrderBand(n))
         
         
    
        return(self)
        
    
     def generateLoyaltySegmentColumn(self, data):
            
        self.createDataFrame(data)
        self.generateProbBandColumn()
        self.calculateAvgOrderValue()
        self.generateAvgOrderBandColumn()
        
        def loyalty_segmentation(prob, spend):
            
            if prob=='Very_Low' :
                if spend in ('Very_High','High','Moderate') :
                    return 'Premium'
                elif spend in ('Low'):
                    return 'Valuable'
                elif spend in ('Very_Low'):
                    return 'Potential'
            elif prob=='Low':
                if spend in ('Very_High','High'):
                    return 'Premium'
                elif spend in ('Moderate','Low'):
                    return 'Valuable'
                else:
                    return 'Potential'
            elif prob=='Moderate':
                if spend in ('Very_High','High'):
                    return 'Valuable'
                else:
                    return 'Potential'
            elif prob=='High':
                if spend in ('Very_High','High'):
                    return 'Potential'
                else:
                    return 'Uncommitted'
            else:
                if spend in ('Very_High','High','Moderate'):
                    return 'Uncommitted'
                else:
                    return 'Lapsers'
         
        
        self.deb_data['6_lvl_loyalty_segment']=self.deb_data[['probability_band','avg_order_band']].apply(lambda x: loyalty_segmentation(*x), axis=1)
          
        self.generateTwoSegmentColumn()
        
        result = self.deb_data[['acct_key','predicted_probability','probability_band','6_lvl_loyalty_segment','2_lvl_loyalty_segment']]
        
        print (result.head(10))
        result.to_csv('avon_representative_segment'+str(self.mrkt_id)+'_'+str(self.X_end)+'.csv')
        
        return (self)
        
     
     def generateTwoSegmentColumn(self):
        
        def generateTwoSegmentColumn(n):
            
            if n in ('Premium','Valuable'):
                return 'Loyal'
            else:
                return 'Non-Loyal'
         
        
      
        self.deb_data['2_lvl_loyalty_segment'] = self.deb_data['6_lvl_loyalty_segment'].apply(lambda n: generateTwoSegmentColumn(n))
      
   