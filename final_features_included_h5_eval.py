#!/usr/bin/env python
# coding: utf-8

# In[1]:


from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb
import os
from glob import glob

os.chdir('D:\\kaggle_competitions\\M5 Competition')


# In[2]:


CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }


# In[3]:


h = 28 
max_lags = 57
tr_last = 1941
fday = datetime(2016,5, 23) 


# In[4]:


catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
dtype = {col: "category" for col in catcols if col != "id"}
sales_train = pd.read_csv("sales_train_evaluation.csv", usecols = catcols, dtype = dtype) ##changes the names
sale_1=sales_train.copy()


# In[5]:


for col in catcols:
        if col != "id":
            sales_train[col]=sales_train[col].astype('category')
            sales_train[col] = sales_train[col].cat.codes.astype("int16")
            sales_train[col] -= sales_train[col].min()


# In[6]:


cols=[f for f in sales_train if 'd' not in f or ('id' in f )]


# In[7]:


sales_train=sales_train[cols]


# In[8]:


sales_train.columns=['id','item_no','dept_no','cat_no','store_no','state_no']


# In[9]:


sale_1=sale_1[cols]
sale_1=pd.merge(sale_1,sales_train,on='id')


# In[10]:


del sale_1['id']
del sale_1['item_id']
sale_1=sale_1.drop_duplicates()


# In[11]:


del sale_1['item_no']
sale_1=sale_1.drop_duplicates()


# In[12]:


state_id1=sale_1[['state_id','state_no']]
state_id1=state_id1.drop_duplicates()
state_id1=state_id1.reset_index()
del state_id1['index']

dept_id1=sale_1[['dept_id','dept_no']]
dept_id1=dept_id1.drop_duplicates()
dept_id1=dept_id1.reset_index()
del dept_id1['index']



store_id1=sale_1[['store_id','store_no']]
store_id1=store_id1.drop_duplicates()
store_id1=store_id1.reset_index()
del store_id1['index']


cat_id1=sale_1[['cat_id','cat_no']]
cat_id1=cat_id1.drop_duplicates()
cat_id1=cat_id1.reset_index()
del cat_id1['index']


# In[13]:


state_id=pd.read_csv('D:\\kaggle_competitions\\M5 Competition\\validation_h5\\state_id.csv')
dept_id=pd.read_csv('D:\\kaggle_competitions\\M5 Competition\\validation_h5\\dept_id.csv')
cat_id=pd.read_csv('D:\\kaggle_competitions\\M5 Competition\\validation_h5\\cat_id.csv')
store_id=pd.read_csv('D:\\kaggle_competitions\\M5 Competition\\validation_h5\\store_id.csv')
store_cat_id=pd.read_csv('D:\\kaggle_competitions\\M5 Competition\\validation_h5\\store_id_cat_id.csv')
store_dept_id=pd.read_csv('D:\\kaggle_competitions\\M5 Competition\\validation_h5\\store_id_dept_id.csv')


# In[14]:


state_id=pd.merge(state_id,state_id1,on='state_id')
del state_id['state_id']
state_id.rename(columns={'state_no':'state_id'},inplace=True)

dept_id=pd.merge(dept_id,dept_id1,on='dept_id')
del dept_id['dept_id']
dept_id.rename(columns={'dept_no':'dept_id'},inplace=True)


cat_id=pd.merge(cat_id,cat_id1,on='cat_id')
del cat_id['cat_id']
cat_id.rename(columns={'cat_no':'cat_id'},inplace=True)

store_id=pd.merge(store_id,store_id1,on='store_id')
del store_id['store_id']
store_id.rename(columns={'store_no':'store_id'},inplace=True)

store_cat_id=pd.merge(store_cat_id,store_id1,on='store_id')
del store_cat_id['store_id']
store_cat_id.rename(columns={'store_no':'store_id'},inplace=True)

store_cat_id=pd.merge(store_cat_id,cat_id1,on='cat_id')
del store_cat_id['cat_id']
store_cat_id.rename(columns={'cat_no':'cat_id'},inplace=True)



store_dept_id=pd.merge(store_dept_id,store_id1,on='store_id')
del store_dept_id['store_id']
store_dept_id.rename(columns={'store_no':'store_id'},inplace=True)

store_dept_id=pd.merge(store_dept_id,dept_id1,on='dept_id')
del store_dept_id['dept_id']
store_dept_id.rename(columns={'dept_no':'dept_id'},inplace=True)


# In[15]:


dt1=state_id
state_id = pd.melt(state_id,
                  id_vars = ['state_id'],
                  value_vars = [col for col in dt1.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales_states")
dept_id = pd.melt(dept_id,
                  id_vars = ['dept_id'],
                  value_vars = [col for col in dt1.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales_dept")
cat_id = pd.melt(cat_id,
                  id_vars = ['cat_id'],
                  value_vars = [col for col in dt1.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales_cat")
store_id = pd.melt(store_id,
                  id_vars = ['store_id'],
                  value_vars = [col for col in dt1.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales_store")
store_cat_id = pd.melt(store_cat_id,
                  id_vars = ['store_id','cat_id'],
                  value_vars = [col for col in dt1.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales_store_cat")

store_dept_id = pd.melt(store_dept_id,
                  id_vars = ['store_id','dept_id'],
                  value_vars = [col for col in dt1.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales_dept_store")



# In[16]:


def create_dt(is_train = True, nrows = None, first_day = 1200):
    prices = pd.read_csv("sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv("calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("sales_train_evaluation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype) ##changes the names
    ###################
    
    
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    dt=dt.merge(state_id,on=['state_id','d'],copy=False)
    dt=dt.merge(dept_id,on=['dept_id','d'],copy=False)
    dt=dt.merge(store_id,on=['store_id','d'],copy=False)
    dt=dt.merge(cat_id,on=['cat_id','d'],copy=False)
    dt=dt.merge(store_cat_id,on=['store_id','cat_id','d'],copy=False)
    dt=dt.merge(store_dept_id,on=['store_id','dept_id','d'],copy=False)
    
    
    dt['var']=dt['d'].str.replace('d_','').astype(int)
    #dt['var2']=dt['var']**2
    return dt


# In[17]:


def create_fea(dt):
    lags = [7,28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7,28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
            
            
            

    
    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
    
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")


# In[18]:


get_ipython().run_cell_magic('time', '', "\nFIRST_DAY = 150  #f you want to load all the data set it to '1' -->  Great  memory overflow  risk ! ##set to 50\n\ndf = create_dt(is_train=True, first_day= FIRST_DAY)\ndf.shape\n\ncreate_fea(df)\ndf.shape\n")


# In[20]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[21]:


df=reduce_mem_usage(df)
df.dropna(inplace = True)

cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id']+["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "date", "sales", "wm_yr_wk", "weekday"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X_train = df[train_cols]
y_train = df["sales"]



# In[22]:


#fake_valid_inds=X_train.index.values[X_train['d'].str.contains('d_1885|d_1886|d_1887|d_1888|d_1889|d_1890|d_1891|d_1892|d_1893|d_1894|d_1895|d_1896|d_1897|d_1898|d_1899|d_1900|d_1901|d_1902|d_1903|d_1904|d_1905|d_1906|d_1907|d_1908|d_1909|d_1910|d_1911|d_1912|d_1913'),True]
fake_valid_inds=X_train.index.values[X_train['d'].str.contains('d_1914|d_1915|d_1916|d_1917|d_1918|d_1919|d_1920|d_1921|d_1922|d_1923|d_1924|d_1925|d_1926|d_1927|d_1928|d_1929|d_1930|d_1931|d_1932|d_1933|d_1934|d_1935|d_1936|d_1937|d_1938|d_1939|d_1940|d_1941'),True]


# In[23]:


len(fake_valid_inds)


# In[24]:


useless_cols = ["id", "date", "sales",'d', "wm_yr_wk", "weekday"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X_train = df[train_cols]
y_train = df["sales"]


# In[31]:


get_ipython().run_cell_magic('time', '', "\nnp.random.seed(786)\n\ntrain_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)\ntrain_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], \n                         categorical_feature=cat_feats, free_raw_data=False)\nfake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],\n                              categorical_feature=cat_feats,\n                 free_raw_data=False)# This is a random sample, we're not gonna apply any time series train-test-split tricks here!")


# In[32]:


del df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()


# In[33]:


params={
                    'boosting_type': 'gbdt',
                    'objective': 'poisson', ###try this with tweedie ##and tweedie reult in 0.493
                    #'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.04,#riginal=0.0409 ###i'll try this with the same lr a
                    'num_leaves': 2**14,
                    'min_data_in_leaf': 2**15,
                    'feature_fraction': 0.5,
                    'max_bin': 100, ###100
                    'n_estimators': 2500,
                    'boost_from_average': False,
                    'verbose': 1,
                } 


# In[39]:


get_ipython().run_cell_magic('time', '', '\nm_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=200) ')


# In[44]:


m_lgb.save_model("lgb_lr_final_evaluation_dfeatures_hp.lgb")


# In[40]:


import matplotlib.pyplot as plt

lgb.plot_importance(m_lgb)

plt.show()


# In[41]:


alphas=[1]
weights = [1/len(alphas)]*len(alphas)
sub = 0.

for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

    te = create_dt(False)
    cols = [f"F{i}" for i in range(1,29)]

    for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        print(tdelta, day)
        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
        create_fea(tst)
        tst = tst.loc[tst.date == day , train_cols]
        te.loc[te.date == day, "sales"] = alpha*m_lgb.predict(tst) # magic multiplier by kyakovlev



    te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
    te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
    te_sub.fillna(0., inplace = True)
    te_sub.sort_values("id", inplace = True)
    te_sub.reset_index(drop=True, inplace = True)
    #te_sub.to_csv(f"submission_{icount}.csv",index=False)
    if icount == 0 :
        sub = te_sub
        sub[cols] *= weight
    else:
        sub[cols] += te_sub[cols]*weight
    print(icount, alpha, weight)


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("evaluation$", "validation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission_eval_h5_features.csv",index=False)


# In[42]:


sub.head()


# In[43]:


cols=[f for f in sub.columns if 'id' not in f]
a=pd.DataFrame(sub[sub['id'].str.contains('evaluation')==True][cols].sum())
a.columns=['sum']
a['sum'].sum()

