import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime

################################################################################################

st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded'
)

################################################################################################
#prohetPred

def tsPredFig(df, item, predictDays):
    """
    This function is to predict the community demand using Prophet Time Series Algorithms. 
    
    Args:
    df (dataframe)  : dataset
    item (str)      : product type that was selected
    predictDays(int): days for prediction
    
    Returns:
    forecast        : dataframe consists of yhat, trend, ds
    fig             : time series graph
    fig1            : trend graph
    
    """

    #transformation
    df = df[['ProductType','OrderDate']]
    df = df.loc[df['ProductType']==str(item)]
    df = df.reset_index(drop=True)
    vc = df['OrderDate'].value_counts(dropna=True, sort=True)
    dfT = pd.DataFrame(vc)
    dfT = dfT.reset_index()
    dfT.columns = ['ds', 'y']
    dfT.sort_values(by=['ds'])


    #initiate Prophet
    m = Prophet()
    m.fit(dfT)

    #make future dataset in days
    future = m.make_future_dataframe(periods=int(predictDays), freq='d')

    #forecast
    forecast = m.predict(future)
    fig = m.plot(forecast,xlabel='Date', ylabel='Value', figsize=(20, 12))
    fig2 = m.plot_components(forecast,figsize=(20, 12))

    return forecast, fig, fig2

################################################################################################
def main():
    
    st.sidebar.image('logo2.png')
    st.sidebar.title('AGA x Happy Tummy')
    st.sidebar.subheader('Zero Hunger, Zero Waste')
    st.sidebar.write('___')
    st.sidebar.info('To make a meaningful contribution through AGA: (1) Have look at our real-time demand orders from the community. (2) Predict the demand of the product in the upcoming day(s).')
    
    #DATASET
    @st.cache(allow_output_mutation=True)
    def dfRead(path, User):
        map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr'}
        df = pd.read_csv(path)
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        df['Month'] = pd.DatetimeIndex(df['OrderDate']).month
        df = df.replace({'Month': map})
        df['User'] = User
        return df

    dfComm  = dfRead('tummyCustomer3.csv', 'Community')
    dfDon   = dfRead('tummyDonor3.csv', 'Donors')
    
    #FOODGROUP
    #Dataset that use to generate historgram "Orders by Community & Donor "
    dfCommFG = list(dfComm['FoodCattegory'].value_counts(normalize=True)*100)
    dfDonFG  = list(dfDon['FoodCattegory'].value_counts(normalize=True)*100)

    #Top5 Table
    dfCommT5 = list(dfComm['ProductType'].value_counts())
    dfDonT5  = list(dfDon['ProductType'].value_counts())

    #SIDEBAR
    st.sidebar.write('## Donation: Product Type List')
    donationList = st.sidebar.multiselect('Please select one or more from the dropdown list.',
        options=['BAKERY', 'BEVERAGES', 'CONDIMENTS', 'COOKIES', 'CRACKERS', 'DOUGH', 'DIARY', 'DELI_ITEM', 'FISH', 'FRESH_POULTRY', 'FRESH_MEAT', 'PROCESSED_POULTRY', 'PROCESSED_MEAT', 'PESTO', 'PASTA', 'SHELF_ITEM', 'SHELLFISH', 'SOY', 'SMOKEDFISH', 'VEGETABLES'])
    st.write('')
    st.sidebar.warning('Click below to predict the demand of the selected product from the community.')
    predDay = st.sidebar.number_input('Nr of Prediction Days', min_value=1)
    predButton = st.sidebar.button('Predict')
    st.sidebar.write('')
    st.sidebar.write('')
    submitButton = st.sidebar.button('Donate Submission')
    if submitButton:
        st.balloons()

    
    #PREDICTION
    if predButton:
        
        st.header('Community Demand Prediction by Product - Time Series')
        st.write('')
        col3, col4 = st.columns((1,1))

        with col3: 
            for don in donationList:
                st.write('#### Predicted Order Demand for Product ', don)
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                forecast, fig, fig2 = tsPredFig(dfComm, item=don, predictDays=predDay)
                forecast = forecast[['ds','yhat']]
                forecast = forecast.rename(columns={"ds": "Datetime", "yhat": "OrderVolume"})
                forecast['Item'] = don
                forecast = forecast.sort_index(ascending=False)
                forecast
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
        
        with col4:

            getGroupCommL   = []
            getGroupDonL    = []

            for i in donationList:

                #COMMUNITY
                groupComm = dfComm.groupby(dfComm['ProductType'])
                getGroupComm = groupComm.get_group(i)
                getGroupComm = getGroupComm[['ProductType', 'Month', 'User']]
                getGroupComm = getGroupComm.reset_index(drop=True)
                getGroupComm['User'] = 'Community'
                getGroupCommL.append(getGroupComm)

                #DONOR
                groupDon = dfDon.groupby(dfDon['ProductType'])
                getGroupDon = groupDon.get_group(i)
                getGroupDon = getGroupDon[['ProductType', 'Month', 'User']]
                getGroupDon = getGroupDon.reset_index(drop=True)
                getGroupDon['User'] = 'Donors'
                getGroupDonL.append(getGroupDon)
                
            dfGetGroupCommJoin = pd.concat(getGroupCommL)
            dfGetGroupCommJoin = dfGetGroupCommJoin.reset_index(drop=True)

            dfGetGroupDonJoin = pd.concat(getGroupDonL)
            dfGetGroupDonJoin = dfGetGroupDonJoin.reset_index(drop=True)

            dfGetGroupList = pd.concat([dfGetGroupCommJoin, dfGetGroupDonJoin], ignore_index=True)

            for t in donationList:
                st.write('#### Order Requests and Donations - ', t)
                histGroup       = dfGetGroupList.groupby(dfGetGroupList['ProductType'])
                getHistGroup    = histGroup.get_group(t)
                getHistGroup    = getHistGroup.value_counts(dropna=True, sort=True)
                getHistDF       = pd.DataFrame(getHistGroup)
                getHistDF       = getHistDF.reset_index()
                
                fig1 = px.bar(getHistDF, x='Month', y=0, color='User', barmode='group', width=600, height=400)
                fig1.update_layout(xaxis={'categoryorder':'array', 'categoryarray':['Jan','Feb','Mar','Apr']})

                st.plotly_chart(fig1)

                prescrVC    = dfGetGroupList[['ProductType','User']].value_counts()
                prescrDF    = pd.DataFrame(prescrVC)
                prescrDF    = prescrDF.reset_index()
                #Donor
                prescrDon   = prescrDF.loc[(prescrDF['ProductType'] == t) & (prescrDF['User'] == 'Donors')]
                prescrDonV  = prescrDon.iat[0,2]
                
                #Community
                prescrCom   = prescrDF.loc[(prescrDF['ProductType'] == t) & (prescrDF['User'] == 'Community')]
                prescrComV  = prescrCom.iat[0,2]

                #Treshhold
                #Positive - Has enough of stocks (Unwanted Waste)
                #Negative - Not enough of order to fill the demand
                t1 = "Hey, it looks like we have enough to share the love. Thank you :)"
                t2 = "Thank you, it looks like this item is running out soon, we are still taking the donation for this item."
                t3 = "Please, we are in critical need of this item. Thank you."

                if (prescrDonV - prescrComV) / prescrComV > 0.5:
                    st.success(t1)
                elif (prescrDonV - prescrComV) / prescrComV >= 0 and (prescrDonV - prescrComV) / prescrComV < 0.5:
                    st.warning(t2)
                else:
                    st.error(t3)

    

    #DASHBOARD

    if not predButton:
        col1, col2 = st.columns(2)

        with col1:
            st.write('#### Summary Orders by Community & Donor')
            st.write('##### Jan-21 - Mar-21')
            foodGroupDF = pd.DataFrame(data={
                'values'    : dfCommFG+dfDonFG,
                'foodCat'   : ['Meat, Fish, Milk, Eggs', 'Fat, Oil, Sugar, Sweets', 'Bread, Cereal, Pasta', 'Vegetables', 'Fruits', 'Meat, Fish, Milk, Eggs', 'Vegetables', 'Bread, Cereal, Pasta', 'Fruits', 'Fat, Oil, Sugar, Sweets'],
                'user'      : ['Community', 'Community', 'Community', 'Community', 'Community', 'Donor', 'Donor', 'Donor', 'Donor', 'Donor']
            })

            fig = px.bar(foodGroupDF, x='foodCat', y='values', color='user', barmode='group', width=600, height=400)
            st.plotly_chart(fig)

        with col2:
            st.write('#### Top 5 Products by Community & Partner')
            st.write('##### Jan-21 - Mar-21')
            st.write('')
            st.write('')
            top5CommDF      = pd.DataFrame(data={
                'Product Type by Community'     : ['SHELF_ITEM', 'VEGETABLES', 'BAKERY', 'DIARY', 'FROZEN_ITEM'],
                'No. Orders(Community)'         : dfCommT5[:5],
            })

            top5DonDF       = pd.DataFrame(data={
                'Product Type by Donors'        : ['DIARY', 'VEGETABLES', 'PROCESSED_MEAT', 'FRUIT', 'SHELF_ITEM'],
                'No. Orders(Donors)'            :  dfDonT5[:5]
            })
            top5CommDF
            top5DonDF
        
        st.subheader('Demand Surge Trend from the Community by Day')
        st.image('trendImg2.png')



if __name__ == '__main__':
    main()
