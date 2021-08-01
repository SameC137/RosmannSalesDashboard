import pandas as pd
import streamlit as st
import holiday 
import bisect

# from streamlit_gallery.utils import readme
import pickle

@st.cache(allow_output_mutation=True)
def loadModel():
    file = open("model2.pkl",'rb')
    model = pickle.load(file)
    return model
@st.cache()
def load_store_data():
    store_df=pd.read_csv('store_cleaned.csv')

    return store_df
# s.dt.dayofweek
def read_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print("file read as csv")
        return df
    except FileNotFoundError:
        print("file not found")

def read_csv_without_index(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print("file read as csv")
        return df
    except FileNotFoundError:
        print("file not found")
def datToAndAfterHoliday(df,Column,holidays):
    
    to=[]
    after=[]
    for a in df[Column]:
        index=bisect.bisect(holidays,a)
        if len(holidays)==index:
            to.append(pd.Timedelta(0, unit='d') )
            after.append(a - holidays[index-1])
        else:
            after.append(holidays[index] - a)
            to.append(a -holidays[index-1])
    return to,after
def startMidEndMonth(x):
    if x<10:
        return 0
    elif x<20:
        return 1
    else:
        return 2
def isWeekend(x):
    if x<6:
        return 0
    else: 
        return 1
def dateExplode(df,column):
    try:
        df['Year'] = pd.DatetimeIndex(df[column]).year
        df['Month'] = pd.DatetimeIndex(df[column]).month
        df['Day'] = pd.DatetimeIndex(df[column]).day  
    except KeyError:
        print("Column couldn't be found")
        return
    return  df
    
def generate_features(df):
    
    df["Date"]=pd.to_datetime(df["Date"])
    
    df["weekend"]= df["DayOfWeek"].apply(isWeekend )
    df["MonthState"]=df["Day"].apply(startMidEndMonth)
    with open('dates.pickle', 'rb') as handle:
        dates = pickle.load(handle)
    df["To"],df["After"]=datToAndAfterHoliday(df,"Date",dates)
    
    df["After"]=pd.to_timedelta(df["After"])
    
    df["To"]=pd.to_timedelta(df["To"])

    df["After"]=pd.to_numeric(df['After'].dt.days, downcast='integer')
    df["To"]=pd.to_numeric(df['To'].dt.days, downcast='integer')

    return  df
    
def merge_store(df):
    store=read_csv("store_cleaned.csv")
    combined = pd.merge(df, store, on=["Store"])
    return combined

def predict(model,csv):
    csv_copy=csv.copy()
    csv_copy.drop("Store",axis=1,inplace=True)
    
    csv_copy.drop("Id",axis=1,inplace=True)
    
    csv_copy.drop("Date",axis=1,inplace=True)
    print(csv_copy.columns)
    prediction=model.predict(csv_copy)
    
    pred_df = csv.copy()

    pred_df["Sales-Prediction"] = prediction
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
    return pred_df


def main():
    dataset = "train.csv"

    df = load_store_data()
    # pr = gen_profile_report(df, explorative=True)
    model=loadModel()
    st.write(f"🔗Preictions based on  [Rossman Sales]({dataset})")
    st.markdown(f"### Sales Prediction")
    method = st.radio("method", ('Upload file', 'Manual'))
    if (method == "Upload file"):
        test_file = st.file_uploader("Upload csv files", type=['csv'])
        test_csv = None
        if (test_file):
            test_csv = read_csv_without_index(test_file)
            # st.write(test_csv)

            if st.button('Predict'):
                dateExplode(test_csv,column="Date")
                test_store=merge_store(test_csv)
                test_store=generate_features(test_store)
                # st.write(test_csv)
                prediction=predict(model,test_store)
                st.write(prediction)

    # st.write(df)
    



    


@st.cache(allow_output_mutation=True)
def gen_profile_report(df, *report_args, **report_kwargs):
    return df.profile_report(*report_args, **report_kwargs)


if __name__ == "__main__":
    main()