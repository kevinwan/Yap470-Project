import streamlit as st
import pandas as pd
import numpy as np
import pickle
import imageio as iio
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Damage Grade Prediction App

This app predicts the **Damage Grade** of structures!

Data obtained from the (https://www.drivendata.org/competitions/57/nepal-earthquake/page/134/) in DrivenData.
""")

st.sidebar.header('User Input Features')


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    data2 = pd.DataFrame()
    def user_input_features():
        data2['geo_level_1_id'] = [st.sidebar.slider('geo_level_1_id', 0,31,7)]
        data2['geo_level_2_id'] = [int(st.sidebar.text_input("geo_level_2_id", value=0))]
        data2['geo_level_3_id'] = [int(st.sidebar.text_input("geo_level_3_id", value=0))]
        data2['count_floors_pre_eq'] = [st.sidebar.slider('count_floors_pre_eq', 1,9,2)]
        data2['age'] = [st.sidebar.slider('age', 0,1000,12)]
        data2['area_percentage'] = [st.sidebar.slider('area_percentage', 1,100,9)]
        data2['height_percentage'] = [st.sidebar.slider('height_percentage', 2,32,6)]
        data2['land_surface_condition'] = [st.sidebar.selectbox('land_surface_condition',('t','n','o'))]
        data2['foundation_type'] = [st.sidebar.selectbox('foundation_type',('r','w','u','i','h'))]
        data2['roof_type'] = [st.sidebar.selectbox('roof_type',('n','q','x'))]
        data2['ground_floor_type'] = [st.sidebar.selectbox('ground_floor_type',('f','x','v','z','m'))]
        data2['other_floor_type'] =[ st.sidebar.selectbox('other_floor_type',('q','x','j','s'))]
        data2['position'] = [st.sidebar.selectbox('position',('s','t','j','o'))]
        data2['plan_configuration'] = [st.sidebar.selectbox('plan_configuration',('d','q','u','s','c','a','o','m','n','f'))]
        data2['has_superstructure_adobe_mud']=[1 if st.sidebar.checkbox('has_superstructure_adobe_mud', value=False)  else 0]
        data2['has_superstructure_mud_mortar_stone']=[1 if st.sidebar.checkbox('has_superstructure_mud_mortar_stone', value=False)  else 0]
        data2['has_superstructure_stone_flag']=[1 if st.sidebar.checkbox('has_superstructure_stone_flag', value=False)  else 0]
        data2['has_superstructure_cement_mortar_stone']=[1 if st.sidebar.checkbox('has_superstructure_cement_mortar_stone', value=False)  else 0]
        data2['has_superstructure_mud_mortar_brick']=[1 if st.sidebar.checkbox('has_superstructure_mud_mortar_brick', value=False)  else 0]
        data2['has_superstructure_cement_mortar_brick']=[1 if st.sidebar.checkbox('has_superstructure_cement_mortar_brick', value=False)  else 0]
        data2['has_superstructure_timber']=[1 if st.sidebar.checkbox('has_superstructure_timber', value=False)  else 0]
        data2['has_superstructure_bamboo']=[1 if st.sidebar.checkbox('has_superstructure_bamboo', value=False)  else 0]
        data2['has_superstructure_rc_non_engineered']=[1 if st.sidebar.checkbox('has_superstructure_rc_non_engineered', value=False)  else 0]
        data2['has_superstructure_rc_engineered']=[1 if st.sidebar.checkbox('has_superstructure_rc_engineered', value=False)  else 0]
        data2['has_superstructure_other']=[1 if st.sidebar.checkbox('has_superstructure_other', value=False)  else 0]
        data2['legal_ownership_status'] = [st.sidebar.selectbox('legal_ownership_status',('v','a','w','r'))]
        data2['count_families'] = [st.sidebar.slider('count_families', 0,9,1)]
        data2['has_secondary_use']=[1 if st.sidebar.checkbox('has_secondary_use', value=False)  else 0]
        data2['has_secondary_use_agriculture']=[1 if st.sidebar.checkbox('has_secondary_use_agriculture', value=False)  else 0]
        data2['has_secondary_use_hotel']=[1 if st.sidebar.checkbox('has_secondary_use_hotel', value=False)  else 0]
        data2['has_secondary_use_rental']=[1 if st.sidebar.checkbox('has_secondary_use_rental', value=False)  else 0]
        data2['has_secondary_use_institution']=[1 if st.sidebar.checkbox('has_secondary_use_institution', value=False)  else 0]
        data2['has_secondary_use_school']=[1 if st.sidebar.checkbox('has_secondary_use_school', value=False)  else 0]
        data2['has_secondary_use_industry']=[1 if st.sidebar.checkbox('has_secondary_use_industry', value=False)  else 0]
        data2['has_secondary_use_health_post']=[1 if st.sidebar.checkbox('has_secondary_use_health_post', value=False)  else 0]
        data2['has_secondary_use_gov_office']=[1 if st.sidebar.checkbox('has_secondary_use_gov_office', value=False)  else 0]
        data2['has_secondary_use_use_police']=[1 if st.sidebar.checkbox('has_secondary_use_use_police', value=False)  else 0]
        data2['has_secondary_use_other']=[1 if st.sidebar.checkbox('has_secondary_use_other', value=False)  else 0]
        return data2
    input_df = user_input_features()


df=input_df
# Displays the user input features
st.subheader('User Input features')

#geo_level_id_1
damage1 = {}
damage2 = {}
damage3 = {}
with open('damage1_geo_1.pickle','rb') as read_file11:
    damage1 = pickle.load(read_file11)
with open('damage2_geo_1.pickle','rb') as read_file12:
    damage2 = pickle.load(read_file12)
with open('damage3_geo_1.pickle','rb') as read_file13:
    damage3 = pickle.load(read_file13)
list1 = []
list2 = []
list3 = []
for i in df['geo_level_1_id']:
    list1.append(damage1.get(i))
    list2.append(damage2.get(i))
    list3.append(damage3.get(i))
df['prob1_geo1'] = list1
df['prob2_geo1'] = list2
df['prob3_geo1'] = list3
#geo_level_id_1
#geo_level_id_2
with open('damage1_geo_2.pickle','rb') as read_file21:
    damage1 = pickle.load(read_file21)
with open('damage2_geo_2.pickle','rb') as read_file22:
    damage2 = pickle.load(read_file22)
with open('damage3_geo_2.pickle','rb') as read_file23:
    damage3 = pickle.load(read_file23)
list1 = []
list2 = []
list3 = []
for i in df['geo_level_2_id']:
    list1.append(damage1.get(i))
    list2.append(damage2.get(i))
    list3.append(damage3.get(i))
df['prob1_geo2'] = list1
df['prob2_geo2'] = list2
df['prob3_geo2'] = list3
#geo_level_id_2
#geo_level_id_3
with open('damage1_geo_3.pickle','rb') as read_file31:
    damage1 = pickle.load(read_file31)
with open('damage2_geo_3.pickle','rb') as read_file32:
    damage2 = pickle.load(read_file32)
with open('damage3_geo_3.pickle','rb') as read_file33:
    damage3 = pickle.load(read_file33)
list1 = []
list2 = []
list3 = []
for i in df['geo_level_3_id']:
    list1.append(damage1.get(i))
    list2.append(damage2.get(i))
    list3.append(damage3.get(i))
df['prob1_geo3'] = list1
df['prob2_geo3'] = list2
df['prob3_geo3'] = list3
#geo_level_id_3
#plan_configuration
with open('damage1_plan.pickle','rb') as read_file41:
    damage1 = pickle.load(read_file41)
with open('damage2_plan.pickle','rb') as read_file42:
    damage2 = pickle.load(read_file42)
with open('damage3_plan.pickle','rb') as read_file43:
    damage3 = pickle.load(read_file43)
list1 = []
list2 = []
list3 = []
for i in df['plan_configuration']:
    list1.append(damage1.get(i))
    list2.append(damage2.get(i))
    list3.append(damage3.get(i))
df['prob1_plan'] = list1
df['prob2_plan'] = list2
df['prob3_plan'] = list3
#plan_configuration

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)


# Reads in saved classification model
load_clf = pickle.load(open('model.pickle', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
damages = np.array(['Damage Grade 1','Damage Grade 2','Damage Grade 3'])
st.write(damages[prediction-1])

st.subheader('Prediction Probability')
st.write(prediction_proba)

img = iio.imread("index.png")
st.image(img)