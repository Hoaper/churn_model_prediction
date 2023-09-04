import streamlit as st

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle


# Настройка страницы
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

if 'df_input' not in st.session_state:
    st.session_state['df_input'] = pd.DataFrame()

if 'df_predicted' not in st.session_state:
    st.session_state['df_predicted'] = pd.DataFrame()

numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender',
'seniorcitizen',
'partner',
'dependents',
'phoneservice',
'multiplelines',
'internetservice',
'onlinesecurity',
'onlinebackup',
'deviceprotection',
'techsupport',
'streamingtv',
'streamingmovies',
'contract',
'paperlessbilling',
'paymentmethod']
le_enc_cols = ['gender', 'partner', 'dependents','paperlessbilling', 'phoneservice']
gender_map = {'male': 0, 'female': 1}
y_n_map = {'yes': 1, 'no': 0}

# Логистическая модель
model_file_path = "models/lr_model_churn_prediction.sav"
model = pickle.load(open(model_file_path, 'rb'))

encoding_model_file_path = "models/lr_model_churn_prediction_encode.sav"
encoding_model = pickle.load(open(encoding_model_file_path, 'rb'))

@st.cache_data
def convert_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def predict_churn(data, treshold):
    scaler = MinMaxScaler()

    df_copy = data.copy()
    df_copy[numerical] = scaler.fit_transform(df_copy[numerical])

    for col in le_enc_cols:
        if col == 'gender':
            df_copy[col] = df_copy[col].map(gender_map)
        else:
            df_copy[col] = df_copy[col].map(y_n_map)

    dicts_df = df_copy[categorical + numerical].to_dict(orient='records')
    X = encoding_model.transform(dicts_df)
    y_pred = model.predict_proba(X)[:, 1]
    churn_descision = (y_pred >= treshold).astype(int)

    df_copy['churn_predicted'] = churn_descision
    df_copy['churn_predicted_probability'] = y_pred

    return df_copy



# SIDEBAR START
with st.sidebar:
    st.title("Ввод данных")

    tab1, tab2 = st.tabs(["Данные из файла", "Ввести вручную"])

    with tab1:
        uploaded_files = st.file_uploader("Выберите CSV файл", accept_multiple_files=False, type=["csv", "xlsx"])
        if uploaded_files is not None:
            treshold = st.slider('Порог вероятности оттока', 0.0, 1.0, 0.5, 0.01, key="treshold")
            predict_button = st.button("Прогнозировать", type="primary", key="predict_button", use_container_width=True)
            st.session_state['df_input'] = pd.read_csv(uploaded_files)

            if predict_button:
                st.session_state['df_predicted'] = predict_churn(st.session_state['df_input'], treshold)
                

    with tab2:
        pass

# SIDEBAR END

# MAIN SECTION START
st.image('https://miro.medium.com/v2/resize:fit:1400/1*WqId29D5dN_8DhiYQcHa2w.png', width=400)
st.header("Прогнозирование оттока клиентов")
with st.expander("Описание проекта"):
    st.write(
        """В данном проекте мы рассмотрим задачу прогнозирования оттока клиентов.
        Для этого мы будем использовать датасет из открытых источников.
        Датасет содержит информацию о клиентах, которые уже ушли или остались в компании.
        Наша задача - построить модель, которая будет предсказывать отток клиентов.
        """)

# Выводим входные данные клиентов.
if len(st.session_state['df_input']) > 0:
    
    if len(st.session_state['df_predicted']) == 0:
        st.subheader("Данные из файла")
        st.write(st.session_state['df_input'])
    else:
        with st.expander("Входные данные"):
            st.write(st.session_state['df_input'])


# Выводим результаты предсказания из файла.
if len(st.session_state['df_predicted']) > 0:
    st.subheader("Результаты прогнозирования оттока клиентов")
    st.write(st.session_state['df_predicted'])
    res_all_csv = convert_to_csv(st.session_state['df_predicted'])
    st.download_button(
        "Скачать результаты",
        data=res_all_csv,
        file_name="result.csv",
        mime="text/csv",
    )

    


# Выводим результаты предсказания для отдельного клиента.