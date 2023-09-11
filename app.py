import streamlit as st

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import plotly.express as px


# Настройка страницы
st.set_page_config(

    page_title="Churn Prediction App",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body {
        background-color: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def reset_session_state():
    st.session_state['df_input'] = pd.DataFrame()
    st.session_state['df_predicted'] = pd.DataFrame()

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

    tab1, tab2 = st.tabs(['📁 Данные из файла', '📝 Ввести вручную'])

    with tab1:
        uploaded_files = st.file_uploader("Выберите CSV файл", accept_multiple_files=False, type=["csv", "xlsx"], on_change=reset_session_state)
        if uploaded_files is not None:
            treshold = st.slider('Порог вероятности оттока', 0.0, 1.0, 0.5, 0.01, key="treshold")
            predict_button = st.button("Прогнозировать", type="primary", key="predict_button", use_container_width=True)
            st.session_state['df_input'] = pd.read_csv(uploaded_files)

            if predict_button:
                st.session_state['df_predicted'] = predict_churn(st.session_state['df_input'], treshold)
                

    with tab2:
        # Вкладка с вводом данных в ручную
        customer_id = st.text_input("Customer ID", placeholder="Enter the customer ID")
        gender = st.selectbox("Gender", ("male", "female"))
        seniorcitizen = st.selectbox("Senior citizen?", ("yes", "no"))
        partner = st.selectbox("Partner", ("yes", "no"))
        dependents = st.selectbox("Dependents", ("yes", "no"))
        tenure = st.number_input("Tenure", min_value=1, max_value=99, value=1)
        phoneservice = st.selectbox("Phone services", ("yes", "no", "no_phone_service"))
        multiple_lines = st.selectbox("Multiple lines", ("no", "dsl", "fiber_optic"))
        internetservice = st.selectbox("Internet Service", ("no", "yes", "no_internet_service"))
        onlinesecurity = st.selectbox("Online Security", ("yes", "no", "no_internet_service"))
        onlinebackup = st.selectbox("Online Backup", ("yes", "no", "no_internet_service"))
        deviceprotection = st.selectbox("Device Protection", ("yes", "no", "no_internet_service"))
        techsupport = st.selectbox("Tech Support", ("yes", "no", "no_internet_service"))
        streamingtv = st.selectbox("Streaming TV", ("yes", "no", "no_internet_service"))
        streamingmovies = st.selectbox("Streaming Movies", ("yes", "no", "no_internet_service"))
        contract = st.selectbox("Contract", ("month-to-month", "one_year", "two_year"))
        paperlessbilling = st.selectbox("Paperless Billing", ("yes", "no"))
        paymentmethod = st.selectbox("Payment method", ("bank_transfer_(automatic)", "credit_card_(automatic)", "electronic_check", "mailed_check"))
        monthlycharges = st.number_input("Monthly charges", min_value=0, value=0)
        totalcharges = st.number_input("Total charges", min_value=0, value=0)

        if customer_id != '':
            treshold = st.slider('Порог вероятности оттока', 0.0, 1.0, 0.5, 0.01, key="treshold")
            predict_button_tab2 = st.button("Прогнозировать", type="primary", key="predict_button", use_container_width=True)

            if predict_button_tab2:

                st.session_state['df_input'] = pd.DataFrame({
                    'customerid': customer_id,
                    'gender': gender,
                    'seniorcitizen': 1 if seniorcitizen == "yes" else 0,
                    'partner': partner,
                    "dependents": dependents,
                    "tenure": tenure,
                    "phoneservice": phoneservice,
                    "multiplelines": multiple_lines,
                    "internetservice": internetservice,
                    "onlinesecurity": onlinesecurity,
                    "onlinebackup": onlinebackup,
                    "deviceprotection": deviceprotection,
                    "techsupport": techsupport,
                    "streamingtv": streamingtv,
                    "streamingmovies": streamingmovies,
                    "contract": contract,
                    "paperlessbilling": paperlessbilling,
                    "paymentmethod": paymentmethod,
                    "monthlycharges": monthlycharges,
                    "totalcharges": totalcharges
                }, index=[0])

                st.session_state['df_predicted'] = predict_churn(st.session_state['df_input'], treshold)
        

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

    fig = px.histogram(st.session_state['df_predicted'], x="churn_predicted")
    st.plotly_chart(fig, use_container_width=True)

    risk_clients = st.session_state['df_predicted'][st.session_state['df_predicted']['churn_predicted'] == 1]

    if len(risk_clients) > 0:
        st.subheader("Клиенты с высоким риском оттока")
        st.write(risk_clients)

        res_risk_clients_csv = convert_to_csv(risk_clients)
        st.download_button(
            "Скачать клиентов в зоне риска",
            data=res_risk_clients_csv,
            file_name="result_risk_clients.csv",
            mime="text/csv",
        )

