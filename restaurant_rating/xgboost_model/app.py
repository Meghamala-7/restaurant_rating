import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

@st.cache_data
def load_data():
    df = pd.read_csv("Dataset .csv") 
    return df

df = load_data()

features = ['Country Code', 'City', 'Cuisines', 'Currency', 'Has Table booking',
            'Has Online delivery', 'Is delivering now', 'Price range', 'Votes']
target = 'Aggregate rating'
country_code_options = sorted(df['Country Code'].dropna().unique())
currency_options = sorted(df['Currency'].dropna().unique())

@st.cache_resource
def train_model(data):
    X = data[features].copy()
    y = data[target]

    for col in ['Has Table booking', 'Has Online delivery', 'Is delivering now']:
        X[col] = X[col].map({'Yes': 1, 'No': 0}).fillna(0)

    label_encoders = {}
    for col in ['City', 'Cuisines', 'Currency']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, label_encoders

model, scaler, label_encoders = train_model(df)

st.title("🍽️ Restaurant Rating Prediction (XGBoost)")
st.markdown("Enter restaurant details to predict **Aggregate Rating** using an optimized XGBoost model.")

st.subheader("📝 Input Restaurant Details")

selected_country = st.selectbox("Country Code", country_code_options)
filtered_data = df[df['Country Code'] == selected_country]

city_options = sorted(filtered_data['City'].dropna().unique())
selected_city = st.selectbox("City", city_options)

filtered_cuisines = filtered_data[filtered_data['City'] == selected_city]['Cuisines'].dropna().unique()
cuisine_options = sorted(filtered_cuisines)
selected_cuisine = st.selectbox("Cuisines", cuisine_options)

currency_options = sorted(filtered_data['Currency'].dropna().unique())
selected_currency = st.selectbox("Currency", currency_options)

has_table_booking = 1 if st.selectbox("Has Table booking", ["Yes", "No"]) == "Yes" else 0
has_online_delivery = 1 if st.selectbox("Has Online delivery", ["Yes", "No"]) == "Yes" else 0
is_delivering_now = 1 if st.selectbox("Is delivering now", ["Yes", "No"]) == "Yes" else 0
price_range = st.number_input("Price range", min_value=1, max_value=4, value=2)
votes = st.number_input("Votes", min_value=0, value=100)

input_df = pd.DataFrame([{
    'Country Code': selected_country,
    'City': selected_city,
    'Cuisines': selected_cuisine,
    'Currency': selected_currency,
    'Has Table booking': has_table_booking,
    'Has Online delivery': has_online_delivery,
    'Is delivering now': is_delivering_now,
    'Price range': price_range,
    'Votes': votes
}])

def preprocess(df_input):
    df_proc = df_input.copy()
    for col, le in label_encoders.items():
        df_proc[col] = df_proc[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    df_proc = scaler.transform(df_proc)
    return df_proc

if st.button("Predict Rating"):
    try:
        processed = preprocess(input_df)
        pred = model.predict(processed)
        pred_clipped = max(1, min(5, pred[0]))
        st.success(f"⭐ Predicted Aggregate Rating: **{pred_clipped:.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
