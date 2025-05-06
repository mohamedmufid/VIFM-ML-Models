import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from io import StringIO
import graphviz

st.set_page_config(layout="wide")
st.title("AI Fundamentals Simulator")
st.markdown("Upload your dataset, choose an algorithm, and explore the results.")

# Algorithm options
model_options = [
    "Decision Tree",
    "Linear Regression",
    "Logistic Regression",
    "Confusion Matrix",
    "K-Means",
    "House Price Prediction"
]

choice = st.selectbox("Select Machine Learning Model", model_options)
file = st.file_uploader("Upload the corresponding dataset", type=["csv", "xlsx"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    if choice == "Decision Tree":
        X = pd.get_dummies(df.drop(columns=['accepted']), drop_first=True)
        y = df['accepted']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        dot_data = StringIO()
        export_graphviz(
            model,
            out_file=dot_data,
            feature_names=X.columns,
            class_names=['Reject', 'Accept'],
            filled=True,
            rounded=True,
            special_characters=True
        )
        st.success(f"Model Accuracy: {acc * 100:.2f}%")
        st.graphviz_chart(dot_data.getvalue())

    elif choice == "Linear Regression":
        df['education_level_numeric'] = df['education_level'].map({'Bachelors': 1, 'Masters': 2, 'PhD': 3})
        X = df[['years_experience', 'education_level_numeric']]
        y = df['salary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        experience_range = np.linspace(df['years_experience'].min(), df['years_experience'].max(), 200).reshape(-1, 1)
        line_input = np.hstack([experience_range, np.ones_like(experience_range) * df['education_level_numeric'].mean()])
        prediction = model.predict(line_input)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs[0].scatter(df['years_experience'], df['salary'], alpha=0.6)
        axs[0].plot(experience_range, prediction, color='red')
        axs[0].set_title("Experience vs. Salary")

        sns.boxplot(x='education_level', y='salary', data=df, palette='Set2', ax=axs[1])
        axs[1].set_title("Education Level vs. Salary")
        st.pyplot(fig)

    elif choice == "Logistic Regression":
        X = df[['price_sensitivity']]
        y = df['purchase']
        model = LogisticRegression()
        model.fit(X, y)

        X_test_vals = np.linspace(1, 10, 300).reshape(-1, 1)
        y_prob = model.predict_proba(X_test_vals)[:, 1]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X, y, alpha=0.6)
        ax.plot(X_test_vals, y_prob, color='red')
        ax.axvline(-model.intercept_[0] / model.coef_[0][0], color='green', linestyle='--')
        ax.set_title("Logistic Regression: Purchase Decision")
        st.pyplot(fig)

    elif choice == "Confusion Matrix":
        X = df.drop(columns=['Outcome'])
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.success(f"Model Accuracy: {acc * 100:.2f}%")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    elif choice == "K-Means":
        cluster_df = df.drop(columns=['Transaction_ID', 'Is_Fraudulent'])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_df)

        kmeans = KMeans(n_clusters=2, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(df['Transaction_Amount'], df['Transaction_Time'], c=df['Cluster'], cmap='viridis', alpha=0.6)
        ax1.set_xlabel('Transaction Amount')
        ax1.set_ylabel('Transaction Time')
        ax1.set_title('K-means Clustering')
        st.pyplot(fig1)

        fig2 = sns.pairplot(df, vars=['Transaction_Amount', 'Transaction_Time', 'Merchant_ID', 'Device_ID'], hue='Cluster', palette='viridis')
        st.pyplot(fig2)

    elif choice == "House Price Prediction":
        st.subheader("🏠 House Price Prediction")

        house_file = st.file_uploader("Upload house dataset", type=["xlsx"], key="housefile")

        if house_file:
            house_df = pd.read_excel(house_file)
            st.write("### Dataset Preview")
            st.dataframe(house_df.head())

            if {'size', 'bedrooms', 'bathrooms', 'price'}.issubset(house_df.columns):
                features = house_df[['size', 'bedrooms', 'bathrooms']].dropna()
                target = house_df['price'].loc[features.index]

                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
                house_model = LinearRegression()
                house_model.fit(X_train, y_train)
                y_pred = house_model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred) ** 0.5

                st.success(f"Model trained successfully!\n\n**MAE:** {mae:.2f}\n\n**RMSE:** {rmse:.2f}")

                st.markdown("### 📥 Enter New House Features")
                col1, col2, col3 = st.columns(3)
                with col1:
                    size = st.number_input("Size (sqm)", min_value=10.0, value=100.0)
                with col2:
                    bedrooms = st.number_input("Bedrooms", min_value=1, value=3, step=1)
                with col3:
                    bathrooms = st.number_input("Bathrooms", min_value=1, value=2, step=1)

                if st.button("Predict Price"):
                    prediction = house_model.predict([[size, bedrooms, bathrooms]])[0]
                    st.success(f"💰 Predicted Price: **{prediction:,.2f}**")
            else:
                st.warning("Dataset must include columns: size, bedrooms, bathrooms, and price.")

    else:
        st.warning("Unsupported model selected.")