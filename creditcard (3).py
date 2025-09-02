
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load data
data = pd.read_csv('credit_card.csv.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create a dictionary to store model names and their corresponding models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# Create a dictionary to store accuracy scores
accuracy_scores = {}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[model_name] = accuracy

color_map = {
    "Logistic Regression": "green",
    "Random Forest": "blue",
    "Decision Tree": "red"
}

# Streamlit app
st.title("Credit Card Fraud Detection Different Algorithms")
st.subheader("Model Comparison")

# Display accuracy scores
st.write("Model Accuracy Scores:")
st.write(accuracy_scores)


# train logistic regression model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

st.title("Credit Card Fraud Detection Using RandomForestClassifier Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.title("Legitimate transaction")
    else:
        st.title("Fraudulent transaction")


# Plot bar chart for accuracy comparison
fig, ax = plt.subplots(figsize=(6, 2))
bars = ax.bar(accuracy_scores.keys(), accuracy_scores.values(), color=[color_map[model] for model in accuracy_scores.keys()])
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
plt.xticks()
st.pyplot(fig)


   # Plot line graph for accuracy comparison
fig, ax = plt.subplots(figsize=(8, 4))  # Adjust figsize as needed
for model_name, color in color_map.items():
    ax.plot(list(accuracy_scores.keys()), list(accuracy_scores.values()), marker='o', linestyle='-', color=color, label=model_name)
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
ax.legend()
plt.xticks()  # Rotate x-axis labels for better readability

# Instead of st.pyplot(), pass the figure directly
st.pyplot(fig)

# Reset the Matplotlib figure after using it with Streamlit
plt.close(fig)

# Confusion Matrix Heatmaps
for model_name, model in models.items():
    st.write(f"Confusion Matrix for {model_name}:")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual Legit', 'Actual Fraud'], columns=['Predicted Legit', 'Predicted Fraud'])
    st.write(cm_df)

    # Plot confusion matrix as heatmap
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d', cbar=False, ax=ax)
    ax.set_title(f'Confusion Matrix for {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)


st.write("Classification Report:")
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(f"Classification Report for {model_name}:")
    st.dataframe(report_df.style.background_gradient(cmap='coolwarm'))

# Histogram Graph
st.title("Histogram for Transaction Amount")
fig, ax = plt.subplots()
sns.histplot(data['Amount'], bins=30, kde=True, ax=ax)
ax.set_xlabel('Amount')
ax.set_ylabel('Frequency')
st.pyplot(fig) 

!streamlit run app.py & npx localtunnel --port 8501




