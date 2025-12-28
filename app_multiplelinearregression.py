import streamlit as st 
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,root_mean_squared_error,adjusted_rand_score


st.set_page_config("Multiple Linear regression",layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("style.css")

#Title
st.markdown("""
 <div class="card">
 <h1>Multiple Linear Regression </h1>
 <p> Predict <b>Tip Amount</b> from <b> Total Bill </b> using Linear Regression </p>
 </div>
 """,unsafe_allow_html=True)

 # Load Data
@st.cache_data
def load_data():
    data = sns.load_dataset("tips")
    return data
df=load_data()

#Dataset Preview
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[["total_bill","size","tip"]].head())
st.markdown('</div>',unsafe_allow_html=True)

# Prepare Data
x,y=df[["total_bill","size"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# Train Model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Metrics
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
rmse=numpy.sqrt(mse)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)


#Visualization
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Total Bill vs Tip(with multiple linear regression) ")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha=0.6)
ax.plot(df["total_bill"],model.predict(scaler.transform(x)),color="red")
ax.set_xlabel("Total bill($)")
ax.set_ylabel("Tip($)")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

# performance Metrics
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader('Model Performance')
c1,c2=st.columns(2)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("RMSE",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("R² Score",f"{r2:.2f}")
c4.metric("Adjusted R²",f"{adj_r2:.2f}")
st.markdown('</div>',unsafe_allow_html=True)

# Intercept and Coefficient
st.markdown(f"""
<div class="card">
<h3>Model Interception</h3>
<p><b>co-efficient:</b> {model.coef_[0]:.3f}<br>
<b>co-efficient(Group size):</b> {model.coef_[1]:.3f}<br>
<b>Intercept:</b> {model.intercept_:.3f}</p>
<p> Tip depends on the<b>Bill Amount</b> and <b>Group Size</b> </p>
</div>
""",unsafe_allow_html=True)


#Prediction
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Predict Tip amount")
bill=st.slider("Total bill($)",float(df.total_bill.min()),float(df.total_bill.max()),30.0)
size=st.slider("Group size",int(df["size"].min()),int(df["size"].max()),2)
tip=model.predict(scaler.transform([[bill,size]]))[0]
st.markdown(f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)
import matplotlib.pyplot as plt
import seaborn as sns