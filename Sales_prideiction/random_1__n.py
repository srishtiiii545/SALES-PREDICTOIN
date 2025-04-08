from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sqlite3 as sql
from flask_session import Session
app = Flask(__name__)

app.config["SESSION_PERMANENT"]= False
app.config["SESSION_TYPE"]= "filesystem"
Session(app)
# Load dataset
big_mart_data = pd.read_csv("train.csv")  # Replace with actual dataset path
n= len(big_mart_data["Item_Type"])
item_cat= []
for i in range(n):
    if big_mart_data["Item_Type"][i] not in item_cat:
        item_cat.append(big_mart_data["Item_Type"][i])

big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)

mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))


miss_values = big_mart_data['Outlet_Size'].isnull()


big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])


big_mart_data['Item_Fat_Content'] = big_mart_data['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})


colums1= ['Outlet_Type', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Fat_Content','Item_Type','Item_Identifier', 'Outlet_Identifier']
features = ['Item_Identifier','Item_Weight' ,'Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']
target = 'Item_Outlet_Sales'
# Encode categorical features
label_encoders = {}
for col in colums1:
    encoder = LabelEncoder()
    big_mart_data[col] = encoder.fit_transform(big_mart_data[col])  # Convert categories to numbers
    label_encoders[col] = encoder  # Save the encoder for later

    big_mart_data[col] = big_mart_data[col].astype(int)
# print(tuple(label_encoders["Item_Type"]))

# Split data into train and test sets
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']
model = XGBRegressor()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model.fit(X_train, Y_train)

n= len(big_mart_data["Item_Type"])
item_num= []
for i in range(n):
    if big_mart_data["Item_Type"][i] not in item_num:
        item_num.append(big_mart_data["Item_Type"][i])


# Train the XGBoost model
# model = XGBRegressor()
# model.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('homie.html')  
@app.route('/signin', methods= ['POST', 'GET'])
def signin():

    if request.method=="POST":
        Email= request.form['email']
        Pass= request.form['pass']
        # db= sql.connect(host="localhost",port=3306,database="Sales_Pric",user="root")
        db = sql.connect('sales_pric.db')
        c= db.cursor()
        data= c.execute(f"Select Name from users where Email = '{Email}'")
        name= data.fetchall()
        
        data=  c.execute(f"Select * from users where email = '{Email}'")
        user= data.fetchone()
        db.close()
        if user:
            if Pass == user[2]:
                session['name']=  name[0][0]
                return redirect('/')
            else:
                error= "Password is incorrect"
                return render_template('signin.html', error= error)
        else:
            error= "Email not found. please register first"
        
            return render_template('signin.html', error= error)
        
    
    return render_template('signin.html', prediction=None, error=None)

@app.route('/signup', methods= ['POST', 'GET'])
def signup():
    if request.method=="POST":
        Name= request.form['name']
        Email= request.form['email']
        Pass= request.form['pass']
        session['name']=  Name
        # db= sql.connect(host="localhost",port=3306,database="Sales_Pric",user="root")
        db= sql.connect('sales_pric.db')
        c= db.cursor()
        c.execute(f"insert into users values('{Name}','{Email}','{Pass}')")
        db.commit()
        db.close()
        return redirect('/')
    return render_template("signup.html")
        
@app.route('/home')
def home2():
    if not session.get('name'):
        return redirect('/')   
    return render_template('random_n.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
        if request.method=="POST":
            n= len(big_mart_data["Item_Type"])
            item_num= []
            for i in range(n):
                if big_mart_data["Item_Type"][i] not in item_num:
                    item_num.append(big_mart_data["Item_Type"][i])

            item_type_dic= {}
            for i in range(len(item_cat)):
                item_type_dic[item_cat[i]]= item_num[i]
            
        # Get form data
            Item_Type= (request.form['item_type'])
            Item_Type= item_type_dic[Item_Type]
            item_mrp = float(request.form['item_mrp'])
            outlet_location = (request.form['outlet_location'])


            if outlet_location == "Developing Urban Cities":
                outlet_location = 2
            elif outlet_location == "Secondary Cities or Emerging Urban Cities":
                outlet_location = 1
            else:
                outlet_location= 0


            outlet_type = (request.form['outlet_type'])

            if outlet_type == "Grocery Store":
                outlet_type= 0
            elif outlet_type== "Supermarket type 1":
                outlet_type= 1
            else:
                outlet_type= 2

            year_established = int(request.form['year_established'])
            outlet_size = (request.form['outlet_size'])

            if outlet_size == "Small":
                outlet_size= 2
            elif outlet_size == "Medium":
                outlet_size= 1
            else:
                outlet_size= 0


            fat_content= 0
        # outlet_size= 2
        # Encode categorical inputs
        # outlet_type = label_encoders['Outlet_Type'].transform([outlet_type])[0]
        # fat_content = label_encoders['Item_Fat_Content'].transform([fat_content])[0]
        # outlet_size = label_encoders['Outlet_Size'].transform([outlet_size])[0]
        # item_type= label_encoders['Item_Type'].transform([item_type])[0]
        
            Item_iden= 156
            avg_item_vis= np.average(big_mart_data["Item_Visibility"])
            Item_Weight= np.average(big_mart_data["Item_Weight"])
            Outlet_Iden= 9

        # imput dictionary 
            input_data = {
            'Item_Identifier':Item_iden,
            'Item_Weight': Item_Weight,
            'Item_Fat_Content': fat_content,
            'Item_Visibility':avg_item_vis,
            'Item_Type': Item_Type,
            'Item_MRP': item_mrp,
            'Outlet_Identifier':Outlet_Iden,
            'Outlet_Establishment_Year': year_established,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location,
            'Outlet_Type': outlet_type
            }
            

        # Create input data frame
            input_df = pd.DataFrame([input_data])
        
        # Predict sales
           
            predicted_sales = model.predict(input_df)
            cost_price= item_mrp * 0.8
            
            profit=  predicted_sales[0]- cost_price
            loss= 0
            if profit < 0:
                
                loss= abs(profit)
                profit= 0
            # if profit_or_loss < 0:
            #     profit_or_loss= abs(profit_or_loss)
            # return render_template('graph.html', prediction=f'Predicted Sales: ₹{predicted_sales[0]:,.2f}', pro_or_loss= profit_or_loss)
            # return redirect(url_for('get_data1', profit=profit, predicted_sales= predicted_sales[0], loss= loss))
            # return jsonify(predicted_sales, profit, loss)
            return render_template('graph.html', profit=f"{profit:,.2f}", predicted_sales= f"{predicted_sales[0]:,.2f}", loss= f"{loss:,.2f}")
        return render_template('random_n.html')

    # except Exception as e:
    #     return render_template('random_n.html', prediction=None, error=str(e))

# @app.route('/get_data')
# def get_data1(profit, predicted_sales, loss):
#     # Replace with actual predicted values
#     data = {
#         "labels": ["Sales", "Profit", "Loss"],  # Categories
#         "values": [predicted_sales, profit, loss]  # Replace these with dynamic values
#     }
#     return jsonify(data)


# @app.route('/graph', methods= ["GET", "POST"])
# def graph():
#     return render_template('graph.html')
@app.route('/logout')
def logout():
    session.pop('name')   
    return redirect('/')


# responses = {
#     "What are the best-selling products?": "Our best-selling products are currently packaged foods, dairy products, and household essentials.",
#     "Can you provide a sales report?": "Sure! Our latest sales report indicates a steady increase in demand for fresh produce and beverages.",
#     "Do you have any discounts available?": "Yes! We currently offer seasonal discounts and special deals on bulk purchases.",
#     "How can I improve my sales strategy?": "To optimize sales, focus on pricing strategies, promotional campaigns, and customer loyalty programs.",
#     "What factors affect Big Mart sales?": "Key factors include store location, product visibility, seasonal demand, and promotional offers.",
#     "Can you predict future sales?": "Yes! Using historical sales data and machine learning, we can estimate future sales trends with high accuracy."
# }

# Predefined queries and responses (intents)
intents = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "good morning", "good evening"],
        "responses": ["Hello! How can I assist you today?", "Hi there! What can I do for you?", "Hey! How can I help?"]
    },
    "help": {
        "patterns": ["help", "need help", "assist", "can you help me", "support"],
        "responses": ["I can assist you with answering general queries. What do you need help with?", "How can I assist you today? What do you need help with?"]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you", "take care"],
        "responses": ["Goodbye! Have a great day!", "See you later! Take care."]
    },
    "best-selling products":{
        "patterns":["best-selling products", "What are the best-selling products?", "What are the best selling products","What are your best-selling products?"],
        "responses": ["Our best-selling products are currently packaged foods, dairy products, and household essentials."]
    },
    "sales report":{
        "patterns":["Can you provide a sales report?", "sales report?","sales report","what are the sales report?","Can you provide a sales report"],
        "responses":["Sure! Our latest sales report indicates a steady increase in demand for fresh produce and beverages."]
    },
    "discounts":{
        "patterns":["Do you have any discounts available?", "Discount","Do you have any discounts available", "discount available"],
        "responses":["Yes! We currently offer seasonal discounts and special deals on bulk purchases."]
    },
    "sales strategy":{
        "patterns":["How can I improve my sales strategy?","How can I improve my sales strategy","improve my sales strategy"],
        "responses":["To optimize sales, focus on pricing strategies, promotional campaigns, and customer loyalty programs."]
    },
    "Big Mart sales":{
        "patterns":["What factors affect Big Mart sales?", "Big Mart sales","factors affecting Big Mart sales","What factors affect Big Mart sales"],
        "responses":["Key factors include store location, product visibility, seasonal demand, and promotional offers."]
    },
    "future sales":{
        "patterns":["Can you predict future sales?","Can you predict future sales"],
        "responses":["Yes! Using historical sales data and machine learning, we can estimate future sales trends with high accuracy."]
    },
    "basic":{
        "patterns":["what you do", "what you can do?","what you can do"],
        "responses":["I can assist you with answering general queries. What do you need help with?", "How can I assist you today? What do you need help with?"]
    },
    "default": {
        "patterns": [],
        "responses": ["Sorry, I didn't understand that. Can you please rephrase?", "I'm not sure how to help with that, could you ask something else?"]
    }

}


# @app.route('/chat', methods=["GET",'POST'])
# def chat():
#     user_message = request.json.get("message", "").strip()
#     bot_response = responses.get(user_message, "I'm still learning, but I'll help you with sales!")
#     return jsonify({"response": bot_response})
def get_intent(user_input):
    user_input = user_input.lower() # Do you have any discounts available?

    for intent, data in intents.items():
        for pattern in data['patterns']:
            if pattern.lower() in user_input:
                return intent
    
    return "default"


# Define a route for the main page (chat interface)

# @app.route('/chatbot')
# def chatbot():
#     return render_template('chatbot.html')

# Define a route for receiving user input and responding

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form.get('user_input')

    if user_input:
        intent = get_intent(user_input)
        response = intents[intent]['responses'][0]  # Get the first response from the matched intent
        return jsonify({'response': f"{response}"})
    else:
        return jsonify({'response': "Please type something to get a response."})


if __name__ == '__main__':
    app.run(debug=True)
