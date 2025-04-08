from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load pre-trained ML model (assuming it's saved as 'model.pkl')

# with open("model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template("chatbot.html")  # Connects to your UI

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        item_mrp = float(request.form['item_mrp'])
        outlet_type = int(request.form['outlet_type'])
        outlet_location = int(request.form['outlet_location'])
        outlet_size = request.form['outlet_size']
        fat_content = request.form['fat_content']
        year_established = int(request.form['year_established'])
        
        # Convert categorical values to numerical (simplified encoding)
        outlet_size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
        fat_content_map = {'Low Fat': 0, 'Regular': 1}
        
        outlet_size = outlet_size_map.get(outlet_size, 0)
        fat_content = fat_content_map.get(fat_content, 0)
        
        # Prepare feature array
        features = np.array([[item_mrp, outlet_type, outlet_location, outlet_size, fat_content, year_established]])
        
        # Predict sales
        prediction = model.predict(features)[0]
        
        return render_template("chatbot.html", prediction=f"Predicted Sales: {round(prediction, 2)}")
    except Exception as e:
        return render_template("chatbot.html", error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)