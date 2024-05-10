from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__, template_folder="template")

# Load the CSV file with predictions
predictions_df = pd.read_csv('testresult.csv')

@app.route('/')
def index():
    # Print the columns of the DataFrame
    print(predictions_df.columns)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            beds = float(request.form['beds'])
            baths = float(request.form['baths'])
            size = float(request.form['size'])
        except ValueError:
            return "Invalid input. Please enter numeric values for beds, baths, and size."

        # Filter the DataFrame based on user inputs
        query_result = predictions_df[(predictions_df['beds'] == beds) & 
                                      (predictions_df['baths'] == baths) & 
                                      (predictions_df['size'] == size)]
        
        # Check if there's a matching row
        if not query_result.empty:
            predicted_price = query_result['PredictedPrice'].iloc[0]  # Corrected column name
            return render_template('result.html', predicted_price=predicted_price)
        else:
            return "No prediction found for the given inputs."

if __name__ == '__main__':
    app.run(debug=True)

