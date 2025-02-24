from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__)
CORS(app)

def load_and_prepare_data():
    # Read CSV data
    df = pd.read_csv('1900_2021_DISASTERS.xlsx - emdat data.csv')
    
    # Convert Year columns to datetime
    df['Start Year'] = pd.to_datetime(df['Start Year'], format='%Y')
    
    return df

def calculate_disaster_probability(historical_data, future_years=10):
    predictions = {}
    probabilities = {}
    current_year = datetime.now().year
    
    for country in historical_data['Country'].unique():
        country_data = historical_data[historical_data['Country'] == country]
        
        # Calculate frequency of each disaster type
        disaster_counts = country_data['Disaster Type'].value_counts()
        total_years = current_year - 1900
        
        # Calculate probability for each disaster type
        for disaster_type in disaster_counts.index:
            yearly_probability = disaster_counts[disaster_type] / total_years
            
            # Adjust probability based on recent trends (last 20 years)
            recent_data = country_data[
                country_data['Start Year'].dt.year >= (current_year - 20)
            ]
            recent_count = len(recent_data[recent_data['Disaster Type'] == disaster_type])
            recent_probability = recent_count / 20
            
            # Weighted average of historical and recent probabilities
            adjusted_probability = (0.3 * yearly_probability + 0.7 * recent_probability)
            
            # Store probabilities
            if country not in probabilities:
                probabilities[country] = {}
            probabilities[country][disaster_type] = adjusted_probability
    
    return probabilities

def predict_future_disasters(probabilities, threshold=0.3):
    current_year = datetime.now().year
    future_years = range(current_year + 1, current_year + 11)
    predictions = []
    
    for country in probabilities:
        for disaster_type, probability in probabilities[country].items():
            for year in future_years:
                # Add random variation to probability (±10%)
                varied_prob = probability * np.random.uniform(0.9, 1.1)
                # Normalize probability
                varied_prob = min(max(varied_prob, 0), 1)
                
                if varied_prob > threshold:
                    risk_level = 'High' if varied_prob > 0.6 else 'Medium'
                    predictions.append({
                        'Country': country,
                        'Year': year,
                        'Disaster_Type': disaster_type,
                        'Probability': varied_prob,
                        'Risk_Level': risk_level
                    })
    
    return pd.DataFrame(predictions)

def create_visualizations(predictions_df, country=None):
    if country:
        predictions_df = predictions_df[predictions_df['Country'] == country]
    
    graphs = []
    
    # 1. Line plot for probability trends
    line_data = []
    for country in predictions_df['Country'].unique():
        country_data = predictions_df[predictions_df['Country'] == country]
        mean_probs = country_data.groupby('Year')['Probability'].mean() * 100
        
        line_data.append({
            'x': mean_probs.index.tolist(),
            'y': mean_probs.values.tolist(),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': country
        })
    
    graphs.append({
        'data': line_data,
        'layout': {
            'title': 'Disaster Probability Trends by Country',
            'xaxis': {'title': 'Year'},
            'yaxis': {'title': 'Probability (%)'}
        }
    })
    
    # 2. Bar plot for disaster types
    disaster_stats = predictions_df.groupby('Disaster_Type')['Probability'].mean() * 100
    
    bar_data = [{
        'x': disaster_stats.index.tolist(),
        'y': disaster_stats.values.tolist(),
        'type': 'bar'
    }]
    
    graphs.append({
        'data': bar_data,
        'layout': {
            'title': 'Average Probability by Disaster Type',
            'xaxis': {'title': 'Disaster Type'},
            'yaxis': {'title': 'Probability (%)'}
        }
    })
    
    return graphs

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        country = data['country']
        year = int(data['year'])
        
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Calculate probabilities
        probabilities = calculate_disaster_probability(df)
        
        # Generate predictions
        predictions_df = predict_future_disasters(probabilities)
        
        # Filter predictions for the specified country and year
        filtered_predictions = predictions_df[
            (predictions_df['Country'] == country) & 
            (predictions_df['Year'] == year)
        ]
        
        # Prepare predictions response
        predictions_list = []
        for _, row in filtered_predictions.iterrows():
            predictions_list.append({
                'disaster_type': row['Disaster_Type'],
                'probability': f"{row['Probability']*100:.1f}",
                'risk_level': row['Risk_Level']
            })
        
        # Generate graphs
        graphs = create_visualizations(predictions_df, country)
        
        return jsonify({
            'predictions': predictions_list,
            'graphs': graphs
        })

    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)