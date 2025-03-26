from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

with open("https://raw.githubusercontent.com/youssef-azam-tach/Tourism-App/main/tourism_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("https://raw.githubusercontent.com/youssef-azam-tach/Tourism-App/main/label_encoders_for_Gradient.pkl", "rb") as encoder_file:
    label_encoders = pickle.load(encoder_file)

mapping = {
    'Country': {0: 'Australia', 1: 'Canada', 2: 'China', 3: 'France', 4: 'Germany', 5: 'India', 6: 'UK', 7: 'USA'},
    'Gender': {0: 'Female', 1: 'Male', 2: 'Other'},
    'Travel_Purpose': {0: 'Business', 1: 'Education', 2: 'Leisure', 3: 'Medical'},
    'Preferred_Destination': {0: 'Adventure Park', 1: 'Beach', 2: 'City', 3: 'Countryside', 4: 'Mountain'},
    'Accommodation_Type': {0: 'Airbnb', 1: 'Hostel', 2: 'Hotel', 3: 'Resort'},
    'With_Family': {0: 'With Family', 1: 'Without Family'},
    'Tourism Type': {0: 'Adventure Program', 1: 'Beach Program', 2: 'Historical Program', 3: 'Major Cities Program', 4: 'Relaxation Program'}
}

# تعريف FastAPI
app = FastAPI()

class TourismInput(BaseModel):
    country: str
    Age: int
    gender: str
    travel_purpose: str
    preferred_destination: str
    accommodation_type: str
    stay_duration: int
    spending_usd: int
    travel_frequency: int
    avg_spending_accommodation: int
    avg_spending_transport: int
    avg_spending_food: int
    avg_cost_per_day_aed: int
    with_family: str

def transform_input(input_data: TourismInput):
    new_sample = {
        'Country': input_data.country,
        'Age': input_data.Age,  # يمكن تعديله
        'Gender': input_data.gender,
        'Travel_Purpose': input_data.travel_purpose,
        'Preferred_Destination': input_data.preferred_destination,
        'Stay_Duration_Days': input_data.stay_duration,
        'Spending_USD': input_data.spending_usd,
        'Accommodation_Type': input_data.accommodation_type,
        'Travel_Frequency_per_Year': input_data.travel_frequency,
        'Average_Spending_Accommodation_USD': input_data.avg_spending_accommodation,
        'Average_Spending_Transport_USD': input_data.avg_spending_transport,
        'Average_Spending_Food_USD': input_data.avg_spending_food,
        'Average_Cost_Per_Day_AED': input_data.avg_cost_per_day_aed,
        'With_Family': input_data.with_family
    }

    for col in label_encoders.keys():
        if col in new_sample:
            new_sample[col] = label_encoders[col].transform([new_sample[col]])[0]

    feature_columns = [col for col in new_sample.keys() if col != 'Tourism Type']
    new_sample_values = np.array([[new_sample.get(col, 0) for col in feature_columns]])
    
    return new_sample_values

@app.post("/predict_tourism_type/")
def predict_tourism_type(input_data: TourismInput):
    new_sample_values = transform_input(input_data)
    
    predicted_class = model.predict(new_sample_values)
    predicted_label = label_encoders['Tourism Type'].inverse_transform(predicted_class)
    
    return {"predicted_tourism_type": predicted_label[0]}
