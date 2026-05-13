import pandas as pd
import pickle

with open("project.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
sfs = saved["sfs"]
scaler = saved["scaler"]
df_new = saved["df"]   

print("Model R2 Score:", saved["r2"])

# Numerical columns
num_cols = [col for col in df_new.columns if df_new[col].nunique() > 2]
num_cols.remove("Price")  

print("\nENTER PROPERTY DETAILS:\n")

# Inputs with proper units
Property_Area = float(input("Property Area (in square feet): "))
Year_Built = int(input("Year Built (e.g. 2015): "))
Distance_to_City_Center = float(input("Distance to City Center (in km): "))
Neighborhood_Quality_Score = int(input("Neighborhood Quality Score (1-10): "))
Parking_Availability = int(input("Parking Availability (0 = No, 1 = Yes): "))
Construction_Quality_Rating = int(input("Construction Quality Rating (1-10): "))
Flood_Risk_Index = float(input("Flood Risk Index (0.0 to 1.0): "))

Renovation_Status = input("Renovation Status (Yes/No): ")
Property_Type = input("Property Type (Apartment/Villa/Independent House): ")

# Convert categorical values
Renovation_Status_Yes = 1 if Renovation_Status.lower() == "yes" else 0

Property_Type_Independent_House = 0
Property_Type_Villa = 0

if Property_Type.lower() == "independent house":
    Property_Type_Independent_House = 1
elif Property_Type.lower() == "villa":
    Property_Type_Villa = 1

# Create input dictionary
input_dict = {}

for col in df_new.drop("Price", axis=1).columns:
    input_dict[col] = 0  

# Update with user values
input_dict.update({
    'Property_Area': Property_Area,
    'Year_Built': Year_Built,
    'Distance_to_City_Center': Distance_to_City_Center,
    'Neighborhood_Quality_Score': Neighborhood_Quality_Score,
    'Parking_Availability': Parking_Availability,
    'Construction_Quality_Rating': Construction_Quality_Rating,
    'Flood_Risk_Index': Flood_Risk_Index,
    'Renovation_Status_Yes': Renovation_Status_Yes,
    'Property_Type_Independent House': Property_Type_Independent_House,
    'Property_Type_Villa': Property_Type_Villa
})

# Convert to dataframe
input_df = pd.DataFrame([input_dict])

# Scale numerical data
input_scaled = input_df.copy()
input_scaled[num_cols] = scaler.transform(input_scaled[num_cols])

# Feature selection
input_selected = sfs.transform(input_scaled)

# Prediction
prediction = model.predict(input_selected)

print("\nPredicted House Price: ₹", round(prediction[0], 2))
