# IMDb Movie Rating Prediction  

This project is part of my **Data Science Internship at CodSoft**.  
The goal is to predict IMDb movie ratings using machine learning models, based on features like **Year, Duration, Genre, Director, Actors, and Votes**.  



## Project Structure
- `IMDb Movies India.csv` → Dataset with Indian movies and their details  
- `IMDb_RATING.PY` → Python script for data preprocessing, model training, and evaluation  

---

## Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn  



## Steps in the Project
1. Load dataset (`IMDb Movies India.csv`)  
2. Data cleaning & preprocessing  
   - Handle missing values  
   - Convert `Year`, `Duration`, `Votes` into proper numeric formats  
   - Extract primary genre  
   - Encode categorical columns (Genre, Director, Actors)  
3. Train-test split (80:20)  
4. Train **Random Forest Regressor**  
5. Evaluate performance with:  
   - Mean Absolute Error (MAE)  
   - Mean Squared Error (MSE)  
   - R² Score  
6. Analyze **feature importance**  
7. Visualize **feature importance** and **Actual vs Predicted ratings**  



## Results
- Model achieved strong predictive performance with **Random Forest Regressor**  
- Key insights on which factors influence IMDb ratings the most  
- Visualizations for feature importance and prediction accuracy  




