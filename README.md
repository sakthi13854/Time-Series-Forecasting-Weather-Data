# Weather Forecast Project

This project demonstrates how to build a **time series forecasting model** for weather data using Python.

##  Project Overview
We use historical weather records (temperature, humidity, precipitation, wind speed) and apply a **SARIMAX time series model** to forecast future weather patterns.

The dataset provided contains records for **San Diego, Los Angeles, and Houston**. The model can be run for one city at a time, or you can extend it to build forecasts for all cities.

## 🚀 Features
- Load and process weather data from CSV  
- Handle multiple cities (filter by `Location`)  
- Resample daily averages for time series consistency  
- Train a SARIMAX forecasting model  
- Evaluate model performance with **MSE** and **RMSE**  
- Visualize actual vs predicted values  

## 📂 Project Structure

## 📑 About the Dataset
The dataset contains weather information for **three U.S. cities**:  
- **San Diego**  
- **Los Angeles**  
- **Houston**

Columns included:
- **Location** → City name (San Diego, Los Angeles, Houston)  
- **Date_Time** → Timestamp of the record (YYYY-MM-DD HH:MM:SS format)  
- **Temperature_C** → Temperature in Celsius  
- **Humidity_pct** → Relative humidity in percentage (%)  
- **Precipitation_mm** → Rainfall in millimeters (mm)  
- **Wind_Speed_kmh** → Wind speed in kilometers per hour (km/h)  

Example row:
San Diego,2025-01-01 00:00:00,18.5,55,0.0,14

