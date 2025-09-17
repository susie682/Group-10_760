# Aurora Ground Visibility Prediction

## Project Overview

This project, developed by Group 10 at the University of Auckland, aims to predict the intensity of auroras visible from the ground. Our goal is to serve tourists, photographers, and aurora enthusiasts by providing accurate ground-level aurora forecasts. Unlike most existing models that rely solely on satellite data, we incorporate ground-based weather and environmental factors to make predictions closer to real-world observations.

## Research Objectives & Motivation

- **Objective 1:** Predict the intensity of auroras observable from the ground.
- **Objective 2:** Integrate ground factors (e.g., weather) into satellite-based models for more realistic predictions.

Auroras result from interactions between the solar wind and Earth’s magnetic field. Studying them helps monitor space weather and supports tourism. However, most research focuses on satellite perspectives, which do not fully reflect ground visibility. Our project bridges this gap by combining satellite and ground data, including weather effects.

## Related Work & Our Innovations

- Previous studies either use the Kp index (a solar wind indicator) for satellite-view aurora prediction or apply machine learning to classify aurora images.
- Our approach:
  1. Fuse ground-based keogram brightness with satellite Rayleigh data to convert satellite intensity to ground intensity.
  2. Incorporate weather factors to model their impact on aurora visibility.
  3. Compare multiple machine learning models for ground-level aurora prediction.

## Data Sources

1. **Keogram Images:** Daily all-sky camera images for ground-based aurora observation. Pixel brightness represents auroral intensity, but images may contain noise (moonlight, clouds, twilight).
2. **DMSP Satellite Data:** Measures auroral radiance, includes magnetic coordinates and timestamps, used to calibrate ground brightness.
3. **Kp Index:** Global geomagnetic activity index, recorded every three hours, used to align ground and satellite data.
4. **ERA5 Weather Data:** Includes cloud coverage, aerosols, and humidity, used to model weather attenuation of aurora visibility.

The dataset is split into 6 years for training, 1 year for validation, and 2 years for testing.

## Methodology

1. **Data Preprocessing**
   - Divide each keogram image into eight horizontal segments to match Kp intervals.
   - Remove invalid periods (dusk, dawn, black frames).
   - Reduce moonlight influence, extract and average pixel brightness over three-hour windows.
   - Merge Kp, weather, and aurora datasets.

2. **Weather Effect Modeling**
   - Train a model to analyze how weather factors (clouds, humidity) reduce aurora intensity from space to ground level.

3. **Prediction**
   - Use a pre-trained public model for satellite-level aurora intensity.
   - Apply the weather model to output ground-level aurora intensity for observers.

## Machine Learning Models

- **Random Forest:** Handles tabular and non-linear data, robust to noise and missing values, interpretable via feature importance and SHAP values.
- **XGBoost:** Uses gradient boosting and regularization, interpretable, robust to missing data, highlights key factors.
- **CNN:** Excels at spatial feature fusion, scale robustness, and capturing temporal patterns.

All models are trained on the same data, tested under different Kp levels, and tuned for hyperparameters (tree number/depth, learning rate, CNN layers/filters). Performance is measured by R² (closer to 1 is better) and Mean Squared Error (lower is better).

## Challenges & Solutions

- **Satellite-Ground Data Alignment:** If fusion fails, we provide a ground-only version using geomagnetic, weather, and aurora logs, all aligned to three-hour windows.
- **Class Imbalance & Label Noise:** We upweight aurora cases, train on nighttime and clear-sky records, and set a conservative alert threshold to reduce false alarms.

## Project Timeline

- **Phase 1:** Data processing and feature extraction (keogram preprocessing, data fusion).
- **Phase 2:** Parallel training and evaluation of XGBoost, Random Forest, and CNN models.
- **Final Phase:** Results integration, model comparison, code and data documentation, and report preparation.

---

For code usage and experiment reproduction, please refer to the scripts and comments in each subdirectory. Contributions and suggestions are welcome!

---

# Group-10_760