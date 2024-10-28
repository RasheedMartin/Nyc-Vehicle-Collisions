# NYC Motor Vehicle Collision Analysis

This repository contains an analysis of motor vehicle collisions in New York City, using a dataset of recorded accidents. The goal of this project is to predict accident severity based on features such as borough, contributing factors, and time-related attributes, and to provide visualizations of the accident data in the form of heatmaps.

## Repository Structure

```markdown
├── data_files/
│   ├── raw_data/
│   │   ├── Motor_Vehicle_Collisions_-_Crashes.csv   # Raw dataset of vehicle collision incidents in NYC
│   │   └── Motor_Vehicle_Collisions_-_Person.csv    # Raw dataset of person details involved in collisions
│   ├── location_data.csv                            # Dataset containing location information for modeling
│   ├── location_model_data.csv                      # Processed dataset for location-based modeling
│   ├── merged_data.csv                              # Merged dataset of vehicle collisions and person details
│   └── person_detail.csv                            # Dataset with person-specific details for analysis
│   └── Borough_Boundaries/
│       ├── geo_borough.cpg
│       ├── geo_borough.dbf
│       ├── geo_borough.prj
│       ├── geo_borough.shp
│       └── geo_borough.shx
├── src/
│   ├── 01_data_cleaning.ipynb                       # Notebook for data cleaning and preprocessing
│   ├── 02_accident_severity_model.ipynb             # Notebook for RandomForestClassifier model predicting accident severity
│   └── 03_heatmap_visualization.ipynb               # Notebook for creating heatmaps using Folium
├── visualizations/
│   ├── ny_accidents_fatal_heatmap.html              # HTML file containing a heatmap of fatal accidents
│   ├── severity_random_forest_Confusion_Matrix.png  # Confusion matrix for RandomForestClassifier model
│   ├── severity_random_forest_smote_Confusion_Matrix.png # Confusion matrix for RandomForest with SMOTE
│   └── severity_regression_Confusion_Matrix.png     # Confusion matrix for severity regression model
└── README.md                                        # Project documentation
```

## Notebooks Overview

### 1. Data Cleaning (01_data_cleaning.ipynb)
This notebook is responsible for loading the raw dataset and preparing it for modeling and visualization. The steps include:

- Handling missing values and removing irrelevant columns.
- Creating new features, such as:
  - ACCIDENT_SEVERITY: A new column that categorizes accidents into "Fatal," "Major Injury," "Minor Injury," and "No Injury" based on the number of people injured or killed.
  - Season: Categorizing accidents based on the season they occurred.
  - Year, Month, Day of the Week, and Weekend: Derived time-related features.
- Encoding categorical variables using LabelEncoder for fields such as contributing factors, boroughs, and accident severity.
- Splitting the data into training (70%) and testing (30%) sets.
- 
### 2. Accident Severity Prediction Model (Prediction Models.ipynb)
This notebook implements a RandomForestClassifier to predict the severity of accidents, using features such as location (borough), contributing factors, and time-related information.

- The model is configured with:
  - 100 estimators
  - Random state of 42
- **Performance Metrics (without SMOTE):**
  - **Accuracy**: 75%
  - **Precision**: 41.6%
  - **Recall**: 35.3%
  - **F1 Score**: 36.9%
  - The model performs well for "No Injury" cases, but struggles with minority classes like "Fatal" and "Major Injury" due to class imbalance.

- **Performance Metrics (with SMOTE applied):**
  - **Accuracy**: 69%
  - **Precision**: 42.9%
  - **Recall**: 47.9%
  - **F1 Score**: 31.5%
  - SMOTE improves recall for minority classes, especially "Fatal" cases, but reduces overall accuracy and precision.

- **Logistic Regression Baseline:**
  - **Accuracy**: 54.3%
  - **Precision**: 32.4%
  - **Recall**: 43.5%
  - **F1 Score**: 28.9%
  - Logistic regression has lower accuracy overall, but slightly better recall for some minority classes compared to RandomForest.

#### Observations:
- The RandomForestClassifier without SMOTE achieves the highest accuracy but struggles with recall for minority classes.
- Applying SMOTE increases recall for minority classes, though it results in a trade-off with precision and accuracy.
- The logistic regression model, while lower in accuracy, provides a balanced alternative but struggles with multi-class separation.

The confusion matrix and other visualizations can be found in the `visualizations/confusion_matrix.png` file. Future improvements may include hyperparameter tuning, experimenting with other resampling techniques, or testing more advanced models like XGBoost to handle class imbalance more effectively.

### 3. Heatmap Visualization (HeatMap.ipynb)

This notebook generates interactive heatmaps using the **folium** library to visualize accident hotspots across NYC.

- The heatmap highlights areas with a high concentration of accidents and allows users to explore patterns geographically.
- The output heatmap is saved as an HTML file (`visualizations/nyc_heatmap.html`) that can be viewed in any web browser.

---

## Visualizations Folder

The `visualizations/` folder contains key output files:

- `ny_accidents_fatal_heatmap.html`: An interactive heatmap showing fatal accident hotspots in New York City.
- `severity_random_forest_Confusion_Matrix.png`: A visual representation of the RandomForestClassifier's performance for predicting accident severity.
- `severity_random_forest_smote_Confusion_Matrix.png`: A confusion matrix for the RandomForest model enhanced with SMOTE (Synthetic Minority Over-sampling Technique).
- `severity_regression_Confusion_Matrix.png`: A confusion matrix for the severity regression model.

---

## How to Use the Repository

### 1. Clone the repository

```bash
git clone https://github.com/RasheedMartin/Nyc-Vehicle-Collisions.git
cd nyc-vehicle-collisions
```

### 2. Install dependencies

This project uses Python and the following libraries:

- **pandas**
- **numpy**
- **scikit-learn**
- **matplotlib**
- **seaborn**
- **folium**

Install the dependencies with:

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks

You can open and run each notebook in your preferred environment (Jupyter, VSCode, Google Colab, etc.).

### 4. Visualize the results

- Check the classification model performance by viewing the confusion matrix in `visualizations/confusion_matrix.png`.
- Open the `visualizations/nyc_heatmap.html` file in a browser to explore the heatmap.

---

## Future Work

- **Improve model performance**: The recall for "Major Injury" cases is low (6%). Future iterations could include hyperparameter tuning or experimenting with different models to improve classification performance.
- **Data enhancement**: Adding more features or balancing the dataset may help in resolving the class imbalance issues.
- **Additional visualizations**: Future efforts could include time-series analysis or spatial clustering to further analyze accident trends.

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
```

### Key Changes:
- Added horizontal lines for better section separation.
- Used code blocks for command line instructions and file paths.
- Made library names bold for emphasis.
- Ensured consistent formatting for lists and sections.