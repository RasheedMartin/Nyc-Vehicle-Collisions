# NYC Motor Vehicle Collision Analysis

This repository contains an analysis of motor vehicle collisions in New York City, using a dataset of recorded accidents. The goal of this project is to predict accident severity based on features such as borough, contributing factors, and time-related attributes, and to provide visualizations of the accident data in the form of heatmaps.

## Repository Structure
``
├── data/
│   └── nyc_motor_vehicle_collisions.csv  # Raw dataset of motor vehicle collisions in NYC
├── src/
│   ├── 01_data_cleaning.ipynb            # Notebook for data cleaning and preprocessing
│   ├── 02_accident_severity_model.ipynb  # Notebook for RandomForestClassifier model predicting accident severity
│   └── 03_heatmap_visualization.ipynb    # Notebook for creating heatmaps using Folium
├── visualizations/
│   ├── confusion_matrix.png              # Confusion matrix for the classification model
│   └── nyc_heatmap.html                  # HTML file containing an interactive heatmap of collision hotspots
└── README.md                             # Project documentation
``

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
### 2. Accident Severity Prediction Model (Prediction Models.ipynb)
This notebook implements a RandomForestClassifier to predict the severity of accidents. It uses features like location (borough), contributing factors, and time-related information.

- The model is configured with:
  - 100 estimators
  - Max depth of 5
  - Random state of 42
- Performance Metrics:
  - Accuracy: 98%
  - Precision: 98%
  - Recall: 76%
  - F1 Score: 76%
  - The model performs well overall, particularly with "Fatal" and "No Injury" classes. However, it struggles to differentiate "Major Injury" from "Minor Injury" cases, suggesting some areas for improvement.
  The confusion matrix can be found in the visualizations/confusion_matrix.png file.
### 3. Heatmap Visualization (HeatMap.ipynb)
This notebook generates interactive heatmaps using the folium library to visualize accident hotspots across NYC.

- The heatmap highlights areas with a high concentration of accidents and allows users to explore patterns geographically.
- The output heatmap is saved as an HTML file (visualizations/nyc_heatmap.html) that can be viewed in any web browser.
## Visualizations Folder

The visualizations/ folder contains key output files:

- confusion_matrix.png: A visual representation of the RandomForestClassifier's performance.
- nyc_heatmap.html: An interactive heatmap showing accident hotspots in New York City.

## How to Use the Repository

### 1. Clone the repository:

`git clone https://github.com/your-username/nyc-vehicle-collisions.git
cd nyc-vehicle-collisions`

### 2. Install dependencies: This project uses Python and the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- folium
Install the dependencies with:

`pip install -r requirements.txt
`

### 3. Run the notebooks: 
You can open and run each notebook in your preferred environment (Jupyter, VSCode, Google Colab, etc.).
### 4. Visualize the results:
- Check the classification model performance by viewing the confusion matrix in the visualizations/confusion_matrix.png.
- Open the visualizations/nyc_heatmap.html file in a browser to explore the heatmap.

## Future Work

- Improve model performance: The recall for "Major Injury" cases is low (6%). Future iterations could include hyperparameter tuning or experimenting with different models to improve classification performance.
- Data enhancement: Adding more features or balancing the dataset may help in resolving the class imbalance issues.
- Additional visualizations: Future efforts could include time-series analysis or spatial clustering to further analyze accident trends.
License

This project is licensed under the MIT License. See the LICENSE file for more details.