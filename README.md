## Flood Risk Prediction Model
### Industry-Based Learning (IBL) Project with IBM Tech
This project builds a machine learning model to predict flood risk using environmental and geographic data provided by the U.S. Environmental Protection Agency through EnviroAtlas. The objective is to support early detection and decision-making for communities that could be vulnerable to flooding. This particularly investigates the University City, MO, area.
Interactive Flood Reporting Map:
https://stlfloodreporting.net/Pages/map.html

### Problem Statement
Flooding causes significant infrastructure damage, economic loss, and public safety risks.
The goal of this project was to:
- Develop a tool that can predictively model flood occurrences.
- Identify high-risk regions using certain features.
- Evaluate model performance using classification metrics.
This work was completed as a part of an Industry-Based Learning (IBL) collaboration with IBM Tech.

### Tech Stack
- Python
- Pandas
- Numpy
- scikit-learn
- MLPClassifier
- joblib

### About the Dataset
- ~1000 observations.
- Geographic and Environmental variables.
- Latitude and longitude used as predictive features.
- Missing values were handled through preprocessing.
- Placeholder values have been replaced and imputed.
This dataset is small in comparison to other datasets, and there is a class imbalance since flooding doesn't happen everywhere. The size can be improved by including other areas around University City or expanding to other regions or cities.

### Methodology
1. Data Cleaning
Replacing placeholder values with NaN, imputing missing values with column means where necessary, removing duplicate identifiers, and veryifying no data leakage.
2. Data Preprocessing
Split 'latlong' column into separate columns, added elevation data based on lat/long, and cleaned up columns. Scaled features to improve performance (although later removed for interpretability over performance).
3. Feature Engineering
Used correlation heatmaps and VIF to decrease dimensionality.
4. Model
Used a Multi-Layer Perceptron, running through numerous grid searches and experimentation with hyperparameters to come upon a model that had the best flood prediction. Ended up using adam as the solver, with 4000 iterations, ReLU activation, and found 0.5 to be a good regularization to reduce overfitting.
5. Evaluation
Cross-validation, confusion matrix, ROC-AUC, and other metrics. Using other metrics than Accuracy made it much easier to understand due to the class imbalance, having a more accurate way to understand the model and its abilities.
