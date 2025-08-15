# Martian Colony Disappearance Prediction

A machine learning solution to predict colonist disappearances on the Martian colony Prosperity-3.

## Project Overview

The colony Prosperity-3 has been experiencing mysterious disappearances. This project develops a predictive model to identify colonists at risk of disappearing, using various clues provided by the Senior Data Scientist including ID detector visits, shift patterns, and demographic information.

## Problem Statement

- **Objective**: Predict which colonists are at risk of disappearing
- **Data**: 6,500+ colonist records with demographic, behavioral, and shift data
- **Challenge**: Highly imbalanced dataset (1:40 ratio of missing to non-missing)
- **Constraint**: Zero-miss policy preferred (no actual disappearances should be missed)

## Key Insights Discovered

### 1. **Family-Staff Connection** 
- Colonists with staff surnames are **2.3x more likely** to disappear
- 27 staff members have missing family members
- Strong indicator of targeted disappearances

### 2. **ID Detector Visits** 
- Last seen venue (hairdresser, counselor, psychologist, doctor, dentist) is highly predictive
- Frequency of visits to these locations correlates with disappearance risk

### 3. **Demographic Patterns** 
- Age, BMI, and sleep patterns show significant associations
- Geographic origin (zip code) influences risk levels
- Recent activity patterns are crucial predictors

## Technical Approach

### Data Engineering
- **Target Creation**: Matched missing persons report with training data (260/276 matches)
- **Feature Engineering**: 
  - Temporal features (age, days since last seen, day of week)
  - Sleep metrics (average, variability, trend)
  - Shift patterns (on-duty flags, frequency counts, last seen venue)
  - Geographic encoding (city → zip code → ordinal)
  - Health metrics (BMI calculation)
  - Family connections (staff surname matching)

### Model Architecture
- **Algorithm**: XGBoost Classifier with SMOTE for class balancing
- **Pipeline**: Preprocessing → SMOTE → XGBoost → Probability Calibration
- **Validation**: Stratified 5-fold cross-validation
- **Hyperparameter Tuning**: RandomizedSearchCV with 40 iterations

### Model Variants & Optimization
- **Full Model**: Uses all engineered features
- **Reduced Model**: Optimized feature subset focusing on the most predictive variables, reducing noise and improving interpretability
- **Threshold Optimization**: Two policy approaches tested:
  - **Zero-miss Policy**: Threshold optimized to achieve 100% recall (no missed disappearances)
  - **F1-max Policy**: Threshold optimized to maximize F1-score (balanced precision/recall)

### Model Performance
| Model | Policy | AUC | Precision | Recall | F1 | Test Alerts |
|-------|--------|-----|-----------|--------|----|-----------:|
| Full | Zero-miss | 0.989 | 0.267 | 1.000 | 0.421 | 450 |
| Full | F1-max | 0.989 | 0.661 | 0.712 | 0.685 | 110 |
| **Reduced** | **Zero-miss** | **0.989** | **0.291** | **1.000** | **0.450** | **406** |
| Reduced | F1-max | 0.989 | 0.679 | 0.692 | 0.686 | 116 |

## Final Solution

**Selected Model**: Reduced Feature Set + Zero-miss Policy
- **Feature Selection**: Optimized subset of most predictive features for better performance and interpretability
- **Threshold**: Calibrated to achieve zero-miss policy (100% recall)
- **AUC**: 0.989 (excellent discrimination)
- **Precision**: 0.291 (1 in 3 alerts is a true case)
- **Recall**: 1.000 (no disappearances missed)
- **Alerts**: 406 out of 4,680 colonists (8.7%)

**Policy Justification**: The zero-miss policy was chosen over F1-maximization because missing an actual disappearance has much higher cost than investigating false positives in this critical safety scenario.

## Feature Importance

Top predictive features identified through XGBoost importance and SHAP analysis:
1. **has_staff_surname** - Family connection to staff
2. **days_since_last_seen** - Recency of last sighting
3. **age** - Colonist age
4. **last_seen_venue** - Specific location of last ID detection
5. **BMI** - Body mass index
6. **sleep_avg** - Average sleep duration
7. **zip_code** - Geographic origin
8. **on_duty_* flags** - ID detector visit history

## Business Impact

The model enables proactive intervention by:
- **Identifying high-risk colonists** before they disappear
- **Optimizing security resources** with 406 targeted alerts vs. monitoring all 4,680 colonists
- **Preventing disappearances** through early warning system
- **Investigating staff-family connections** for potential security breaches

## Recommendations

1. **Immediate Action**: Implement enhanced monitoring for the 406 flagged colonists
2. **Investigation**: Focus on staff members with missing family members
3. **Security**: Increase surveillance around ID detector locations
4. **Data Collection**: Gather more granular temporal data for pattern analysis
5. **Model Updates**: Retrain monthly as new disappearance data becomes available

## Notes

- This solution prioritizes recall over precision to ensure no disappearances are missed
- The model identified suspicious family-staff connections warranting investigation
- Temporal clustering of disappearances suggests coordinated events
- High AUC (0.989) indicates excellent model discrimination capability

## Repository Structure

```
colonist_disappearance/
├── Niklas_Mellgren_Prosperity-3_Case_Solution.ipynb
├── README.md
├── data/
│   ├── data_train_fin.csv
│   ├── data_test_fin.csv
│   ├── missing_report.csv
│   └── shift_report.csv
└── outputs/
    ├── martian_disappearance_predictions.csv
    └── martian_disappearance_predictions_clean.csv
```

## Key Technologies

- **Python 3.11**
- **Machine Learning**: XGBoost, scikit-learn, imbalanced-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Explainability**: SHAP
- **Environment**: Google Colab

