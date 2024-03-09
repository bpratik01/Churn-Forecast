## Churn Forecast

### Project Overview
Churn Forecast is a data science project focused on predicting customer churn rates for a telecommunications company. Leveraging machine learning techniques, including decision trees and random forest algorithms, the project aims to provide valuable insights into customer behavior and churn patterns.

### Workflow
1. **Data Preprocessing:** The project begins with data preprocessing steps, handling missing values, encoding categorical variables, and scaling numerical features to prepare the data for analysis.
2. **Exploratory Data Analysis (EDA):** Exploring the dataset to uncover key insights into customer demographics, behavior, and characteristics.
3. **Model Building:** Implementing decision trees and random forest algorithms to predict churn rates. Hyperparameter tuning is performed using grid search CV to optimize model performance.
4. **Model Evaluation:** Evaluating model performance using metrics such as R2 score and F1 score on both training and testing datasets to assess predictive accuracy.
5. **Insights:** Drawing actionable insights from the model results to understand customer churn behavior and identify potential strategies for retention.

### Key Insights
- **Customer Demographics:** 
  - Analysis reveals that 16.21% of customers are senior citizens, highlighting the importance of understanding their churn patterns.
  - The average tenure of customers is 32.37 months, with significant variability, suggesting diverse customer relationships.
  - The median tenure of 29 months indicates the typical duration of customer engagements with the company.

- **Monthly Charges:**
  - The average monthly charge is $64.76, with a median of $70.35, indicating the range of pricing plans.
  - Notably, customers with monthly charges ranging from $0 to $40 exhibit significantly lower churn rates, indicating potential retention strategies for this segment.

### Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Imblearn

### How to Run
1. Clone the repository.
2. Install the required dependencies.
3. Execute the main script to perform churn prediction and generate insights.

### Results and Analysis
The project results provide deep insights into customer churn behavior, helps in understanding the factors that influence churn rates and guide towards strategic decision-making. Decision trees and random forest algorithms prove effective in predicting churn, with random forest demonstrating superior performance. 


---

Feel free to adjust and expand the README as needed to reflect the specific insights and findings of your project.