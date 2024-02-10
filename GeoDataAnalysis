# Assessing Profitability and Risk in Resource Extraction: A Comparative Analysis of Three Regions

This report presents a comprehensive analysis of the profitability and associated risks in resource extraction across three distinct regions. By evaluating key performance indicators such as profit margins, the accuracy of reserve volume predictions, and the overall risk of loss, we aim to provide a nuanced understanding of each region's potential for investment. Region 0 emerges as the standout for profitability, achieving the highest returns despite predictions suggesting otherwise due to its lower-than-expected volume of reserves. Contrarily, Region 1, while demonstrating the highest accuracy in reserve predictions reflected by the lowest Root Mean Square Error (RMSE), yields lower profits. Region 2 offers a balance, with moderate profits and risks, positioning it as a potentially attractive option for investors seeking a middle ground.

Furthermore, our risk assessment sheds light on the likelihood of loss across these regions, revealing that despite previous assumptions of a high risk of loss, actual figures indicate a relatively low risk, with Region 1 showing the most promising profile for risk-averse strategies. This analysis not only challenges previous perceptions but also underscores the importance of a multi-faceted approach to evaluating investment opportunities in resource extraction. Through a detailed examination of profitability, prediction accuracy, and risk, this report aims to guide stakeholders in making informed decisions that align with their financial goals and risk tolerance levels.

## Prepare the Data


```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the datasets
geo_data_0 = pd.read_csv('/datasets/geo_data_0.csv')
geo_data_1 = pd.read_csv('/datasets/geo_data_1.csv')
geo_data_2 = pd.read_csv('/datasets/geo_data_2.csv')

# Function to prepare data (perform initial exploration and data cleaning)
def prepare_data(data):
    # Basic Exploration
    print(data.head())
    print("\nData Information:")
    data.info()
    print("\nData Description:")
    print(data.describe())

    # Checking for missing values and duplicates
    missing_values = data.isnull().sum()
    print("\nMissing Values:")
    print(missing_values)
    duplicates = data.duplicated().sum()
    print("\nNumber of Duplicates:", duplicates)

# Prepare each dataset
print("Data 0:")
prepare_data(geo_data_0)
print("\nData 1:")
prepare_data(geo_data_1)
print("\nData 2:")
prepare_data(geo_data_2)
```

    Data 0:
          id        f0        f1        f2     product
    0  txEyH  0.705745 -0.497823  1.221170  105.280062
    1  2acmU  1.334711 -0.340164  4.365080   73.037750
    2  409Wp  1.022732  0.151990  1.419926   85.265647
    3  iJLyR -0.032172  0.139033  2.978566  168.620776
    4  Xdl7t  1.988431  0.155413  4.751769  154.036647
    
    Data Information:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 5 columns):
     #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
     0   id       100000 non-null  object 
     1   f0       100000 non-null  float64
     2   f1       100000 non-null  float64
     3   f2       100000 non-null  float64
     4   product  100000 non-null  float64
    dtypes: float64(4), object(1)
    memory usage: 3.8+ MB
    
    Data Description:
                      f0             f1             f2        product
    count  100000.000000  100000.000000  100000.000000  100000.000000
    mean        0.500419       0.250143       2.502647      92.500000
    std         0.871832       0.504433       3.248248      44.288691
    min        -1.408605      -0.848218     -12.088328       0.000000
    25%        -0.072580      -0.200881       0.287748      56.497507
    50%         0.502360       0.250252       2.515969      91.849972
    75%         1.073581       0.700646       4.715088     128.564089
    max         2.362331       1.343769      16.003790     185.364347
    
    Missing Values:
    id         0
    f0         0
    f1         0
    f2         0
    product    0
    dtype: int64
    
    Number of Duplicates: 0
    
    Data 1:
          id         f0         f1        f2     product
    0  kBEdx -15.001348  -8.276000 -0.005876    3.179103
    1  62mP7  14.272088  -3.475083  0.999183   26.953261
    2  vyE1P   6.263187  -5.948386  5.001160  134.766305
    3  KcrkZ -13.081196 -11.506057  4.999415  137.945408
    4  AHL4O  12.702195  -8.147433  5.004363  134.766305
    
    Data Information:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 5 columns):
     #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
     0   id       100000 non-null  object 
     1   f0       100000 non-null  float64
     2   f1       100000 non-null  float64
     3   f2       100000 non-null  float64
     4   product  100000 non-null  float64
    dtypes: float64(4), object(1)
    memory usage: 3.8+ MB
    
    Data Description:
                      f0             f1             f2        product
    count  100000.000000  100000.000000  100000.000000  100000.000000
    mean        1.141296      -4.796579       2.494541      68.825000
    std         8.965932       5.119872       1.703572      45.944423
    min       -31.609576     -26.358598      -0.018144       0.000000
    25%        -6.298551      -8.267985       1.000021      26.953261
    50%         1.153055      -4.813172       2.011479      57.085625
    75%         8.621015      -1.332816       3.999904     107.813044
    max        29.421755      18.734063       5.019721     137.945408
    
    Missing Values:
    id         0
    f0         0
    f1         0
    f2         0
    product    0
    dtype: int64
    
    Number of Duplicates: 0
    
    Data 2:
          id        f0        f1        f2     product
    0  fwXo0 -1.146987  0.963328 -0.828965   27.758673
    1  WJtFt  0.262778  0.269839 -2.530187   56.069697
    2  ovLUW  0.194587  0.289035 -5.586433   62.871910
    3  q6cA6  2.236060 -0.553760  0.930038  114.572842
    4  WPMUX -0.515993  1.716266  5.899011  149.600746
    
    Data Information:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 5 columns):
     #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
     0   id       100000 non-null  object 
     1   f0       100000 non-null  float64
     2   f1       100000 non-null  float64
     3   f2       100000 non-null  float64
     4   product  100000 non-null  float64
    dtypes: float64(4), object(1)
    memory usage: 3.8+ MB
    
    Data Description:
                      f0             f1             f2        product
    count  100000.000000  100000.000000  100000.000000  100000.000000
    mean        0.002023      -0.002081       2.495128      95.000000
    std         1.732045       1.730417       3.473445      44.749921
    min        -8.760004      -7.084020     -11.970335       0.000000
    25%        -1.162288      -1.174820       0.130359      59.450441
    50%         0.009424      -0.009482       2.484236      94.925613
    75%         1.158535       1.163678       4.858794     130.595027
    max         7.238262       7.844801      16.739402     190.029838
    
    Missing Values:
    id         0
    f0         0
    f1         0
    f2         0
    product    0
    dtype: int64
    
    Number of Duplicates: 0


## Train and Test the Model for Each Region


```python
# Constants
BUDGET = 100000000  # Total budget: 100 million USD
REVENUE_PER_BARREL = 4.5  # Revenue per barrel: 4.5 USD
NUMBER_OF_WELLS = 200  # Number of oil wells to develop
COST_PER_WELL = BUDGET / NUMBER_OF_WELLS  # Cost to develop one oil well

# Function to train and evaluate the model
def train_and_evaluate(data):
    X = data[['f0', 'f1', 'f2']]
    y = data['product']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12345)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))
    avg_volume = np.mean(predictions)

    profit_data = pd.DataFrame({
        'product': y_valid,
        'predictions': predictions
    })
    profit_data['individual_profit'] = (profit_data['product'] * 1000 * REVENUE_PER_BARREL) - COST_PER_WELL

    return rmse, avg_volume, profit_data

results = {}
for i, geo_data in enumerate([geo_data_0, geo_data_1, geo_data_2]):
    rmse, avg_volume, profit_data = train_and_evaluate(geo_data)
    results[i] = {'RMSE': rmse, 'Average Volume': avg_volume}

# Calculating break-even volume of reserves per well
BREAK_EVEN_VOLUME = COST_PER_WELL / (REVENUE_PER_BARREL * 1000)  # multiplied by 1000 to convert to thousand barrels
print(f"Break-even Volume: {BREAK_EVEN_VOLUME:.2f} thousand barrels")

for i in range(3):
    avg_volume = results[i]['Average Volume']
    print(f"Average Volume in Region {i}: {avg_volume:.2f} thousand barrels")
```

    Break-even Volume: 111.11 thousand barrels
    Average Volume in Region 0: 92.59 thousand barrels
    Average Volume in Region 1: 68.73 thousand barrels
    Average Volume in Region 2: 94.97 thousand barrels


### Analysis:
Region 0:
RMSE: 37.58
Average Predicted Volume: 92.59 thousand barrels
Analysis: The RMSE is relatively high, suggesting that the model's predictions might not be very accurate. However, the average predicted volume of reserves is quite substantial.

Region 1:
RMSE: 0.89
Average Predicted Volume: 68.73 thousand barrels
Analysis: The RMSE is significantly lower than in the other regions, indicating a high level of accuracy in the predictions. However, the average predicted volume of reserves is lower than in Region 0.

Region 2:
RMSE: 40.03
Average Predicted Volume: 94.97 thousand barrels
Analysis: Similar to Region 0, the RMSE is high, suggesting less accuracy in predictions. The average predicted volume is the highest among the three regions.

Considerations for Decision Making:
Accuracy vs. Volume: Region 1 offers the most accurate predictions but with a lower average reserve volume. Regions 0 and 2 predict higher volumes but with less accuracy.
Risk Assessment: High RMSE in Regions 0 and 2 might indicate higher risks in decision-making based on these predictions. Accurate predictions (like in Region 1) reduce the risk of investing in wells with less than expected reserves.
Economic Implications: A balance needs to be struck between the potential volume of reserves and the confidence in those predictions. High volumes with high uncertainty might not be as desirable as lower volumes with more certainty.

## Prepare for Profit Calculation


```python


# Display the key values
print(f"Total Budget: {BUDGET} USD")
print(f"Revenue per Barrel: {REVENUE_PER_BARREL} USD")
print(f"Number of Wells: {NUMBER_OF_WELLS}")
print(f"Cost per Well: {COST_PER_WELL} USD")


```

    Total Budget: 100000000 USD
    Revenue per Barrel: 4.5 USD
    Number of Wells: 200
    Cost per Well: 500000.0 USD



```python
# Calculating break-even volume of reserves per well
BREAK_EVEN_VOLUME = COST_PER_WELL / (REVENUE_PER_BARREL * 1000)  # multiplied by 1000 to convert to thousand barrels

print(f"Break-even Volume: {BREAK_EVEN_VOLUME:.2f} thousand barrels")
for i in range(3):
    avg_volume = results[i]['Average Volume']
    print(f"Average Volume in Region {i}: {avg_volume:.2f} thousand barrels")
```

    Break-even Volume: 111.11 thousand barrels
    Average Volume in Region 0: 92.59 thousand barrels
    Average Volume in Region 1: 68.73 thousand barrels
    Average Volume in Region 2: 94.97 thousand barrels


None of the Regions Meet Break-even Requirements: All three regions have average reserve volumes below the calculated break-even volume. This implies a risk of financial losses if wells are developed based on the average reserve volumes in these regions.

## Function to Calculate Profit


```python
from sklearn.model_selection import train_test_split


# Function to calculate profit
def calculate_profit(predictions, profit_data, top_wells_count):
    top_wells = profit_data.nlargest(top_wells_count, 'predictions')
    total_reserves = top_wells['product'].sum()
    profit = (total_reserves * 1000 * REVENUE_PER_BARREL) - BUDGET
    return profit

# Apply the function to each region
region_profits = {}
for i, geo_data in enumerate([geo_data_0, geo_data_1, geo_data_2]):
    rmse, avg_volume, profit_data = train_and_evaluate(geo_data)
    profit = calculate_profit(profit_data['predictions'], profit_data, NUMBER_OF_WELLS)
    region_profits[i] = profit
    print(f"Region {i}: RMSE: {rmse}, Average Volume: {avg_volume}, Profit: {profit}")

# Identify the most profitable region
most_profitable_region = max(region_profits, key=region_profits.get)
print(f"The most profitable region is Region {most_profitable_region} with a profit of {region_profits[most_profitable_region]}")

```

    Region 0: RMSE: 37.5794217150813, Average Volume: 92.59256778438035, Profit: 33208260.43139851
    Region 1: RMSE: 0.893099286775617, Average Volume: 68.728546895446, Profit: 24150866.966815114
    Region 2: RMSE: 40.02970873393434, Average Volume: 94.96504596800489, Profit: 27103499.635998324
    The most profitable region is Region 0 with a profit of 33208260.43139851


Region 0:
Profit: $34,941,041.92
Analysis: Region 0 is the most profitable among the three. Despite its average volume of reserves being below the break-even point, the high values in some of its wells likely contributed to this higher profit.

Region 1:
Profit: $24,150,866.97
Analysis: Although Region 1 had the highest accuracy in predictions (lowest RMSE), its lower average volume of reserves led to a lower overall profit compared to Region 0.

Region 2:
Profit: $25,714,106.32
Analysis: Region 2 showed a reasonable profit, higher than Region 1 but lower than Region 0. Its proximity to the break-even volume might have contributed to this moderate performance.

In summary, while the profit calculations point to Region 0 as the most lucrative option, a comprehensive risk assessment and consideration of other non-model factors are crucial in making a well-informed decision.

## Calculate Risks and Rrofit for Each Region


```python
def bootstrap_analysis(profit_data, n_iterations=1000, initial_sample_size=500, top_wells_count=200):
    np.random.seed(123)  # For reproducibility
    profits = []

    for _ in range(n_iterations):
        initial_sample = profit_data.sample(n=initial_sample_size, replace=True)
        top_wells = initial_sample.nlargest(top_wells_count, 'predictions')
        total_profit = top_wells['individual_profit'].sum()
        profits.append(total_profit)

    profits = np.array(profits)
    average_profit = np.mean(profits)
    lower_bound = np.percentile(profits, 2.5)
    upper_bound = np.percentile(profits, 97.5)
    risk_of_loss = np.mean(profits < 0) * 100

    return average_profit, (lower_bound, upper_bound), risk_of_loss

# Prepare profit_data for each region
profit_data_regions = [train_and_evaluate(geo_data)[2] for geo_data in [geo_data_0, geo_data_1, geo_data_2]]

# Apply the bootstrap analysis to each region's profit data
for i, profit_data in enumerate(profit_data_regions):
    avg_profit, confidence_interval, loss_risk = bootstrap_analysis(profit_data)
    print(f"Region {i}: Average Profit: {avg_profit}, 95% CI: {confidence_interval}, Risk of Loss: {loss_risk}%")
```

    Region 0: Average Profit: 3960114.2729595117, 95% CI: (-1522459.761304759, 9125304.69099573), Risk of Loss: 7.6%
    Region 1: Average Profit: 4540382.903699188, 95% CI: (743203.7575858902, 8345096.315601907), Risk of Loss: 0.6%
    Region 2: Average Profit: 3934129.6371716387, 95% CI: (-1077121.3185743103, 8973662.610705575), Risk of Loss: 6.0%


### Analysis and Recommendations:
Region 0:
Profit: $33,208,260.43
Analysis: Region 0 emerges as the most profitable, with a significant profit despite its average volume of reserves being below the break-even point. The higher RMSE indicates less accuracy in predictions, but the actual profits are substantial.

Region 1:
Profit: $24,150,866.97
Analysis: Region 1, with the lowest RMSE, indicates the highest accuracy in predictions. However, its lower average volume of reserves has resulted in a lower profit compared to Region 0.

Region 2:
Profit: $27,103,499.64
Analysis: Region 2 shows a reasonable profit, higher than Region 1 but lower than Region 0. The average volume of reserves and RMSE positions it as a moderately risky option.
Risk Assessment:

Region 0: Risk of Loss: 7.6%
Region 1: Risk of Loss: 0.6%
Region 2: Risk of Loss: 6.0%
The risk of loss is relatively low for all regions, with Region 1 showing the lowest risk. This contrasts with the earlier conclusion of 100% risk of loss, suggesting that the risk assessment might need a more nuanced approach.

### Conclusion:

Profitability and Risk Balance: Region 0 is the most profitable option but comes with a higher risk compared to Region 1. Region 1, despite being the least risky, offers lower profits.
Consideration of Accuracy and Volume: Region 1’s high accuracy in predictions (low RMSE) did not translate into the highest profits, indicating that factors other than just prediction accuracy (like the volume of reserves) play a crucial role in determining profitability.
Region 1 as a Potential Alternative: Given its low risk of loss and reasonable profit potential, Region 1 could be considered a viable alternative, especially for risk-averse investment strategies.
Further Analysis and Consideration: It’s important to revisit the project's assumptions and models, considering these new insights. Additional factors, such as operational costs, environmental impacts, and regulatory aspects, should also be factored into the final decision.
In summary, while Region 0 shows the highest profit potential, the decision to invest should carefully balance profit potential against the risk of loss. The updated analysis suggests that Region 1, with its lower risk, might also be a viable option, especially if risk mitigation is a priority.
