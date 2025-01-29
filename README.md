Project Overview:
I designed and implemented an A/B test for Urban Wear to optimize email sign-ups on their pre-launch page. The test evaluated whether changing the submit button color from blue (control) to green (treatment) would lead to a higher conversion rate. My goal was to provide actionable, data-driven recommendations to help maximize sign-ups before the full website launch.

Steps I Took:
Data Exploration and Cleaning:

Loaded and explored pretest and test datasets, checking for missing values and formatting inconsistencies.
Converted date columns for proper time-series analysis and ensured data integrity through initial validation.
Segmentation of Data:

Split the dataset into control (blue button) and treatment (green button) groups.
Verified balanced group assignment and corrected any data imbalance using stratified sampling.
Calculating Conversion Rates:

Compute the Conversion rate as 
Number of Sign-ups/Total Visitors
 
Statistical Hypothesis Testing:

Formulated hypotheses:
Null Hypothesis (H₀): No significant difference in conversion rates between control and treatment.
Alternative Hypothesis (H₁): The green button has a higher conversion rate than the blue button.
Performed a Z-test for proportions using Python’s statsmodels to assess statistical significance.
Visualization and Business Recommendations:

Created conversion rate plots and summary tables using Seaborn and Matplotlib.
Presented findings to the product team, emphasizing actionable insights and confidence intervals.
Results:
The treatment group (green button) had a higher conversion rate compared to the control group, with a statistically significant p-value (< 0.05).
Based on the results, I recommended deploying the green button, which is projected to increase email sign-ups and improve pre-launch engagement.
Challenges I Overcame:
Data Imbalance: The initial dataset had unequal group sizes, which I addressed using stratified sampling to ensure fair comparisons.
Handling Missing Data: Some sign-up records were incomplete, so I used imputation techniques and conducted sensitivity analysis to validate that the missing data didn’t affect the final outcome.
Multiple Metrics: Balancing the immediate goal (email sign-ups) with potential long-term business outcomes was a key consideration when providing recommendations.
Key Takeaways:
This project demonstrated my ability to design and execute experiments, clean and analyze data, apply statistical methods, and provide business-driven recommendations. It reinforced the importance of careful experimental design and stakeholder communication when driving business decisions.

Tools & Technologies:
Data Analysis & Cleaning: Python (Pandas, NumPy)
Statistical Testing: SciPy, Statsmodels
Visualization: Seaborn, Matplotlib
Hypothesis Testing: Z-test for proportions, confidence intervals
