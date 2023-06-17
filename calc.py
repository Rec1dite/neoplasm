# Simple script to calculate the t-statistic and p-value for arbitrary models
import scipy.stats as stats

# [Accuracy, F-Measure]
model1 = [0.7586, 0.8627]   # ANN
model2 = [0.7586, 0.8542]   # GP
model3 = [0.7517, 0.7150]   # C4.5

# Perform pairwise t-tests
t_statistic, p_value = stats.ttest_ind(model1, model2)
print("Model 1 vs Model 2:")
print("T-Statistic:", t_statistic)
print("P-value:", p_value)

t_statistic, p_value = stats.ttest_ind(model1, model3)
print("Model 1 vs Model 3:")
print("T-Statistic:", t_statistic)
print("P-value:", p_value)

t_statistic, p_value = stats.ttest_ind(model2, model3)
print("Model 2 vs Model 3:")
print("T-Statistic:", t_statistic)
print("P-value:", p_value)
