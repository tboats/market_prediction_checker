import json
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def calculate_p_value_pearson(r, n):
    """Calculate p-value for Pearson correlation using Student's t approximation."""
    if abs(r) >= 1.0:
        return 0.0
    t_stat = r * math.sqrt((n - 2) / (1 - r**2))
    df = n - 2
    # Simple numerical approximation of Student's t cumulative distribution function
    # for two-tailed p-value
    def t_pdf(u, df):
        coeff = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
        return coeff * (1 + (u**2) / df) ** (- (df + 1) / 2)
        
    steps = 1000
    lower = abs(t_stat)
    upper = max(10.0, lower + 5.0)
    h = (upper - lower) / steps
    
    integral = 0.5 * (t_pdf(lower, df) + t_pdf(upper, df))
    for i in range(1, steps):
        integral += t_pdf(lower + i * h, df)
    area = integral * h
    p_val = min(1.0, 2.0 * area)
    return p_val

def analyze():
    # Paths
    results_path = os.path.join(os.path.dirname(__file__), 'results.json')
    repo_dir = os.path.dirname(__file__)
    
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    # Filter for elapsed predictions
    elapsed = [p for p in data if p.get('is_elapsed')]
    
    predicted = []
    actual = []
    
    for p in elapsed:
        midpoint = sum(p['predicted_range']) / 2.0
        predicted.append(midpoint)
        actual.append(p['actual_return'])
        
    predicted = np.array(predicted)
    actual = np.array(actual)
    n = len(elapsed)
    
    # 1. Pearson Correlation
    mean_p = np.mean(predicted)
    mean_a = np.mean(actual)
    
    cov = np.sum((predicted - mean_p) * (actual - mean_a))
    var_p = np.sum((predicted - mean_p) ** 2)
    var_a = np.sum((actual - mean_a) ** 2)
    
    pearson_r = cov / math.sqrt(var_p * var_a)
    pearson_p = calculate_p_value_pearson(pearson_r, n)
    
    # 2. Linear Regression (Actual = slope * Predicted + intercept)
    slope = cov / var_p
    intercept = mean_a - slope * mean_p
    r_squared = pearson_r ** 2
    
    # 3. Error Metrics
    errors = actual - predicted
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mean_bias = np.mean(errors)
    
    # Print statistics
    print("===============================================")
    print("📊 STATISTICAL EVALUATION RESULTS")
    print(f"   Number of evaluated predictions: {n}")
    print("===============================================")
    print(f" Pearson Correlation (r):  {pearson_r:.4f}")
    print(f"   p-value:                {pearson_p:.6f} ({'Significant' if pearson_p < 0.05 else 'Not Significant'} at 5% level)")
    print("-----------------------------------------------")
    print(f" Linear Regression (y = Actual, x = Predicted):")
    print(f"   Formula:   Actual = {slope:.4f} * Predicted + {intercept:.4f}%")
    print(f"   R-squared: {r_squared:.4f}")
    print("-----------------------------------------------")
    print(f" Forecast Accuracy Metrics:")
    print(f"   Mean Absolute Error (MAE): {mae:.2f} percentage points")
    print(f"   Root Mean Squared Error (RMSE): {rmse:.2f} percentage points")
    print(f"   Mean Forecast Bias (Actual - Predicted): {mean_bias:+.2f} percentage points")
    print("===============================================")
    
    # 4. Generate Calibration Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Diagonal 1:1 perfect calibration line
    min_val = min(predicted.min(), actual.min()) - 2
    max_val = max(predicted.max(), actual.max()) + 2
    x_diag = np.linspace(min_val, max_val, 100)
    ax.plot(x_diag, x_diag, color='#555555', linestyle='--', linewidth=1.5, label='Perfect Calibration (Actual = Predicted)')
    
    # Regression line
    x_reg = np.linspace(predicted.min() - 1, predicted.max() + 1, 100)
    y_reg = slope * x_reg + intercept
    ax.plot(x_reg, y_reg, color='#38bdf8', linestyle='-', linewidth=2, label=f'OLS Regression Line (R² = {r_squared:.2f})')
    
    # Scatter points colored by return type
    nominal_idx = [i for i, p in enumerate(elapsed) if p['return_type'] == 'nominal']
    real_idx = [i for i, p in enumerate(elapsed) if p['return_type'] == 'real']
    
    ax.scatter(predicted[nominal_idx], actual[nominal_idx], color='#fbbf24', s=80, alpha=0.85, edgecolors='white', linewidths=0.5, label='Nominal Predictions')
    ax.scatter(predicted[real_idx], actual[real_idx], color='#34d399', s=80, alpha=0.85, edgecolors='white', linewidths=0.5, label='Real (Inflation-Adj) Predictions')
    
    # Annotate details
    ax.set_title('Prediction Calibration Plot\nActual Returns vs. Predicted Midpoint (Elapsed Forecasts)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Return Midpoint (%)', fontsize=11, labelpad=10)
    ax.set_ylabel('Actual Realized Return (CAGR %)', fontsize=11, labelpad=10)
    
    # Display statistics box
    stats_text = (
        f"Pearson r: {pearson_r:.2f}\n"
        f"p-value: {pearson_p:.4f}\n"
        f"R²: {r_squared:.2f}\n"
        f"MAE: {mae:.2f} pp\n"
        f"Mean Bias: {mean_bias:+.2f} pp"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e293b', alpha=0.7, edgecolor='#475569'))
    
    ax.legend(loc='lower right', frameon=True, facecolor='#1e293b', edgecolor='#475569')
    ax.grid(True, linestyle=':', alpha=0.3, color='#475569')
    
    # Set tight layout
    plt.tight_layout()
    
    # Save plots
    plt.savefig(os.path.join(repo_dir, 'calibration.png'), dpi=300)
    print("📈 Calibration plot saved successfully!")

if __name__ == "__main__":
    analyze()
