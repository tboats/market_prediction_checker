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

def calculate_sub_stats(predicted_subset, actual_subset):
    n_sub = len(predicted_subset)
    if n_sub < 3:
        return None
        
    mean_p = np.mean(predicted_subset)
    mean_a = np.mean(actual_subset)
    
    cov = np.sum((predicted_subset - mean_p) * (actual_subset - mean_a))
    var_p = np.sum((predicted_subset - mean_p) ** 2)
    var_a = np.sum((actual_subset - mean_a) ** 2)
    
    if var_p == 0 or var_a == 0:
        return None
        
    r = cov / math.sqrt(var_p * var_a)
    p_val = calculate_p_value_pearson(r, n_sub)
    
    slope = cov / var_p
    intercept = mean_a - slope * mean_p
    r_squared = r ** 2
    
    errors = actual_subset - predicted_subset
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(errors)
    
    return {
        'n': n_sub,
        'r': r,
        'p': p_val,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'mae': mae,
        'rmse': rmse,
        'bias': bias
    }

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
    
    # 1. Combined Stats
    comb_stats = calculate_sub_stats(predicted, actual)
    
    # 2. Split Stats
    nominal_indices = [i for i, p in enumerate(elapsed) if p['return_type'] == 'nominal']
    real_indices = [i for i, p in enumerate(elapsed) if p['return_type'] == 'real']
    
    nom_stats = calculate_sub_stats(predicted[nominal_indices], actual[nominal_indices])
    real_stats = calculate_sub_stats(predicted[real_indices], actual[real_indices])
    
    # Print statistics
    print("===============================================")
    print("📊 STATISTICAL EVALUATION RESULTS")
    print(f"   Number of evaluated predictions: {n}")
    print("===============================================")
    
    if comb_stats:
        print("COMBINED VIEW:")
        print(f" Pearson Correlation (r):  {comb_stats['r']:.4f}")
        print(f"   p-value:                {comb_stats['p']:.6f} ({'Significant' if comb_stats['p'] < 0.05 else 'Not Significant'} at 5% level)")
        print(f" Formula:                  Actual = {comb_stats['slope']:.4f} * Predicted + {comb_stats['intercept']:.4f}%")
        print(f" R-squared (R²):           {comb_stats['r_squared']:.4f}")
        print(f" MAE:                      {comb_stats['mae']:.2f} pp")
        print(f" Mean Forecast Bias:       {comb_stats['bias']:+.2f} pp")
    
    print("-----------------------------------------------")
    if nom_stats:
        print("NOMINAL PREDICTIONS ONLY:")
        print(f" n:                        {nom_stats['n']}")
        print(f" Pearson Correlation (r):  {nom_stats['r']:.4f}")
        print(f"   p-value:                {nom_stats['p']:.6f} ({'Significant' if nom_stats['p'] < 0.05 else 'Not Significant'} at 5% level)")
        print(f" Formula:                  Actual = {nom_stats['slope']:.4f} * Predicted + {nom_stats['intercept']:.4f}%")
        print(f" R-squared (R²):           {nom_stats['r_squared']:.4f}")
        print(f" MAE:                      {nom_stats['mae']:.2f} pp")
        print(f" Mean Forecast Bias:       {nom_stats['bias']:+.2f} pp")
        
    print("-----------------------------------------------")
    if real_stats:
        print("REAL PREDICTIONS ONLY (Inflation-Adjusted):")
        print(f" n:                        {real_stats['n']}")
        print(f" Pearson Correlation (r):  {real_stats['r']:.4f}")
        print(f"   p-value:                {real_stats['p']:.6f} ({'Significant' if real_stats['p'] < 0.05 else 'Not Significant'} at 5% level)")
        print(f" Formula:                  Actual = {real_stats['slope']:.4f} * Predicted + {real_stats['intercept']:.4f}%")
        print(f" R-squared (R²):           {real_stats['r_squared']:.4f}")
        print(f" MAE:                      {real_stats['mae']:.2f} pp")
        print(f" Mean Forecast Bias:       {real_stats['bias']:+.2f} pp")
    print("===============================================")
    
    # 4. Generate Calibration Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Diagonal 1:1 perfect calibration line
    min_val = min(predicted.min(), actual.min()) - 2
    max_val = max(predicted.max(), actual.max()) + 2
    x_diag = np.linspace(min_val, max_val, 100)
    ax.plot(x_diag, x_diag, color='#555555', linestyle='--', linewidth=1.5, label='Perfect Calibration (Actual = Predicted)')
    
    # Combined Regression line
    if comb_stats:
        x_reg = np.linspace(predicted.min() - 1, predicted.max() + 1, 100)
        y_reg = comb_stats['slope'] * x_reg + comb_stats['intercept']
        ax.plot(x_reg, y_reg, color='#38bdf8', linestyle='-', linewidth=2, label=f'OLS Combined (R² = {comb_stats["r_squared"]:.2f})')
        
    # Scatter points colored by return type
    ax.scatter(predicted[nominal_indices], actual[nominal_indices], color='#fbbf24', s=80, alpha=0.85, edgecolors='white', linewidths=0.5, label='Nominal Predictions')
    ax.scatter(predicted[real_indices], actual[real_indices], color='#34d399', s=80, alpha=0.85, edgecolors='white', linewidths=0.5, label='Real (Inflation-Adj) Predictions')
    
    # Annotate details
    ax.set_title('Prediction Calibration Plot\nActual Returns vs. Predicted Midpoint (Elapsed Forecasts)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Return Midpoint (%)', fontsize=11, labelpad=10)
    ax.set_ylabel('Actual Realized Return (CAGR %)', fontsize=11, labelpad=10)
    
    # Display statistics box
    if comb_stats:
        stats_text = (
            f"Combined Pearson r: {comb_stats['r']:.2f}\n"
            f"Combined p-value: {comb_stats['p']:.4f}\n"
            f"Nominal Pearson r: {nom_stats['r']:.2f} (p={nom_stats['p']:.4f})\n"
            f"Real Pearson r: {real_stats['r']:.2f} (p={real_stats['p']:.4f})"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e293b', alpha=0.7, edgecolor='#475569'))
    
    ax.legend(loc='lower right', frameon=True, facecolor='#1e293b', edgecolor='#475569')
    ax.grid(True, linestyle=':', alpha=0.3, color='#475569')
    
    plt.tight_layout()
    plt.savefig(os.path.join(repo_dir, 'calibration.png'), dpi=300)
    print("📈 Calibration plot saved successfully!")
    
    # 5. Generate Histogram Plot
    all_nominal_mids = [sum(p['predicted_range']) / 2.0 for p in data if p['return_type'] == 'nominal']
    all_real_mids = [sum(p['predicted_range']) / 2.0 for p in data if p['return_type'] == 'real']
    
    fig_hist, ax_hist = plt.subplots(figsize=(9, 6))
    
    # Create bin boundaries
    all_mids = all_nominal_mids + all_real_mids
    min_bin = math.floor(min(all_mids)) - 1
    max_bin = math.ceil(max(all_mids)) + 2
    bins = np.arange(min_bin, max_bin, 1)
    
    ax_hist.hist(all_nominal_mids, bins=bins, alpha=0.75, color='#fbbf24', edgecolor='#1e1e24', label='Nominal Predictions', rwidth=0.85)
    ax_hist.hist(all_real_mids, bins=bins, alpha=0.75, color='#34d399', edgecolor='#1e1e24', label='Real (Inflation-Adj) Predictions', rwidth=0.85)
    
    ax_hist.set_title('Distribution of Stock Return Predictions', fontsize=14, fontweight='bold', pad=15)
    ax_hist.set_xlabel('Predicted CAGR Midpoint (%)', fontsize=11, labelpad=10)
    ax_hist.set_ylabel('Number of Predictions', fontsize=11, labelpad=10)
    ax_hist.legend(loc='upper right', frameon=True, facecolor='#1e293b', edgecolor='#475569')
    ax_hist.grid(True, linestyle=':', alpha=0.3, color='#475569')
    
    plt.tight_layout()
    plt.savefig(os.path.join(repo_dir, 'predictions_distribution.png'), dpi=300)
    print("📈 Distribution histogram saved successfully!")

if __name__ == "__main__":
    analyze()
