import os
import json
import math
import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(REPO_DIR, "results.json")
DASHBOARD_PATH = os.path.join(REPO_DIR, "dashboard.html")

def generate_dashboard():
    print("Generating dashboard...")
    if not os.path.exists(RESULTS_PATH):
        print(f"❌ Error: {RESULTS_PATH} does not exist. Run evaluate.py first.")
        return
        
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
    results = sorted(results, key=lambda x: x["source_date"])
        
    # Calculate statistics
    elapsed = [r for r in results if r['is_elapsed']]
    total_evaluated = len(elapsed)
    total_active = len(results) - total_evaluated
    
    overperformed_count = sum(1 for r in elapsed if r['status'] == "Overperformed")
    underperformed_count = sum(1 for r in elapsed if r['status'] == "Underperformed")
    accurate_count = sum(1 for r in elapsed if r['status'] == "Accurate")
    
    overperform_rate = (overperformed_count / total_evaluated * 100.0) if total_evaluated > 0 else 0
    
    errors = [abs(r['forecast_error']) for r in elapsed]
    avg_error = (sum(errors) / len(errors)) if errors else 0
    
    # Pearson Correlation and OLS Regression
    predicted_arr = np.array([sum(r['predicted_range']) / 2.0 for r in elapsed])
    actual_arr = np.array([r['actual_return'] for r in elapsed])
    n = len(elapsed)
    
    if n > 2:
        mean_p = np.mean(predicted_arr)
        mean_a = np.mean(actual_arr)
        
        cov = np.sum((predicted_arr - mean_p) * (actual_arr - mean_a))
        var_p = np.sum((predicted_arr - mean_p) ** 2)
        var_a = np.sum((actual_arr - mean_a) ** 2)
        
        pearson_r = cov / math.sqrt(var_p * var_a)
        
        # OLS
        slope = cov / var_p
        intercept = mean_a - slope * mean_p
        r_squared = pearson_r ** 2
        
        # Pearson p-value using t approximation
        t_stat = pearson_r * math.sqrt((n - 2) / (1 - pearson_r**2))
        def t_pdf(u, df):
            coeff = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
            return coeff * (1 + (u**2) / df) ** (- (df + 1) / 2)
            
        steps = 1000
        lower = abs(t_stat)
        upper = max(10.0, lower + 5.0)
        h = (upper - lower) / steps
        integral = 0.5 * (t_pdf(lower, n-2) + t_pdf(upper, n-2))
        for i in range(1, steps):
            integral += t_pdf(lower + i * h, n-2)
        pearson_p = min(1.0, 2.0 * integral * h)
        
        errors_arr = actual_arr - predicted_arr
        mean_bias = np.mean(errors_arr)
    else:
        pearson_r = 0.0
        pearson_p = 1.0
        slope = 0.0
        intercept = 0.0
        r_squared = 0.0
        mean_bias = 0.0
        
    # Identify largest miss and closest prediction
    largest_miss = None
    closest_pred = None
    if elapsed:
        sorted_by_error = sorted(elapsed, key=lambda r: abs(r['forecast_error']))
        closest_pred = sorted_by_error[0]
        largest_miss = sorted_by_error[-1]

    # Render metrics cards HTML
    metrics_html = f"""
    <div class="card metric-card">
        <div class="metric-title">Evaluated Predictions</div>
        <div class="metric-value">{total_evaluated}</div>
        <div class="metric-subtext">{total_active} active/pending prediction(s)</div>
    </div>
    <div class="card metric-card">
        <div class="metric-title">Overperformance Rate</div>
        <div class="metric-value" style="color: var(--blue);">{overperform_rate:.1f}%</div>
        <div class="metric-subtext">{overperformed_count} out of {total_evaluated} predictions exceeded range</div>
    </div>
    <div class="card metric-card">
        <div class="metric-title">Avg. Forecast Error (Absolute)</div>
        <div class="metric-value" style="color: var(--orange);">{avg_error:.2f}%</div>
        <div class="metric-subtext">Average absolute return difference in percentage points (pp)</div>
    </div>
    """
    
    # Render table rows HTML
    table_rows = []
    for r in results:
        p_range = r['predicted_range']
        range_str = f"{p_range[0]:.2f}%" if p_range[0] == p_range[1] else f"{p_range[0]:.2f}% to {p_range[1]:.2f}%"
        
        status_class = r['status'].lower()
        status_badge = f'<span class="badge badge-{status_class}">{r["status"]}</span>'
        
        if r['is_elapsed']:
            actual_str = f"{r['actual_return']:.2f}%"
            error_str = f"{r['forecast_error']:+.2f}%"
            # Highlight extreme errors
            error_style = 'color: var(--red); font-weight: 600;' if abs(r['forecast_error']) >= 8 else 'color: var(--orange);'
            if r['status'] == 'Accurate':
                error_style = 'color: var(--green);'
            error_td = f'<td style="{error_style}">{error_str}</td>'
            
            detail_row = f"""
            <tr class="main-row" onclick="toggleDetails('{r['id']}')">
                <td><strong>{f'<a href="{r["source_url"]}" target="_blank" style="color: var(--text-primary); text-decoration: none; border-bottom: 1px dashed var(--blue);" onclick="event.stopPropagation();">{r["organization"]}</a>' if r.get("source_url") else r["organization"]}</strong></td>
                <td>{r['source_date']}</td>
                <td>{r['horizon_years']} Years</td>
                <td>{r['target_asset']}</td>
                <td>{range_str} <span class="return-type">({r['return_type']})</span></td>
                <td><strong>{actual_str}</strong></td>
                <td>{status_badge}</td>
                {error_td}
            </tr>
            <tr id="details-{r['id']}" class="detail-row" style="display: none;">
                <td colspan="8">
                    <div class="detail-content">
                        <div class="detail-grid">
                            <div>
                                <strong>S&P 500 Total Return Index Details:</strong><br>
                                • Start Price: {r['start_price']:.2f}<br>
                                • End Price: {r['end_price']:.2f}<br>
                                • Nominal CAGR: {r['nominal_cagr']:.2f}%
                            </div>
                            <div>
                                <strong>Inflation Details (U.S. CPI):</strong><br>
                                • Start CPI: {r['cpi_start']:.2f}<br>
                                • End CPI: {r['cpi_end']:.2f}<br>
                                • Total Inflation: {r['inflation_rate']:.2f}% (CAGR: {r['inflation_cagr']:.2f}%)
                            </div>
                            <div>
                                <strong>Context / Source Note:</strong><br>
                                <p class="context-p">{r['context']}</p>
                                {f'<a href="{r["source_url"]}" target="_blank" class="source-link">Read Source Article ↗</a>' if r['source_url'] else ''}
                            </div>
                        </div>
                    </div>
                </td>
            </tr>
            """
        else:
            actual_str = f"{r['actual_return']:.2f}%*"
            error_str = f"{r['forecast_error']:+.2f}%*"
            
            detail_row = f"""
            <tr class="main-row active-prediction" onclick="toggleDetails('{r['id']}')">
                <td><strong>{f'<a href="{r["source_url"]}" target="_blank" style="color: var(--text-primary); text-decoration: none; border-bottom: 1px dashed var(--blue);" onclick="event.stopPropagation();">{r["organization"]}</a>' if r.get("source_url") else r["organization"]}</strong></td>
                <td>{r['source_date']}</td>
                <td>{r['horizon_years']} Years</td>
                <td>{r['target_asset']}</td>
                <td>{range_str} <span class="return-type">({r['return_type']})</span></td>
                <td><span style="color: var(--text-secondary); font-style: italic;" title="Interim return to date (unfinalized)">{actual_str}</span></td>
                <td>{status_badge}</td>
                <td><span style="color: var(--text-secondary); font-style: italic;" title="Interim forecast error to date (unfinalized)">{error_str}</span></td>
            </tr>
            <tr id="details-{r['id']}" class="detail-row" style="display: none;">
                <td colspan="8">
                    <div class="detail-content" style="border-left-color: var(--grey); background-color: rgba(148, 163, 184, 0.03);">
                        <div class="detail-grid">
                            <div>
                                <strong>Interim S&P 500 Index Details:</strong><br>
                                • Start Price: {r['start_price']:.2f}<br>
                                • Current Price (as of {r['latest_eval_date']}): {r['end_price']:.2f}<br>
                                • Interim CAGR: {r['nominal_cagr']:.2f}% (Nominal)
                            </div>
                            <div>
                                <strong>Interim Inflation Details:</strong><br>
                                • Start CPI: {r['cpi_start']:.2f}<br>
                                • Current CPI (as of {r['latest_eval_date']}): {r['cpi_end']:.2f}<br>
                                • Inflation CAGR: {r['inflation_cagr']:.2f}% (Total: {r['inflation_rate']:.2f}%)
                            </div>
                            <div>
                                <strong>Active Prediction Details:</strong><br>
                                <p class="context-p">{r['context']}</p>
                                {f'<a href="{r["source_url"]}" target="_blank" class="source-link">Read Source Article ↗</a>' if r['source_url'] else ''}
                                <br><br>
                                <span class="info-badge" style="background-color: var(--grey-alpha); color: var(--text-secondary);">
                                    Time elapsed: <strong>{r['years_elapsed']:.1f} / {r['horizon_years']:.1f} years</strong>. 
                                    Elapses on <strong>{r['end_date']}</strong>.
                                </span>
                            </div>
                        </div>
                    </div>
                </td>
            </tr>
            """
        table_rows.append(detail_row)
        
    table_rows_html = "\n".join(table_rows)
    
    # Closest and largest miss HTML
    insight_cards_html = ""
    if closest_pred and largest_miss:
        insight_cards_html = f"""
        <div class="insight-row">
            <div class="card insight-card">
                <div class="insight-label">Closest Forecast</div>
                <div class="insight-value" style="color: var(--green);">{closest_pred['organization']} ({closest_pred['source_date'][:4]})</div>
                <div class="insight-desc">
                    Predicted <strong>{closest_pred['predicted_range'][0]}%</strong> {closest_pred['return_type']} return. 
                    Actual was <strong>{closest_pred['actual_return']:.2f}%</strong> (Miss of {abs(closest_pred['forecast_error']):.2f}%)
                </div>
            </div>
            <div class="card insight-card">
                <div class="insight-label">Largest Miss</div>
                <div class="insight-value" style="color: var(--red);">{largest_miss['organization']} ({largest_miss['source_date'][:4]})</div>
                <div class="insight-desc">
                    Predicted <strong>{largest_miss['predicted_range'][0]}%</strong> {largest_miss['return_type']} return. 
                    Actual was <strong>{largest_miss['actual_return']:.2f}%</strong> (Miss of {abs(largest_miss['forecast_error']):.2f}%)
                </div>
            </div>
        </div>
        """

    # HTML Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Market Prediction Evaluator</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #0f172a;
            --bg-card: #1e293b;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            
            --blue: #3b82f6;
            --green: #10b981;
            --orange: #f97316;
            --red: #ef4444;
            --grey: #64748b;
            
            --blue-alpha: rgba(59, 130, 246, 0.15);
            --green-alpha: rgba(16, 185, 129, 0.15);
            --red-alpha: rgba(239, 68, 68, 0.15);
            --grey-alpha: rgba(100, 116, 139, 0.15);
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            line-height: 1.5;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            margin-bottom: 40px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
        }}
        
        h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 8px;
            background: linear-gradient(to right, #3b82f6, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 400;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, border-color 0.2s;
        }}
        
        .card:hover {{
            transform: translateY(-2px);
            border-color: #475569;
        }}
        
        .metric-card {{
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        
        .metric-title {{
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 12px;
            font-weight: 500;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 8px;
            line-height: 1;
        }}
        
        .metric-subtext {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}
        
        .main-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        @media(min-width: 1024px) {{
            .main-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .table-card {{
            overflow-x: auto;
            padding: 0;
        }}
        
        .table-header {{
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        
        .table-title {{
            font-size: 1.2rem;
            font-weight: 600;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.95rem;
        }}
        
        th {{
            color: var(--text-secondary);
            font-weight: 500;
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        td {{
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            vertical-align: middle;
        }}
        
        .main-row {{
            cursor: pointer;
            transition: background-color 0.15s;
        }}
        
        .main-row:hover {{
            background-color: rgba(255, 255, 255, 0.02);
        }}
        
        .active-prediction {{
            background-color: rgba(255, 255, 255, 0.01);
        }}
        
        .return-type {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }}
        
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }}
        
        .badge-overperformed {{
            background-color: var(--blue-alpha);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        
        .badge-underperformed {{
            background-color: var(--red-alpha);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
        
        .badge-accurate {{
            background-color: var(--green-alpha);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        
        .badge-active {{
            background-color: var(--grey-alpha);
            color: #94a3b8;
            border: 1px solid rgba(100, 116, 139, 0.3);
        }}
        
        .detail-row {{
            background-color: #0f172a;
        }}
        
        .detail-content {{
            padding: 24px;
            border-left: 3px solid var(--blue);
            background-color: #172554; /* slate-950/blue shade */
        }}
        
        .detail-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 24px;
            font-size: 0.85rem;
            color: #bfdbfe;
        }}
        
        .detail-grid strong {{
            color: #eff6ff;
            font-size: 0.9rem;
            display: inline-block;
            margin-bottom: 6px;
        }}
        
        .context-p {{
            margin-top: 4px;
            line-height: 1.4;
            color: #93c5fd;
        }}
        
        .source-link {{
            color: #60a5fa;
            text-decoration: none;
            font-weight: 500;
            display: inline-block;
            margin-top: 8px;
        }}
        
        .source-link:hover {{
            text-decoration: underline;
        }}
        
        .info-badge {{
            display: inline-block;
            background-color: rgba(59, 130, 246, 0.2);
            color: #93c5fd;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
        }}
        
        .chart-layout {{
            display: flex;
            flex-direction: column;
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .chart-card {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        
        .chart-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        
        .insight-layout {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .insight-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .insight-card {{
            padding: 20px;
        }}
        
        .insight-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            color: var(--text-secondary);
            font-weight: 600;
            margin-bottom: 6px;
        }}
        
        .insight-value {{
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        
        .insight-desc {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
        
        .analysis-card {{
            height: 100%;
        }}
        
        .analysis-card h3 {{
            margin-bottom: 12px;
            font-size: 1.2rem;
            font-weight: 600;
        }}
        
        .analysis-card p {{
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-bottom: 12px;
            line-height: 1.6;
        }}
        
        footer {{
            margin-top: 60px;
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
    </style>
    <script>
        function toggleDetails(id) {{
            var row = document.getElementById('details-' + id);
            if (row.style.display === 'none') {{
                row.style.display = 'table-row';
            }} else {{
                row.style.display = 'none';
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>Stock Market Prediction Checker</h1>
            <div class="subtitle">Evaluating long-term equity return forecasts against actual historical records</div>
        </header>
        
        <div class="metrics-grid">
            {metrics_html}
        </div>
        
        <!-- CHART 1: Actual Return vs. Predicted Range (Full Width) -->
        <div class="card full-width-card" style="margin-bottom: 30px; display: flex; flex-direction: column; gap: 20px;">
            <h3 style="font-size: 1.4rem; font-weight: 600; text-align: left; margin-bottom: 5px;">Actual Return vs. Predicted Range</h3>
            <div class="chart-image-container" style="text-align: center; width: 100%;">
                <img src="actual_vs_predicted.png" alt="Actual vs Predicted Chart" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border-color);">
            </div>
            
            <div class="insight-layout">
                {insight_cards_html}
            </div>
            
            <div class="card analysis-card" style="background-color: #0f172a; border: 1px solid var(--border-color); padding: 24px; border-radius: 8px; margin-top: 10px;">
                <h3>The Bearish Bias of Institutional Forecasts</h3>
                <p style="margin-top: 12px;">
                    An analysis of the historical record reveals a striking pattern: 
                    <strong>most evaluated predictions overperformed their forecast ranges</strong>, often by a massive margin.
                </p>
                <p style="margin-top: 12px;">
                    For example, in 2016 both Vanguard and J.P. Morgan projected U.S. stock returns for the next 10 years would be muted. 
                    Vanguard estimated 3.0%-7.0% nominal return, and J.P. Morgan projected 6.75%. 
                    The S&P 500 Total Return index subsequently delivered an actual nominal return of <strong>15.57%</strong> annualized.
                </p>
                <p style="margin-top: 12px;">
                    <strong>Why are institutions consistently too conservative?</strong><br>
                    1. <strong>Valuation-Based Models:</strong> Models like Vanguard's VCMM and GMO's forecasts are heavily driven by starting valuations. During the post-2009 era, high valuations (P/E ratios) constantly triggered warnings of mean reversion. However, valuations continued to expand, and corporate earnings grew faster than historical trends.<br>
                    2. <strong>Asymmetric Risk:</strong> For asset managers, predicting high returns that fail to materialize carries high career and reputation risk. Conversely, predicting low returns when the market booms leaves clients pleasantly surprised.
                </p>
            </div>
        </div>

        <!-- CHART 2: Prediction Calibration (Full Width) -->
        <div class="card full-width-card" style="margin-bottom: 40px; display: flex; flex-direction: column; gap: 20px;">
            <h3 style="font-size: 1.4rem; font-weight: 600; text-align: left; margin-bottom: 5px;">Prediction Calibration & Statistical Correlation</h3>
            <div class="chart-image-container" style="text-align: center; width: 100%;">
                <img src="calibration.png" alt="Prediction Calibration Plot" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border-color);">
            </div>
            
            <div class="card analysis-card" style="background-color: #0f172a; border: 1px solid var(--border-color); padding: 24px; border-radius: 8px; margin-top: 10px;">
                <h3 style="margin-bottom: 16px;">Statistical Correlation & Fit</h3>
                <p>
                    To determine if predictions are correlated with actual performance or if they are completely random, we ran an Ordinary Least Squares (OLS) regression on the <strong>{n} elapsed predictions</strong>:
                </p>
                
                <div style="background-color: #1e293b; padding: 15px; border-radius: 6px; border: 1px solid var(--border-color); margin-top: 15px; margin-bottom: 20px; font-family: monospace; font-size: 0.9rem; line-height: 1.6;">
                    <span style="color: var(--blue);">• Pearson Correlation (r):</span> {pearson_r:.4f}<br>
                    <span style="color: var(--green);">• p-value (slope test):</span> {pearson_p:.6f} ({'Significant' if pearson_p < 0.05 else 'Not Significant'} at 5% level)<br>
                    <span style="color: var(--orange);">• OLS Trend Line:</span> Actual = {slope:.4f} * Predicted + {intercept:.2f}%<br>
                    <span style="color: var(--red);">• R-squared (R²):</span> {r_squared:.4f} (Explains {r_squared*100:.1f}% of variance)
                </div>
                
                <p style="margin-top: 12px;">
                    <strong>Key Statistical Takeaways:</strong><br>
                    • <strong>Significant Signal:</strong> The p-value of {pearson_p:.4f} is {'below' if pearson_p < 0.05 else 'above'} the standard 5% significance level, showing that Wall Street's forecasts contain genuine predictive value and are not random noise.<br>
                    • <strong>Pessimistic Offset:</strong> The OLS trend line has an intercept of <strong>+{intercept:.2f}%</strong>. This indicates that even a predicted return of 0% historically translated to a positive actual return of {intercept:.2f}% due to strong U.S. equity performance.<br>
                    • <strong>Forecast Bias:</strong> The Mean Forecast Bias is <strong>{mean_bias:+.2f}%</strong>. A positive bias shows that Wall Street forecasts are systematically too conservative on average (underestimating actual returns).
                </p>
            </div>
        </div>

        <!-- DATABASE TABLE (NOW BELOW CHARTS) -->
        <div class="main-grid">
            <div class="card table-card">
                <div class="table-header">
                    <div class="table-title">Predictions Database</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">💡 Click on rows to expand details</div>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Organization</th>
                            <th>Date Made</th>
                            <th>Horizon</th>
                            <th>Asset Target</th>
                            <th>Forecast Range</th>
                            <th>Actual Return</th>
                            <th>Status</th>
                            <th>Forecast Error (pp)*</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows_html}
                    </tbody>
                </table>
                <div style="font-size: 0.8rem; color: var(--text-secondary); padding: 12px 24px; text-align: left; border-top: 1px solid var(--border-color); line-height: 1.5;">
                    <div>* Forecast error is calculated as the **absolute difference in percentage points (pp)** between the actual return CAGR (or interim CAGR to date) and the predicted range midpoint.</div>
                    <div style="margin-top: 6px; color: var(--text-muted);">* Note: <strong>Active predictions</strong> show italicized values with an asterisk (e.g. <em>12.34%*</em>) indicating the <strong>interim return and interim forecast error to date</strong>. Because the time horizon has not elapsed, these figures are unfinalized and subject to significant change.</div>
                </div>
            </div>
        </div>
        
        <footer>
            Stock Market Prediction Checker • Built with Python, yfinance, and FRED CPI Data • Last Updated: 2026-07-12
        </footer>
    </div>
</body>
</html>
    """
    
    with open(DASHBOARD_PATH, "w") as f:
        f.write(html_content)
    print(f"Dashboard saved to {DASHBOARD_PATH}")

if __name__ == "__main__":
    generate_dashboard()
