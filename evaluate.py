import os
import json
import datetime
import urllib.request
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# File paths
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(REPO_DIR, "predictions.json")
RESULTS_PATH = os.path.join(REPO_DIR, "results.json")
CPI_CACHE_PATH = os.path.join(REPO_DIR, "cpi_cache.csv")
MARKET_CACHE_PATH = os.path.join(REPO_DIR, "market_cache.csv")
CHART_PATH = os.path.join(REPO_DIR, "actual_vs_predicted.png")

# Data sources
CPI_URL = "https://r2.datahub.io/clt9801yk0003jm081nzgs9iy/main/raw/data/cpiai.csv"
MARKET_TICKER = "^SP500TR"

def fetch_cpi_data():
    """Fetch CPI data from cache or remote URL."""
    if os.path.exists(CPI_CACHE_PATH):
        print("Loading CPI data from cache...")
        return pd.read_csv(CPI_CACHE_PATH)
    
    print(f"Downloading CPI data from {CPI_URL}...")
    try:
        req = urllib.request.Request(CPI_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = response.read().decode('utf-8')
        
        # Save to cache
        with open(CPI_CACHE_PATH, "w") as f:
            f.write(data)
        
        return pd.read_csv(CPI_CACHE_PATH)
    except Exception as e:
        print(f"⚠️ Error fetching CPI data: {e}. Attempting to fallback.")
        raise e

def fetch_market_data():
    """Fetch monthly S&P 500 Total Return data from cache or yfinance."""
    if os.path.exists(MARKET_CACHE_PATH):
        print("Loading S&P 500 TR data from cache...")
        df = pd.read_csv(MARKET_CACHE_PATH)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        return df.set_index('Date')
    
    print(f"Fetching S&P 500 TR data ({MARKET_TICKER}) from yfinance...")
    ticker = yf.Ticker(MARKET_TICKER)
    hist = ticker.history(period="max", interval="1mo")
    
    # Save to cache
    hist.to_csv(MARKET_CACHE_PATH)
    return hist

def parse_date(date_str):
    """Parse date string into datetime object."""
    return datetime.datetime.strptime(date_str, "%Y-%m-%d")

def get_cpi_value(date, cpi_df):
    """Get U.S. CPI value for a given date (month)."""
    # format date to YYYY-MM
    target_ym = date.strftime("%Y-%m")
    
    # In CPI dataset, Date is YYYY-MM-DD
    # Find match where Date starts with target_ym
    match = cpi_df[cpi_df['Date'].str.startswith(target_ym)]
    if not match.empty:
        return float(match.iloc[0]['Index'])
    
    # Fallback to closest available date
    # Convert dates to datetime
    cpi_df_temp = cpi_df.copy()
    cpi_df_temp['ParsedDate'] = pd.to_datetime(cpi_df_temp['Date'])
    cpi_df_temp['Diff'] = (cpi_df_temp['ParsedDate'] - date).abs()
    closest = cpi_df_temp.sort_values('Diff').iloc[0]
    print(f"  ⚠️ Exact CPI match not found for {target_ym}. Using closest: {closest['Date']}")
    return float(closest['Index'])

def get_market_price(date, market_df):
    """Get monthly close price for S&P 500 TR."""
    target_ym = date.strftime("%Y-%m")
    
    # Convert index of market_df to string format for matching
    market_df_str = market_df.copy()
    market_df_str.index = market_df_str.index.strftime('%Y-%m')
    
    if target_ym in market_df_str.index:
        row = market_df_str.loc[target_ym]
        # In case of multiple rows for that month, take the last close
        if isinstance(row, pd.DataFrame):
            return float(row.iloc[-1]['Close'])
        return float(row['Close'])
    
    # Fallback to closest month
    market_df_temp = market_df.copy()
    market_df_temp['Diff'] = (market_df_temp.index.tz_localize(None) - date).abs()
    closest = market_df_temp.sort_values('Diff').iloc[0]
    closest_date = market_df_temp.sort_values('Diff').index[0].strftime('%Y-%m')
    print(f"  ⚠️ Exact market match not found for {target_ym}. Using closest: {closest_date}")
    return float(closest['Close'])

def evaluate_all():
    print("Evaluating predictions...")
    with open(PREDICTIONS_PATH, "r") as f:
        predictions = json.load(f)
        
    cpi_df = fetch_cpi_data()
    market_df = fetch_market_data()
    
    # Latest available dates in data
    latest_cpi_date = pd.to_datetime(cpi_df['Date']).max()
    latest_market_date = market_df.index.tz_localize(None).max()
    
    results = []
    
    for pred in predictions:
        print(f"Evaluating {pred['id']} ({pred['organization']})...")
        
        start_date = parse_date(pred['source_date'])
        horizon = pred['horizon_years']
        
        # Calculate target end date
        end_date = start_date.replace(year=start_date.year + horizon)
        
        # Check if elapsed
        is_elapsed = end_date <= latest_market_date and (pred['return_type'] == 'nominal' or end_date <= latest_cpi_date)
        
        eval_result = pred.copy()
        eval_result['end_date'] = end_date.strftime("%Y-%m-%d")
        eval_result['is_elapsed'] = is_elapsed
        
        if is_elapsed:
            p_start = get_market_price(start_date, market_df)
            p_end = get_market_price(end_date, market_df)
            
            nominal_return_ratio = p_end / p_start
            nominal_cagr = (nominal_return_ratio ** (1.0 / horizon) - 1.0) * 100.0
            
            cpi_start = get_cpi_value(start_date, cpi_df)
            cpi_end = get_cpi_value(end_date, cpi_df)
            inflation_ratio = cpi_end / cpi_start
            inflation_rate = (inflation_ratio - 1.0) * 100.0
            inflation_cagr = (inflation_ratio ** (1.0 / horizon) - 1.0) * 100.0
            
            real_return_ratio = nominal_return_ratio / inflation_ratio
            real_cagr = (real_return_ratio ** (1.0 / horizon) - 1.0) * 100.0
            
            # Determine actual return based on prediction type
            actual_return = real_cagr if pred['return_type'] == 'real' else nominal_cagr
            
            # Determine accuracy
            p_range = pred['predicted_range']
            
            # If point forecast, define range with +/- 0.5% tolerance
            if p_range[0] == p_range[1]:
                low_bound = p_range[0] - 0.5
                high_bound = p_range[0] + 0.5
            else:
                low_bound = p_range[0]
                high_bound = p_range[1]
                
            if actual_return < low_bound:
                status = "Underperformed"
            elif actual_return > high_bound:
                status = "Overperformed"
            else:
                status = "Accurate"
                
            eval_result.update({
                "start_price": p_start,
                "end_price": p_end,
                "nominal_cagr": nominal_cagr,
                "cpi_start": cpi_start,
                "cpi_end": cpi_end,
                "inflation_rate": inflation_rate,
                "inflation_cagr": inflation_cagr,
                "real_cagr": real_cagr,
                "actual_return": actual_return,
                "status": status,
                "forecast_error": actual_return - ((p_range[0] + p_range[1]) / 2.0)
            })
        else:
            # For active predictions, calculate the interim metrics
            latest_eval_date = min(latest_market_date, latest_cpi_date)
            
            p_start = get_market_price(start_date, market_df)
            p_latest = get_market_price(latest_eval_date, market_df)
            nominal_return_ratio = p_latest / p_start
            
            cpi_start = get_cpi_value(start_date, cpi_df)
            cpi_latest = get_cpi_value(latest_eval_date, cpi_df)
            inflation_ratio = cpi_latest / cpi_start
            inflation_rate = (inflation_ratio - 1.0) * 100.0
            
            years_elapsed = (latest_eval_date - start_date).days / 365.25
            
            if years_elapsed > 0:
                nominal_interim_cagr = (nominal_return_ratio ** (1.0 / years_elapsed) - 1.0) * 100.0
                inflation_cagr = (inflation_ratio ** (1.0 / years_elapsed) - 1.0) * 100.0
                real_interim_cagr = ((nominal_return_ratio / inflation_ratio) ** (1.0 / years_elapsed) - 1.0) * 100.0
            else:
                nominal_interim_cagr = 0.0
                inflation_cagr = 0.0
                real_interim_cagr = 0.0
                
            actual_interim = real_interim_cagr if pred['return_type'] == 'real' else nominal_interim_cagr
            
            eval_result.update({
                "status": "Active",
                "start_price": p_start,
                "end_price": p_latest,
                "nominal_cagr": nominal_interim_cagr,
                "cpi_start": cpi_start,
                "cpi_end": cpi_latest,
                "inflation_rate": inflation_rate,
                "inflation_cagr": inflation_cagr,
                "real_cagr": real_interim_cagr,
                "actual_return": actual_interim,
                "forecast_error": actual_interim - ((pred['predicted_range'][0] + pred['predicted_range'][1]) / 2.0),
                "years_elapsed": years_elapsed,
                "latest_eval_date": latest_eval_date.strftime("%Y-%m-%d")
            })
            
        results.append(eval_result)
        
    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")
    
    # Generate chart
    generate_chart(results)

def generate_chart(results):
    """Generate actual vs predicted return chart."""
    elapsed_results = [r for r in results if r['is_elapsed']]
    if not elapsed_results:
        print("No elapsed results to plot.")
        return
        
    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#1e1e24")
    ax.set_facecolor("#1e1e24")
    
    labels = []
    predicted_means = []
    predicted_errors_low = []
    predicted_errors_high = []
    actuals = []
    colors = []
    
    for i, r in enumerate(elapsed_results):
        label = f"{r['organization']}\n({r['source_date'][:4]}, {r['horizon_years']}y {r['return_type']})"
        labels.append(label)
        
        p_range = r['predicted_range']
        p_mean = sum(p_range) / 2.0
        predicted_means.append(p_mean)
        
        # Error bar values
        predicted_errors_low.append(p_mean - p_range[0])
        predicted_errors_high.append(p_range[1] - p_mean)
        
        actuals.append(r['actual_return'])
        
        # Color based on status
        if r['status'] == "Accurate":
            colors.append("#2ecc71") # Green
        elif r['status'] == "Overperformed":
            colors.append("#3498db") # Blue
        else:
            colors.append("#e74c3c") # Red
            
    x = range(len(labels))
    
    # Plot predicted ranges as error bars
    ax.errorbar(x, predicted_means, yerr=[predicted_errors_low, predicted_errors_high], 
                fmt='o', color='#95a5a6', ecolor='#7f8c8d', elinewidth=3, capsize=8, 
                label='Predicted Range', markersize=8)
                
    # Plot actual values as scatter dots
    ax.scatter(x, actuals, color=colors, s=150, zorder=5, label='Actual Return')
    
    # Draw annotations
    for i, r in enumerate(elapsed_results):
        ax.annotate(f"{actuals[i]:.1f}%", (i, actuals[i]), textcoords="offset points", 
                    xytext=(0,10), ha='center', color='white', fontweight='bold', fontsize=9)
        p_range = r['predicted_range']
        range_str = f"{p_range[0]}%" if p_range[0] == p_range[1] else f"{p_range[0]}-{p_range[1]}%"
        ax.annotate(f"Pred: {range_str}", (i, sum(p_range)/2), textcoords="offset points", 
                    xytext=(15,-5), ha='left', color='#b2bec3', fontsize=8)
                    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='white', fontsize=8, rotation=45, ha='right')
    ax.tick_params(colors='white')
    
    # Title & Labels
    ax.set_title("Historical Stock Return Predictions vs. Actual Outcomes", color='white', fontsize=14, pad=20, fontweight='bold')
    ax.set_ylabel("Annualized Return (%)", color='white', fontsize=12)
    
    # Grid & Spines
    ax.grid(True, linestyle='--', alpha=0.1, color='white')
    for spine in ax.spines.values():
        spine.set_color('#2d3436')
        
    # Legend
    legend = ax.legend(facecolor='#2d3436', edgecolor='none', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Chart saved to {CHART_PATH}")

if __name__ == "__main__":
    evaluate_all()
