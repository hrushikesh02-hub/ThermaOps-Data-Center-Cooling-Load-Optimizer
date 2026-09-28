import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path


from ml.config import DATA_DIR, FIG_EDA_DIR, NUMERICAL_FEATURES
from ml.utils import load_data, setup_logger

logger = setup_logger("eda")

def plot_distribution(df, col, title, xlabel, filename):
    if col not in df.columns:
        logger.warning(f"Column {col} not in dataset, skipping plot.")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(FIG_EDA_DIR / filename, dpi=300)
    plt.close()

def plot_scatter(df, x_col, y_col, title, xlabel, ylabel, filename):
    if x_col not in df.columns or y_col not in df.columns:
        logger.warning(f"Columns {x_col} or {y_col} not in dataset, skipping plot.")
        return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIG_EDA_DIR / filename, dpi=300)
    plt.close()
    
def plot_time_series(df, time_col, y_col, title, ylabel, filename):
    if time_col not in df.columns or y_col not in df.columns:
        logger.warning(f"Columns {time_col} or {y_col} not in dataset, skipping plot.")
        return
    plt.figure(figsize=(14, 6))
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    # Take a sample if too large, or plot all. 7000 rows is fine for a line plot.
    plt.plot(df_sorted[time_col], df_sorted[y_col], alpha=0.8, linewidth=0.5)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIG_EDA_DIR / filename, dpi=300)
    plt.close()

def plot_boxplot(df, x_col, y_col, title, xlabel, ylabel, filename):
    if x_col not in df.columns or y_col not in df.columns:
        logger.warning(f"Columns {x_col} or {y_col} not in dataset, skipping plot.")
        return
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[x_col], y=df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIG_EDA_DIR / filename, dpi=300)
    plt.close()

def run_eda():
    logger.info("Starting EDA...")
    
    # We use the train dataset for EDA to avoid looking at the test set
    train_path = DATA_DIR / "train.csv"
    df = load_data(train_path)
    
    # 1. Cooling load distribution
    plot_distribution(df, 'cooling_load_kw', 'Cooling Load Distribution', 'Cooling Load (kW)', '1_cooling_load_dist.png')
    
    # 2. Cooling power distribution
    plot_distribution(df, 'cooling_power_kw', 'Cooling Power Distribution', 'Cooling Power (kW)', '2_cooling_power_dist.png')
    
    # 3. Server inlet temperature distribution
    plot_distribution(df, 'server_inlet_temperature_c', 'Server Inlet Temperature Distribution', 'Server Inlet Temp (°C)', '3_server_inlet_temp_dist.png')
    
    # 4. Hotspot temperature distribution
    plot_distribution(df, 'hotspot_temperature_c', 'Hotspot Temperature Distribution', 'Hotspot Temp (°C)', '4_hotspot_temp_dist.png')
    
    # 5. CPU utilization vs cooling load
    plot_scatter(df, 'cpu_utilization_pct', 'cooling_load_kw', 'CPU Utilization vs Cooling Load', 'CPU Utilization (%)', 'Cooling Load (kW)', '5_cpu_vs_cooling_load.png')
    
    # 6. IT power vs cooling load
    plot_scatter(df, 'it_power_kw', 'cooling_load_kw', 'IT Power vs Cooling Load', 'IT Power (kW)', 'Cooling Load (kW)', '6_it_power_vs_cooling_load.png')
    
    # 7. Outdoor temperature vs cooling load
    plot_scatter(df, 'outdoor_temperature_c', 'cooling_load_kw', 'Outdoor Temp vs Cooling Load', 'Outdoor Temp (°C)', 'Cooling Load (kW)', '7_outdoor_temp_vs_cooling_load.png')
    
    # 8. Airflow vs cooling load
    plot_scatter(df, 'airflow_cfm', 'cooling_load_kw', 'Airflow vs Cooling Load', 'Airflow (CFM)', 'Cooling Load (kW)', '8_airflow_vs_cooling_load.png')
    
    # 9. Supply-air temperature vs cooling load
    plot_scatter(df, 'supply_air_temperature_c', 'cooling_load_kw', 'Supply Air Temp vs Cooling Load', 'Supply Air Temp (°C)', 'Cooling Load (kW)', '9_supply_air_vs_cooling_load.png')
    
    # 10. Cooling load over time
    plot_time_series(df, 'timestamp', 'cooling_load_kw', 'Cooling Load Over Time', 'Cooling Load (kW)', '10_cooling_load_over_time.png')
    
    # 11. Hotspot temperature over time
    plot_time_series(df, 'timestamp', 'hotspot_temperature_c', 'Hotspot Temperature Over Time', 'Hotspot Temp (°C)', '11_hotspot_temp_over_time.png')
    
    # 12. Correlation matrix
    logger.info("Generating correlation matrix...")
    plt.figure(figsize=(16, 12))
    # Select available numerical columns and add targets
    cols_for_corr = [c for c in NUMERICAL_FEATURES if c in df.columns]
    for t in ['cooling_load_kw', 'server_inlet_temperature_c', 'hotspot_temperature_c', 'cooling_power_kw']:
        if t in df.columns and t not in cols_for_corr:
            cols_for_corr.append(t)
            
    corr = df[cols_for_corr].corr()
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
    plt.title('Correlation Matrix (Numerical Features & Targets)')
    plt.tight_layout()
    plt.savefig(FIG_EDA_DIR / '12_correlation_matrix.png', dpi=300)
    plt.close()
    
    # 13. Cooling load by season
    plot_boxplot(df, 'season', 'cooling_load_kw', 'Cooling Load by Season', 'Season', 'Cooling Load (kW)', '13_cooling_load_by_season.png')
    
    # 14. Cooling load by workload level (binning cpu_utilization)
    if 'cpu_utilization_pct' in df.columns:
        df['workload_level'] = pd.qcut(df['cpu_utilization_pct'], q=3, labels=['Low', 'Medium', 'High'])
        plot_boxplot(df, 'workload_level', 'cooling_load_kw', 'Cooling Load by Workload Level', 'Workload Level', 'Cooling Load (kW)', '14_cooling_load_by_workload.png')
        
    logger.info("EDA complete. Figures saved.")

if __name__ == "__main__":
    run_eda()
