import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing


file_path = r"C:\VS Code\Python\survey\SurveySchema.csv"
df = pd.read_csv(file_path, low_memory=False)


language_cols = [col for col in df.columns if 'Language' in col]
framework_cols = [col for col in df.columns if 'ML Framework' in col or 'Q28' in col]
job_role_col = next((col for col in df.columns if 'Job Role' in col or 'Q5' in col), None)


def summarize_selection(df, columns, label):
    usage_counts = df[columns].apply(pd.Series.value_counts).sum(axis=1)
    usage_df = usage_counts.reset_index()
    usage_df.columns = [label, 'Count']
    usage_df = usage_df.sort_values(by='Count', ascending=False)
    return usage_df

language_usage = summarize_selection(df, language_cols, 'Language')
framework_usage = summarize_selection(df, framework_cols, 'Framework')
job_roles = df[job_role_col].value_counts().reset_index()
job_roles.columns = ['Job Role', 'Count']


def plot_top(df, x_col, y_col, title, top_n=10):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.head(top_n), x='Count', y=x_col, palette='coolwarm')
    plt.title(title)
    plt.xlabel('Count')
    plt.ylabel(x_col)
    plt.tight_layout()
    plt.show()

plot_top(language_usage, 'Language', 'Count', 'Top Programming Languages (2025)')
plot_top(framework_usage, 'Framework', 'Count', 'Top ML Frameworks (2025)')
plot_top(job_roles, 'Job Role', 'Count', 'Top Job Roles (2025)')


years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
python_usage = [50, 60, 72, 83, 91, 102, 118, 135]

df_trend = pd.DataFrame({
    'Year': years,
    'Python_Users': python_usage
}).set_index('Year')


model = ExponentialSmoothing(df_trend['Python_Users'], trend='add', seasonal=None)
fit = model.fit()
forecast_2026 = fit.forecast(1)
df_trend.loc[2026] = forecast_2026.values[0]


plt.figure(figsize=(8, 5))
plt.plot(df_trend.index, df_trend['Python_Users'], marker='o', label='Python Users')
plt.axvline(2025.5, color='gray', linestyle='--', label='Forecast Start')
plt.title('Forecast of Python Usage (2026)')
plt.xlabel('Year')
plt.ylabel('User Count (normalized)')
plt.legend()
plt.tight_layout()
plt.show()

print("📈 Forecast for Python Users in 2026:", round(forecast_2026.values[0]))
