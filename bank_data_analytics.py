# Add libraries
import pandas as pd
import numpy as np

# 1. Detailed Analysis Logic
def analyze_transactions(df):
    # Separate Income from Expenses using the Amount column
    income_df = df[df['amount'] > 0]
    expense_df = df[df['amount'] < 0]

    # Overall Totals
    total_income = income_df['amount'].sum()
    total_expenses = abs(expense_df['amount'].sum()) # Use absolute value for cleaner report
    net_savings = total_income - total_expenses

    # Group by Main Category
    # NOTE: main_category is blank until we label transactions by hand or run
    # our ML models (ml_models.py), so unlabeled rows are dropped here.
    category_totals = (
        expense_df.dropna(subset=["main_category"])
        .groupby("main_category")["amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
    )
    
    return {
        "income": total_income,
        "expenses": total_expenses,
        "savings": net_savings,
        "categories": category_totals
    }

# 2. Pretty Summary Report
def generate_summary_report(stats):
    report = []
    report.append("="*30)
    report.append("      FINANCIAL SUMMARY      ")
    report.append("="*30)
    report.append(f"Total Income:   ${stats['income']:>10.2f}")
    report.append(f"Total Expenses: ${stats['expenses']:>10.2f}")
    report.append(f"Net Savings:    ${stats['savings']:>10.2f}")
    report.append("-"*30)
    report.append("TOP SPENDING CATEGORIES:")
    
    # Loop through the categorized totals
    if stats["categories"].empty:
        report.append("   (no categorized spending yet - label rows or run ml_models.py)")
    else:
        for cat, amt in stats["categories"].items():
            report.append(f" - {cat:<15}: ${amt:>8.2f}")
    
    report.append("="*30)
    return "\n".join(report)

# 3. Updated Main Process
def process_bank_statement():
    
    # Grab the finished file from bank_statement_processor
    df = pd.read_csv("all_banks_final_categorized.csv")
    
    # Run the analysis
    stats = analyze_transactions(df)
    
    # Print the pretty report
    print(generate_summary_report(stats))
    
    # Optional: Save the categorized CSV for your records
    # df.to_csv("categorized_spending.csv", index=False)


# Run the analytics functions
process_bank_statement()