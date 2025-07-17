import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the transaction data from the JSON file.

    Args:
        file_path (str): The path to the user-transactions.json file.

    Returns:
        pandas.DataFrame: A preprocessed DataFrame with relevant columns.
    """
    print("Loading and preprocessing data...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

    df = pd.json_normalize(data)

    # Select and rename columns for clarity
    df = df[['userWallet', 'action', 'timestamp', 'actionData.amount', 'actionData.assetPriceUSD']]
    df.columns = ['wallet', 'action', 'timestamp', 'amount', 'price_usd']

    # Convert data types
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['price_usd'] = pd.to_numeric(df['price_usd'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Calculate the USD value of each transaction
    df['amount_usd'] = df['amount'] * df['price_usd']
    
    # Drop rows with missing values that are critical for our analysis
    df.dropna(subset=['amount_usd', 'wallet', 'action'], inplace=True)

    print("Data loaded and preprocessed successfully.")
    return df

def feature_engineering(df):
    """
    Engineers features for each wallet based on their transaction history.

    Args:
        df (pandas.DataFrame): The preprocessed transaction DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame with engineered features for each wallet.
    """
    print("Engineering features...")
    # Group by wallet to aggregate features
    wallets_df = df.groupby('wallet').agg(
        total_transactions=('action', 'count'),
        first_transaction=('timestamp', 'min'),
        last_transaction=('timestamp', 'max')
    ).reset_index()

    # Calculate wallet age in days
    wallets_df['wallet_age_days'] = (wallets_df['last_transaction'] - wallets_df['first_transaction']).dt.days

    # Calculate transaction frequency (transactions per day)
    wallets_df['transaction_frequency'] = wallets_df['total_transactions'] / (wallets_df['wallet_age_days'] + 1) # Add 1 to avoid division by zero

    # Calculate total USD value for different actions
    action_pivot = df.pivot_table(index='wallet', columns='action', values='amount_usd', aggfunc='sum').fillna(0)
    wallets_df = wallets_df.merge(action_pivot, on='wallet', how='left').fillna(0)

    # Rename columns for clarity
    wallets_df.rename(columns={
        'deposit': 'total_deposited_usd',
        'borrow': 'total_borrowed_usd',
        'repay': 'total_repaid_usd',
        'redeemunderlying': 'total_redeemed_usd',
        'liquidationcall': 'total_liquidated_usd'
    }, inplace=True)
    
    # Count liquidations
    liquidation_counts = df[df['action'] == 'liquidationcall'].groupby('wallet').size().reset_index(name='liquidation_count')
    wallets_df = wallets_df.merge(liquidation_counts, on='wallet', how='left').fillna(0)


    # Calculate financial ratios
    wallets_df['borrow_to_deposit_ratio'] = wallets_df['total_borrowed_usd'] / (wallets_df['total_deposited_usd'] + 1e-6)
    wallets_df['repayment_to_borrow_ratio'] = wallets_df['total_repaid_usd'] / (wallets_df['total_borrowed_usd'] + 1e-6)

    print("Feature engineering complete.")
    return wallets_df

def calculate_credit_scores(features_df):
    """
    Calculates credit scores for each wallet based on engineered features.

    Args:
        features_df (pandas.DataFrame): DataFrame with engineered features.

    Returns:
        pandas.DataFrame: DataFrame with an added 'credit_score' column.
    """
    print("Calculating credit scores...")
    # Define weights for each feature
    weights = {
        'wallet_age_days': 0.15,
        'total_transactions': 0.1,
        'total_deposited_usd': 0.2,
        'repayment_to_borrow_ratio': 0.3,
        'liquidation_count': -0.25, # Negative weight for liquidations
    }

    # Normalize the features to a 0-1 scale
    scaler = MinMaxScaler()
    
    # Select only the features that will be used in scoring
    scoring_features = features_df[list(weights.keys())].copy()
    
    # Handle the case where a feature might not exist (e.g., no liquidations in the dataset)
    for feature in weights.keys():
        if feature not in scoring_features.columns:
            scoring_features[feature] = 0

    scoring_features_scaled = scaler.fit_transform(scoring_features)
    
    # Convert back to a DataFrame
    scoring_features_scaled = pd.DataFrame(scoring_features_scaled, columns=list(weights.keys()))

    # Calculate the weighted score
    weighted_score = np.dot(scoring_features_scaled, list(weights.values()))
    
    # Scale the scores to be between 0 and 1000
    score_scaler = MinMaxScaler(feature_range=(0, 1000))
    features_df['credit_score'] = score_scaler.fit_transform(weighted_score.reshape(-1, 1))
    
    print("Credit scores calculated.")
    return features_df

def generate_analysis(scored_df):
    """
    Generates and saves a score distribution graph.
    """
    print("Generating analysis graph...")
    plt.figure(figsize=(12, 6))
    sns.histplot(scored_df['credit_score'], bins=20, kde=True)
    plt.title('Distribution of Credit Scores')
    plt.xlabel('Credit Score')
    plt.ylabel('Number of Wallets')
    plt.grid(True)
    plt.savefig('score_distribution.png')
    print("Analysis graph 'score_distribution.png' saved.")

def main():
    parser = argparse.ArgumentParser(description="Generate DeFi credit scores from Aave V2 transaction data.")
    parser.add_argument("json_file", help="Path to the user-transactions.json file.")
    args = parser.parse_args()

    # 1. Load and preprocess data
    df = load_and_preprocess_data(args.json_file)
    if df is None:
        return

    # 2. Engineer features
    features_df = feature_engineering(df)

    # 3. Calculate credit scores
    scored_df = calculate_credit_scores(features_df)

    # 4. Save the results
    output_file = 'wallet_scores.csv'
    scored_df[['wallet', 'credit_score']].to_csv(output_file, index=False)
    print(f"Scores saved to {output_file}")

    # 5. Generate analysis
    generate_analysis(scored_df)

if __name__ == "__main__":
    main()