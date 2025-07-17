# DeFi Credit Scoring Model for Aave V2

This project provides a machine learning model to generate credit scores for wallets interacting with the Aave V2 protocol. The model analyzes historical transaction data to assign a score between 0 and 1000, where higher scores indicate more reliable and responsible on-chain behavior.

## Table of Contents
- [Methodology](#methodology)
- [Architecture and Processing Flow](#architecture-and-processing-flow)
- [Features](#features)
- [How to Run](#how-to-run)
- [Analysis](#analysis)

## Methodology

The credit scoring model is a **heuristic-based system**. This approach was chosen for its transparency and explainability, which are crucial for any credit scoring system. The model calculates a score for each wallet by analyzing several key features engineered from their transaction history. Each feature is assigned a weight that reflects its importance in determining a user's creditworthiness.

The final score is a weighted sum of these normalized features, which is then scaled to a range of 0 to 1000.

## Architecture and Processing Flow

The entire process is encapsulated in a single Python script (`generate_scores.py`) and follows these steps:

1.  **Data Loading:** The script loads the raw transaction data from the provided JSON file.
2.  **Preprocessing:** The data is cleaned and preprocessed. This includes parsing nested JSON, converting data types, and calculating the USD value of each transaction.
3.  **Feature Engineering:** The script computes a set of features for each unique wallet address based on its historical activity.
4.  **Scoring:** The engineered features are normalized and combined using a weighted formula to produce a raw score.
5.  **Scaling:** The raw scores are scaled to the final 0-1000 range.
6.  **Output:** The script outputs a CSV file (`wallet_scores.csv`) containing each wallet and its calculated credit score. It also generates a plot (`score_distribution.png`) showing the distribution of the scores.

## Features

The following features are engineered for each wallet and used in the scoring model:

* **`wallet_age_days` (Weight: 0.15):** The number of days between a wallet's first and last transaction. A longer history is a sign of a more established user.
* **`total_transactions` (Weight: 0.10):** The total number of transactions. Higher activity can indicate a more engaged user.
* **`total_deposited_usd` (Weight: 0.20):** The total USD value of all deposits. This serves as a proxy for the user's overall financial capacity and trust in the protocol.
* **`repayment_to_borrow_ratio` (Weight: 0.30):** The ratio of total repaid USD to total borrowed USD. A ratio greater than or equal to 1 is a strong positive signal of responsible borrowing.
* **`liquidation_count` (Weight: -0.25):** The number of times a wallet has been liquidated. This is a significant negative indicator, as it signals high-risk behavior.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Download the data:**
    Download the `user-wallet-transactions.json` file and place it in the root of the project directory.

4.  **Run the script:**
    ```bash
    python generate_scores.py user-wallet-transactions.json
    ```

This will generate two files:
* `wallet_scores.csv`: A CSV file with wallet addresses and their credit scores.
* `score_distribution.png`: A PNG image showing the distribution of the credit scores.

## Analysis

For a detailed analysis of the scoring results, please see the `analysis.md` file.
