import numpy as np
import pandas as pd
import os

OUTPUT = "demo_creditcard.csv"
N_SAMPLES = 2000
N_FEATURES = 10
FRAUD_RATIO = 0.02

np.random.seed(42)
X = np.random.normal(0, 1, size=(N_SAMPLES, N_FEATURES))
amount = np.abs(np.random.exponential(scale=50, size=N_SAMPLES))
time = np.random.randint(0, 86400, size=N_SAMPLES)

# Create imbalance
y = np.zeros(N_SAMPLES, dtype=int)
fraud_count = max(1, int(N_SAMPLES * FRAUD_RATIO))
fraud_idx = np.random.choice(N_SAMPLES, size=fraud_count, replace=False)
y[fraud_idx] = 1

cols = [f"V{i+1}" for i in range(N_FEATURES)] + ["Amount", "Time", "Class"]
df = pd.DataFrame(np.hstack([X, amount.reshape(-1,1), time.reshape(-1,1), y.reshape(-1,1)]), columns=cols)
# Ensure numeric types
for c in cols:
    df[c] = pd.to_numeric(df[c])

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Writing {OUTPUT} with {len(df)} samples ({fraud_count} fraud cases)")
df.to_csv(OUTPUT, index=False)
print("Done")
