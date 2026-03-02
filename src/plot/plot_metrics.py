# %%
# CELL 1: Data Loading (Run this once)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load your persisted predictions
# We use dummy data here to simulate your CSV load
# df = pd.read_csv('predictions.csv')
df = pd.DataFrame(
    {"y_true": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1], "y_pred": [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]}
)
print("Data loaded! Ready to plot.")

# %%
# CELL 2: The Confusion Matrix (Iterate on this cell)
# 1. Calculate the raw matrix numbers using Scikit-Learn
cm = confusion_matrix(df["y_true"], df["y_pred"])

# 2. Clear out any previous grid lines for a clean heatmap
sns.set_theme(style="white")
plt.figure(figsize=(6, 5))

# 3. Draw the beautiful Seaborn heatmap
sns.heatmap(
    cm,
    annot=True,  # Show the actual numbers inside the boxes
    fmt="d",  # 'd' formats the numbers as integers (no messy decimals)
    cmap="Blues",  # A highly professional color palette
    cbar=False,  # Hides the colorbar on the side for a cleaner look
    square=True,  # Forces the boxes to be perfect squares
    linewidths=1,  # Adds a tiny gridline between boxes
    linecolor="black",
)

# 4. Add clear labels
plt.title("Model Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)

# 5. Display in the interactive window
plt.show()

# %%
