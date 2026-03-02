# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load your training data
df = pd.read_csv(
    "/home/niklas/focus_offset_public/logs/jiang2018/fourier_domain/metrics.csv"
)

# %%
# 2. Plot the Training Loss
sns.lineplot(data=df, x="epoch", y="train_loss", label="Train Loss")

# 3. Plot the Validation Loss on the exact same graph
sns.lineplot(data=df, x="epoch", y="val_loss", label="Val Loss")
# 4. Add your title
plt.title("Jiang Fourier Domain")
# 5. Display the plot in your VS Code Interactive Window
plt.show()

# %%
