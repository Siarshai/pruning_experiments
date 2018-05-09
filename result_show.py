import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show_results_against_compression(compressions, accuracies, losses):
    df = pd.DataFrame({
        "compressions": compressions,
        "accuracies": accuracies,
        "losses": losses,
    })
    unique_compressions = np.asarray(df.groupby("compressions")["compressions"].mean())
    mean_accuracies = np.asarray(df.groupby("compressions")["accuracies"].mean())
    plt.plot(unique_compressions, mean_accuracies)
    plt.show()

