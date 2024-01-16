import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def get_summary_statistics(df_col):
    """
    Returns a dictionary of summary statistics for a given column in a dataframe.
    """
    return {
        'mean': df_col.mean(),
        'median': df_col.median(),
        'min': df_col.min(),
        'max': df_col.max(),
        'std': df_col.std()
    }


def plot_distribution(df_col):
    """
    Plots a histogram and boxplot for a given column in a dataframe.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df_col, ax=ax[0])
    sns.boxplot(df_col, ax=ax[1])
    plt.show()


def get_target_correlation(df, var, var_type, target='pitch_outcome_id'):
    if var_type == 'categorical':
        corr_dict = {}

        for val in df[var].unique():
            val_df = df[df[var] == val]
            noval_df = df[df[var] != val]
            strike_pct_val = len(val_df[val_df[target] == 8]) / len(val_df)
            strike_pct_noval = len(noval_df[noval_df[target] == 8]) / len(noval_df)
            
            ttest = stats.ttest_ind(a=val_df[target], b=noval_df[target], equal_var=False)
            
            corr_dict[val] = (round(strike_pct_val - strike_pct_noval, 3), round(ttest[1], 3))

        return corr_dict

    else:
        corr_dict = {}

        strike_df = df[df[target] == 8]
        ball_df = df[df[target] == 2]

        strike_var_avg = strike_df[var].mean()
        ball_var_avg = ball_df[var].mean()

        diff_pct_mean = (strike_var_avg - ball_var_avg) / (df[var].max() - df[var].min())

        ttest = stats.ttest_ind(a=strike_df[var], b=ball_df[var], equal_var=False)
        corr_dict[var] = (round(strike_var_avg - ball_var_avg, 3), round(diff_pct_mean * 100, 2), round(ttest[1], 3))

        return corr_dict
        
