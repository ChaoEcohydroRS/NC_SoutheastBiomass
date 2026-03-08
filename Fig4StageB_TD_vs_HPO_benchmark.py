# import warnings
# warnings.filterwarnings(
#     "ignore",
#     message="X does not have valid feature names, but StandardScaler was fitted with feature names"
# )

# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import RepeatedKFold
# from sklearn.metrics import mean_squared_error
# from scipy.stats import wilcoxon, ttest_rel
# from matplotlib.lines import Line2D

# # Tuned-default regressors
# from pytabkit import (
#     XGB_TD_Regressor, CatBoost_TD_Regressor,
#     LGBM_TD_Regressor, RealMLP_TD_Regressor
# )
# # HPO wrappers
# from pytabkit import (
#     XGB_HPO_Regressor, CatBoost_HPO_Regressor,
#     LGBM_HPO_Regressor, RealMLP_HPO_Regressor
# )

# from model_utils import load_and_preprocess_data
# from model_config import model_feature_sets

# # ─────────────────────────────────────────────────────────────────────────────
# # 1. Helper to run one policy (TD or HPO) for one learner across CV folds
# # ─────────────────────────────────────────────────────────────────────────────
# def eval_policy(model_cls, policy_kwargs, X, y, cv):
#     """Return lists: nrmse_per_fold, train_time_per_fold."""
#     nrmse_folds = []
#     time_folds  = []
#     for train_idx, test_idx in cv.split(X, y):
#         model = model_cls(**policy_kwargs)
#         t0 = time.time()
#         model.fit(X.iloc[train_idx], y.iloc[train_idx])
#         elapsed = time.time() - t0
#         preds = model.predict(X.iloc[test_idx])
#         rmse  = np.sqrt(mean_squared_error(y.iloc[test_idx], preds))
#         nrmse = rmse / y.iloc[test_idx].mean()
#         # nrmse = rmse / np.std(y.iloc[test_idx])
#         nrmse_folds.append(nrmse)
#         time_folds.append(elapsed)
#     return nrmse_folds, time_folds

# # Wrap plotting in a reusable function for both nRMSE and time
# def plot_metric_box(ax, df, metric, title, ylabel, color_map, add_stars=False):
#     for algo, grp in df.groupby("algo"):
#         data = grp.pivot(columns="policy", values=metric)
#         positions = [list(color_map).index(algo)*2, list(color_map).index(algo)*2+1]
#         ax.boxplot(
#             [data["TD"], data["HPO"]],
#             positions=positions, widths=0.6,
#             boxprops=dict(color=color_map[algo]),
#             medianprops=dict(color="black")
#         )
#         if add_stars:
#             p_w, _ = wilcoxon(data["TD"], data["HPO"], alternative="greater")
#             star = "**" if p_w<0.01 else ("*" if p_w<0.05 else "")
#             ax.text(np.mean(positions), max(data.max())*1.02, star,
#                     ha="center", va="bottom", color=color_map[algo], fontsize=14)
#     ax.set_xticks([i*2+0.5 for i in range(len(color_map))], list(color_map.keys()))
#     ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.grid(axis="y", linestyle="--", alpha=0.4)

# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     # paths and data load
#     output_dir = "/work/users/w/a/wayne128/Biomass_ML/Dataset/OutBiomassRaster"
#     cleaned_df, short_tag = load_and_preprocess_data(
#         "unc_chao_fia_data.xlsx",
#         na_values=["1.#QNB","1.#INF","-1.#INF","nan","NaN","inf","-inf"]
#     )

#     # target and Stage-A winner
#     feat_id = "G1c"
#     y = cleaned_df["total_biomass_tons_ha"]
#     X_raw = cleaned_df[model_feature_sets[feat_id]].dropna()
#     y = y.loc[X_raw.index]
#     X_raw = X_raw.loc[y.index]

#     # scale features
#     scaler = StandardScaler()
#     X_scaled = pd.DataFrame(
#         scaler.fit_transform(X_raw),
#         index=X_raw.index, columns=X_raw.columns
#     )

#     # CV scheme
#     cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)

#     # define learners and policies
#     learners = {
#         "XGB":     (XGB_TD_Regressor,     {'n_threads':64, "device":"cpu"}, 
#                     XGB_HPO_TPE_Regressor,     {'n_threads':64, "device":"cpu"}),
#         "CatBoost":(CatBoost_TD_Regressor,{'n_threads':64, "device":"cpu"}, 
#                     CatBoost_HPO_TPE_Regressor,{'n_threads':64, "device":"cpu"}),
#         "LGBM":    (LGBM_TD_Regressor,    {'n_threads':64, "verbosity":-1, "device":"cpu"}, 
#                     LGBM_HPO_TPE_Regressor,    {'n_threads':64, "verbosity":-1, "device":"cpu"}),
#         "RealMLP": (RealMLP_TD_Regressor, {'n_threads':64, "device":"cpu"}, 
#                     RealMLP_HPO_Regressor, {'n_threads':64, "n_hyperopt_steps":30,"device":"cpu"})
#     }

#     # collect per-fold metrics
#     records = []
#     print("Stage B: TD-opt vs HPO-opt across learners")
#     for name, (TD_cls, TD_kwargs, HPO_cls, HPO_kwargs) in learners.items():
#         print(f"\nEvaluating {name}")
#         td_nrmse, td_time = eval_policy(TD_cls, TD_kwargs, X_scaled, y, cv)
#         hp_nrmse, hp_time = eval_policy(HPO_cls, HPO_kwargs, X_scaled, y, cv)
#         # append
#         for val, t in zip(td_nrmse, td_time):
#             records.append({"algo":name, "policy":"TD",  "nrmse":val, "time":t})
#         for val, t in zip(hp_nrmse, hp_time):
#             records.append({"algo":name, "policy":"HPO", "nrmse":val, "time":t})
#         # statistical test
#         p_w, _ = wilcoxon(td_nrmse, hp_nrmse, alternative="greater")
#         p_t, _ = ttest_rel(td_nrmse, hp_nrmse)
        
#         sig = "**" if p_w<0.01 else ("*" if p_w<0.05 else "")
#         # print(f"  Wilcoxon TD>HPO p = {p_w:.3g} {sig}")
#         print(f"  Wilcoxon TD>HPO p = {p_w:.4f} {sig}")
#         print(f"  Test TD>HPO p = {p_t:.4f} {sig}")

#     df = pd.DataFrame(records)
    
#     # Print final summary of mean ± std for each model and policy
#     summary = df.groupby(["algo", "policy"])["nrmse"].agg(["mean", "std"]).round(4)
#     print("\nnRMSE Summary:\n", summary.unstack())

#     df.to_csv(f"{output_dir}/metrics_TD_vs_HPO_{feat_id}.csv", index=False)


#     # ─────────────────────────────────────────────────────────────────────────
#     # 5. Plot Figure 4a: nRMSE boxplots
#     # ─────────────────────────────────────────────────────────────────────────
#     fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))

#     color_map = {
#         "XGB":"#3E7CB1","CatBoost":"#66A182",
#         "LGBM":"#F5A623","RealMLP":"#D65A31"
#     }
    
#     # a) nRMSE
#     plot_metric_box(ax1, df, "nrmse", "(a) Accuracy (nRMSE)", "nRMSE", color_map, add_stars=True)
#     plot_metric_box(ax2, df, "time",  "(b) Training time per fold (s)", "Time (s)", color_map)


#     plt.tight_layout()
#     # save figure
#     fig_out = f"{output_dir}/FigStageB_TD_vs_HPO_{feat_id}.png"
#     fig.savefig(fig_out, dpi=300)
#     plt.show()
#     print(f"\nSaved Figure 4 to: {fig_out}")
    
    
    
    
######################################
import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)




import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import wilcoxon, ttest_rel
from matplotlib.lines import Line2D

# Tuned-default regressors
from pytabkit import (
    XGB_TD_Regressor, CatBoost_TD_Regressor,
    LGBM_TD_Regressor, RealMLP_TD_Regressor
)
# HPO wrappers
from pytabkit import (
    XGB_HPO_TPE_Regressor, CatBoost_HPO_TPE_Regressor,
    LGBM_HPO_TPE_Regressor, RealMLP_HPO_Regressor
)

from model_utils import load_and_preprocess_data
from model_config import model_feature_sets


def eval_policy(model_cls, policy_kwargs, X, y, cv, mute_fit=False):
    """Evaluate one training policy over CV folds."""
    nrmse_folds, time_folds = [], []
    for train_idx, test_idx in cv.split(X, y):
        model = model_cls(**policy_kwargs)
        t0 = time.time()
        
        if mute_fit:
            # suppress anything that model.fit prints:
            with contextlib.redirect_stdout(io.StringIO()):
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
        else:
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
        elapsed = time.time() - t0
        preds = model.predict(X.iloc[test_idx])
        rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], preds))
        nrmse = rmse / y.iloc[test_idx].mean()
        nrmse_folds.append(nrmse)
        time_folds.append(elapsed)
    return nrmse_folds, time_folds


def plot_metric_box(ax, df, metric, title, ylabel, color_map, add_stars=False):
    """Plot grouped boxplots for given metric."""
    for algo, grp in df.groupby("algo"):
        data = grp.pivot(columns="policy", values=metric)
        base = list(color_map.keys()).index(algo) * 2
        positions = [base, base + 1]
        ax.boxplot(
            [data['TD'], data['HPO']],
            positions=positions, widths=0.6,
            boxprops=dict(color=color_map[algo]),
            medianprops=dict(color='black')
        )
        if add_stars:
            p_w, _ = wilcoxon(data['TD'], data['HPO'], alternative='greater')
            star = '**' if p_w < 0.01 else ('*' if p_w < 0.05 else '')
            ax.text(
                np.mean(positions), data.values.max() * 1.02,
                star, ha='center', va='bottom', color=color_map[algo], fontsize=14
            )

    ticks = [i * 2 + 0.5 for i in range(len(color_map))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(list(color_map.keys()))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.4)


if __name__ == '__main__':
    # import logging

    # # Patch Logger.log to suppress all `log(-1, ...)` messages
    # _original_log = logging.Logger.log
    
    # def patched_log(self, level, msg, *args, **kwargs):
    #     if level == -1:
    #         return  # suppress noisy message
    #     return _original_log(self, level, msg, *args, **kwargs)
    
    # logging.Logger.log = patched_log

    # 1. Load data
    output_dir = "/work/users/w/a/wayne128/Biomass_ML/Dataset/OutBiomassRaster"
    cleaned_df, short_tag = load_and_preprocess_data(
        "unc_chao_fia_data.xlsx",
        na_values=["1.#QNB","1.#INF","-1.#INF","nan","NaN","inf","-inf"]
    )

    # 2. Prepare features and target
    feat_id = 'G1c'
    y = cleaned_df['total_biomass_tons_ha']
    X_raw = cleaned_df[model_feature_sets[feat_id]].dropna()
    y = y.loc[X_raw.index]
    X_raw = X_raw.loc[y.index]

    # 3. Scale features
    X_raw = X_raw.sort_index(axis=1)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw),
        index=X_raw.index, columns=X_raw.columns
    )

    # 4. CV scheme
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)

    # 5. Define learners with consistent random_state and verbosity
    base_kwargs = {'random_state': 0, 'n_threads': 64,  'device': 'cpu'}
    learners = {
        'XGB': (
            XGB_TD_Regressor,     {**base_kwargs},
            XGB_HPO_TPE_Regressor,    {**base_kwargs, 'n_hyperopt_steps': 50, 'verbosity': 0}
        ),
        'CatBoost': (
            CatBoost_TD_Regressor,{**base_kwargs},
            CatBoost_HPO_TPE_Regressor,{**base_kwargs, 'n_hyperopt_steps': 50, 'verbosity': 0}
        ),
        'LGBM': (
            LGBM_TD_Regressor,    {**base_kwargs, 'verbosity': -1},
            LGBM_HPO_TPE_Regressor,   {**base_kwargs, 'n_hyperopt_steps': 50, 'verbosity': -1}
        ),
        'RealMLP': (
            RealMLP_TD_Regressor, {**base_kwargs},
            RealMLP_HPO_Regressor,{**base_kwargs, 'n_hyperopt_steps': 50, 'verbosity': 0}
        )
    }

    # 6. Evaluate policies
    records = []
    print('Stage B: TD-opt vs HPO-opt across learners')
    for name, (TD_cls, TD_kwargs, HPO_cls, HPO_kwargs) in learners.items():
        print(f'\nEvaluating {name}')
        td_nrmse, td_time = eval_policy(TD_cls, TD_kwargs, X_scaled, y, cv)
        hp_nrmse, hp_time = eval_policy(HPO_cls, HPO_kwargs, X_scaled, y, cv)

        # record metrics
        for val, t in zip(td_nrmse, td_time):
            records.append({'algo': name, 'policy': 'TD',  'nrmse': val, 'time': t})
        for val, t in zip(hp_nrmse, hp_time):
            records.append({'algo': name, 'policy': 'HPO', 'nrmse': val, 'time': t})

        # statistical tests
        p_w, _ = wilcoxon(td_nrmse, hp_nrmse, alternative='greater')
        p_t, _ = ttest_rel(td_nrmse, hp_nrmse)
        sig = '**' if p_w < 0.01 else ('*' if p_w < 0.05 else '')

        print(f'  Wilcoxon TD>HPO p = {p_w:.4f} {sig}')
        print(f'  Paired t-test TD>HPO p = {p_t:.4f} {sig}')

    df = pd.DataFrame(records)

    # 7. Summary table
    summary = df.groupby(['algo', 'policy'])['nrmse'].agg(['mean', 'std']).round(4)
    print('\nnRMSE Summary:\n', summary.unstack())

    # 8. Save raw metrics
    df.to_csv(f"{output_dir}/metrics_TD_vs_HPO_{feat_id}.csv", index=False)

    # 9. Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    color_map = {
        'XGB': '#3E7CB1', 'CatBoost': '#66A182',
        'LGBM':'#F5A623', 'RealMLP':'#D65A31'
    }

    plot_metric_box(ax1, df, 'nrmse', '(a) Accuracy (nRMSE)', 'nRMSE', color_map, add_stars=True)
    plot_metric_box(ax2, df, 'time',   '(b) Training time per fold (s)', 'Time (s)', color_map)

    # policy legend
    legend_elems = [
        Line2D([0],[0], marker='s', color='w', label='TD-opt',  markerfacecolor='gray',     markersize=10),
        Line2D([0],[0], marker='s', color='w', label='HPO-opt', markerfacecolor='lightgray',markersize=10)
    ]
    ax1.legend(handles=legend_elems, loc='upper right')

    plt.tight_layout()
    fig_out = f"{output_dir}/FigStageB_TD_vs_HPO_{feat_id}.png"
    fig.savefig(fig_out, dpi=300)
    plt.show()
    print(f"\nSaved Figure to: {fig_out}")

