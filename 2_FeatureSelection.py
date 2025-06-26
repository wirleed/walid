import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, VarianceThreshold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from boruta import BorutaPy
from sklearn_genetic import GAFeatureSelectionCV
import warnings

warnings.filterwarnings('ignore')



class HousingFeatureSelector:

    def __init__(self, data_path=None, target_column='totalPrice'):
        self.data_path = data_path
        self.target_column = target_column
        self.X = None
        self.y = None
        self.X_scaled = None
        self.feature_names = None
        self.results = {}
        self.scaler = StandardScaler()

    def load_preprocessed_data(self, df=None):
        """Load and prepare preprocessed data"""
        if df is not None:
            self.df = df.copy()
        else:
            print("Please provide preprocessed DataFrame or data path")
            return

        # Separate features and target
        self.y = self.df[self.target_column]
        self.X = self.df.drop(columns=[self.target_column])

        # Handle categorical variables ensure properly encoded
        categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns

        # Use label encoding for categorical variables (preserves them as features)
        le_dict = {}
        for col in categorical_columns:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col].astype(str))
            le_dict[col] = le

        self.feature_names = list(self.X.columns)

        # Scale features for methods that require it
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.feature_names,
            index=self.X.index
        )

        print(f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Categorical features encoded: {list(categorical_columns)}")

    def univariate_selection(self, k=10):
        """1. Univariate Feature Selection using f_regression"""
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(self.X, self.y)

        selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
        scores = selector.scores_[selector.get_support()]

        return {
            'method': 'Univariate (f_regression)',
            'features': selected_features,
            'scores': scores,
            'n_features': len(selected_features)
        }

    def rf_importance_selection(self, k=10):
        """2. Random Forest Feature Importance"""
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X, self.y)

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        selected_features = feature_importance.head(k)['feature'].tolist()
        scores = feature_importance.head(k)['importance'].values

        return {
            'method': 'Random Forest Importance',
            'features': selected_features,
            'scores': scores,
            'n_features': len(selected_features)
        }

    def lasso_selection(self, alpha=None):
        """3. LASSO Feature Selection"""
        if alpha is None:
            lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
        else:
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=alpha, random_state=42)

        lasso.fit(self.X_scaled, self.y)

        selected_mask = lasso.coef_ != 0
        selected_features = [self.feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]]
        scores = lasso.coef_[selected_mask]

        return {
            'method': 'LASSO',
            'features': selected_features,
            'scores': np.abs(scores),
            'n_features': len(selected_features)
        }

    def variance_correlation_selection(self, var_threshold=0.01, corr_threshold=0.95):
        """4. Variance + Correlation Filter"""
        # Remove low variance features
        var_selector = VarianceThreshold(threshold=var_threshold)
        X_var = var_selector.fit_transform(self.X)
        var_features = [self.feature_names[i] for i in var_selector.get_support(indices=True)]

        # Remove highly correlated features
        X_var_df = pd.DataFrame(X_var, columns=var_features)
        corr_matrix = X_var_df.corr().abs()

        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]

        selected_features = [f for f in var_features if f not in to_drop]

        return {
            'method': 'Variance + Correlation Filter',
            'features': selected_features,
            'scores': np.ones(len(selected_features)),  # No specific scores for this method
            'n_features': len(selected_features)
        }

    def rfe_selection(self, k=10):
        """5. Recursive Feature Elimination"""
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=k, step=1)
        selector.fit(self.X_scaled, self.y)

        selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
        # Use ranking as inverse score (lower rank = higher score)
        scores = 1.0 / selector.ranking_[selector.get_support()]

        return {
            'method': 'RFE (Linear Regression)',
            'features': selected_features,
            'scores': scores,
            'n_features': len(selected_features)
        }

    def linear_importance_selection(self, k=10):
        """6. Linear Regression Coefficient Importance"""
        lr = LinearRegression()
        lr.fit(self.X_scaled, self.y)

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(lr.coef_)
        }).sort_values('importance', ascending=False)

        selected_features = feature_importance.head(k)['feature'].tolist()
        scores = feature_importance.head(k)['importance'].values

        return {
            'method': 'Linear Regression Importance',
            'features': selected_features,
            'scores': scores,
            'n_features': len(selected_features)
        }

    def mutual_info_selection(self, k=10):
        """7. Mutual Information Feature Selection"""
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_selected = selector.fit_transform(self.X, self.y)

        selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
        scores = selector.scores_[selector.get_support()]

        return {
            'method': 'Mutual Information',
            'features': selected_features,
            'scores': scores,
            'n_features': len(selected_features)
        }

    def permutation_importance_selection(self, k=10):
        """8. Permutation Importance"""
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(self.X, self.y)

        # Calculate permutation importance
        perm_importance = permutation_importance(rf, self.X, self.y, n_repeats=5, random_state=42, n_jobs=-1)

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)

        selected_features = feature_importance.head(k)['feature'].tolist()
        scores = feature_importance.head(k)['importance'].values

        return {
            'method': 'Permutation Importance',
            'features': selected_features,
            'scores': scores,
            'n_features': len(selected_features)
        }

    def boruta_selection(self, max_iter=100):
        """9. Boruta Feature Selection"""
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        # Initialize Boruta
        boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0,
                                   random_state=42, max_iter=max_iter)

        boruta_selector.fit(self.X.values, self.y.values)

        # Get selected features
        selected_mask = boruta_selector.support_
        selected_features = [self.feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]]

        # Get feature rankings as scores (lower rank = higher importance)
        scores = 1.0 / (boruta_selector.ranking_[selected_mask] + 1e-10)

        return {
            'method': 'Boruta',
            'features': selected_features,
            'scores': scores,
            'n_features': len(selected_features)
        }

    def genetic_algorithm_selection(self, k=10):
        """10. Genetic Algorithm Feature Selection"""
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        # Use genetic algorithm for feature selection
        selector = GAFeatureSelectionCV(
            estimator=estimator,
            cv=3,
            scoring='neg_mean_squared_error',
            max_features=k,
            population_size=20,
            generations=10,
            n_jobs=-1,
            verbose=False,
        )

        selector.fit(self.X, self.y)

        selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
        # Use uniform scores for genetic selection
        scores = np.ones(len(selected_features))

        return {
            'method': 'Genetic Algorithm',
            'features': selected_features,
            'scores': scores,
            'n_features': len(selected_features)
        }

    def run_all_methods(self, k=10):
        """Run all feature selection methods"""
        print("Running feature selection methods...")

        methods = [
            ('univariate', lambda: self.univariate_selection(k)),
            ('rf_importance', lambda: self.rf_importance_selection(k)),
            ('lasso', lambda: self.lasso_selection()),
            ('variance_correlation', lambda: self.variance_correlation_selection()),
            ('rfe', lambda: self.rfe_selection(k)),
            ('linear_importance', lambda: self.linear_importance_selection(k)),
            ('mutual_info', lambda: self.mutual_info_selection(k)),
            ('permutation', lambda: self.permutation_importance_selection(k)),
            ('boruta', lambda: self.boruta_selection()),
            ('genetic', lambda: self.genetic_algorithm_selection(k))
        ]

        for name, method in methods:
            try:
                print(f"Running {name}...")
                result = method()
                self.results[name] = result
                print(f"✓ {result['method']}: {result['n_features']} features selected")
            except Exception as e:
                print(f"✗ Error in {name}: {str(e)}")

        return self.results

    def evaluate_methods(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        evaluation_results = {}

        for method_name, result in self.results.items():
            if len(result['features']) == 0:
                continue

            try:
                # Select features
                X_train_selected = X_train[result['features']]
                X_test_selected = X_test[result['features']]

                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                evaluation_results[method_name] = {
                    'method': result['method'],
                    'n_features': result['n_features'],
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2
                }

            except Exception as e:
                print(f"Error evaluating {method_name}: {str(e)}")

        return evaluation_results

    def plot_comparison(self, save_path=None):
        # Get evaluation results
        eval_results = self.evaluate_methods()

        if not eval_results:
            print("No evaluation results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # Prepare data
        methods = list(eval_results.keys())
        short_names_map = {
            'Univariate (f_regression)': 'Univariate',
            'Random Forest Importance': 'RF',
            'LASSO': 'Lasso',
            'Variance + Correlation Filter': 'Variance\n+Corr',
            'RFE (Linear Regression)': 'RFE',
            'Linear Regression Importance': 'Linear',
            'Mutual Information': 'Mutual\nInfo',
            'Permutation Importance': 'Perm',
            'Boruta': 'Boruta',
            'Genetic Algorithm': 'Genetic'
        }
        method_names = [short_names_map.get(eval_results[m]['method'], eval_results[m]['method']) for m in methods]

        n_features = [eval_results[m]['n_features'] for m in methods]
        r2_scores = [eval_results[m]['r2'] for m in methods]
        rmse_scores = [eval_results[m]['rmse'] for m in methods]

        # 1. Number of Features Selected
        axes[0, 0].bar(range(len(methods)), n_features, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Number of Features Selected', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_xticks(range(len(methods)))
        axes[0, 0].set_xticklabels([name.replace(' ', '\n') for name in method_names],
                                   rotation=45, ha='right', fontsize=9)

        for i, v in enumerate(n_features):
            axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

        # 2. R² Score Comparison
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        bars = axes[0, 1].bar(range(len(methods)), r2_scores, color=colors, alpha=0.8)
        axes[0, 1].set_title('R² Score Comparison', fontweight='bold')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_xticks(range(len(methods)))
        axes[0, 1].set_xticklabels([name.replace(' ', '\n') for name in method_names],
                                   rotation=45, ha='right', fontsize=9)
        axes[0, 1].set_ylim(0, 1)

        # Add value labels
        for i, v in enumerate(r2_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. RMSE Comparison
        axes[1, 0].bar(range(len(methods)), rmse_scores, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('RMSE Comparison (Lower is Better)', fontweight='bold')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_xticks(range(len(methods)))
        axes[1, 0].set_xticklabels([name.replace(' ', '\n') for name in method_names],
                                   rotation=45, ha='right', fontsize=9)

        for i, v in enumerate(rmse_scores):
            axes[1, 0].text(i, v + max(rmse_scores) * 0.01, f'{v:.0f}',
                            ha='center', va='bottom', fontweight='bold')

        # 4. Efficiency Score (R² / Number of Features)
        efficiency_scores = [r2_scores[i] / n_features[i] if n_features[i] > 0 else 0
                             for i in range(len(methods))]

        axes[1, 1].bar(range(len(methods)), efficiency_scores, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Efficiency Score (R² / # Features)', fontweight='bold')
        axes[1, 1].set_ylabel('Efficiency Score')
        axes[1, 1].set_xticks(range(len(methods)))
        axes[1, 1].set_xticklabels([name.replace(' ', '\n') for name in method_names],
                                   rotation=45, ha='right', fontsize=9)

        for i, v in enumerate(efficiency_scores):
            axes[1, 1].text(i, v + max(efficiency_scores) * 0.01, f'{v:.4f}',
                            ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

        self.print_summary_table(eval_results)

    def print_summary_table(self, eval_results):
        print("\n" + "=" * 80)
        print("FEATURE SELECTION METHODS COMPARISON SUMMARY")
        print("=" * 80)

        df_summary = pd.DataFrame({
            'Method': [eval_results[m]['method'] for m in eval_results.keys()],
            'Features': [eval_results[m]['n_features'] for m in eval_results.keys()],
            'R² Score': [f"{eval_results[m]['r2']:.4f}" for m in eval_results.keys()],
            'RMSE': [f"{eval_results[m]['rmse']:.2f}" for m in eval_results.keys()],
            'Efficiency': [f"{eval_results[m]['r2'] / eval_results[m]['n_features']:.4f}"
                           for m in eval_results.keys()]
        })

        # Sort by R² score
        df_summary = df_summary.sort_values('R² Score', ascending=False)
        df_summary.reset_index(drop=True, inplace=True)
        df_summary.index = df_summary.index + 1

        print(df_summary.to_string())
        # Find best method
        best_r2_idx = df_summary['R² Score'].astype(float).idxmax()
        best_efficiency_idx = df_summary['Efficiency'].astype(float).idxmax()

        print(f"\nBEST R² SCORE: {df_summary.loc[best_r2_idx, 'Method']} "
              f"(R² = {df_summary.loc[best_r2_idx, 'R² Score']})")
        print(f"BEST EFFICIENCY: {df_summary.loc[best_efficiency_idx, 'Method']} "
              f"(Efficiency = {df_summary.loc[best_efficiency_idx, 'Efficiency']})")


if __name__ == "__main__":

    selector = HousingFeatureSelector()
    df = pd.read_csv('Data_For_Model.csv')
    selector.load_preprocessed_data(df)
    results = selector.run_all_methods(k=15)  # Select top 15 features for methods that need k
    selector.plot_comparison(save_path='feature_selection_comparison.png')

    import json

    # Automatically pick best R² method
    best_method_key = max(results, key=lambda k: selector.evaluate_methods()[k]['r2'])

    # Extract best features
    best_features = results[best_method_key]['features']

    # Save to JSON
    with open('best_features.json', 'w') as f:
        json.dump(best_features, f)

    print(f"\n✅ Saved top {len(best_features)} features from {results[best_method_key]['method']} to best_features.json")
