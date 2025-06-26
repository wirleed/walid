import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
import warnings
import time
import random
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')


def random_forest_feature_selection(X_train, y_train, X_test, n_features=None, threshold=None):

    print("Performing Random Forest feature selection...")

    # Train Random Forest for feature importance
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_selector.fit(X_train, y_train)

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_selector.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f" Top 10 most important features:")
    print(feature_importance.head(10))

    # Select features based on criteria
    if n_features is not None:
        # Select top n features
        selected_features = feature_importance.head(n_features)['feature'].tolist()
        print(f"Selected top {n_features} features")
    elif threshold is not None:
        # Select features above threshold
        selected_features = feature_importance[feature_importance['importance'] >= threshold]['feature'].tolist()
        print(f"Selected {len(selected_features)} features above threshold {threshold}")
    else:
        # Default: select features that contribute to 95% of cumulative importance
        cumulative_importance = feature_importance['importance'].cumsum()
        total_importance = feature_importance['importance'].sum()
        cutoff_index = (cumulative_importance <= 0.95 * total_importance).sum()
        selected_features = feature_importance.head(cutoff_index + 1)['feature'].tolist()
        print(f"Selected {len(selected_features)} features contributing to 95% of importance")

    # Filter datasets
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    return X_train_selected, X_test_selected

def main():
    train_test_data = readFile()
    model = Training(*train_test_data)
    saveModel(model)
    return

def readFile():
    # Load data
    data = pd.read_csv("Data_For_Model.csv", index_col=0)

    import json

    # Load selected features from JSON
    with open("best_features.json", "r") as f:
        selected_features = json.load(f)

    # Ensure all features exist in the data
    selected_features = [f for f in selected_features if f in data.columns]

    # Filter the dataset to use only selected features
    X = data[selected_features]
    y = data['totalPrice']

    print("✅ Training model using these features:")
    print(selected_features)

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=2)

    return X_train, X_test, y_train, y_test

def Training(X_train, X_test, y_train, y_test):
    # Models to compare
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=2),
        "KNN Regression": KNeighborsRegressor(n_neighbors=5),
        "XGBoost": xgb.XGBRegressor(),
    }

    # Train, predict, and store metrics
    results = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        training_time = end - start
        y_pred = model.predict(X_test)
        results[name] = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            "model": model,
            'Training Time': training_time
        }
        print(f"{name} trained.")
    metrics_df = ModelVisualisation(results) # Up to five model results, otherwise it will error out
    print("\nModel Comparison Metrics:\n", metrics_df.round(4))

    # Balancing the accuracy and speed (Identify the highest similar accuracy model with 10 times less training time and consider it as the best model)
    best_r2 = metrics_df['R2'].max()
    best_model_name = metrics_df['R2'].idxmax()
    best_time = metrics_df[metrics_df['R2'] == best_r2].iloc[0]['Training Time']
    candidates = metrics_df[metrics_df['Training Time'] <= best_time / 10]

    if not candidates.empty:
        best_model_name = candidates['R2'].idxmax()
    best_model = results[best_model_name]['model']

    print(f"Best model is {best_model_name} with R2 = {metrics_df.loc[best_model_name, 'R2']:.4f} and Training Time = {metrics_df.loc[best_model_name, 'Training Time']:.4f}s")
    
    # This hyperparameter tuning only applied to xgboost
    if best_model_name.lower().strip() == 'xgboost':
        # best_params = hyperparameter(X_train, y_train)
        tuningModel = xgb.XGBRegressor(**best_params)
        tuningModel.fit(X_train,y_train)
        y_pred = tuningModel.predict(X_test)
        results = {best_model_name: results[best_model_name]}
        del results[best_model_name]['Training Time']
        results['xgboost after tuning'] = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            'model': tuningModel,
        }
        metrics_df = ModelVisualisation(results)
        print("\nTuning Model Comparison Metrics:\n", metrics_df.round(4))
        if metrics_df['R2'].idxmax()==metrics_df['MAE'].idxmin()==metrics_df['RMSE'].idxmin():
            best_model_name = metrics_df['R2'].idxmax()
            best_model = results[best_model_name]['model']
        else:
            best_model = results[best_model_name]['model']
        print("Best model:", best_model_name)
    print("✅ Model training complete.")
    return best_model


def saveModel(model):
    with open('best_housing_model.pkl', 'wb') as f:
        pickle.dump(model, f)


# Hyperparameter turning using Genetic Algorithm for xgboost
def hyperparameter(X_train, y_train):
    # Create fitness function (we minimize RMSE, so negative score)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Parameter bounds (low, high)
    param_bounds = {
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'n_estimators': (50, 300),
        'min_child_weight': (1, 10),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'gamma': (0, 5)
    }
    param_keys = list(param_bounds.keys())

    # Create an Individual (chromosome) with random parameter number
    def create_individual():
        return [random.uniform(*param_bounds[k]) if 'float' in str(type(param_bounds[k][0])) else random.randint(*param_bounds[k]) for k in param_keys]

    def clip_bounds(individual):
        for i, key in enumerate(param_keys):
            low, high = param_bounds[key]
            # Ensure the new individual's parameter doesn't exceed the bound
            individual[i] = max(low, min(high, individual[i]))
            # Confirm the two parameter remain integer
            if key in ['max_depth', 'n_estimators']:
                individual[i] = int(round(individual[i]))
    
    def custom_mutate(individual):
        tools.mutGaussian(individual, mu=0, sigma=0.2, indpb=0.2)
        clip_bounds(individual)
        return individual,

    # Evaluation function
    def evaluate(individual):
        clip_bounds(individual)
        params = dict(zip(param_keys, individual))
        params['verbosity'] = 0
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=3) #3-fold cross-validation
        return (abs(scores.mean()),)  # Minimize RMSE

    # Genetic Algorithm required functions
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", custom_mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Run GA
    population = toolbox.population(n=20)
    NGEN = 10

    #eaSimple is genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.3, ngen=NGEN, verbose=True)

    # Get the best individual
    best_ind = tools.selBest(population, 1)[0]
    best_params = dict(zip(param_keys, best_ind))
    return best_params


def ModelVisualisation(results):
    df = pd.DataFrame(results).T
    df = df.drop(columns="model")

    # Models and colors
    models = df.index.tolist()
    model_colors = ['royalblue','orange', 'pink','yellow','red']
    # Metrics to plot
    metrics = ['R2', 'MAE', 'RMSE']

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = df[metric].values
        bars = ax.bar(models, values, color=[model_color for model_color in model_colors])
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                    f"{height:.4f}", ha='center', va='bottom', fontsize=10)
        ax.set_title(metric)
        ax.set_ylabel('Score')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("Model Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
    return df


best_params = {'max_depth': 10,
'learning_rate': 0.08835175974447863,
'n_estimators': 267,
'min_child_weight': 3.1086708008538624,
'subsample': 0.9152854359485246,
'colsample_bytree': 0.9784521339979477,
'gamma': 4.6828016617073835
}

if __name__ == '__main__':
    main()