import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# Citirea datelor
file_path = "abalone.data"
columns = ["Sex", "Length", "Diameter", "Height", "WholeWeight",
           "ShuckedWeight", "VisceraWeight", "ShellWeight", "Rings"]

df = pd.read_csv(file_path, names=columns)
df = pd.get_dummies(df, drop_first=True)

# Analiza distributiei țintei
df['Rings'].hist(bins=20)
plt.xlabel('Număr de inele')
plt.ylabel('Frecvență')
plt.title('Distribuția valorilor Rings')
plt.show()

print("Statistici pentru 'Rings':")
print(df['Rings'].agg(['min', 'max', 'mean', 'std']))

# Separare X și y
X = df.drop('Rings', axis=1)
y = df['Rings']

# Împărțire în set de antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Standardizare
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model inițial
mlp = MLPRegressor(max_iter=1000, random_state=150)

# Set hiperparametri
param_grid = {
    'hidden_layer_sizes': [(100,), (200,), (200, 100), (100, 50), (50, 25)],
    'learning_rate_init': [0.1, 0.01],
}

# Grid search
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5,
                           scoring='neg_mean_squared_error', n_jobs=-1, verbose=0,
                           return_train_score=True)
grid_search.fit(X_train_scaled, y_train)

# Rezultate detaliate
results = pd.DataFrame(grid_search.cv_results_)
results["mean_test_score"] = -results["mean_test_score"]  # transformăm scorul în MSE pozitiv

summary = results[[
    "param_hidden_layer_sizes",
    "param_learning_rate_init",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]]

summary = summary.sort_values(by="rank_test_score").reset_index(drop=True)
summary["mean_test_score"] = summary["mean_test_score"].round(4)
summary["std_test_score"] = summary["std_test_score"].round(4)
summary.columns = ["Hidden Layer Sizes", "Learning Rate", "Mean Test MSE", "Std MSE", "Rank"]

# Afișăm tabelul complet
print("\nTabel complet cu rezultatele hiperparametrilor testați:\n")
print(summary)

# Vizibil și în Variable Explorer din Spyder
summary_for_display = summary.copy()

# Alegem modelul cel mai bun
best_params = grid_search.best_params_
print("\nCei mai buni parametri găsiți:")
print(best_params)

best_mlp = grid_search.best_estimator_
y_pred = best_mlp.predict(X_test_scaled)
y_pred_years = y_pred * 1.5  # conversie în ani

# Metrici de performanță
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("\nPerformanța modelului ales (în inele):")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Explained Variance Score (EVS): {evs:.4f}")

# Grafic: adevărat vs prezis
plt.figure(figsize=(8, 6))
plt.scatter(y_test * 1.5, y_pred_years, color='blue', alpha=0.5)
plt.plot([min(y_test * 1.5), max(y_test * 1.5)],
         [min(y_test * 1.5), max(y_test * 1.5)], color='red', linestyle='--')
plt.xlabel('Valori adevărate (ani)')
plt.ylabel('Valori prezise (ani)')
plt.title('Adevărat vs. Prezis')
plt.grid(True)
plt.show()

# Grafic reziduuri
residuals = (y_test * 1.5) - y_pred_years
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_years, residuals, color='blue', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Valori prezise (ani)')
plt.ylabel('Erori (Residuals)')
plt.title('Graficul erorilor')
plt.grid(True)
plt.show()

# Histograma erorilor
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.xlabel('Erori')
plt.ylabel('Frecvență')
plt.title('Distribuția erorilor')
plt.grid(True)
plt.show()

# Learning curve
train_errors, test_errors = [], []
for m in range(10, len(X_train_scaled), int(len(X_train_scaled) / 10)):
    best_mlp.fit(X_train_scaled[:m], y_train[:m])
    train_pred = best_mlp.predict(X_train_scaled[:m])
    test_pred = best_mlp.predict(X_test_scaled)
    train_errors.append(mean_squared_error(y_train[:m], train_pred))
    test_errors.append(mean_squared_error(y_test, test_pred))

plt.figure(figsize=(8, 6))
plt.plot(np.sqrt(train_errors), label="Eroare antrenament (RMSE)")
plt.plot(np.sqrt(test_errors), label="Eroare testare (RMSE)")
plt.legend()
plt.xlabel('Număr de exemple de antrenament')
plt.ylabel('Rădăcina Erorii Pătratice Medii (RMSE)')
plt.title('Learning Curve')
plt.grid(True)
plt.show()

# Histograma valorilor prezise
plt.figure(figsize=(8, 6))
plt.hist(y_pred_years, bins=30, color='blue', alpha=0.7)
plt.xlabel('Valori prezise (ani)')
plt.ylabel('Frecvență')
plt.title('Distribuția valorilor prezise')
plt.grid(True)
plt.show()
