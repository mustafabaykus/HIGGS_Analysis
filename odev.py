# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# HIGGS veri setini yükleme (ilk 100.000 örnek)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
cols = ['label'] + [f'feature_{i}' for i in range(1, 29)]
df = pd.read_csv(url, names=cols, nrows=100000)

# İlk birkaç satırı inceleyelim
df.head()

# Aykırı değerleri analiz et (yalnızca özellikler)
features = df.columns[1:]

# Aykırı değer sayısını bul ve grafiğini çiz
outlier_counts = {}

for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_counts[col] = len(outliers)

# Görselleştirme
plt.figure(figsize=(12,6))
sns.barplot(x=list(outlier_counts.keys()), y=list(outlier_counts.values()))
plt.xticks(rotation=90)
plt.title("Özniteliklere Göre Aykırı Değer Sayısı")
plt.ylabel("Aykırı Değer Sayısı")
plt.tight_layout()
plt.show()

# Aykırıları sınırlarla değiştir
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))
    

# MinMaxScaler ile [0, 1] aralığına dönüştürme
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Ölçeklenmiş verinin ön izlemesi
df.head()


#### Yöntem : Mutual Information ile Özellik Seçimi ####

from sklearn.feature_selection import mutual_info_classif

# MI skorlarını hesapla
mi_scores = mutual_info_classif(X, y, random_state=42)

# Sonuçları DataFrame ile göster
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_df = mi_df.sort_values(by='MI Score', ascending=False)

# En iyi 15 özelliği seç
selected_features_mi = mi_df.head(15)['Feature'].tolist()

print("Seçilen Özellikler (Mutual Information):")
print(selected_features_mi)

# MI skorlarını görselleştir
plt.figure(figsize=(10,6))
sns.barplot(data=mi_df.head(15), x='MI Score', y='Feature')
plt.title('En Bilgilendirici 15 Özellik (Mutual Information)')
plt.tight_layout()
plt.show()


#####  Nested Cross-Validation ####

results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'ROC AUC': []
}

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'KNN': (
        KNeighborsClassifier(),
        {'n_neighbors': [5]}
    ),
    'MLP': (
        MLPClassifier(max_iter=500, early_stopping=True, random_state=42),
        {'hidden_layer_sizes': [(50,)], 'activation': ['relu']}
    ),
    'XGBoost': (
        XGBClassifier(eval_metric='logloss', verbosity=0),
        {'n_estimators': [50], 'max_depth': [3]}
    )
}

for model_name, (clf, params) in models.items():
    print(f"\n⏳ Training {model_name}...")
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline([('clf', clf)])
        param_grid = {f'clf__{k}': v for k, v in params.items()}
        grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        results['Model'].append(model_name)
        results['Accuracy'].append(accuracy_score(y_test, y_pred))
        results['Precision'].append(precision_score(y_test, y_pred))
        results['Recall'].append(recall_score(y_test, y_pred))
        results['F1 Score'].append(f1_score(y_test, y_pred))
        results['ROC AUC'].append(roc_auc_score(y_test, y_prob))

#Sonuçları Göster
results_df = pd.DataFrame(results)
print("\n📊 Ortalama Performans Metrikleri:")
print(results_df.groupby("Model").mean().round(4))


### ROC Eğrilerini Hesapla ve Çiz ###

# Her model için ROC eğrilerini ayrı ayrı tutmak için sözlük
roc_data = {
    'KNN': [],
    'MLP': [],
    'XGBoost': []
}

# Dış katmanları yeniden başlatmak için
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, (clf, params) in models.items():
    print(f"\n📊 ROC hesaplanıyor: {model_name}")
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipe = Pipeline([('clf', clf)])
        param_grid = {f'clf__{k}': v for k, v in params.items()}
        grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_score = best_model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    roc_data[model_name] = (mean_fpr, mean_tpr, mean_auc)



plt.figure(figsize=(8,6))
for model_name, (fpr, tpr, auc_score) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrileri - Model Karşılaştırması')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()