#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANÁLISIS COMPARATIVO DE MODELOS - delfosA1C8
Integración del análisis predictivo con estructura modular
"""

import sys
import os
import warnings
from pathlib import Path

# Añadir directorio raíz al path
project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                           accuracy_score, classification_report, confusion_matrix)

# Modelos de sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Imports modulares del proyecto
from scripts.generate_synthetic_data import generate_synthetic_data
from src.data.preprocessor import DiabetesDataPreprocessor

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

def preparar_datos_analisis(n_samples=500, seed=42):
    """Genera datos sintéticos compatibles con el formato del análisis"""
    print("🔄 Generando datos sintéticos para análisis comparativo...")
    
    # Generar datos usando componente modular
    df_raw = generate_synthetic_data(n_samples, seed)
    
    # Guardar como outputglucosa.csv para compatibilidad
    output_path = Path('data/raw/outputglucosa.csv')
    df_raw.to_csv(output_path, index=False)
    print(f"✅ Datos guardados en {output_path}")
    
    return df_raw

def categorizar_glucosa(valor):
    """Clasificación de niveles de glucosa"""
    if pd.isna(valor):
        return np.nan
    elif valor < 100:
        return 0  # Normal
    elif valor <= 126:
        return 1  # Prediabetes
    else:
        return 2  # Diabetes


def definir_modelos_comparativos():
    """Define los 7 modelos del análisis comparativo"""
    models = {
        'Regresión Lineal': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(  # Usar params del proyecto
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            random_state=42
        ),
        'Support Vector Machine': SVR(
            kernel='rbf',
            C=100,
            gamma='scale'
        ),
        'Red Neuronal (MLP)': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
    }
    return models

def entrenar_y_evaluar_modelos(X_train, X_test, y_train, y_test, models):
    """Entrena todos los modelos y calcula métricas completas"""
    results = {}
    predictions = {}
    
    print("\n🤖 Entrenando modelos comparativos...")
    
    for name, model in models.items():
        print(f"  Entrenando {name}...")
        
        # Entrenar
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Métricas principales
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'CV_R²_mean': cv_scores.mean(),
            'CV_R²_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"    ✓ R² = {r2:.4f}, RMSE = {rmse:.2f}")
    
    return results, predictions

def visualizar_comparacion_modelos(results, y_test, model_names):
    """Crea los 4 gráficos principales del análisis"""
    
    # 1. Comparación de valores reales vs predichos
    fig = plt.figure(figsize=(20, 12))
    n_models = len(model_names)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    for idx, name in enumerate(model_names, 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        y_pred = results[name]['predictions']
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        
        # Línea perfecta
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                label='Predicción Perfecta')
        
        # Línea de tendencia
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_test, p(y_test), "b-", alpha=0.8, lw=2, label='Tendencia')
        
        # Métricas en el gráfico
        r2 = results[name]['R²']
        rmse = results[name]['RMSE']
        mae = results[name]['MAE']
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Glucosa Observada (mg/dL)')
        ax.set_ylabel('Glucosa Estimada (mg/dL)')
        ax.set_title(f'{name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Zonas de clasificación
        ax.axhline(y=100, color='orange', linestyle=':', alpha=0.5, lw=1)
        ax.axhline(y=126, color='red', linestyle=':', alpha=0.5, lw=1)
        ax.axvline(x=100, color='orange', linestyle=':', alpha=0.5, lw=1)
        ax.axvline(x=126, color='red', linestyle=':', alpha=0.5, lw=1)
    
    plt.suptitle('Comparación de Modelos: Valores Observados vs Estimados', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('docs/images/comparacion_modelos_diabetes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Métricas comparativas (3 gráficos)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # R²
    r2_scores = [results[name]['R²'] for name in model_names]
    bars = axes[0].bar(range(len(model_names)), r2_scores, color='skyblue', edgecolor='navy')
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Comparación de R² entre Modelos')
    axes[0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
    axes[0].axhline(y=0.9, color='gold', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # RMSE
    rmse_scores = [results[name]['RMSE'] for name in model_names]
    bars = axes[1].bar(range(len(model_names)), rmse_scores, color='lightcoral', edgecolor='darkred')
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].set_ylabel('RMSE (mg/dL)')
    axes[1].set_title('Comparación de RMSE entre Modelos')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Validación cruzada
    cv_means = [results[name]['CV_R²_mean'] for name in model_names]
    cv_stds = [results[name]['CV_R²_std'] for name in model_names]
    x_pos = np.arange(len(model_names))
    axes[2].errorbar(x_pos, cv_means, yerr=cv_stds, fmt='o', capsize=5, capthick=2, 
                     markersize=8, color='green', ecolor='gray', alpha=0.7)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[2].set_ylabel('CV R² Score')
    axes[2].set_title('R² con Validación Cruzada (μ ± σ)')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
    
    plt.suptitle('Métricas Comparativas de Rendimiento', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('docs/images/metricas_comparativas_diabetes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Análisis de residuos
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, name in enumerate(model_names):
        if idx < len(axes):
            ax = axes[idx]
            y_pred = results[name]['predictions']
            residuals = y_test - y_pred
            
            ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='red', linestyle='--', lw=2)
            ax.set_xlabel('Residuos (mg/dL)')
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'{name}\nμ={residuals.mean():.2f}, σ={residuals.std():.2f}')
            ax.grid(True, alpha=0.3)
    
    if len(model_names) < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle('Análisis de Residuos por Modelo', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('docs/images/analisis_residuos_diabetes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Importancia de características (Random Forest)
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.xlabel('Importancia')
        plt.title('Top 10 Características Más Importantes (Random Forest)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('docs/images/importancia_caracteristicas_diabetes.png', dpi=300, bbox_inches='tight')
        plt.show()

def analizar_clasificacion_categorica(results, y_test, model_names):
    """Análisis de clasificación categórica de diabetes"""
    print("\n📊 ANÁLISIS DE CLASIFICACIÓN CATEGÓRICA")
    print("-" * 60)
    
    class_labels = {0: 'Normal', 1: 'Prediabetes', 2: 'Diabetes'}
    y_test_cat = [categorizar_glucosa(val) for val in y_test]
    
    for name in model_names:
        y_pred = results[name]['predictions']
        y_pred_cat = [categorizar_glucosa(val) for val in y_pred]
        
        accuracy = accuracy_score(y_test_cat, y_pred_cat)
        print(f"\n{name}:")
        print(f"  Precisión en clasificación: {accuracy:.3f}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test_cat, y_pred_cat)
        
        # Métricas por clase
        for i, label in class_labels.items():
            if i < len(cm):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                print(f"  Sensibilidad {label}: {sensitivity:.3f}")

def main_analisis_comparativo():
    """Función principal del análisis comparativo integrado"""
    print("="*80)
    print(" ANÁLISIS COMPARATIVO DE MODELOS - delfosA1C8 ".center(80))
    print("="*80)
    
    # 1. Preparar datos
    df = preparar_datos_analisis(n_samples=500, seed=42)
    
    # Estadísticas básicas
    print(f"\n📊 Dataset preparado: {df.shape[0]} registros, {df.shape[1]} columnas")
    print(f"\n📈 Estadísticas de Glucosa en Ayunas:")
    print(df['Resultado'].describe())
    
    # Crear categorías
    df['categoria_diabetes'] = df['Resultado'].apply(categorizar_glucosa)
    
    # Distribución de clases
    print("\n🎯 Distribución de Clases:")
    class_labels = {0: 'Normal', 1: 'Prediabetes', 2: 'Diabetes'}
    for clase, label in class_labels.items():
        count = (df['categoria_diabetes'] == clase).sum()
        pct = count / len(df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # 2. Inicializar preprocessor
    preprocessor = DiabetesDataPreprocessor()
    
    # 3. Ingeniería de features avanzada
    df_features = preprocessor.ingenieria_features_avanzada(df)
    
    # Definir features para el análisis
    global feature_columns
    feature_columns = [
        'edad', 'talla', 'peso', 'imc', 'tas', 'tad',
        'perimetro_abdominal', 'puntaje_total', 'riesgo_dm',
        'presion_arterial_media', 'ratio_cintura_altura',
        'indice_masa_corporal_cat'
    ]
    
    # Limpiar y seleccionar features
    df_clean = df_features[feature_columns + ['Resultado']].dropna()
    print(f"\n📊 Datos limpios: {len(df_clean)} registros")
    
    # 4. División 80/20
    X = df_clean[feature_columns]
    y = df_clean['Resultado']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=df_clean['Resultado'].apply(categorizar_glucosa)
    )
    
    print(f"\n📊 División de datos:")
    print(f"  Entrenamiento: {len(X_train)} registros (80%)")
    print(f"  Prueba: {len(X_test)} registros (20%)")
    
    # Normalización
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # 5. Entrenar modelos
    models = definir_modelos_comparativos()
    results, predictions = entrenar_y_evaluar_modelos(
        X_train_scaled, X_test_scaled, y_train, y_test, models
    )
    
    # 6. Tabla de comparación
    print("\n📊 COMPARACIÓN DE MODELOS:")
    print("-" * 90)
    print(f"{'Modelo':<25} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'CV R² (μ±σ)':>20}")
    print("-" * 90)
    
    model_names = list(results.keys())
    for name in model_names:
        metrics = results[name]
        cv_str = f"{metrics['CV_R²_mean']:.3f}±{metrics['CV_R²_std']:.3f}"
        print(f"{name:<25} {metrics['R²']:>8.4f} {metrics['RMSE']:>10.2f} "
              f"{metrics['MAE']:>10.2f} {cv_str:>20}")
    
    # Mejor modelo
    best_model = max(results.items(), key=lambda x: x[1]['R²'])
    print(f"\n🏆 Mejor modelo por R²: {best_model[0]} (R² = {best_model[1]['R²']:.4f})")
    
    # 7. Visualizaciones
    print("\n📈 Generando visualizaciones...")
    visualizar_comparacion_modelos(results, y_test, model_names)
    
    # 8. Análisis categórico
    analizar_clasificacion_categorica(results, y_test, model_names)
    
    # 9. Reporte final
    print("\n" + "="*80)
    print(" REPORTE FINAL ".center(80))
    print("="*80)
    
    print(f"\n🏆 MODELO RECOMENDADO: {best_model[0]}")
    print(f"   - R² Score: {best_model[1]['R²']:.4f}")
    print(f"   - RMSE: {best_model[1]['RMSE']:.2f} mg/dL")
    print(f"   - MAE: {best_model[1]['MAE']:.2f} mg/dL")
    print(f"   - CV R²: {best_model[1]['CV_R²_mean']:.3f} ± {best_model[1]['CV_R²_std']:.3f}")
    
    print("\n📊 CONCLUSIONES:")
    print("   1. Integración exitosa del análisis comparativo en delfosA1C8")
    print("   2. Reutilización efectiva de componentes modulares existentes")
    print("   3. Análisis completo con 7 algoritmos y métricas avanzadas")
    print("   4. Visualizaciones generadas y guardadas en docs/images/")
    
    print("\n💾 Archivos generados:")
    print("   - data/raw/outputglucosa.csv (datos para análisis)")
    print("   - docs/images/comparacion_modelos_diabetes.png")
    print("   - docs/images/metricas_comparativas_diabetes.png")
    print("   - docs/images/analisis_residuos_diabetes.png")
    print("   - docs/images/importancia_caracteristicas_diabetes.png")
    
    print("\n✅ Análisis comparativo completado exitosamente")
    print("="*80)

if __name__ == "__main__":
    main_analisis_comparativo()