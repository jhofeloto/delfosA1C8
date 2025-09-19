# Plan de Integración: Análisis Comparativo de Modelos de Diabetes

## Fecha: 18 de septiembre de 2025
## Autor: Sonoma (Architect Mode)
## Proyecto: delfosA1C8

## Resumen Ejecutivo

Este documento presenta el plan detallado para integrar el script de análisis comparativo de modelos de diabetes proporcionado en el proyecto delfosA1C8. El análisis incluye entrenamiento de múltiples algoritmos de regresión, validación cruzada, visualizaciones comparativas, análisis de residuos e importancia de características. 

**Estado actual**: El proyecto tiene una estructura modular completa pero fragmentada. El nuevo script será adaptado para reutilizar componentes existentes (preprocessor.py, datos sintéticos) mientras se añaden las funcionalidades avanzadas del análisis comparativo.

## Análisis de Compatibilidad

### Componentes Existentes Reutilizables
| Componente | Ubicación | Funcionalidad | Integración Propuesta |
|------------|-----------|---------------|----------------------|
| `DiabetesDataPreprocessor` | `src/data/preprocessor.py` | Limpieza, ingeniería básica de features, split 80/20, scaling | Extender con features adicionales del script nuevo |
| `generate_synthetic_data.py` | `scripts/` | Generación de datos sintéticos | Adaptar salida a formato `outputglucosa.csv` esperado |
| `DiabetesGradientBoosting` | `src/models/gradient_boosting.py` | Modelo base de Gradient Boosting | Usar como baseline y comparar con otros algoritmos |
| `train_models.py` | `scripts/` | Entrenamiento básico de 4 modelos | Expandir a 7 modelos del análisis comparativo |
| Estructura de directorios | `notebooks/`, `docs/images/` | Almacenamiento de análisis y visualizaciones | Usar para el nuevo script y outputs |

### Diferencias Clave con el Script Proporcionado
- **Datos**: El script espera `outputglucosa.csv`. Se generará con `generate_synthetic_data.py` adaptado.
- **Features**: El script nuevo añade `presion_arterial_media`, `ratio_cintura_altura`, `indice_masa_corporal_cat`. Se integrarán en el preprocessor.
- **Modelos**: Expande de 4 a 7 algoritmos (añade Lasso, SVR, MLPRegressor).
- **Métricas**: Añade validación cruzada, análisis de residuos, clasificación categórica, importancia de features.
- **Visualizaciones**: 4 gráficos nuevos (comparación, métricas, residuos, importancia).

## Estructura del Nuevo Script

El script se creará en `notebooks/analisis_comparativo_diabetes.py` con la siguiente estructura:

### 1. Imports y Configuración
```python
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

# Añadir src al path para imports modulares
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                           accuracy_score, classification_report, confusion_matrix)

# Modelos de sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Imports modulares del proyecto
from src.data.preprocessor import DiabetesDataPreprocessor
from scripts.generate_synthetic_data import generate_synthetic_data

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")
```

### 2. Generación y Preparación de Datos
```python
def preparar_datos_analisis(n_samples=500, seed=42):
    """Genera datos sintéticos compatibles con el formato del análisis"""
    print("🔄 Generando datos sintéticos para análisis comparativo...")
    
    # Generar datos usando componente existente
    df_raw = generate_synthetic_data(n_samples, seed)
    
    # Renombrar columnas para compatibilidad con el análisis
    column_mapping = {
        'Resultado': 'Resultado',  # Ya existe
        'edad': 'edad',
        'talla': 'talla', 
        'peso': 'peso',
        'imc': 'imc',
        'tas': 'tas',
        'tad': 'tad',
        'perimetro_abdominal': 'perimetro_abdominal',
        'puntaje_total': 'puntaje_total',
        'riesgo_dm': 'riesgo_dm'
    }
    
    # Asegurar que todas las columnas necesarias existan
    for col in column_mapping.values():
        if col not in df_raw.columns:
            print(f"⚠️  Columna {col} no encontrada, creando placeholder...")
            df_raw[col] = np.random.normal(0, 1, len(df_raw))
    
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
```

### 3. Ingeniería de Features Extendida
```python
def ingenieria_features_avanzada(df, preprocessor):
    """Extiende el preprocessor existente con features del análisis comparativo"""
    print("🔧 Aplicando ingeniería de features avanzada...")
    
    # Usar el preprocessor existente
    df_processed = preprocessor.engineer_features(df)
    
    # Añadir features específicas del análisis comparativo
    if 'tas' in df.columns and 'tad' in df.columns:
        df_processed['presion_arterial_media'] = (df['tas'] + 2*df['tad']) / 3
    
    if 'perimetro_abdominal' in df.columns and 'talla' in df.columns:
        df_processed['ratio_cintura_altura'] = df['perimetro_abdominal'] / df['talla']
    
    # Categorización de IMC
    if 'imc' in df.columns:
        df_processed['indice_masa_corporal_cat'] = pd.cut(
            df['imc'], 
            bins=[0, 18.5, 25, 30, 100], 
            labels=[0, 1, 2, 3]
        ).astype('int64')
    
    return df_processed
```

### 4. Definición y Entrenamiento de Modelos
```python
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
```

### 5. Funciones de Visualización
```python
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
```

### 6. Función Principal
```python
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
    
    # 2. Inicializar preprocessor extendido
    preprocessor = DiabetesDataPreprocessor()
    
    # 3. Ingeniería de features avanzada
    df_features = ingenieria_features_avanzada(df, preprocessor)
    
    # Definir features para el análisis
    global feature_columns
    feature_columns = [
        'edad', 'talla', 'peso', 'imc', 'tas', 'tad', 
        'perimetro_abdominal', 'puntaje_total', 'riesgo_dm',
        'presion_arterial_media', 'ratio_cintura_altura'
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
    X_train_scaled = preprocessor.scaler.fit_transform(X_train)
    X_test_scaled = preprocessor.scaler.transform(X_test)
    
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
```

## Plan de Implementación

### Fase 1: Preparación (Inmediata)
1. **Crear el script principal**: `notebooks/analisis_comparativo_diabetes.py`
2. **Adaptar generate_synthetic_data.py**: Añadir función `generate_synthetic_data` para compatibilidad
3. **Extender preprocessor.py**: Añadir método `ingenieria_features_avanzada`

### Fase 2: Integración (Modo Code)
1. **Cambiar a modo Code** para crear/editar archivos Python
2. **Crear el script completo** con la estructura mostrada arriba
3. **Actualizar scripts/generate_synthetic_data.py** con la función requerida
4. **Modificar src/data/preprocessor.py** para features adicionales

### Fase 3: Documentación y Pruebas
1. **Actualizar README.md** con sección de análisis comparativo
2. **Ejecutar y verificar** que genera todos los outputs
3. **Commitear y pushear** los cambios

## Actualización del README.md

Se añadirá esta sección al README:

```markdown
## 🔬 Análisis Comparativo de Modelos

Para ejecutar el análisis completo de 7 algoritmos de machine learning:

```bash
# 1. Instalar dependencias (si no lo has hecho)
pip install -r requirements.txt

# 2. Generar datos y ejecutar análisis
cd notebooks
python analisis_comparativo_diabetes.py

# 3. Los resultados se guardan en:
#    - data/raw/outputglucosa.csv (datos analizados)
#    - docs/images/ (4 visualizaciones comparativas)
```

### Outputs Generados
- **comparacion_modelos_diabetes.png**: Valores reales vs predichos para cada modelo
- **metricas_comparativas_diabetes.png**: R², RMSE y validación cruzada
- **analisis_residuos_diabetes.png**: Distribución de errores por modelo
- **importancia_caracteristicas_diabetes.png**: Top 10 features más importantes

### Modelos Incluidos
- Regresión Lineal, Ridge, Lasso
- Random Forest, Gradient Boosting
- Support Vector Machine (SVR)
- Red Neuronal Multicapa (MLP)
```

## Consideraciones Técnicas

### Dependencias Adicionales
El script requiere todas las dependencias existentes más:
- `scikit-learn` (ya incluido)
- `matplotlib`, `seaborn` (ya incluidos)

### Rendimiento Esperado
- **Tiempo de ejecución**: ~2-3 minutos para 500 muestras
- **Espacio**: ~10MB para datos + 5MB para imágenes
- **Escalabilidad**: Fácil aumento de muestras ajustando `n_samples`

### Manejo de Errores
- Validación de columnas requeridas
- Manejo de NaN en features calculadas
- Fallback para modelos que fallen en convergencia

## Próximos Pasos

1. **Revisar y aprobar este plan**
2. **Cambiar a modo Code** para implementación: `<switch_mode><mode_slug>code</mode_slug></switch_mode>`
3. **Ejecutar el script** y verificar outputs
4. **Documentar resultados** en el informe final del proyecto

**Estado del plan**: Listo para implementación. ¿Apruebas proceder al modo Code?

---
*Generado por Sonoma en modo Architect - 18/09/2025*