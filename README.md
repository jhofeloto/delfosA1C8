# 🏥 delfosA1C8 - Sistema Predictivo de Diabetes Mellitus Tipo 2

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)

Sistema de Machine Learning para predicción de diabetes tipo 2.

## 🚀 Instalación Rápida

```bash
# Instalar dependencias
pip install -r requirements.txt

# Generar datos y entrenar
python scripts/generate_synthetic_data.py
python scripts/train_models.py

# Iniciar API
uvicorn src.api.app:app --reload
```

## 📊 Resultados

- **Mejor modelo**: Gradient Boosting
- **R² Score**: 0.9823
- **RMSE**: 9.12 mg/dL

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

## 🌐 API

Documentación disponible en: http://localhost:8000/docs

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE)
