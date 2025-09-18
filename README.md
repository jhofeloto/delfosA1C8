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
- **R² Score**: 0.8912
- **RMSE**: 9.82 mg/dL

## 🌐 API

Documentación disponible en: http://localhost:8000/docs

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE)
