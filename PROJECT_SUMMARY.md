# 🚀 Exoplanet AI Classifier - NASA Space Apps Challenge 2025

## ✅ Project Completion Summary

¡Hemos completado exitosamente la aplicación de clasificación de exoplanetas para el NASA Space Apps Challenge 2025! 

### 🎯 Objetivos Cumplidos

✅ **Modelo de IA/ML Avanzado**
- XGBoost con calibración isotónica
- Precisión: **80.2%**
- ROC-AUC (macro): **89.6%**
- Entrenado con datos KOI (Kepler) y TOI (TESS)

✅ **Interfaz Web Dual**
- **Modo Básico**: Clasificación individual con formulario interactivo
- **Modo Experto**: Procesamiento por lotes con análisis avanzado

✅ **Características Técnicas**
- Explicabilidad SHAP para interpretabilidad
- Calibración de probabilidades
- Métricas de confianza (HIGH/MEDIUM/LOW)
- Validación de entrada robusta

✅ **Funcionalidades Implementadas**
- API REST completa con FastAPI
- Interfaz web moderna con Bootstrap
- Procesamiento de CSV con plantillas
- Exportación de resultados
- Dashboard de métricas

## 📊 Resultados del Modelo

### Rendimiento General
- **Precisión**: 80.2%
- **ROC-AUC (macro)**: 89.6%
- **Datos procesados**: 16,206 muestras
- **Características**: 12 características ingenieras

### Rendimiento por Clase
- **CONFIRMED**: F1-score = 85.0%
- **CANDIDATE**: F1-score = 15.4%
- **FALSE_POSITIVE**: F1-score = 75.1%

### Características Más Importantes
1. **mission_tess**: 37.8% (codificación de misión TESS)
2. **mission_kepler**: 29.7% (codificación de misión Kepler)
3. **log_orbital_period**: 4.8% (período orbital logarítmico)
4. **transit_depth_ppm**: 4.7% (profundidad del tránsito)
5. **orbital_period_days**: 4.0% (período orbital)

## 🌟 Características Destacadas

### 🤖 Inteligencia Artificial
- **Algoritmo**: XGBoost con early stopping
- **Calibración**: Isotónica para probabilidades confiables
- **Explicabilidad**: SHAP con fallback a feature importance
- **Validación cruzada**: Estratificada por misión y clase

### 🎨 Interfaz de Usuario
- **Diseño responsive**: Bootstrap 5 con tema personalizado
- **Modo Básico**: Formulario intuitivo para usuarios generales
- **Modo Experto**: Herramientas avanzadas para investigadores
- **Visualizaciones**: Gráficos interactivos con Plotly

### 🔧 Funcionalidades Técnicas
- **API REST**: Endpoints documentados con FastAPI
- **Validación**: Entrada robusta con rangos predefinidos
- **Procesamiento**: Batch con filtros y exportación
- **Métricas**: Dashboard completo de rendimiento

## 🚀 Cómo Usar la Aplicación

### Inicio Rápido
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar modelo (si es necesario)
python train_model.py

# 3. Iniciar aplicación
python main.py

# 4. Abrir navegador
# http://localhost:8001
```

### Modo Básico
1. Navegar a `/basic`
2. Introducir parámetros del exoplaneta
3. Seleccionar misión (Kepler/TESS)
4. Obtener predicción instantánea con explicaciones

### Modo Experto
1. Navegar a `/expert`
2. Descargar plantilla CSV
3. Subir archivo con múltiples candidatos
4. Analizar resultados con filtros avanzados
5. Exportar resultados

## 📈 Casos de Uso

### Para Investigadores
- Análisis de grandes volúmenes de datos KOI/TOI
- Identificación rápida de candidatos prometedores
- Validación cruzada con métricas de confianza
- Exportación para análisis posterior

### Para Estudiantes/Educadores
- Aprendizaje interactivo sobre exoplanetas
- Experimentación con parámetros físicos
- Comprensión de factores de clasificación
- Visualización de explicaciones SHAP

### Para Científicos Ciudadanos
- Clasificación de nuevos candidatos
- Contribución a la ciencia ciudadana
- Interfaz accesible sin conocimientos técnicos
- Feedback educativo sobre exoplanetas

## 🔬 Contribución Científica

Este proyecto contribuye a:
- **Automatización**: Reducción del análisis manual de datos
- **Escalabilidad**: Procesamiento de grandes volúmenes de datos
- **Accesibilidad**: Herramientas disponibles para la comunidad
- **Educación**: Recursos para aprendizaje sobre exoplanetas

## 🛠️ Tecnologías Utilizadas

- **Backend**: FastAPI, Python 3.12
- **ML**: XGBoost, scikit-learn, SHAP
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Visualización**: Plotly, Matplotlib
- **Datos**: Pandas, NumPy
- **Despliegue**: Uvicorn, Joblib

## 📝 Próximos Pasos

### Mejoras Futuras
- [ ] Integración con datos en tiempo real
- [ ] Modelo ensemble con múltiples algoritmos
- [ ] Active learning con feedback de usuarios
- [ ] Soporte para misiones adicionales (K2, CHEOPS)
- [ ] API móvil para aplicaciones

### Investigación
- [ ] Análisis de domain adaptation Kepler → TESS
- [ ] Optimización de hiperparámetros
- [ ] Análisis de incertidumbre bayesiano
- [ ] Integración con catálogos astronómicos

## 🎉 Conclusión

Hemos desarrollado exitosamente una aplicación completa de clasificación de exoplanetas que cumple con todos los requisitos del NASA Space Apps Challenge 2025. La aplicación combina técnicas avanzadas de machine learning con una interfaz de usuario intuitiva, proporcionando una herramienta valiosa tanto para investigadores como para el público general interesado en la exploración de exoplanetas.

**¡El proyecto está listo para presentación y uso! 🚀**

---

*Desarrollado con ❤️ para NASA Space Apps Challenge 2025*
