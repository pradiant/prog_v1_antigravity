# Proyecto Final: Modelado Predictivo de Clasificación y Pipeline de Datos

**Asignatura:** Programación Avanzada en Ciencia de Datos — Licenciatura en Ciencia de Datos  
**Opción Elegida:** Opción A — Mercado Inmobiliario y Tipificación de Propiedades (CABA)  
**Dataset:** Propiedades e inmuebles en venta (Portal de Datos Abiertos de la Ciudad de Buenos Aires - BA Data, 2001–2016).  

---

## 📌 Descripción del Proyecto

El objetivo de este proyecto es implementar una solución punta a punta (*end-to-end*) aplicando buenas prácticas de programación modular, control de versiones, integración con bases de datos y desarrollo de modelos predictivos de clasificación mediante **scikit-learn**.

El modelo predice si un departamento puesto en venta en la Ciudad Autónoma de Buenos Aires califica en el segmento de **Gama Alta / Lujo (1)** o **Estándar / Accesible (0)**, en función de sus metros cuadrados, antigüedad, cantidad de ambientes, baños, comodidades (cochera, baulera, lavadero, terraza), barrio y comuna.

---

## 🏗️ Arquitectura del Repositorio

```
taller_integrador/
├── .venv/                      # Entorno virtual de Python
├── .gitignore                  # Exclusiones de Git
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Documentación general y guía de ejecución
├── DOCUMENTACION.md            # Guía didáctica y técnica completa al 100%
├── planificacion.md            # Registro de planificación y seguimiento
├── config.py                   # Rutas del sistema, mapeo de columnas y URLs
├── data/
│   ├── raw/                    # 16 datasets anuales de BA Data (2001 a 2016)
│   ├── processed/              # Archivos procesados intermedios
│   └── inmuebles.db            # Base de datos SQLite generada con las 3 tablas
├── src/
│   ├── __init__.py             # Inicializador del paquete
│   ├── logger_setup.py         # Logger rotativo a consola y archivo
│   ├── data_loader.py          # Lector y unificador de datasets anuales
│   ├── cleaning.py             # Limpieza, outliers y target binario ES_GAMA_ALTA
│   ├── features.py             # ColumnTransformer (imputación + StandardScaler + OneHotEncoder)
│   ├── models.py               # LogisticRegression, RandomForestClassifier y GridSearchCV
│   ├── evaluation.py           # Métricas oficiales (Accuracy, Precision, Recall, F1) y matriz de confusión
│   ├── db_sql.py               # Persistencia de las 3 tablas en SQLite
│   ├── db_mongo.py             # Persistencia de experimentos y logs en MongoDB
│   └── visualization.py        # Generación y guardado de gráficos
├── notebooks/
│   ├── 01_eda.ipynb            # Análisis exploratorio detallado de los 16 años
│   └── 02_modelado.ipynb       # Experimentación interactiva celda por celda
├── app/
│   ├── app.py                  # Dashboard interactivo en Streamlit con simulador
│   └── modelo_ganador.joblib   # Modelo óptimo serializado
└── main.py                     # Script orquestador de ejecución autónoma
```

---

## ⚙️ Instalación y Requisitos

1. **Clonar el repositorio o situarse en la carpeta:**
   ```bash
   cd taller_integrador
   ```

2. **Crear y activar el entorno virtual:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # En Linux/macOS
   # .venv\Scripts\activate   # En Windows
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Ejecución del Proyecto

### 1. Ejecutar el Pipeline Completo por Consola
Para unificar los 16 años de datos, limpiarlos, generar el target, entrenar los dos modelos (`LogisticRegression` y `RandomForestClassifier`), buscar hiperparámetros con `GridSearchCV`, evaluar métricas y guardar las 3 tablas en la base de datos:
```bash
python main.py
```

### 2. Lanzar el Dashboard Interactivo (Streamlit)
Para abrir la interfaz web interactiva con métricas y simulador de predicción en tiempo real:
```bash
streamlit run app/app.py
```

### 3. Ejecutar los Notebooks de Jupyter
```bash
jupyter lab
# Abrir notebooks/01_eda.ipynb o notebooks/02_modelado.ipynb
```

---

## 🗄️ Estructura de la Base de Datos (`data/inmuebles.db`)

El proyecto cumple con las 3 tablas exigidas por la consigna del TP:
1. **`departamentos_limpios` (Tabla 1):** Registros filtrados, sin outliers de precio/superficie, con coordenadas validadas dentro de CABA y columna objetivo `ES_GAMA_ALTA`.
2. **`modelo_predicciones` y `modelo_metricas_globales` (Tabla 2):** Predicciones fila por fila sobre el conjunto de test comparadas contra la clase real, junto con el resumen de métricas globales (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
3. **`modelo_parametros` (Tabla 3):** Registro de configuración e hiperparámetros óptimos seleccionados por `GridSearchCV`.

---

## 📘 Guía Pedagógica
Para un entendimiento paso a paso pensado para quien quiere aprender desde cero cómo funciona cada etapa técnica y matemática, consultá el archivo:
👉 **[`DOCUMENTACION.md`](DOCUMENTACION.md)**
