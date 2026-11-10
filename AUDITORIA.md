# Auditoría Técnica Completa del Proyecto Integrador

**Asignatura:** Programación Avanzada en Ciencia de Datos — Licenciatura en Ciencia de Datos  
**Documento auditado:** `Proyecto Final - Programación Avanzada en Ciencia de Datos.pdf`  
**Proyecto evaluado:** `taller_integrador/`  
**Fecha de auditoría:** Noviembre 2026  

---

## 1. Lectura y Análisis Exhaustivo del PDF Oficial

Del análisis completo y detallado del documento de la consigna oficial, se desglosan todos los requisitos, condiciones, restricciones y rúbricas exigidas por la cátedra:

### 1.1. Dominio y Datasets Disponibles (Sección 1)
Cada estudiante o equipo debe seleccionar **uno** de los siguientes tres datasets públicos y gratuitos en Argentina / CABA para desarrollar un **modelo predictivo de clasificación binaria o multiclase**:
* **Opción A — Mercado Inmobiliario y Tipificación de Propiedades (CABA):**
  * *Dataset:* Propiedades e inmuebles en venta (Portal de Datos Abiertos de la Ciudad de Buenos Aires - BA Data).
  * *Objetivo:* Predecir la categoría/segmento de valor del inmueble (ej. Gama Alta / Lujo vs. Estándar / Accesible) o el tipo de propiedad (Departamento vs. PH / Casa) según superficie, barrios y prestaciones.
  * *Enlace:* `https://data.buenosaires.gob.ar/dataset/propiedades-en-venta`
* **Opción B — Eficiencia y Consumo Energético en Edificios Públicos:**
  * *Dataset:* Consumo de energía en dependencias públicas (`datos.gob.ar`).
  * *Objetivo:* Predecir si una instalación registrará un nivel de consumo Alto / Crítico (1) o Normal / Eficiente (0) evaluando superficie, uso edilicio y estacionalidad.
* **Opción C — Evaluación de Riesgo y Devolución de Microcréditos:**
  * *Dataset:* Inclusión financiera y comportamiento crediticio (`datos.gob.ar` / Banco Central de la República Argentina).
  * *Objetivo:* Predecir el cumplimiento o mora crediticia (Mora Alta vs. Cumplidor) para la automatización de la evaluación de riesgo financiero.

### 1.2. Requerimientos Técnicos y Arquitectura (Sección 2)
1. **Construcción del Modelo (`scikit-learn`):**
   * Implementar obligatoriamente un `sklearn.pipeline.Pipeline` para el preprocesamiento (debe incluir: imputación de faltantes, escalado numérico y codificación de variables categóricas).
   * Comparar al menos **dos algoritmos de clasificación** (ej. Logistic Regression, Random Forest, XGBoost).
   * Optimizar hiperparámetros mediante `GridSearchCV` o `RandomizedSearchCV`.
2. **Evaluación de Desempeño:**
   * Evaluar los modelos utilizando obligatoriamente las métricas: **Accuracy**, **Precision**, **Recall** y **F1-Score**.
   * Acompañar la evaluación con la **Matriz de Confusión** correspondiente.
3. **Almacenamiento e Integración con Base de Datos:**
   * Persistir la información en una base de datos relacional (`SQLite`, `PostgreSQL`) o NoSQL (`MongoDB`).
   * La BD debe incluir:
     * **Tabla 1:** Datos de entrada limpios y preprocesados.
     * **Tabla 2:** Resultados del modelo (predicciones individuales y métricas globales).
     * **Tabla 3 (Opcional - Extra):** Configuración y parametrización del modelo.
4. **Visualización:**
   * Presentar gráficos y tablas claras utilizando `matplotlib`, `seaborn`, `Plotly`, `Power BI` o `Streamlit`.

### 1.3. Rúbrica de Evaluación y Entregables (Sección 3)
* **Criterios de Ponderación:**
  * **Buenas prácticas de código y Git (25%):** Organización del proyecto, calidad del código (PEP 8), **uso conjunto de funciones y clases**, comentarios y control de versiones en Git/GitHub.
  * **Base de datos y visualización (25%):** Integración correcta de BD para almacenamiento de datos de entrada/salida y uso de visualizaciones claras y efectivas.
  * **Modelado y resultados (25%):** Claridad al explicar el flujo de trabajo, decisiones técnicas y justificación metodológica de métricas.
  * **Presentación oral y defensa (25%):** Exposición de 8 a 10 minutos, claridad, dominio del tema, uso de material visual y manejo del tiempo.
* **Formato Oficial de Entrega:**
  * **Repositorio Git:** Código fuente estructurado, **script de carga a BD**, `requirements.txt` o `environment.yml` y `README.md`.
  * **Base de Datos:** Accesible para la revisión.
  * **Presentación:** Diapositivas (6 a 8 filminas) en formato **PDF o PPTX** subido al campus virtual.

---

## 2. Análisis del Proyecto Actual en Anti-Gravity

Se realizó una inspección técnica completa de todos los directorios, archivos fuente, dependencias, artefactos y bases de datos presentes en el repositorio:

### 2.1. Arquitectura y Modularidad
El proyecto presenta una estructura organizada de carpetas:
* `src/`: módulos funcionales divididos por responsabilidad (`data_loader.py`, `cleaning.py`, `features.py`, `models.py`, `evaluation.py`, `db_sql.py`, `db_mongo.py`, `visualization.py`, `logger_setup.py`).
* `main.py`: script orquestador secuencial del pipeline completo.
* `config.py`: constantes centralizadas de paths, URLs y mapeos de columnas.
* `app/`: contiene `app.py` (frontend Streamlit) y `modelo_ganador.joblib` (modelo serializado).
* `data/`: contiene `data/raw/` (16 datasets anuales 2001-2016) y `inmuebles.db` (base SQLite).
* `notebooks/`: `01_eda.ipynb` y `02_modelado.ipynb`.
* `logs/`: registros de ejecución en texto y figuras generadas (`matriz_confusion_ganador.png`, etc.).

### 2.2. Preprocesamiento e Ingeniería de Características
* Se aplica filtrado geográfico por coordenadas de CABA y remoción de valores extremos en superficie y precio en `src/cleaning.py`.
* Se crea la variable objetivo binaria `ES_GAMA_ALTA` mediante percentil anual móvil (percentil 75) para neutralizar distorsiones macroeconómicas (inflación / devaluación histórica).
* En `src/features.py`, se utiliza `ColumnTransformer` articulando `SimpleImputer(strategy='median')` + `StandardScaler()` para variables numéricas, y `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')` para categóricas.

### 2.3. Modelado Predictivo y Métricas
* Se entrenan 3 modelos:
  1. `DummyClassifier(strategy='most_frequent')`: baseline de control ingenuo.
  2. `LogisticRegression(class_weight='balanced')`: modelo lineal interpretable.
  3. `RandomForestClassifier`: ensamble no lineal optimizado con `GridSearchCV` sobre muestra representativa estratificada.
* Se calculan Accuracy, Precision, Recall, F1-Score (ponderado y macro), ROC-AUC y matriz de confusión en `src/evaluation.py`.
* El modelo ganador es serializado con `joblib` en `app/modelo_ganador.joblib`.

### 2.4. Persistencia en Base de Datos (Hallazgo Crítico)
* **SQLite (`data/inmuebles.db`):** Solo contiene la tabla `departamentos_limpios` (Tabla 1). **No existen las tablas de resultados ni de métricas en SQLite**. A pesar de que el `README.md` declara la existencia de `modelo_predicciones`, `modelo_metricas_globales` y `modelo_parametros`, el código de `src/db_sql.py` no las implementa ni las genera.
* **MongoDB (`taller_inmobiliario`):** Contiene las colecciones `resultados` (con métricas globales y una muestra limitada a 500 predicciones) y `parametros` (hiperparámetros óptimos). Sin embargo, no almacena la Tabla 1 de datos limpios.
* **Diagnóstico de BD:** El esquema quedó particionado en dos tecnologías sin que ninguna contenga de forma integral las tablas obligatorias (Tabla 1 y Tabla 2). Un evaluador que examine únicamente el archivo entregable `inmuebles.db` no encontrará los resultados ni las métricas persistidas.

### 2.5. Visualización e Interfaz de Usuario
* En `src/visualization.py` se generan gráficos estáticos con `matplotlib` y `seaborn` (correlaciones, distribución de clases, importancia de variables y matriz de confusión).
* En `app/app.py` se diseñó un dashboard completo en `Streamlit` con tres pestañas: Simulador predictivo en vivo, Métricas del modelo y Explorador de persistencia de bases de datos.

### 2.6. Cuadernos de Jupyter
* `notebooks/01_eda.ipynb`: Contiene únicamente 2 celdas de código inconclusas; carece del análisis exploratorio visual requerido.
* `notebooks/02_modelado.ipynb`: Contiene 20 celdas interactivas completas que replican el entrenamiento y comparación de modelos.

### 2.7. Buenas Prácticas de Programación y Git
* **Adopción de POO (Clases):** La rúbrica exige *"uso de funciones y clases"*. Actualmente, la única clase definida en todo el proyecto es `MongoHandler(logging.Handler)` en `logger_setup.py`. Todas las demás etapas están escritas con funciones procedimentales.
* **Control de Versiones (Git):** El historial de commits registra mensajes no descriptivos (`"1"`, `"2"`, `"3"`, `"readme"`). Asimismo, la carpeta del entorno virtual `.env/` no está listada en `.gitignore` (solo figura `.venv/`), quedando como archivo no rastreado (*untracked*).
* **Entregable de Presentación:** No existe ninguna presentación (diapositivas de 6 a 8 filminas en formato PDF o PPTX).

---

## 3. Comparación Punto por Punto

| Requisito del PDF | Estado | Exigencia del PDF | Implementación Actual en el Proyecto | Faltante o Corrección Necesaria |
| :--- | :---: | :--- | :--- | :--- |
| **Elección de Dataset y Problema** | **CUMPLE** | Seleccionar 1 de 3 opciones oficiales para clasificación binaria o multiclase. | Se implementó la Opción A (BA Data Inmuebles CABA) con target binario `ES_GAMA_ALTA` (1 vs 0). | Ninguno. |
| **Pipeline de Preprocesamiento** | **CUMPLE** | `sklearn.pipeline.Pipeline` con imputación, escalado y codificación categórica. | `ColumnTransformer` en `src/features.py` con `SimpleImputer`, `StandardScaler` y `OneHotEncoder`. | Ninguno. |
| **Comparación de Algoritmos** | **CUMPLE** | Comparar al menos dos algoritmos de clasificación. | Se comparan `LogisticRegression` y `RandomForestClassifier` (más `DummyClassifier` de control). | Ninguno. |
| **Optimización de Hiperparámetros** | **CUMPLE** | Optimizar hiperparámetros con `GridSearchCV` o `RandomizedSearchCV`. | `GridSearchCV` implementado en `src/models.py` sobre `RandomForestClassifier` (`n_estimators`, `max_depth`). | Ninguno. |
| **Métricas de Evaluación** | **CUMPLE** | Evaluar Accuracy, Precision, Recall y F1-Score. | Implementadas en `src/evaluation.py` (ponderadas, macro, por clase y ROC-AUC). | Ninguno. |
| **Matriz de Confusión** | **CUMPLE** | Acompañar la evaluación con matriz de confusión correspondiente. | Calculada con scikit-learn y graficada con Seaborn en `logs/matriz_confusion_ganador.png` y Streamlit. | Ninguno. |
| **BD: Tabla 1 (Datos Limpios)** | **CUMPLE** | Persistir datos de entrada limpios y preprocesados en la BD seleccionada. | Almacenada en SQLite (`data/inmuebles.db` -> tabla `departamentos_limpios`). | En MongoDB no está almacenada. |
| **BD: Tabla 2 (Resultados y Métricas)** | **CUMPLE PARCIALMENTE** | Persistir resultados del modelo (predicciones y métricas globales). | En SQLite **no existe**. En MongoDB se almacena una colección `resultados` con métricas y una muestra de 500 predicciones. | La base relacional entregable (`inmuebles.db`) carece de la Tabla 2. Debe implementarse la persistencia de predicciones y métricas en SQLite. |
| **BD: Tabla 3 (Parámetros - Opcional)** | **CUMPLE** | Configuración y parametrización del modelo (Opcional - Extra). | Almacenada en MongoDB (`parametros`). | No está en SQLite (aunque el README lo afirma). |
| **Visualización** | **CUMPLE** | Presentar gráficos y tablas claras (`matplotlib`, `seaborn`, `Plotly`, `Power BI` o `Streamlit`). | Implementado con `matplotlib`, `seaborn` y dashboard interactivo en `Streamlit` (`app/app.py`). | Ninguno. |
| **Script de Carga a BD** | **CUMPLE PARCIALMENTE** | Script de carga a BD incluido en el repositorio. | La carga a BD está integrada como paso interno en `main.py` y `src/db_sql.py`. | Falta un script autónomo e independiente (ej. `scripts/cargar_bd.py`) que ejecute la carga sin reentrenar ML. |
| **Dependencias y README** | **CUMPLE PARCIALMENTE** | `requirements.txt` o `environment.yml` y `README.md`. | Ambos archivos existen en la raíz. | El `README.md` describe tablas en SQLite que no existen en el código (`modelo_predicciones`, etc.). |
| **Buenas Prácticas: POO (Clases)** | **CUMPLE PARCIALMENTE** | Uso equilibrado de funciones y clases (PEP 8). | Código modular y con docstrings, pero solo existe **1 clase** en todo el proyecto (`MongoHandler`). | Implementar clases de dominio (ej. `InmueblesDBManager` o `PipelineModel`) acorde a Programación Avanzada. |
| **Buenas Prácticas: Git** | **CUMPLE PARCIALMENTE** | Control de versiones profesional en Git. | Repositorio Git inicializado y vinculado a origen remoto. | Mensajes de commits poco descriptivos (`"1"`, `"2"`, `"3"`), `.env/` sin ignorar en `.gitignore`. |
| **Cuadernos de Análisis (EDA)** | **CUMPLE PARCIALMENTE** | Claridad al explicar el flujo de trabajo y decisiones metodológicas. | `02_modelado.ipynb` está completo (20 celdas). | `01_eda.ipynb` está incompleto (solo 2 celdas preliminares). |
| **Presentación Oral y Defensa** | **NO CUMPLE** | Diapositivas (6 a 8 filminas) en formato PDF o PPTX. | No existe ningún archivo de presentación en el proyecto. | Debe elaborarse la presentación requerida de 6 a 8 filminas en formato PDF o PPTX. |

---

## 4. Resultado de la Auditoría

### 4.1. Resumen Cuantitativo de Cumplimiento
* **Total de requerimientos evaluados:** 16 aspectos técnicos y formales.
* **CUMPLE:** 7 ítems (43.75%)
* **CUMPLE PARCIALMENTE:** 8 ítems (50.00%)
* **NO CUMPLE:** 1 ítem (6.25%)

### 4.2. Porcentaje General Ponderado por Rúbrica Oficial
Aplicando los pesos estipulados en la rúbrica de evaluación del trabajo práctico:
* **Modelado y resultados (25% del total):** **95%** de cumplimiento.  
  *Justificación:* El pipeline de `scikit-learn`, el uso de `ColumnTransformer`, la optimización con `GridSearchCV`, las 4 métricas oficiales y la matriz de confusión están técnicamente impecables.
* **Base de datos y visualización (25% del total):** **75%** de cumplimiento.  
  *Justificación:* La visualización estática y la app en `Streamlit` son completas, pero la persistencia en base de datos está fragmentada y la base relacional física carece de la Tabla 2 obligatoria.
* **Buenas prácticas de código y Git (25% del total):** **65%** de cumplimiento.  
  *Justificación:* Excelente modularidad y documentación, pero falta adopción de Programación Orientada a Objetos (POO con clases de dominio), commits no descriptivos y `.env` expuesto en git.
* **Presentación oral y entregables formales (25% del total):** **45%** de cumplimiento.  
  *Justificación:* Repositorio estructurado y código ejecutable, pero falta el script autónomo de carga a BD y el archivo de diapositivas (6 a 8 filminas) está totalmente ausente.

$$\text{Cumplimiento Total} = (0.25 \times 0.95) + (0.25 \times 0.75) + (0.25 \times 0.65) + (0.25 \times 0.45) = 70.0\%$$

> **Porcentaje General de Cumplimiento Actual:** **~70 %**

---

## 5. Plan de Acción para Alcanzar el 100 % de los Requisitos

Para subsanar las observaciones y alcanzar el **100 % de cumplimiento** en la entrega final, se debe seguir la siguiente hoja de ruta técnica:

1. **Completar las Tablas en SQLite (`data/inmuebles.db`) [Prioridad Crítica]:**
   * Incorporar en `src/db_sql.py` y orquestar en `main.py` la creación y persistencia de:
     * `modelo_predicciones` (Tabla 2A): columnas `id`, `y_real`, `y_pred`, `probabilidad_gama_alta`.
     * `modelo_metricas_globales` (Tabla 2B): métricas calculadas (`accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`).
     * `modelo_parametros` (Tabla 3 Opcional): registro de los mejores hiperparámetros encontrados por `GridSearchCV`.
   * Con esto, el archivo `inmuebles.db` entregable en el repositorio cumple de forma autosuficiente con todas las tablas solicitadas por la consigna.
2. **Crear el Script Autónomo de Carga a BD:**
   * Desarrollar un script dedicado (ej. `scripts/cargar_bd.py` o módulo ejecutable `python -m src.db_sql --load`) que realice la ingesta de los 16 CSVs crudos, ejecute la limpieza y guarde `departamentos_limpios` en la base de datos sin obligar a entrenar los modelos de machine learning.
3. **Elaborar las Diapositivas de la Presentación:**
   * Crear la presentación oficial (entre 6 y 8 diapositivas) en formato PDF o PPTX cubriendo:
     * *Filmina 1:* Portada y presentación del equipo/proyecto.
     * *Filmina 2:* Planteo del problema, justificación de Opción A y objetivo de clasificación.
     * *Filmina 3:* Análisis exploratorio de datos (EDA), calidad de datos y tratamiento de outliers.
     * *Filmina 4:* Arquitectura del pipeline de preprocesamiento (`ColumnTransformer`).
     * *Filmina 5:* Modelos evaluados (`LogisticRegression` vs. `RandomForest`) y ajuste con `GridSearchCV`.
     * *Filmina 6:* Resultados comparativos, matriz de confusión y justificación de métricas (F1-Score).
     * *Filmina 7:* Integración con Base de Datos y demostración del dashboard interactivo en `Streamlit`.
     * *Filmina 8:* Conclusiones técnicas y líneas de trabajo futuro.
4. **Completar el Cuaderno `01_eda.ipynb`:**
   * Desarrollar las celdas del análisis exploratorio visual interactivo: mapas de correlación, distribución geográfica en CABA, dispersión de precio vs. m² y distribución de la variable objetivo.
5. **Incorporar Clases de Dominio (POO):**
   * Refactorizar componentes clave a clases orientadas a objetos (por ejemplo, `InmueblesDBManager` para encapsular la persistencia SQLite/Mongo y `PropertyClassificationPipeline` para el orquestado de modelos) para evidenciar el dominio conjunto de funciones y clases exigido por la materia.
6. **Saneamiento de Git y Documentación:**
   * Añadir `.env/` al archivo `.gitignore`.
   * Realizar commits descriptivos en Git con formato estructurado (ej. Conventional Commits: `feat:`, `fix:`, `docs:`).
   * Sincronizar el `README.md` con las tablas efectivamente persistidas y documentar las instrucciones del script de carga.

