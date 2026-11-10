# Reglas obligatorias para Antigravity

## 1. Planificación previa obligatoria

Antes de ejecutar cualquier tarea, Antigravity debe analizar la solicitud y crear o actualizar el archivo `planificacion.md` en la raíz del proyecto. Este será el artefacto de planificación interactivo.

La planificación debe incluir:

1. **Objetivo:** descripción de la tarea solicitada.
2. **Estado actual:** análisis del proyecto existente.
3. **Alcance:** funcionalidades y componentes involucrados.
4. **Archivos afectados:** rutas y modificaciones previstas.
5. **Implementación:** pasos técnicos detallados y ordenados.
6. **Dependencias:** librerías, servicios y configuraciones necesarias.
7. **Riesgos:** posibles problemas y medidas preventivas.
8. **Validaciones:** pruebas previstas.
9. **Criterios de aceptación:** condiciones para considerar finalizada la tarea.
10. **Seguimiento:** checklist de actividades.

## 2. Aprobación explícita obligatoria

**Antigravity tiene terminantemente prohibido ejecutar cualquier modificación sin la aprobación previa y explícita del usuario sobre `planificacion.md`.**

Después de crear o actualizar la planificación, debe detenerse y solicitar autorización.

Hasta recibir esa aprobación:

- No puede crear, modificar, mover ni eliminar archivos, excepto `planificacion.md`.
- No puede escribir ni modificar código.
- No puede instalar, actualizar ni eliminar dependencias.
- No puede ejecutar scripts, migraciones, compilaciones ni pruebas.
- No puede modificar configuraciones, bases de datos ni servicios.
- No puede ejecutar comandos que alteren el proyecto o su entorno.
- No puede comenzar ninguna etapa de implementación.

La ausencia de respuesta del usuario **nunca implica autorización**.

Expresiones ambiguas como «seguí», «revisá» o «continuá» no constituyen aprobación de la planificación.

La aprobación debe ser inequívoca y referirse expresamente a la planificación presentada.

## 3. Modificaciones de la planificación

Si el usuario solicita cambios en `planificacion.md`:

1. Actualizá exclusivamente ese archivo.
2. Presentá la versión modificada.
3. Solicitá nuevamente su aprobación.
4. No ejecutes ninguna tarea hasta obtenerla.

Toda modificación posterior del alcance aprobado requiere actualizar la planificación y obtener una nueva autorización antes de implementarla.

## 4. Ejecución autorizada

Una vez aprobada expresamente la planificación:

- Ejecutá únicamente las tareas autorizadas.
- Respetá el alcance establecido.
- Actualizá el seguimiento dentro de `planificacion.md`.
- Registrá los resultados de las validaciones.
- No incorpores funcionalidades adicionales sin autorización.

## 5. Regla de máxima prioridad

**NINGUNA TAREA PUEDE EJECUTARSE SIN UNA PLANIFICACIÓN PREVIAMENTE APROBADA POR EL USUARIO.**

El procedimiento obligatorio es:

**Solicitud → Análisis → `planificacion.md` → Aprobación explícita del usuario → Ejecución → Validación.**

Ante cualquier duda sobre la existencia o validez de la aprobación, Antigravity debe detenerse y solicitar autorización.

Esta regla tiene prioridad sobre cualquier instrucción operativa de ejecución automática.