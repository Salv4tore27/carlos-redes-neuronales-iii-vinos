# Redes Neuronales III — Carlos

Clasificación de vinos mediante 13 mediciones y una red densa 13-64-32-3. Se contrasta AdamW desacoplado con SGD y RMSprop; se informa la fragilidad de una prueba de 36 observaciones.

Este proyecto conserva el código que produjo las figuras y cifras del informe. Los entrenamientos se ejecutaron localmente en CPU, con la prueba aislada de la selección. El cuaderno es autónomo y se puede abrir en Google Colab; no se atribuye a Colab una ejecución que no se hizo allí.

## Archivos

- Carlos_Redes_Informe.pdf: informe de entrega en PDF.
- Carlos_Redes.ipynb: cuaderno con salida guardada.
- Script .py: ejecución desde línea de comandos.
- resultados.json y CSV/PNG: mediciones y evidencias de la ejecución.
- main.tex y archivo Overleaf .zip: fuente editable del informe.

## Repetir el experimento

Instala las dependencias con python -m pip install -r requirements.txt y ejecuta:

    python experimento_redes.py --persona Carlos --salida resultados

En Colab, instala las dependencias que falten y ejecuta el cuaderno de arriba abajo. La corrida reconstruye la carpeta resultados. Para comparar, revisa los CSV/JSON sin modificar la partición de prueba.

Informe en Overleaf: https://www.overleaf.com/project/6aca70b7e714829b5f2d561d

Repositorio: https://github.com/Salv4tore27/carlos-redes-neuronales-iii-vinos
