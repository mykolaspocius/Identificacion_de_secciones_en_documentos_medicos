Identificación de secciones en textos médicos en español.
Se puede probar el funcionamiento del sistema en: https://huggingface.co/spaces/mpocius/Identificacion_de_secciones_en_textos_medicos?logs=container
# Descripción del problema
  Existe un tipo de textos concreto en el dominio de textos médicos que se denomina notas médicas
y se suele denotar con el acrónimo EHR (Electronical Health Records). Son documentos que
contienen información referente al estado de salud de los pacientes y les sirve a los profesionales
médicos como método de seguimiento de la evolución de este estado en el tiempo y como registro
de acciones realizadas y sus efectos sobre el mismo. Este tipo de documentos suelen carecer de
estructura rígida. Sin embargo, contienen información que se puede clasificar como perteneciente
a ciertas clases concretas bien definidas. Un sistema que permita la consulta rápida y eficiente
de alguno de estos tipos de información contenida en el documento de manera automática
sin tener que leer el documento completo podría facilitar el trabajo de estos profesionales y
mejorar la calidad de atención al paciente. Estas clases o tipos de información contenidas en
la nota médica se suelen denominar como secciones presentes en el texto. Durante los  ́ultimos
años hubo muchas iniciativas para detectar la presencia de estas secciones en los documentos
de todo tipo. Se usaron estrategias diversas que fueron evolucionando con el tiempo. En el
presente trabajo se propone como objetivo crear uno de estos sistemas basados en la tecnología
transformers. Se enfocará la tarea como clasificación de tokens. Para ello se ha partido de un
modelo preentrenado sobre un corpus de textos en español pertenecientes al dominio biomédico
y se realizó un ajuste fino (fine-tuning) para la tarea de clasificación de secciones usando un
corpus especialmente preparado para la tarea. A la hora de crear los modelos se probaron
técnicas de aumento de datos, diferentes modelos preentrenados como partida y reducción de
la complejidad del modelo mediante la congelación de ciertos parámetros antes de comenzar el
entrenamiento. Los resultados obtenidos estaban en el rango de lo esperado.

# Como se creó el datset aumentado
En la plataforma Google Colab, una vez importadas todas las dependencias se ejecutan las siguientes
dos funciones (Los directorios donde aparece el dataset clinais.train.json no se incluye, ya que no es un dataset de dominio público. Para obtenerlo, tiene que ponerse en contacto con los organizadores de la competición ClinAIS).:
translate_dataset_and_save("./ClinAIS_dataset/clinais.train.json","./ClinAIS_dataset/clinais.train.translated.json")
create_augmented_dataset("./ClinAIS_dataset/clinais.train.json","./ClinAIS_dataset/clinais.train.translated.json","./ClinAIS_dataset/clinais.train.augmented.json")
