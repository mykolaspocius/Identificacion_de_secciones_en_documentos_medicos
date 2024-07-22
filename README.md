Identificación de secciones en textos médicos en español.
Se puede probar el funcionamiento del sistema en: https://huggingface.co/spaces/mpocius/Identificacion_de_secciones_en_textos_medicos?logs=container
* Descripción del problema
  Existe un tipo de textos concreto en el dominio de textos m ́edicos que se denomina notas m ́edicas
y se suele denotar con el acr ́onimo EHR (Electronical Health Records). Son documentos que
contienen informaci ́on referente al estado de salud de los pacientes y les sirve a los profesionales
m ́edicos como m ́etodo de seguimiento de la evoluci ́on de este estado en el tiempo y como registro
de acciones realizadas y sus efectos sobre el mismo. Este tipo de documentos suelen carecer de
estructura r ́ıgida. Sin embargo, contienen informaci ́on que se puede clasificar como perteneciente
a ciertas clases concretas bien definidas. Un sistema que permita la consulta r ́apida y eficiente
de alguno de estos tipos de informaci ́on contenida en el documento de manera autom ́atica
sin tener que leer el documento completo podr ́ıa facilitar el trabajo de estos profesionales y
mejorar la calidad de atenci ́on al paciente. Estas clases o tipos de informaci ́on contenidas en
la nota m ́edica se suelen denominar como secciones presentes en el texto. Durante los  ́ultimos
a ̃nos hubo muchas iniciativas para detectar la presencia de estas secciones en los documentos
de todo tipo. Se usaron estrategias diversas que fueron evolucionando con el tiempo. En el
presente trabajo se propone como objetivo crear uno de estos sistemas basados en la tecnolog ́ıa
transformers. Se enfocar ́a la tarea como clasificaci ́on de tokens. Para ello se ha partido de un
modelo preentrenado sobre un corpus de textos en espa ̃nol pertenecientes al dominio biom ́edico
y se realiz ́o un ajuste fino (fine-tuning) para la tarea de clasificaci ́on de secciones usando un
corpus especialmente preparado para la tarea. A la hora de crear los modelos se probaron
t ́ecnicas de aumento de datos, diferentes modelos preentrenados como partida y reducci ́on de
la complejidad del modelo mediante la congelaci ́on de ciertos par ́ametros antes de comenzar el
entrenamiento. Los resultados obtenidos estaban en el rango de lo esperado.
