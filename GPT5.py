from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nombre_modelo = "t5-small"
convertir_vectores = AutoTokenizer.from_pretrained(nombre_modelo)
modelo = AutoModelForSeq2SeqLM.from_pretrained(nombre_modelo)

# Pedir el texto por consola
texto = input("Introduce el texto que quieres resumir: ")

# Preprocesar texto agregando el prefijo "summarize: " para ordenarle al modelo que "resuma" el texto
texto = "summarize: " + texto

vectores_entrada = convertir_vectores.encode(texto, return_tensors="pt", max_length=1024, truncation=True)

# Generar resumen usando el modelo

# Primero generamos el texto de salida en formato de vectores
vectores_salida = modelo.generate(vectores_entrada, max_length=1024, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)

# Luego decodificamos esos vectores de salida para obtener el texto resumido
resumen = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)

# Mostrar el texto resumido
print("\n Texto resumido:")
print(resumen)
