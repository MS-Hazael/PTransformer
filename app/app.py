from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration


nombre_modelo = 't5-base'
convertir_vectores = T5Tokenizer.from_pretrained(nombre_modelo)
modelo = T5ForConditionalGeneration.from_pretrained(nombre_modelo)


app = Flask(__name__)

@app.route('/') # ruta base para el menu de opciones
def index():
    # return "Hola muando! - probando el servidor"
    return render_template('index.html')

@app.route('/traducir', methods=['GET', 'POST']) # ruta para la accion de Traducir texto
def generar_traduccion():
    if request.method == 'GET':
        return render_template('traductor.html')
    
    try:
        datos = request.get_json()
        texto = datos.get("texto", "")
        #idioma_origen = datos.get("idioma_origen", "English")
        #idioma_destino = datos.get("idioma_destino", "Spanish")

        if not texto:
            return jsonify({"error": "No se proporcionó un texto"}), 400
        
        texto = f"translate English to Spanish: {texto}"
        vectores_entrada = convertir_vectores.encode(texto, return_tensors="pt", max_length=512, truncation=True)

        vectores_salida = modelo.generate(vectores_entrada, max_length=512, num_beams=6, early_stopping=True)
        traduccion = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)

        return jsonify({"traduccion": traduccion})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/resumen', methods=['GET','POST'])
def generar_resumen():
    if request.method == 'GET':
        return render_template('resumen.html')

    try:
        datos = request.get_json()
        texto = datos.get("texto", "")

        if not texto:
            return jsonify({"error": "No se proporcionó un texto"}), 400
        
        texto = "summarize: " + texto
        vectores_entrada = convertir_vectores.encode(texto, return_tensors="pt", max_length=1024, truncation=True)

        vectores_salida = modelo.generate(vectores_entrada, max_length=1024, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
        resumen = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)

        return jsonify({"resumen": resumen})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/pregunta', methods=['GET','POST'])
def resolver_pregunta():
    if request.method == 'GET':
        return render_template('pregunta.html')
    
    try:
        datos = request.get_json()
        contexto = datos.get("contexto", "").strip()
        pregunta = datos.get("pregunta", "").strip()

        if not contexto or not pregunta:
            return jsonify({"error": "Se requiere contexto y pregunta"}), 400
        
        texto_procesado = f"question: {pregunta} context: {contexto}"
        vectores_entrada = convertir_vectores.encode(texto_procesado, return_tensors="pt", max_length=512, truncation=True)

        vectores_salida = modelo.generate(vectores_entrada, max_length=100, num_beams=6, early_stopping=True)
        respuesta = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)

        return jsonify({"respuesta": respuesta})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/generar_preguntas', methods=['GET','POST'])
def generate_question():
    if request.method == 'GET':
        return render_template('generar_pregunta.html')
    
    try:
        datos = request.get_json()
        texto = datos.get("texto", "").strip()

        if not texto:
            return jsonify({"error": "Se requiere un texto"}), 400

        texto_procesado = f"generate question: {texto}"
        vectores_entrada = convertir_vectores.encode(texto_procesado, return_tensors="pt", max_length=512, truncation=True)

        vectores_salida = modelo.generate(vectores_entrada, max_length=100, num_return_sequences=3, num_beams=6, early_stopping=True)
        preguntas_generadas = [convertir_vectores.decode(v, skip_special_tokens=True) for v in vectores_salida]

        return jsonify({"preguntas": preguntas_generadas})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__=='__main__':
    app.run(debug=True, port=5001)