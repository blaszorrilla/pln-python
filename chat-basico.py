import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# 1. Preparación de datos y descargas
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

# Nuestra base de conocimientos (Corpus)
data = [
    "Hola, ¿en que puedo ayudarte?",
    "Buenos dias, espero que estés teniendo un excelente día.",
    "Mi nombre es ChatBot y estoy aquí para asistirle.",
    "Nuestros productos son desayunos y almuerzos ejecutivos",
    "Puede comunicarse al telefono 0984582xxx",
    "Aceptamos pagos con tarjeta de crédito, débito y giros Tigo.",
    "Nuestro horario de atención es de lunes a viernes de 8:00 a 18:00.",
    "Nuestra ubicacion es Tomas R. Pereira entre Carlos A. Lopez y Lomas Valentinas 488"
]

#Funciones de Preprocesamiento
lemmer = nltk.stem.WordNetLemmatizer()

def LemNormalize(text):
    # Convierte a minúsculas, quita puntuación
    tokens = nltk.word_tokenize(text.lower().translate(dict((ord(punct), None) for punct in string.punctuation)))
    return [lemmer.lemmatize(token) for token in tokens]

#Lógica de respuesta
def get_response(user_input, corpus):
    robo_response = ''
    corpus.append(user_input) # Añadimos la pregunta al final del corpus temporalmente
    
    # Creamos la matriz TF-IDF
    tfidf_vec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopwords.words('spanish'),token_pattern=None)
    tfidf_matrix = tfidf_vec.fit_transform(corpus)
    
    #Comparamos la última entrada (usuario) con todas las anteriores
    vals = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    #Buscamos el índice con mayor similitud
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    score = flat[-1]
    
    if score == 0:
        robo_response = "Lo siento, no encuentro información sobre eso."
    else:
        robo_response = corpus[idx]
    
    corpus.pop() #Quitamos la pregunta del usuario para no ensuciar el corpus
    return robo_response

# 4. Bucle de ejecución en consola
print("--- Chatbot Iniciado (Escribe 'salir' para terminar) ---")

while True:
    user_input = input("Tú: ").lower()
    if user_input != 'salir':
        if user_input in ['gracias', 'muchas gracias']:
            print("ChatBot: ¡De nada!")
            break
        else:
            print("ChatBot:", get_response(user_input, data))
    else:
        print("ChatBot: ¡Adiós! Que tengas un buen día.")
        break