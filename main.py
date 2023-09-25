import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image

#Import chatbot.py
import chatbot



#Page Configuration
st.set_page_config(page_title="CVIA", page_icon="📑", layout="wide")

def load_lottieurl(url):
    """
    Loads a Lottie animation from a URL.

    Args:
        url (str): The URL of the Lottie animation.

    Returns:
        dict: The JSON data of the Lottie animation, or None if the URL is invalid.
    """
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

def local_css(file_name):
    """
    Lee un archivo de CSS e inyecta su contenido en el proyecto

    Parameters:
    file_name (str): The path to the local CSS file.

    Returns:
    None
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
        


local_css("style/style.css")
lottie_file ="https://assets9.lottiefiles.com/packages/lf20_ggwq3ysg.json"



#Introduction Container
with st.container():
    st.subheader("Bienvenidos a la Revolución en la Selección de Talento: ")
    st.title("CVIA")
    #image = Image.open("images/logo.png")
    #st.image(image, use_column_width=True)
    st.write(
        "En CVIA, fusionamos la vanguardia tecnológica con la excelencia en recursos humanos para ofrecerte una solución innovadora en la gestión de talento. Somos el puente entre tu empresa y los candidatos ideales, impulsando tu éxito en el mundo laboral."
    )

#Doc loader Container
with st.container():
    st.write("---")
    st.header("Subir Documentos")
    uploaded_files = st.file_uploader(
        "Sube tus documentos aquí", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write("Documentos subidos:")
        for file in uploaded_files:
            st.write(f"Nombre: {file.name}, Tamaño: {file.size} bytes")
    
        print(uploaded_files)
        qa = chatbot.vector_qa(uploaded_files)
            
            
            
#Chatbot Containers
with st.container():
        response = " "
        st.write("---")
        st.header("Chatbot")
        st.write(
            """
            Hola, soy el chatbot de CVIA, estoy aquí para ayudarte a analizar tus documentos y encontrar el mejor talento para tu empresa. 
            """
        )
                    
    
with st.container():
                #Espacio para la pregunta del usuario
                user_question = st.text_input("Hazme una pregunta sobre tus documentos:")
                
                #Imprimir la pregunta en consola
                if user_question:
                    print(user_question)
                
                #Espacio para la respuesta del chatbot
                if user_question:
                    response = qa.run(user_question)
                    
                    #Imprimir en la terminal
                    print(response)
                    
with st.container():
    if response:            
        st.write(response)
            
            


#About Us Container
with st.container():
    st.write("---")
    left_column, right_column= st.columns((2))
    with left_column:
        st.header("Sobre nosotros")
        st.write(
            """
            En CVIA, somos pioneros en el análisis y gestión de currículums (CVs)
            mediante inteligencia artificial. Nuestra misión es simplificar y potenciar el proceso
            de selección de talento en las empresas. ¿En qué podemos ser de ayuda?

            - 🚀 Si quieres optimizar tus procesos de selección y ahorrar tiempo y recursos valiosos mediante la automatización del análisis de CVs.
            - 📈 Si buscas tomar decisiones basadas en datos sólidos para tus procesos de contratación, CVIA te proporciona insights precisos sobre los candidatos.
            - 🌟 Si deseas mejorar la experiencia de tus candidatos y garantizar una contratación más efectiva que te diferencie en el mercado, estamos aquí para ayudarte.
            - 🔧 Si usas herramientas de software antiguas o poco eficientes o procesos en los que usas papel y bolígrafo

            ***Si esto suena interesante para ti puedes contactarnos a través del formulario que encontrarás al final de la página para aplicar CVIA directamente en tu empresa: *** 
            """
        )
        st.write("[Más sobre nosotros>](https://www.google.com/forms/about/)")
    with right_column:
        st_lottie(load_lottieurl(lottie_file), height=400)