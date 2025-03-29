import streamlit as st
import pandas as pd
import os
import uuid
from datetime import datetime

IMAGE_DIR = "../data/external/streamlit_images"
CSV_FILE = "../data/external/user_descriptions.csv"


class Client:
    def __init__(self):
        pass

    def show(self):
        # Title of the form
        st.title("Axa")

        # Description
        st.write("Hola, por favor completa este formulario con la información detallada de tu incidencia. Asegúrate de incluir todos los datos relevantes en la descripción y añadir una imagen. Esto nos ayudará a gestionar la incidencia de manera más eficaz.")

        # Streamlit form
        with st.form(key='my_form'):
            # Title and Description fields inside the form
            st.subheader("Formulario de Incidencia")
            title = st.text_input("Inserta un título:")
            description = st.text_area("Inserta una descripción:")

            # Image upload
            image = st.file_uploader("Sube tu imagen", type=["png", "jpg", "jpeg"])

            # Submit button
            submit_button = st.form_submit_button(label='Submit')

            # Display the entered data after submission
            if submit_button:
                print("Submitted")

                unique_id = str(uuid.uuid4())
                form_data = {
                    'ID': [unique_id],
                    'Title': [title],
                    'Description': [description]
                }

                image_filename = f"{unique_id}_{image.name}"
                image_path = os.path.join(IMAGE_DIR, image_filename)
                with open(image_path, "wb") as f:
                    print("Img saved")
                    f.write(image.getbuffer())

                df = pd.DataFrame(form_data)
                if os.path.exists(CSV_FILE):
                    df.to_csv(CSV_FILE, mode='a', header=False, index=False)
                else:
                    df.to_csv(CSV_FILE, mode='w', header=True, index=False)
                print("File saved")

                st.write(f"**Título:** {title}")
                st.write(f"**Descripción:** {description}")
                if image:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                else:
                    st.write("No hay imagen.")