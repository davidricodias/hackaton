import streamlit as st
from PIL import Image
from streamlit_app.pages import axa_view, client_view

def axa_view_show():
    st.write("Hola AXA")

def client_view_show():
    st.write("Hola CLIENT")


def main():
    home = axa_view.Home()
    client = client_view.Client()

    st.sidebar.title("AXA")

    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
        home.show()
    if st.sidebar.button("Client_view"):
        st.session_state.page = "Upload"
        client.show()

    if "page" not in st.session_state:
        st.session_state.page = "Home"


if __name__ == "__main__":
    main()