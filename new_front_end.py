import streamlit as st
st.set_page_config(layout="wide")

st.markdown(
    """
        <style>
                .stAppHeader {
                    background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
                    visibility: visible;  /* Ensure the header is visible */
                }

               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 2rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 30px;">ChemRAG</p>'


col1H, col2H, col3H, col4H, col5H, col6H = st.columns( [0.15, 0.3, 0.09, 0.12, 0.102 , 0.2] , vertical_alignment="center")

col1H.markdown(new_title, unsafe_allow_html=True)
user_query = col2H.chat_input("Say something")

toggle_label = (
    "Keyword Search"
    if st.session_state.get("my_toggle", True)
    else "Semantic Search"
)
toggle_value = st.session_state.get("my_toggle", True)

search = col3H.button("Search")
is_toggle = col4H.toggle(toggle_label, value=toggle_value, key="my_toggle")

col5H.caption("Choose NER model")
with col5H:
    optionModel = st.selectbox(
        "Choose NER model",
        ("Catalysis", "Solid State", "Wet Lab", "PcMSP", "CHEMU", "MsMention"),
        key="Model",
        label_visibility="collapsed",
    )

st.text("casa")
st.title("asa")




onix= st.text_input(
        "Placeholder for the other text input widget",
        "This is a placeholder",
        key="placeholdesr",
    )
# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False


col1, col2 = st.columns(2)

with col1:
    st.checkbox("Disable text input widget", key="disabled")
    st.radio(
        "Set text input label visibility 👉",
        key="visibility",
        options=["visible", "hidden", "collapsed"],
    )
    st.text_input(
        "Placeholder for the other text input widget",
        "This is a placeholder",
        key="placeholder",
    )

with col2:
    text_input = st.text_input(
        "Enter some text 👇",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )

    if text_input:
        st.write("You entered: ", text_input)
# Using object notation
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )