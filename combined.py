import streamlit as st

# Create the navigation structure
pages = {
    "ASL Recognition Models": [
        st.Page("fingersigning.py", title="ASL Fingersigning Model"),
        st.Page("gesture.py", title="ASL Gesture Recognition Model")
    ]
}

# Initialize navigation
pg = st.navigation(pages)
pg.run()
