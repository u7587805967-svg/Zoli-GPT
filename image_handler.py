import re
import requests
import streamlit as st

PIXABAY_API_KEY = st.secrets["PIXABAY_API_KEY"]

def get_pixabay_image_url(query: str) -> str | None:
    """
    Kép URL lekérése a Pixabay API segítségével a megadott keresőszó alapján.
    """
    if not PIXABAY_API_KEY or PIXABAY_API_KEY == "A_TI_PIXABAY_API_KULCSOD":
        st.warning("Hiányzik a Pixabay API kulcs!")
        return None

    url = "https://pixabay.com/api/"
    params = {
        "key": PIXABAY_API_KEY,
        "q": query.strip(),
        "image_type": "photo",
        "orientation": "horizontal",
        "per_page": 3,
        "safesearch": "true"  # Biztonságos találatok szűrése
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            hits = data.get("hits", [])
            if hits:
                # Visszaadjuk a legrelevánsabb kép nagy felbontású webes URL-jét (webformatURL)
                return hits[0].get("webformatURL")
    except Exception as e:
        st.error(f"Hiba a Pixabay API hívásakor: {e}")
    
    return None

def render_smart_content(content: str):
    """
    Feldolgozza Zoli válaszát: Ha talál benne [IMAGE: ...] taget,
    lekéri a képet a Pixabay-ről és beszúrja a szöveg közé.
    """
    pattern = r"\[IMAGE:\s*(.*?)\]"
    match = re.search(pattern, content)

    if match:
        image_query = match.group(1)
        parts = re.split(pattern, content, maxsplit=1)
        
        first_part = parts[0]
        second_part = parts[2] if len(parts) > 2 else ""

        # 1. Szöveg első része
        if first_part.strip():
            st.markdown(first_part)

        # 2. Kép lekérése a Pixabay-ről és megjelenítése
        img_url = get_pixabay_image_url(image_query)
        if img_url:
            st.image(img_url, caption=f"Kép: {image_query} (Pixabay)", use_container_width=True)

        # 3. Szöveg második része
        if second_part.strip():
            st.markdown(second_part)
    else:
        # Ha Zoli nem kérte kép beszúrását
        st.markdown(content)