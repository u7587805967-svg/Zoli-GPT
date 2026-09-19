import re
import requests
import streamlit as st

PIXABAY_API_KEY = st.secrets.get("PIXABAY_API_KEY", "")

def get_pixabay_image_url(query: str) -> str | None:
    if not PIXABAY_API_KEY:
        return None
    url = "https://pixabay.com/api/"
    params = {
        "key": PIXABAY_API_KEY,
        "q": query.strip(),
        "image_type": "photo",
        "orientation": "horizontal",
        "per_page": 3,
        "safesearch": "true"
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            hits = response.json().get("hits", [])
            if hits:
                return hits[0].get("webformatURL")
    except Exception:
        pass
    return None

def render_smart_content(content: str):
    # Importáljuk az eredeti markdown-t a rekurzió (lefagyás) elkerülésére
    from chat_renderer import _real_markdown
    
    pattern = r"\[IMAGE:\s*(.*?)\]"
    parts = re.split(pattern, content)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                _real_markdown(part, unsafe_allow_html=True)
        else:
            image_query = part.strip()
            if image_query:
                img_url = get_pixabay_image_url(image_query)
                if img_url:
                    st.image(img_url, caption=f"Kép: {image_query} (Pixabay)", use_container_width=True)