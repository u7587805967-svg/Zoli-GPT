import streamlit as st
import asyncio
import time
import pytz
from datetime import datetime

# --- SAJÁT MODULOK IMPORTÁLÁSA ---
from config import AppConfig
from database import DatabaseRepository, hash_password
from ai_engine import AsyncAIEngine, get_embedding_model
from search_engine import hajzsalpontos_web_kereses, generald_a_hajszalpontos_valaszt
from maps_engine import render_gps_navigation, show_route_widget

# --- ALAPBEÁLLÍTÁSOK ---
st.set_page_config(
    page_title="Zoli GPT - AI Asszisztens",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konfiguráció és Adatbázis inicializálása
cfg = AppConfig()
db_repo = DatabaseRepository(cfg.DB_FILE)
ai_engine = AsyncAIEngine(db_repo=db_repo, config=cfg)

# --- MUNKamenet (SESSION STATE) INICIALIZÁLÁSA ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "current_thread" not in st.session_state:
    st.session_state.current_thread = "default"
if "generating" not in st.session_state:
    st.session_state.generating = False

# --- BEJELENTKEZÉSI LOGIKA ---
def login_screen():
    st.title("🔐 Bejelentkezés - Zoli GPT")
    tab1, tab2 = st.tabs(["Bejelentkezés", "Regisztráció"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Felhasználónév")
            password = st.text_input("Jelszó", type="password")
            submit = st.form_submit_button("Belépés")
            
            if submit:
                user_data = db_repo.get_user(username)
                if user_data:
                    stored_hash, salt = user_data
                    computed_hash, _ = hash_password(password, salt)
                    if computed_hash == stored_hash:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("Sikeres bejelentkezés!")
                        st.rerun()
                    else:
                        st.error("Hibás jelszó!")
                else:
                    st.error("Nem létezik ilyen felhasználó!")

    with tab2:
        with st.form("reg_form"):
            new_user = st.text_input("Új Felhasználónév")
            new_pass = st.text_input("Új Jelszó", type="password")
            reg_submit = st.form_submit_button("Regisztráció")
            
            if reg_submit:
                if new_user and new_pass:
                    if db_repo.get_user(new_user):
                        st.warning("Ez a felhasználónév már foglalt!")
                    else:
                        pwd_hash, salt = hash_password(new_pass)
                        db_repo.create_user(new_user, pwd_hash, salt)
                        st.success("Sikeres regisztráció! Most már bejelentkezhetsz.")
                else:
                    st.error("Minden mezőt tölts ki!")

# --- FŐ ALKALMAZÁS FELÜLET ---
def main_app():
    # Oldalsáv (Sidebar)
    with st.sidebar:
        st.title(f"👤 Üdv, {st.session_state.username}!")
        
        if st.button("🚪 Kijelentkezés"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()
            
        st.divider()
        st.subheader("⚙️ Beállítások")
        selected_model = st.selectbox("AI Modell", ai_engine.get_available_models())
        
        st.divider()
        st.subheader("🗺️ Gyors Térkép / Navigáció")
        start_loc = st.text_input("Kiindulópont", "Budapest")
        end_loc = st.text_input("Célállomás", "Miskolc")
        if st.button("Útvonal tervezése"):
            show_route_widget(start_loc, end_loc)

    # Fő chat felület
    st.title("🤖 Zoli GPT Asszisztens")
    
    # Aktuális chat szál (thread) kezelése
    thread_id = st.session_state.get("current_thread", "default")
    messages = db_repo.get_chat_history(st.session_state.username, thread_id=thread_id)

    # Korábbi üzenetek kirajzolása
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        msg_type = msg.get("type", "text")
        
        with st.chat_message(role):
            if msg_type == "image":
                st.image(content)
            else:
                st.markdown(content)

    # Felhasználói bemenet kezelése
    if prompt := st.chat_input("Írj ide egy üzenetet vagy kérdést..."):
        with st.chat_message("user"):
            st.markdown(prompt)
            
        db_repo.log_message(st.session_state.username, "user", prompt, "text", thread_id=thread_id)
        
        # AI válasz generálása
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            start_time = time.perf_counter()
            
            try:
                # Keresés ellenőrzése / Webes keresés integrálása ha szükséges
                web_kontextus = ""
                if "keress rá" in prompt.lower() or "mi a hír" in prompt.lower():
                    with st.status("🔍 Webes keresés folyamatban...", expanded=False):
                        web_kontextus = hajzsalpontos_web_kereses(prompt)
                
                # Válasz generálása a modellel
                full_response = generald_a_hajszalpontos_valaszt(
                    prompt=prompt,
                    web_kontextus=web_kontextus,
                    doc_kontextus="",
                    model=selected_model,
                    groq_api_key=ai_engine.groq_api_key
                )
                
                response_placeholder.markdown(full_response)
                
                # Naplózás
                end_time = time.perf_counter()
                db_repo.log_latency(end_time - start_time)
                db_repo.log_message(st.session_state.username, "assistant", full_response, "text", thread_id=thread_id)
                
            except Exception as e:
                st.error(f"Hiba történt a válasz generálása közben: {e}")

# --- FŐPROGRAM INDÍTÁSA ---
if not st.session_state.authenticated:
    login_screen()
else:
    main_app()