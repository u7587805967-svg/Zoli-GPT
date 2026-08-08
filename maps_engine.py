import json
import urllib.parse
import requests
import streamlit as st
import streamlit.components.v1 as components

def render_gps_navigation(dest_name="", dest_lat=None, dest_lng=None):
    """
    Dinamikus GPS térkép beágyazása:
    - Lekéri a böngészőből a felhasználó aktuális GPS koordinátáit.
    - Ráfókuszál a felhasználóra (kék pulzáló pont).
    - Ha meg van adva célállomás (dest_lat, dest_lng), kirajzolja az útvonalat.
    """
    
    dest_data_json = json.dumps({
        "name": dest_name,
        "lat": dest_lat,
        "lng": dest_lng
    })

    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        
        <link rel="stylesheet" href="https://unpkg.com/leaflet-routing-machine@latest/dist/leaflet-routing-machine.css" />
        <script src="https://unpkg.com/leaflet-routing-machine@latest/dist/leaflet-routing-machine.js"></script>

        <style>
            body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
            #map { height: 480px; width: 100%; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
            
            .gps-status {
                padding: 10px 14px;
                background-color: #f0f2f6;
                border-left: 4px solid #007bff;
                border-radius: 6px;
                margin-bottom: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }

            /* Pulzáló kék GPS jelölő a felhasználó pozíciójához */
            .user-gps-dot {
                width: 18px;
                height: 18px;
                background-color: #007bff;
                border: 3px solid #ffffff;
                border-radius: 50%;
                box-shadow: 0 0 10px rgba(0, 123, 255, 0.9);
                animation: pulse 1.6s infinite;
            }

            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(0, 123, 255, 0.7); }
                70% { box-shadow: 0 0 0 14px rgba(0, 123, 255, 0); }
                100% { box-shadow: 0 0 0 0 rgba(0, 123, 255, 0); }
            }
        </style>
    </head>
    <body>
        <div id="status" class="gps-status">📡 GPS kapcsolat keresése...</div>
        <div id="map"></div>

        <script>
            const destData = __DEST_DATA_JSON__;
            const statusDiv = document.getElementById('status');

            // Alapértelmezett térkép (Budapest központ fallback)
            const map = L.map('map').setView([47.4979, 19.0402], 13);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19,
                attribution: '© OpenStreetMap'
            }).addTo(map);

            //  Böngésző GPS Helymeghatározása
            if ("geolocation" in navigator) {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        const userLat = position.coords.latitude;
                        const userLng = position.coords.longitude;

                        statusDiv.innerHTML = "✅ GPS pozíció beérkezve! Ráállás a helyzetedre...";

                        // Kék GPS ikon létrehozása
                        const userIcon = L.divIcon({
                            className: 'user-gps-dot',
                            iconSize: [18, 18],
                            iconAnchor: [9, 9]
                        });

                        // Felhasználó megjelölése
                        L.marker([userLat, userLng], { icon: userIcon })
                         .addTo(map)
                         .bindPopup("<b> Az Ön jelenlegi pozíciója</b>")
                         .openPopup();

                        // GPS Fókusz a felhasználóra
                        map.setView([userLat, userLng], 15);

                        // 🏁 Ha van célállomás, útvonal kirajzolása
                        if (destData.lat && destData.lng) {
                            L.Routing.control({
                                waypoints: [
                                    L.latLng(userLat, userLng),
                                    L.latLng(destData.lat, destData.lng)
                                ],
                                router: L.Routing.osrmv1({
                                    serviceUrl: 'https://router.project-osrm.org/route/v1'
                                }),
                                routeWhileDragging: false,
                                show: true,
                                collapsible: true,
                                createMarker: function(i, wp, n) {
                                    if (i === 0) return null;
                                    return L.marker(wp.latLng).bindPopup("<b>🏁 Célállomás: " + (destData.name || "Cél") + "</b>");
                                }
                            }).addTo(map);

                            statusDiv.innerHTML = "🏁 Útvonal megtervezve a célállomáshoz: <b>" + (destData.name || "Cél") + "</b>";
                        }
                    },
                    (error) => {
                        console.error("GPS Hiba:", error);
                        statusDiv.innerHTML = "⚠️ Nem sikerült lekérni a GPS pozíciót. Kérjük engedélyezd a helymeghatározást a böngészőben!";
                    },
                    {
                        enableHighAccuracy: true,
                        timeout: 10000,
                        maximumAge: 0
                    }
                );
            } else {
                statusDiv.innerHTML = "❌ A böngésződ nem támogatja a GPS helymeghatározást.";
            }
        </script>
    </body>
    </html>
    """
    
    html_code = html_code.replace("__DEST_DATA_JSON__", dest_data_json)
    components.html(html_code, height=530)

class MapRoutingEngine:
    @staticmethod
    def geocode(location_name: str):
        """Helyszín név átalakítása GPS koordinátákká (Nominatim API)."""
        headers = {'User-Agent': 'ZoliGPT-MapApp/1.0'}
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(location_name)}&format=json&limit=1"
        try:
            resp = requests.get(url, headers=headers, timeout=8.0)
            data = resp.json()
            if data and len(data) > 0:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                display_name = data[0].get('display_name', location_name)
                return lat, lon, display_name
        except Exception:
            pass
        return None, None, None

    @staticmethod
    def get_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float, profile: str = "driving"):
        """Útvonal, távolság, menetidő és navigációs lépések lekérése (OSRM API)."""
        url = f"http://router.project-osrm.org/route/v1/{profile}/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson&steps=true"
        try:
            resp = requests.get(url, timeout=10.0)
            data = resp.json()
            if data.get('code') == 'Ok' and data.get('routes'):
                route = data['routes'][0]
                distance_km = round(route['distance'] / 1000.0, 1)
                duration_min = round(route['duration'] / 60.0)
                
                coords = route['geometry']['coordinates']
                polyline_coords = [[c[1], c[0]] for c in coords] # [lat, lon]
                
                steps = []
                legs = route.get('legs', [])
                for leg in legs:
                    for step in leg.get('steps', []):
                        name = step.get('name', '')
                        maneuver = step.get('maneuver', {}).get('type', '')
                        modifier = step.get('maneuver', {}).get('modifier', '')
                        dist = round(step.get('distance', 0))
                        
                        if dist > 0:
                            instr = f"{maneuver} {modifier}".strip().capitalize()
                            if name:
                                instr += f" -> {name}"
                            steps.append({"instruction": instr, "distance": dist})
                            
                return {
                    "distance_km": distance_km,
                    "duration_min": duration_min,
                    "polyline": polyline_coords,
                    "steps": steps
                }
        except Exception:
            pass
        return None

    @staticmethod
    def render_map_html(start_lat, start_lon, end_lat, end_lon, polyline_coords, start_name="Indulás", end_name="Cél"):
        """Interaktív Leaflet.js HTML térkép generálása."""
        coords_json = json.dumps(polyline_coords)
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                #map {{ height: 400px; width: 100%; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); border: 1px solid rgba(14, 165, 233, 0.3); }}
                body {{ margin: 0; padding: 0; background-color: transparent; }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map');
                
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© OpenStreetMap contributors'
                }}).addTo(map);

                var polylinePoints = {coords_json};
                var polyline = L.polyline(polylinePoints, {{color: '#0ea5e9', weight: 5, opacity: 0.85}}).addTo(map);

                var startIcon = L.icon({{
                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                    iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
                }});

                var endIcon = L.icon({{
                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                    iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
                }});

                L.marker([{start_lat}, {start_lon}], {{icon: startIcon}}).addTo(map).bindPopup("<b> Indulás:</b> {start_name}");
                L.marker([{end_lat}, {end_lon}], {{icon: endIcon}}).addTo(map).bindPopup("<b>🏁 Érkezés:</b> {end_name}");

                map.fitBounds(polyline.getBounds(), {{padding: [40, 40]}});
            </script>
        </body>
        </html>
        """

def show_route_widget(start_loc: str, end_loc: str):
    """Integrált útvonal kijelző Streamlit widget (külső linkek nélkül)."""
    s_lat, s_lon, s_name = MapRoutingEngine.geocode(start_loc)
    e_lat, e_lon, e_name = MapRoutingEngine.geocode(end_loc)
    
    if not s_lat or not e_lat:
        st.error(f"❌ Nem sikerült azonosítani a helyszíneket: '{start_loc}' vagy '{end_loc}'")
        return
        
    route = MapRoutingEngine.get_route(s_lat, s_lon, e_lat, e_lon)
    if not route:
        st.error("❌ Nem sikerült útvonalat tervezni a megadott pontok között.")
        return
        
    st.markdown(f"### 🗺️ Útvonal: **{s_name.split(',')[0]}** ➔ **{e_name.split(',')[0]}**")
    
    # Távolság és idő kijelzése
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📏 Távolság", f"{route['distance_km']} km")
    with col2:
        st.metric("⏱️ Várható idő", f"{route['duration_min']} perc")
        
    map_html = MapRoutingEngine.render_map_html(s_lat, s_lon, e_lat, e_lon, route['polyline'], s_name, e_name)
    st.components.v1.html(map_html, height=420)
    
    with st.expander(" Lépésről lépésre útbaigazítás", expanded=False):
        for idx, step in enumerate(route['steps'], 1):
            st.write(f"**{idx}.** {step['instruction']} *({step['distance']} m)*")
