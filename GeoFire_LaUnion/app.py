import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MiniMap, Fullscreen, HeatMap, MeasureControl
from streamlit_folium import st_folium
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────────────────────
# Configuración general
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="GeoFire La Unión", page_icon="🔥", layout="wide")

DATA = Path("data")

# ─────────────────────────────────────────────────────────────
# Estilos y paletas
# ─────────────────────────────────────────────────────────────
COLORMAP_DEM = [
    (0.00, "#006400"),
    (0.15, "#228B22"),
    (0.30, "#9ACD32"),
    (0.45, "#DAA520"),
    (0.60, "#CD853F"),
    (0.75, "#8B4513"),
    (0.88, "#D2B48C"),
    (1.00, "#FFFAFA"),
]

ESTILOS_CAMINOS = {
    "motorway": {"color": "#d7191c", "weight": 5.5, "label": "Ruta principal / autopista"},
    "trunk": {"color": "#fdae61", "weight": 4.5, "label": "Ruta troncal"},
    "primary": {"color": "#f46d43", "weight": 4.0, "label": "Primaria"},
    "secondary": {"color": "#fee08b", "weight": 3.0, "label": "Secundaria"},
    "tertiary": {"color": "#ffffbf", "weight": 2.5, "label": "Terciaria"},
    "residential": {"color": "#bdbdbd", "weight": 1.4, "label": "Residencial/local"},
    "service": {"color": "#969696", "weight": 1.2, "label": "Servicio"},
    "track": {"color": "#8c6d31", "weight": 1.2, "label": "Huella/camino menor"},
    "path": {"color": "#8c6d31", "weight": 1.0, "label": "Sendero"},
    "default": {"color": "#cccccc", "weight": 1.3, "label": "Otros caminos"},
}

# ─────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────
def normalizar_columnas_fecha(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    for col in gdf.columns:
        if str(gdf[col].dtype).startswith("datetime"):
            gdf[col] = gdf[col].dt.strftime("%Y-%m-%d")
    return gdf


def cargar_capas():
    archivos_vec = list(DATA.glob("*.gpkg")) + list(DATA.glob("*.shp")) + list(DATA.glob("*.geojson"))
    capas = {}
    for archivo in archivos_vec:
        nombre = archivo.stem.replace("_", " ")
        try:
            gdf = gpd.read_file(archivo)
            if gdf.crs is not None:
                gdf = gdf.to_crs(4326)
            gdf = normalizar_columnas_fecha(gdf)
            capas[nombre] = gdf
        except Exception as e:
            st.warning(f"No fue posible cargar {archivo.name}: {e}")

    archivos_raster = list(DATA.glob("*.tif")) + list(DATA.glob("*.tiff")) + list(DATA.glob("*.img"))
    rasters = {archivo.stem.replace("_", " "): archivo for archivo in archivos_raster}
    return capas, rasters


def encontrar_capa(capas, palabras):
    for nombre, gdf in capas.items():
        n = nombre.lower()
        if any(p in n for p in palabras):
            return nombre, gdf
    return None, None


def obtener_columna(gdf, opciones):
    if gdf is None:
        return None
    cols = {c.lower(): c for c in gdf.columns}
    for op in opciones:
        if op.lower() in cols:
            return cols[op.lower()]
    return None


def preparar_incendios(gdf):
    if gdf is None or len(gdf) == 0:
        return gdf
    gdf = gdf.copy()
    fecha_col = obtener_columna(gdf, ["ACQ_DATE", "acq_date", "fecha"])
    if fecha_col:
        fechas = pd.to_datetime(gdf[fecha_col], errors="coerce", dayfirst=True)
        # Si dayfirst falla para muchos casos, reintenta con formato usual ISO/US.
        if fechas.isna().mean() > 0.5:
            fechas = pd.to_datetime(gdf[fecha_col], errors="coerce")
        gdf["fecha_fire"] = fechas
        gdf["anio_fire"] = fechas.dt.year
        gdf["fecha_texto"] = fechas.dt.strftime("%d-%m-%Y")
    else:
        gdf["fecha_fire"] = pd.NaT
        gdf["anio_fire"] = np.nan
        gdf["fecha_texto"] = "Sin fecha"
    return gdf


def calcular_longitud_km(gdf):
    if gdf is None or len(gdf) == 0:
        return 0.0
    try:
        return float(gdf.to_crs(32718).geometry.length.sum() / 1000)
    except Exception:
        return 0.0


def calcular_area_km2(gdf):
    if gdf is None or len(gdf) == 0:
        return 0.0
    try:
        return float(gdf.to_crs(32718).geometry.area.sum() / 1_000_000)
    except Exception:
        return 0.0


def agregar_leyenda(m):
    html = """
    <div style="position: fixed; bottom: 34px; left: 55px; z-index: 9999;
                background: rgba(255,255,255,0.88); padding: 12px 14px;
                border-radius: 8px; border: 1px solid #999; font-size: 12px;
                box-shadow: 1px 1px 5px rgba(0,0,0,0.35); color:#222;">
      <b>🗺️ Leyenda</b><br>
      <span style="color:#b30000;">●</span> Focos de calor<br>
      <span style="color:#d7191c;">━</span> Rutas principales<br>
      <span style="color:#fee08b;">━</span> Caminos secundarios / locales<br>
      <span style="color:#1e88e5;">━</span> Cursos de agua<br>
      <span style="color:#d4a017;">▱</span> Límite comunal
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def leyenda_dem_html(dem_min, dem_max):
    stops = ", ".join([f"{color} {int(pct*100)}%" for pct, color in COLORMAP_DEM])
    gradient = f"linear-gradient(to top, {stops})"
    return f"""
    <div style="position: fixed; top: 80px; right: 20px; z-index: 9999;
        background: rgba(255,255,255,0.92); padding: 10px 14px; border-radius: 8px;
        border: 1px solid #999; box-shadow: 1px 1px 5px rgba(0,0,0,0.35); color:#222;">
      <b>🏔️ Elevación (m)</b><br>
      <div style="display:flex; gap:8px; margin-top:6px;">
        <div style="width:22px; height:135px; background:{gradient}; border:1px solid #777;"></div>
        <div style="display:flex; flex-direction:column; justify-content:space-between; font-size:11px;">
          <span><b>{int(dem_max)} m</b></span>
          <span>{int((dem_min + dem_max)/2)} m</span>
          <span><b>{int(dem_min)} m</b></span>
        </div>
      </div>
    </div>
    """

# ─────────────────────────────────────────────────────────────
# Raster DEM
# ─────────────────────────────────────────────────────────────
def aplicar_colormap_dem(band, nodata):
    posiciones = [p for p, _ in COLORMAP_DEM]
    colores = [c for _, c in COLORMAP_DEM]
    cmap = mcolors.LinearSegmentedColormap.from_list("dem", list(zip(posiciones, colores)))

    if nodata is not None:
        mascara = (band == nodata) | np.isnan(band)
    else:
        mascara = np.isnan(band)

    valid = band[~mascara]
    dem_min = float(np.nanmin(valid)) if valid.size else 0
    dem_max = float(np.nanmax(valid)) if valid.size else 1

    norm = mcolors.Normalize(vmin=dem_min, vmax=dem_max)
    rgba = cmap(norm(band))
    rgba[mascara, 3] = 0
    rgba[~mascara, 3] = 0.72
    return (rgba * 255).astype(np.uint8), dem_min, dem_max


def raster_a_overlay(raster_path):
    with rasterio.open(raster_path) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            data = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
            bounds_wgs84 = rasterio.transform.array_bounds(height, width, transform)
        else:
            data = src.read(1).astype(np.float32)
            bounds_wgs84 = src.bounds

        img_array, dem_min, dem_max = aplicar_colormap_dem(data, src.nodata)
        img_pil = Image.fromarray(img_array)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
        return img_b64, bounds, dem_min, dem_max

# ─────────────────────────────────────────────────────────────
# Cargar datos
# ─────────────────────────────────────────────────────────────
capas, rasters = cargar_capas()

nombre_limite, gdf_limite = encontrar_capa(capas, ["limite"])
nombre_caminos, gdf_caminos = encontrar_capa(capas, ["camino", "road", "vial"])
nombre_rios, gdf_rios = encontrar_capa(capas, ["curso", "hidro", "rio", "rios"])
nombre_inc, gdf_inc = encontrar_capa(capas, ["incendio", "fire", "firms", "foco"])

# Respaldo: asegurar que el límite comunal sea una capa poligonal.
# Esto evita confundirlo con capas que también contienen "launion" en el nombre.
if gdf_limite is None or not any(gdf_limite.geometry.geom_type.str.contains("Polygon", case=False, na=False)):
    for nombre_tmp, gdf_tmp in capas.items():
        if any(gdf_tmp.geometry.geom_type.str.contains("Polygon", case=False, na=False)):
            if any(c.upper() in ["COMUNA", "SUPERFICIE", "CUT_COM"] for c in gdf_tmp.columns):
                nombre_limite, gdf_limite = nombre_tmp, gdf_tmp
                break

gdf_inc = preparar_incendios(gdf_inc)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
st.sidebar.title("Capas disponibles")
st.sidebar.write("Activa o desactiva las capas del geoportal.")

show_limite = st.sidebar.checkbox("🟧 Límite comunal", value=True)
show_caminos = st.sidebar.checkbox("🛣️ Caminos y carreteras", value=True)
show_rios = st.sidebar.checkbox("💧 Cursos de agua", value=True)
show_inc = st.sidebar.checkbox("🔥 Focos de calor", value=True)
show_dem = st.sidebar.checkbox("⛰️ DEM La Unión", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros de focos de calor")

inc_filtrado = gdf_inc.copy() if gdf_inc is not None else None
anios = []
if inc_filtrado is not None and "anio_fire" in inc_filtrado.columns:
    anios = sorted([int(a) for a in inc_filtrado["anio_fire"].dropna().unique()])

anio_opcion = "Todos"
if anios:
    anio_opcion = st.sidebar.selectbox("Año", ["Todos"] + anios)
    if anio_opcion != "Todos":
        inc_filtrado = inc_filtrado[inc_filtrado["anio_fire"] == int(anio_opcion)]

modo_incendios = st.sidebar.radio(
    "Visualización de focos",
    ["Puntos", "Mapa de calor", "Puntos + mapa de calor"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Versión 3: heatmap, popups, descarga y simbología mejorada.")
st.sidebar.markdown("---")
st.sidebar.write(f"Capas vectoriales detectadas: **{len(capas)}**")
st.sidebar.write(f"Rasters detectados: **{len(rasters)}**")

# ─────────────────────────────────────────────────────────────
# Encabezado
# ─────────────────────────────────────────────────────────────
st.title("🔥 GeoFire La Unión")
st.markdown("""
**GeoVisualizador Web de focos de calor y factores territoriales asociados al riesgo de incendios forestales**  
Comuna de **La Unión**, Región de Los Ríos.

Esta aplicación integra focos de calor NASA FIRMS, red vial, cursos de agua, límite comunal y un Modelo Digital de Elevación. Además, incorpora estadísticas, filtros, gráficos, mapa de calor y descarga de datos.
            
Los datos disponibles para observar en el GeoVisualizador, tienen fecha desde el 01/01/2020 hasta el 01/01/2026.
            
**Brightness**: representa la intensidad térmica detectada por el satélite.
**FRP** (Fire Radiative Power): mide cuánta energía está liberando el fuego.

**¿Por qué se eligieron estos indicadores?**
            
🌡️Brightness promedio responde a: ¿Qué tan calientes fueron, en promedio, los focos detectados?

⚡FRP máximo responde a: ¿Cuál fue el evento de mayor intensidad energética registrado?""")


# ─────────────────────────────────────────────────────────────
# Estadísticas
# ─────────────────────────────────────────────────────────────
num_inc = len(inc_filtrado) if inc_filtrado is not None else 0
km_caminos = calcular_longitud_km(gdf_caminos)
km_rios = calcular_longitud_km(gdf_rios)
area_comunal = calcular_area_km2(gdf_limite)

brightness_col = obtener_columna(gdf_inc, ["BRIGHTNESS", "brightness"])
frp_col = obtener_columna(gdf_inc, ["FRP", "frp"])

brightness_prom = None
frp_max = None
if inc_filtrado is not None and len(inc_filtrado) > 0:
    if brightness_col in inc_filtrado.columns:
        brightness_prom = pd.to_numeric(inc_filtrado[brightness_col], errors="coerce").mean()
    if frp_col in inc_filtrado.columns:
        frp_max = pd.to_numeric(inc_filtrado[frp_col], errors="coerce").max()

st.header("📊 Resumen territorial")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔥 Focos de calor", f"{num_inc:,}".replace(",", "."))
c2.metric("🛣️ Red vial", f"{km_caminos:,.1f} km".replace(",", "X").replace(".", ",").replace("X", "."))
c3.metric("💧 Cursos de agua", f"{km_rios:,.1f} km".replace(",", "X").replace(".", ",").replace("X", "."))
c4.metric("🟧 Área comunal", f"{area_comunal:,.1f} km²".replace(",", "X").replace(".", ",").replace("X", "."))

b1, b2 = st.columns(2)
b1.info(f"🌡️ Brightness promedio: {brightness_prom:.1f} K" if brightness_prom == brightness_prom else "🌡️ Brightness promedio: sin datos")
b2.info(f"⚡ FRP máximo: {frp_max:.1f} MW" if frp_max == frp_max else "⚡ FRP máximo: sin datos")

# ─────────────────────────────────────────────────────────────
# Mapa
# ─────────────────────────────────────────────────────────────
centro = [-40.30, -72.85]
if gdf_limite is not None and len(gdf_limite) > 0:
    c = gdf_limite.unary_union.centroid
    centro = [c.y, c.x]

m = folium.Map(location=centro, zoom_start=10, tiles="OpenStreetMap")
folium.TileLayer("CartoDB positron", name="Mapa claro").add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Mapa oscuro").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satélite Esri",
).add_to(m)

# DEM
if show_dem and rasters:
    raster_nombre, raster_path = next(iter(rasters.items()))
    try:
        img_b64, bounds, dem_min, dem_max = raster_a_overlay(raster_path)
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=bounds,
            opacity=0.75,
            name="⛰️ Modelo Digital de Elevación",
        ).add_to(m)
        m.get_root().html.add_child(folium.Element(leyenda_dem_html(dem_min, dem_max)))
    except Exception as e:
        st.warning(f"No fue posible cargar el DEM: {e}")

# Límite
if show_limite and gdf_limite is not None:
    folium.GeoJson(
        gdf_limite,
        name="🟧 Límite comunal",
        style_function=lambda f: {"fillColor": "#d4a017", "color": "#d4a017", "weight": 3, "fillOpacity": 0.08},
        tooltip=folium.GeoJsonTooltip(fields=[c for c in ["COMUNA", "PROVINCIA", "REGION", "SUPERFICIE"] if c in gdf_limite.columns]),
    ).add_to(m)

# Caminos
if show_caminos and gdf_caminos is not None:
    highway_col = obtener_columna(gdf_caminos, ["highway", "tipo", "TIPO"])
    name_col = obtener_columna(gdf_caminos, ["name", "Nombre", "NOMBRE"])
    surface_col = obtener_columna(gdf_caminos, ["surface", "superficie"])

    def style_camino(feature):
        tipo = str(feature["properties"].get(highway_col, "default")).lower() if highway_col else "default"
        estilo = ESTILOS_CAMINOS.get(tipo, ESTILOS_CAMINOS["default"])
        return {"color": estilo["color"], "weight": estilo["weight"], "opacity": 0.9}

    fields = [c for c in [highway_col, name_col, surface_col, "ref"] if c and c in gdf_caminos.columns]
    aliases = []
    for f in fields:
        aliases.append({highway_col: "Tipo", name_col: "Nombre", surface_col: "Superficie", "ref": "Referencia"}.get(f, f))

    folium.GeoJson(
        gdf_caminos,
        name="🛣️ Caminos",
        style_function=style_camino,
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases) if fields else None,
    ).add_to(m)

# Ríos
if show_rios and gdf_rios is not None:
    nombre_rio = obtener_columna(gdf_rios, ["nombre_bcn", "Nombre", "NOMBRE", "nom_cuen"])
    tipo_rio = obtener_columna(gdf_rios, ["tipo", "TIPO", "clase"])
    fields = [c for c in [nombre_rio, tipo_rio, "stralher_n", "nom_cuen"] if c and c in gdf_rios.columns]
    folium.GeoJson(
        gdf_rios,
        name="💧 Cursos de agua",
        style_function=lambda f: {"color": "#1e88e5", "weight": 2.1, "opacity": 0.9},
        tooltip=folium.GeoJsonTooltip(fields=fields) if fields else None,
    ).add_to(m)

# Incendios: puntos y/o heatmap
if show_inc and inc_filtrado is not None and len(inc_filtrado) > 0:
    lat_col = obtener_columna(inc_filtrado, ["LATITUDE", "latitude", "lat"])
    lon_col = obtener_columna(inc_filtrado, ["LONGITUDE", "longitude", "lon", "long"])
    sat_col = obtener_columna(inc_filtrado, ["SATELLITE", "satellite"])
    time_col = obtener_columna(inc_filtrado, ["ACQ_TIME", "acq_time"])
    daynight_col = obtener_columna(inc_filtrado, ["DAYNIGHT", "daynight"])

    if lat_col and lon_col and modo_incendios in ["Mapa de calor", "Puntos + mapa de calor"]:
        heat_data = []
        for _, row in inc_filtrado.iterrows():
            lat = pd.to_numeric(row.get(lat_col), errors="coerce")
            lon = pd.to_numeric(row.get(lon_col), errors="coerce")
            peso = pd.to_numeric(row.get(frp_col), errors="coerce") if frp_col else 1
            if pd.notna(lat) and pd.notna(lon):
                heat_data.append([float(lat), float(lon), float(peso) if pd.notna(peso) else 1])
        if heat_data:
            HeatMap(heat_data, name="🔥 Mapa de calor", radius=18, blur=22, min_opacity=0.35).add_to(m)

    if modo_incendios in ["Puntos", "Puntos + mapa de calor"]:
        grupo = folium.FeatureGroup(name="🔥 Focos de calor NASA FIRMS")
        for _, row in inc_filtrado.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            lat, lon = geom.y, geom.x
            fecha = row.get("fecha_texto", "Sin fecha")
            brightness = row.get(brightness_col, "Sin dato") if brightness_col else "Sin dato"
            frp = row.get(frp_col, "Sin dato") if frp_col else "Sin dato"
            sat = row.get(sat_col, "Sin dato") if sat_col else "Sin dato"
            hora = row.get(time_col, "Sin dato") if time_col else "Sin dato"
            dn = row.get(daynight_col, "Sin dato") if daynight_col else "Sin dato"
            popup_html = f"""
            <div style='font-family:Arial; font-size:13px; min-width:220px;'>
              <b style='font-size:15px;'>🔥 Foco de calor</b><br><hr style='margin:5px 0;'>
              <b>📅 Fecha:</b> {fecha}<br>
              <b>🕒 Hora:</b> {hora}<br>
              <b>🌡️ Brightness:</b> {brightness} K<br>
              <b>⚡ FRP:</b> {frp} MW<br>
              <b>🛰️ Satélite:</b> {sat}<br>
              <b>🌗 Día/Noche:</b> {dn}<br>
              <b>📍 Coordenadas:</b> {lat:.5f}, {lon:.5f}
            </div>
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="#7f0000",
                weight=1,
                fill=True,
                fill_color="#d7301f",
                fill_opacity=0.72,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"🔥 {fecha}",
            ).add_to(grupo)
        grupo.add_to(m)

MiniMap(toggle_display=True, position="bottomright").add_to(m)
Fullscreen(position="topleft").add_to(m)
MeasureControl(position="topleft", primary_length_unit="kilometers", primary_area_unit="sqmeters").add_to(m)
agregar_leyenda(m)
folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=1200, height=700)

# ─────────────────────────────────────────────────────────────
# Gráfico y tabla
# ─────────────────────────────────────────────────────────────
st.header("📈 Análisis de focos de calor")

if gdf_inc is not None and len(gdf_inc) > 0 and "anio_fire" in gdf_inc.columns:
    conteo = gdf_inc.dropna(subset=["anio_fire"]).copy()
    conteo["anio_fire"] = conteo["anio_fire"].astype(int)
    serie = conteo.groupby("anio_fire").size().sort_index()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(serie.index.astype(str), serie.values)
    ax.set_title("Focos de calor por año")
    ax.set_xlabel("Año")
    ax.set_ylabel("Cantidad de focos")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)
else:
    st.info("No se encontraron fechas suficientes para construir el gráfico por año.")

st.subheader("📋 Tabla de focos de calor filtrados")
if inc_filtrado is not None and len(inc_filtrado) > 0:
    columnas_preferidas = [
        "fecha_texto", "anio_fire", "LATITUDE", "LONGITUDE", "BRIGHTNESS", "FRP",
        "SATELLITE", "ACQ_TIME", "DAYNIGHT", "CONFIDENCE"
    ]
    cols = [c for c in columnas_preferidas if c in inc_filtrado.columns]
    tabla = inc_filtrado[cols].copy() if cols else inc_filtrado.drop(columns="geometry").copy()
    st.dataframe(tabla, use_container_width=True, height=280)

    csv = tabla.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar focos filtrados como CSV",
        data=csv,
        file_name="focos_calor_la_union_filtrados.csv",
        mime="text/csv",
    )
else:
    st.info("No hay focos de calor para el filtro seleccionado.")

with st.expander("ℹ️ Información de las capas cargadas"):
    st.write("Capas vectoriales detectadas:", list(capas.keys()))
    st.write("Rasters detectados:", list(rasters.keys()))


st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#8a8f98; font-size:13px; padding:10px 0 20px 0;'>
        🔥 <b>GeoFire La Unión</b> · Aplicaciones SIG – Creado por Mauro Leal · 2026<br>
        GeoVisualizador desarrollado con Python, Streamlit, GeoPandas, Folium y datos geoespaciales oficiales / NASA FIRMS.
    </div>
    """,
    unsafe_allow_html=True,
)
