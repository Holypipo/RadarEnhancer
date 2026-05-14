import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image

def array_to_base64_png(array, vmin=-32, vmax=75, cmap="viridis"):
    """
    Converts a 2D numpy array to a base64 encoded PNG image for Folium.
    """
    # Normalize array for colormap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # Get RGBA image
    rgba = sm.to_rgba(array, bytes=True)
    
    # Convert to PIL Image
    img = Image.fromarray(rgba)
    
    # Save to buffer
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    
    # Encode to base64
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

def get_radar_bounds(radar_obj, source="nexrad"):
    """
    Returns [[lat_min, lon_min], [lat_max, lon_max]]
    """
    if source == "nexrad":
        # Extract from metadata if possible, else use default + offset
        try:
            lat = float(radar_obj.latitude["data"][0])
            lon = float(radar_obj.longitude["data"][0])
            # NEXRAD range is ~230-460km. Let's say ~4 degrees
            return [[lat - 2.5, lon - 3.5], [lat + 2.5, lon + 3.5]]
        except:
            return [[30, -100], [40, -90]]
    elif source == "aemet":
        # AEMET National approx
        return [[34.0, -10.0], [44.5, 5.0]]
    else:
        # Default/Fallback
        return [[35.0, -5.0], [45.0, 5.0]]
