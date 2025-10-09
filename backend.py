import os
import requests
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import json
import math
import io
import numpy as np
import traceback

# Imports
from astropy.coordinates import get_icrs_coordinates, SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS
from PIL import Image
from reproject import reproject_interp

# --- Configuration ---
Image.MAX_IMAGE_PIXELS = None
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# API Constants & Key Loading
ASTROMETRY_API_URL = 'http://nova.astrometry.net/api'
try:
    from config import API_KEY
except ImportError:
    API_KEY = None

# Global State for pre-loading the skymap
session_key = None
input_data = None
input_wcs = None

# --- Core Service Functions ---
def load_skymap_data():
    """Pre-loads the single skymap image and computes its WCS at server startup."""
    global input_data, input_wcs
    
    composite_map_path = 'composite_skymap_16k.png'
    print(f"Attempting to pre-load skymap: {composite_map_path}")

    if not os.path.exists(composite_map_path):
        print(f"FATAL: Skymap file not found at '{composite_map_path}'. The skymap feature will not work.")
        return

    try:
        composite_image = Image.open(composite_map_path)
        # Separate the RGB channels for reprojection
        r, g, b = composite_image.split()
        input_data = (np.array(r), np.array(g), np.array(b))

        # WCS Definition for the single full-sky image
        w = WCS(naxis=2)
        w.wcs.crpix = [(composite_image.width + 1) / 2, (composite_image.height + 1) / 2]
        w.wcs.cdelt = np.array([-360. / composite_image.width, -180. / composite_image.height])
        w.wcs.crval = [0, 0]
        w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
        input_wcs = w
        
        print("Skymap data pre-loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR during skymap pre-load: {e}")
        input_data = None
        input_wcs = None


def get_apparent_coords(data):
    # Calculate apparent RA/Dec from observer location and time
    if 'ra' in data and 'dec' in data and data['ra'] is not None and data['dec'] is not None:
        return SkyCoord(ra=data['ra']*u.degree, dec=data['dec']*u.degree)

    location = EarthLocation(lon=data['longitude']*u.deg, lat=data['latitude']*u.deg)
    obstime = Time.now()
    base_coords = get_icrs_coordinates(data['target_name'])
    altaz_frame = AltAz(obstime=obstime, location=location)
    apparent_icrs = base_coords.transform_to(altaz_frame).transform_to('icrs')
    return apparent_icrs

# --- Static File Routes ---
@app.route('/')
def serve_index():
    # Serve main HTML file
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    # Serve other static files (CSS, etc.)
    return send_from_directory('.', path)


# --- API Endpoints ---
@app.route('/api/get_skymap_crop', methods=['POST'])
def get_skymap_crop():
    # Reproject a view of the sky using the pre-loaded single skymap
    if input_data is None or input_wcs is None:
        return jsonify({'error': 'Skymap data is not loaded on the server. Check server logs.'}), 500
        
    try:
        data = request.get_json()
        focal_length = data.get('focal_length')
        sensor_width_px = data.get('sensor_width_px')
        sensor_height_px = data.get('sensor_height_px')
        pixel_pitch = data.get('pixel_pitch')

        if not all([focal_length, sensor_width_px, sensor_height_px, pixel_pitch]):
            return jsonify({'error': 'Missing required camera parameters.'}), 400

        coords = get_apparent_coords(data)
        
        # FOV Calculation
        fov_width_deg = 2 * math.degrees(math.atan((pixel_pitch * sensor_width_px / 2000) / focal_length))
        fov_height_deg = 2 * math.degrees(math.atan((pixel_pitch * sensor_height_px / 2000) / focal_length))
        
        # Output WCS Definition
        output_wcs = WCS(naxis=2)
        output_wcs.wcs.crpix = [(sensor_width_px + 1) / 2, (sensor_height_px + 1) / 2]
        output_wcs.wcs.cdelt = np.array([-fov_width_deg / sensor_width_px, fov_height_deg / sensor_height_px])
        output_wcs.wcs.crval = [coords.ra.degree, coords.dec.degree]
        output_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Reprojection
        print(f"Reprojecting for target at RA {coords.ra.degree:.2f}, Dec {coords.dec.degree:.2f}...")
        output_shape = (sensor_height_px, sensor_width_px)
        
        # DEFINITIVE FIX: Reproject each color channel individually.
        # The input to reproject_interp must be a tuple of (data, wcs).
        r_data, _ = reproject_interp((input_data[0], input_wcs), output_wcs, shape_out=output_shape)
        g_data, _ = reproject_interp((input_data[1], input_wcs), output_wcs, shape_out=output_shape)
        b_data, _ = reproject_interp((input_data[2], input_wcs), output_wcs, shape_out=output_shape)
        
        # Stack the reprojected channels back into an RGB image.
        reprojected_data = np.stack([r_data, g_data, b_data], axis=-1)

        reprojected_data = np.nan_to_num(reprojected_data).astype(np.uint8)
        print("Reprojection complete.")

        # Prepare and Send Response
        output_image = Image.fromarray(reprojected_data)
        img_io = io.BytesIO()
        output_image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        print("--- AN ERROR OCCURRED DURING SKYMAP PROCESSING ---")
        traceback.print_exc()
        print("-------------------------------------------------")
        return jsonify({'error': 'An internal error occurred during image processing. Check server logs.'}), 500


# Other API routes...
@app.route('/api/get_declination', methods=['POST'])
def get_declination():
    # Fetches celestial coordinates for a target
    data = request.get_json()
    try:
        coords = get_apparent_coords(data)
        return jsonify({'declination': coords.dec.degree})
    except Exception as e:
        return jsonify({'error': f"Could not find coordinates. {e}"}), 404

def get_session_key():
    # Logs into Astrometry.net API
    global session_key
    if session_key: return session_key
    
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        raise Exception("Astrometry.net API key not configured in config.py.")

    response = requests.post(f'{ASTROMETRY_API_URL}/login', data={'request-json': json.dumps({'apikey': API_KEY})})
    response.raise_for_status()
    data = response.json()
    if data.get('status') == 'success':
        session_key = data['session']
        return session_key
    else:
        raise Exception(f"API login failed: {data.get('errormessage')}")

@app.route('/api/upload', methods=['POST'])
def upload_image():
    # Uploads an image file for analysis
    try:
        current_session_key = get_session_key()
        if 'image' not in request.files: return jsonify({'error': 'No image file in request.'}), 400
        
        file = request.files['image']
        upload_data = {'session': current_session_key, "publicly_visible": "n"}
        files = {'file': (file.filename, file.read(), file.mimetype)}
        
        response = requests.post(f'{ASTROMETRY_API_URL}/upload', data={'request-json': json.dumps(upload_data)}, files=files)
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'success': 
            return jsonify({'sub_id': data['subid']})
        else: 
            return jsonify({'error': data.get('errormessage', 'Upload failed')}), 500
    except Exception as e: 
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<int:sub_id>', methods=['GET'])
def get_status(sub_id):
    # Polls for the status of an analysis job
    try:
        response = requests.get(f'{ASTROMETRY_API_URL}/submissions/{sub_id}')
        response.raise_for_status()
        data = response.json()
        job_id, status, annotated_url = None, 'pending', None
        if data.get('jobs') and data['jobs'] and data['jobs'][0] is not None:
            job_id = data['jobs'][0]
            job_response = requests.get(f'{ASTROMETRY_API_URL}/jobs/{job_id}/info')
            job_response.raise_for_status()
            job_data = job_response.json()
            status = job_data.get('status', 'unknown')
            if status == 'success':
                annotated_url = f"http://nova.astrometry.net/annotated_display/{job_id}"
        return jsonify({'status': status, 'job_id': job_id, 'annotated_image_url': annotated_url})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/results/<int:job_id>', methods=['GET'])
def get_results(job_id):
    # Retrieves the final annotations for a successful job
    try:
        response = requests.get(f'{ASTROMETRY_API_URL}/jobs/{job_id}/annotations')
        response.raise_for_status()
        return jsonify({'annotations': response.json().get('annotations', [])})
    except Exception as e: return jsonify({'error': str(e)}), 500

# --- Server Startup ---
if __name__ == '__main__':
    print("Starting Flask server...")
    load_skymap_data()
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("\nWARNING: Astrometry.net API key not found.")
        print("Please create and fill in the 'config.py' file with your API key.\n")
    print("Navigate to http://127.0.0.1:5000")
    app.run(port=5000, debug=True)

