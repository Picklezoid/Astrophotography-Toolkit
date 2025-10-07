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

# Global State
session_key = None

# --- Static File Routes ---
@app.route('/')
def serve_index():
    # Serve main HTML file
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    # Serve other static files (CSS, etc.)
    return send_from_directory('.', path)

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

# --- API Endpoints ---
@app.route('/api/get_skymap_crop', methods=['POST'])
def get_skymap_crop():
    # Reproject a view of the sky using a tiled skymap system
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
        
        # FOV Corner Identification
        half_fov_w = fov_width_deg / 2
        half_fov_h = fov_height_deg / 2
        center_ra, center_dec = coords.ra.degree, coords.dec.degree

        corners = [
            (center_ra - half_fov_w, center_dec + half_fov_h), (center_ra + half_fov_w, center_dec + half_fov_h),
            (center_ra - half_fov_w, center_dec - half_fov_h), (center_ra + half_fov_w, center_dec - half_fov_h),
        ]

        # Required Tile Calculation
        needed_tiles = set()
        for ra, dec in corners:
            ra = ra % 360
            dec = max(-90, min(90, dec))
            col = math.floor(ra / 45)
            row = 0 if dec >= 45 else 1 if dec >= 0 else 2 if dec >= -45 else 3
            needed_tiles.add((row, col))

        if not needed_tiles:
             return jsonify({'error': 'Could not determine required tiles.'}), 500

        # In-Memory Tile Stitching
        min_row, max_row = min(r for r,c in needed_tiles), max(r for r,c in needed_tiles)
        min_col, max_col = min(c for r,c in needed_tiles), max(c for r,c in needed_tiles)
        
        rows, cols = (max_row - min_row + 1), (max_col - min_col + 1)
        tile_width, tile_height = 2048, 2048
        composite_image = Image.new('RGB', (cols * tile_width, rows * tile_height))

        for r_idx, r in enumerate(range(min_row, max_row + 1)):
            for c_idx, c in enumerate(range(min_col, max_col + 1)):
                tile_path = os.path.join('skymapsplit', f'skymap_tile_{r}_{c}.jpg')
                if (r,c) in needed_tiles and os.path.exists(tile_path):
                    tile_img = Image.open(tile_path)
                    composite_image.paste(tile_img, (c_idx * tile_width, r_idx * tile_height))
        
        input_data = np.array(composite_image)

        # WCS Definition for Stitched Composite
        stitched_wcs = WCS(naxis=2)
        stitched_wcs.wcs.crpix = [((cols * tile_width) + 1) / 2, ((rows * tile_height) + 1) / 2]
        stitched_center_ra = (min_col * 45) + (cols * 45 / 2)
        dec_centers = [67.5, 22.5, -22.5, -67.5]
        stitched_center_dec = sum(dec_centers[r] for r in range(min_row, max_row + 1)) / rows
        stitched_wcs.wcs.crval = [stitched_center_ra, stitched_center_dec]
        stitched_wcs.wcs.cdelt = np.array([-45. / tile_width, 45. / tile_height])
        stitched_wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
        
        # Output WCS Definition
        output_wcs = WCS(naxis=2)
        output_wcs.wcs.crpix = [(sensor_width_px + 1) / 2, (sensor_height_px + 1) / 2]
        output_wcs.wcs.cdelt = np.array([-fov_width_deg / sensor_width_px, fov_height_deg / sensor_height_px])
        output_wcs.wcs.crval = [coords.ra.degree, coords.dec.degree]
        output_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Reprojection
        print(f"Reprojecting for target at RA {coords.ra.degree:.2f}, Dec {coords.dec.degree:.2f}...")
        output_shape = (sensor_height_px, sensor_width_px)
        reprojected_data_r, _ = reproject_interp((input_data[:,:,0], stitched_wcs), output_wcs, shape_out=output_shape)
        reprojected_data_g, _ = reproject_interp((input_data[:,:,1], stitched_wcs), output_wcs, shape_out=output_shape)
        reprojected_data_b, _ = reproject_interp((input_data[:,:,2], stitched_wcs), output_wcs, shape_out=output_shape)

        reprojected_data = np.stack([reprojected_data_r, reprojected_data_g, reprojected_data_b], axis=-1)
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
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("\nWARNING: Astrometry.net API key not found.")
        print("Please create and fill in the 'config.py' file with your API key.\n")
    print("Navigate to http://127.0.0.1:5000")
    app.run(port=5000, debug=True)

