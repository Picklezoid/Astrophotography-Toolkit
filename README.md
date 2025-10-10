Astrophotography Toolkit (Work in Progress)-
This is a web-based toolkit for astrophotographers. It includes an NPF exposure calculator with a skymap preview and an image analyzer powered by Astrometry.net.

How It Works-
The application runs as a local web server on your machine. The backend is built with Python (Flask) and uses libraries like Astropy and Reproject to handle astronomical calculations. The frontend is simple HTML, CSS, and JavaScript.

The skymap preview feature works by intelligently loading and reprojecting sections from a high-resolution, tiled sky map to match your camera's field of view.

Local Setup
1. Skymap Tiles
~~Download the skymapsplit folder and composite_skymap_16k.png from "https://drive.google.com/drive/folders/1rrDNBE-_NzLOH3oTllvlRFs0W_YL9cPu?usp=sharing" and place it in the root folder. (Optional for sensor visual representation)~~
Download only the composite_skymap_16k.png from "https://drive.google.com/drive/folders/1rrDNBE-_NzLOH3oTllvlRFs0W_YL9cPu?usp=sharing" and place it in the root folder.(Optional for sensor visual representation)

2. API Key
The image analyzer requires a free API key from nova.astrometry.net. You must place this key in a config.py file.

3. Dependencies
With Python and Pip installed, run the following in your project folder:

pip install -r requirements.txt

4. Run Server
Finally, start the backend server:

python backend.py

You can then access the toolkit at http://127.0.0.1:5000 in your browser.
