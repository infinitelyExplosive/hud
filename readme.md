# Pokerstars HUD

Displays VPIP/PFR for players at a single table. 

## Installing
In Powershell:

1. `python -m venv .venv`
2. `Set-ExecutionPolicy Unrestricted -Scope Process`
3. `.venv\Scripts\activate`
4. `pip install -r .\requirements.txt`
5. Download the correct version of tesserocr from https://github.com/simonflueckiger/tesserocr-windows_build/releases
6. Download the correct version of tesseract from https://github.com/UB-Mannheim/tesseract/wiki
   - You may need to add the install location (e.g. `C:\Program Files\Tesseract-OCR`) to your PATH
7. `pip install .\tesserocr.whl`


## Running

The virtualenv must be activated every time Powershell is re-opened with:

1. `Set-ExecutionPolicy Unrestricted -Scope Process`
2. `.venv\Scripts\activate`

Then run

`python .\hud.py`

To quit, press `q`. To re-scan for new players, press `r`.

## Todo

* Better position recognition
* Fast re-scan
* Detect when players sit down at table
* Code comments