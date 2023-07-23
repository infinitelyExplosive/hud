# PokerStars HUD

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

In PokerStars, disable throwables. *Settings > Table Appearance > Animation > Throwables*. 

## Running

The virtualenv must be activated every time Powershell is re-opened with:

1. `Set-ExecutionPolicy Unrestricted -Scope Process`
2. `.venv\Scripts\activate`

Then run

`python .\hud.py`

**Do not resize the PokerStars window.** To quit, press `q`. To re-scan for new players, press `r`.

## Todo

**Known Bugs**:

* Players that fail to be detected can be re-created, losing a portion of their hands. 

**Important changes:**
* Detect when players sit down at table
* Better UI
* More stats

**Other changes:**
* Better position recognition
* Fast re-scan
* Code comments
* Refactor to make less horrible
* Support resizing windows
* Support multiple windows
* Support throwables