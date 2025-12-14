"""
macOS Screen Capture and OCR Tool
Captures screenshots at regular intervals and reads text from specific regions
"""

import threading
import time
import subprocess
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple
import pytesseract
import shutil
import sys

class ScreenReader:
    def __init__(self, interval=0.1, region=None, save_screenshots=True):
        """
        Initialize screen reader
        
        Args:
            interval: Capture interval in seconds (default: 1)
            region: Tuple (x, y, width, height) for specific screen region. None = full screen
            save_screenshots: Whether to save screenshot files (default: False)
        """
        self.interval = interval
        self.region = region  # (x, y, width, height)
        self.save_screenshots = save_screenshots
        self.running = False
        self.data = []
        self.lock = threading.Lock()
        self.screenshots_dir = Path("screenshots")
        
        if save_screenshots:
            self.screenshots_dir.mkdir(exist_ok=True)
    
    def capture_screenshot(self):
        """
        Capture a screenshot using macOS screencapture command
        Returns PIL Image object
        """
        temp_file = "/tmp/screen_capture.png"
        
        if self.region:
            x, y, width, height = self.region
            # screencapture -R x,y,width,height
            cmd = ["screencapture", "-R", f"{x},{y},{width},{height}", temp_file]
        else:
            cmd = ["screencapture", temp_file]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            image = Image.open(temp_file)
            return image
        except subprocess.CalledProcessError as e:
            print(f"Error capturing screenshot: {e}")
            return None
        except Exception as e:
            print(f"Error processing screenshot: {e}")
            return None

    def getVoouParaLonge(self, image):
        region3 = (1064, 450, 352, 53) # Voou para longe
        try:
            tess_path = shutil.which("tesseract")
            if tess_path:
                pytesseract.pytesseract.tesseract_cmd = tess_path
            else:
                raise EnvironmentError("tesseract is not installed or it's not in your PATH")
            
            x, y, w, h = region3
            cropped = image.crop((x, y, x + w, y + h))
            text3 = pytesseract.image_to_string(cropped).strip()
            if text3 == "VOOU PARA LONGE!":
                return True
        except Exception as e:
            print(f"Error extracting text: {e}. Install tesseract (macOS): `brew install tesseract`")
            return False        
        
    def capturePremioTotal(self, image):
        region = (600, 413, 140, 39)    # Total pago

        try:
            tess_path = shutil.which("tesseract")
            if tess_path:
                pytesseract.pytesseract.tesseract_cmd = tess_path
            else:
                raise EnvironmentError("tesseract is not installed or it's not in your PATH")

            x, y, w, h = region
            cropped = image.crop((x, y, x + w, y + h))
            text = pytesseract.image_to_string(cropped).strip()
            if text:
                 return text
            
        except Exception as e:
            print(f"Error extracting text: {e}. Install tesseract (macOS): `brew install tesseract`")
            return "" 
          
    def extract_text2(self, image):

        region = (745, 359, 857, 38)      # Multiplicadores
        
        try:
            tess_path = shutil.which("tesseract")
            if tess_path:
                pytesseract.pytesseract.tesseract_cmd = tess_path
            else:
                raise EnvironmentError("tesseract is not installed or it's not in your PATH")
            voou = self.getVoouParaLonge(image)
            ultimovalor = None
            if voou:
                print("Voou para longe detected")
                opa = self.capturePremioTotal(image)
                if opa:
                    print("Valor do premio total: ", opa)
                    ultimovalor = opa
            texts = []
            x, y, w, h = region
            cropped = image.crop((x, y, x + w, y + h))
            text = pytesseract.image_to_string(cropped).strip()
            
            return ultimovalor, text
        except Exception as e:
            print(f"Error extracting text: {e}. Install tesseract (macOS): `brew install tesseract`")
            return ""    

    def parse_multipliers(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse a string containing multiplier values like
        "1.10x 1.12x 1.14x ..." and return the first two numbers as floats.

        Returns a tuple (first, second) where each value is a float or None
        if that value is not present / couldn't be parsed.
        """
        import re

        if not text:
            return None, None

        # Prefer numbers followed by 'x' (case-insensitive)
        matches = re.findall(r"(\d+(?:\.\d+)?)(?=\s*[xX])", text)

        # Fallback: any numbers if none found with 'x'
        if not matches:
            matches = re.findall(r"(\d+(?:\.\d+)?)", text)

        floats = []
        for m in matches:
            try:
                floats.append(float(m))
            except Exception:
                continue
            if len(floats) >= 2:
                break

        first = floats[0] if len(floats) >= 1 else None
        second = floats[1] if len(floats) >= 2 else None
        return first, second
    
    def capture_loop(self):
        """Main capture loop running in background thread"""
        
        primeirovalorNovo= None
        segundoValorNovo = None
        
        primeirovalorAntigo = None
        segundoValorAntigo = None
        
        valorPremioAntigo = None
        ultimoValorPremio = None
        
        while self.running:
            try:
                # Capture screenshot
                image = self.capture_screenshot()
                if image is None:
                    time.sleep(self.interval)
                    continue
                
                # Extract text
                valorPremioAntigo = ultimoValorPremio
                ultimoValorPremio,text = self.extract_text2(image)
                
                primeirovalorAntigo = primeirovalorNovo
                segundoValorAntigo = segundoValorNovo
                
                primeirovalorNovo, segundoValorNovo = self.parse_multipliers(text)
                
                print("Texto extraido: ", text)
                                
                print("Ultimo valor capturado: ", ultimoValorPremio)

                print("segundoValorAntigo: ", primeirovalorAntigo)
                print("segundoValorAntigo: ", segundoValorAntigo)
                
                print("primeirovalorNovo: ", primeirovalorNovo)
                print("segundoValorNovo: ", segundoValorNovo)
                
                if primeirovalorNovo == primeirovalorAntigo and segundoValorNovo == segundoValorAntigo:
                    if valorPremioAntigo:
                        print("EITA: ",valorPremioAntigo, primeirovalorNovo, segundoValorNovo)
                        valorPremioAntigo = None
                if text:
                    # Create data entry
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "text": text,
                        "region": self.region
                    }
                
                    # Save screenshot if enabled
                    if self.save_screenshots:
                        filename = self.screenshots_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                    image.save(filename)
                    entry["screenshot"] = str(filename)
                    #por enquanto nao salva a imagem
                    # Append to data array
                    with self.lock:
                        self.data.append(entry)
                
                    #print(f"[{entry['timestamp']}] Captured: {text[:20]}...")
               # else:
                #    print("No text extracted.")
                
            except Exception as e:
                print(f"Error in capture loop: {e}")
            
            time.sleep(self.interval)
    
    def start(self):
        """Start capturing screenshots"""
        if self.running:
            print("Screen reader already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()
        print(f"Screen reader started (interval: {self.interval}s)")
    
    def stop(self):
        """Stop capturing screenshots"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        print("Screen reader stopped")
    
    def get_data(self):
        """Get all captured data"""
        with self.lock:
            return self.data.copy()
    
    def clear_data(self):
        """Clear captured data"""
        with self.lock:
            self.data.clear()
    
    def get_latest(self, count=1):
        """Get latest N entries"""
        with self.lock:
            return self.data[-count:] if self.data else []
    
    def export_data(self, filename="output.txt"):
        """Export captured data to file"""
        import json
        with self.lock:
            with open(filename, 'w') as f:
                json.dump(self.data, f, indent=2)
        print(f"Data exported to {filename}")


def main():
    """Example usage"""
    
    # Option 1: Capture full screen
    # reader = ScreenReader(interval=1.0, save_screenshots=True)
    
    # Option 2: Capture specific region (e.g., top-left 400x300 area starting at 0,0)
    # Get Google Chrome window bounds and capture only that area
    def get_chrome_window_bounds():
        """Get the bounds of the frontmost Google Chrome window"""
        script = '''tell application "Google Chrome"
            if (count of windows) = 0 then
                return ""
            end if
            set b to bounds of front window
            return b
        end tell'''
        try:
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                raise Exception(f"AppleScript failed: {result.stderr}")
            bounds_str = result.stdout.strip()
            if not bounds_str:
                raise Exception("No bounds returned")
            bounds = list(map(int, bounds_str.split(", ")))
            x, y, w, h = bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]
            return (x, y, w, h)
        except Exception as e:
            print(f"Error getting Chrome bounds: {e}")
            print("Using fallback region (0, 0, 800, 600)")
            return (0, 0, 800, 600)
    
    chrome_region = get_chrome_window_bounds()
    reader = ScreenReader(interval=1.0, region=chrome_region, save_screenshots=True)
    
    reader.start()
    
    try:
        # Let it run for 10 seconds
        time.sleep(3600)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        reader.stop()
        
        # Print results
        data = reader.get_data()
        print(f"\n\nCaptured {len(data)} entries:")
        for i, entry in enumerate(data, 1):
            print(f"\n{i}. [{entry['timestamp']}]")
            print(f"   Text: {entry['text'][:100]}...")
            if 'screenshot' in entry:
                print(f"   Screenshot: {entry['screenshot']}")
        
        # Export data
        reader.export_data("screen_data.json")


if __name__ == "__main__":
    main()
