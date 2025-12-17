"""
Avaliação em tempo real usando Tesseract e modelo treinado
Captura screenshots, extrai texto e avalia probabilidade em tempo real
"""

import threading
import time
import subprocess
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
import pytesseract
import shutil
import sys
import numpy as np
from tensor import NextMultiplierModel, parse_totalvalue  # Importar o modelo e a função de parsing

class RealTimeEvaluator:
    def __init__(self, interval=1.0, region=None, model_path="next_multiplier_model.keras"):
        """
        Initialize real-time evaluator
        
        Args:
            interval: Capture interval in seconds (default: 1)
            region: Tuple (x, y, width, height) for specific screen region. None = full screen
            model_path: Path to the trained model file
        """
        self.interval = interval
        self.region = region  # (x, y, width, height)
        self.running = False
        
        # Load the trained model
        self.model = NextMultiplierModel()
        try:
            self.model.load(model_path)
            print(f"Modelo carregado de {model_path}")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            sys.exit(1)
    
    def capture_screenshot(self):
        """
        Capture a screenshot using macOS screencapture command
        Returns PIL Image object
        """
        temp_file = "/tmp/evaluate.png"
        
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
        region3 = (1110, 452, 346, 49) # Voou para longe
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
        region = (645, 413, 145, 33)    # Total pago

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

        region = (745, 359, 245, 38)      # Multiplicadores
        
        try:
            tess_path = shutil.which("tesseract")
            if tess_path:
                pytesseract.pytesseract.tesseract_cmd = tess_path
            else:
                raise EnvironmentError("tesseract is not installed or it's not in your PATH")
            voou = self.getVoouParaLonge(image)
            ultimovalor = None
            if voou:
                #print("Voou para longe detected")
                opa = self.capturePremioTotal(image)
                if opa:
                    #print("Valor do premio total: ", opa)
                    ultimovalor = opa
            texts = []
            x, y, w, h = region
            cropped = image.crop((x, y, x + w, y + h))
            text = pytesseract.image_to_string(cropped).strip()
            
            return ultimovalor, text
        except Exception as e:
            print(f"Error extracting text: {e}. Install tesseract (macOS): `brew install tesseract`")
            return ""    

    def parse_multipliers(self, text: str):
        """
        Parse a string containing multiplier values like
        "1.10x 1.12x 1.14x ..." and return the first two numbers as floats.
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
    
    def evaluate_loop(self):
        """Main evaluation loop running in background thread"""
        
        primeirovalorNovo = None
        segundoValorNovo = None
        
        primeirovalorAntigo = None
        segundoValorAntigo = None
        
        valorPremioValido = None
        ultimoValorPremio = None
        
        while self.running:
            try:
                # Capture screenshot
                image = self.capture_screenshot()
                if image is None:
                    time.sleep(self.interval)
                    continue
                
                # Extract text
                ultimoValorPremio, text = self.extract_text2(image)
                #print("Texto extraido: ", text, "timestamp: ", datetime.now().isoformat())
                #print("Ultimo valor capturado: ", ultimoValorPremio)

                if ultimoValorPremio:
                    valorPremioValido = ultimoValorPremio
            
                primeirovalorAntigo = primeirovalorNovo
                segundoValorAntigo = segundoValorNovo
                
                primeirovalorNovo, segundoValorNovo = self.parse_multipliers(text)
                
                #print("primeirovalorAntigo: ", primeirovalorAntigo)
                #print("segundoValorAntigo: ", segundoValorAntigo)
                
                #print("primeirovalorNovo: ", primeirovalorNovo)
                #print("segundoValorNovo: ", segundoValorNovo)
                
                if primeirovalorNovo == primeirovalorAntigo and segundoValorNovo == segundoValorAntigo:
                    if valorPremioValido and not ultimoValorPremio:
                        #print("EITA: ", valorPremioValido, primeirovalorNovo, segundoValorNovo)
                        print("Avaliando com: ", primeirovalorNovo, valorPremioValido)
                        # Aqui, ao invés de armazenar, avaliar a probabilidade
                        try:
                            # Usar a mesma função de parsing do tensor.py
                            totalvalue_parsed = parse_totalvalue(valorPremioValido)
                            print(f"Valores para predição: multiplier={primeirovalorNovo}, totalvalue_parsed={totalvalue_parsed}")
                            if not np.isnan(totalvalue_parsed) and primeirovalorNovo is not None:
                                prob = self.model.predict_prob(primeirovalorNovo, totalvalue_parsed)
                                #print("prob: ", prob)
                                print(f"Probabilidade do próximo multiplicador > 2x: {prob:.4f}")
                                
                                # Avaliar se probabilidade > 2.0 (nota: probabilidades são <=1, talvez você queira >0.5?)
                                if prob > 0.6:  # Isso nunca será verdadeiro, pois prob <=1
                                    print("ALERTA: Probabilidade alta detectada!")
                            else:
                                print("Valores inválidos, pulando avaliação")
                        except Exception as e:
                            print(f"Erro na predição: {e}")
                        
                        valorPremioValido = None
                
            except Exception as e:
                print(f"Error in evaluation loop: {e}")
            
            time.sleep(self.interval)
    
    def start(self):
        """Start evaluating"""
        if self.running:
            print("Evaluator already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.evaluate_loop, daemon=True)
        self.thread.start()
        print(f"Real-time evaluator started (interval: {self.interval}s)")
    
    def stop(self):
        """Stop evaluating"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        print("Real-time evaluator stopped")


def main():
    """Example usage"""
    
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
    evaluator = RealTimeEvaluator(interval=0.1, region=chrome_region, model_path="next_multiplier_model.keras")
    
    evaluator.start()
    
    try:
        # Let it run for 10 seconds
        time.sleep(3600)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        evaluator.stop()


if __name__ == "__main__":
    main()
