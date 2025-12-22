
# -*- coding: utf-8 -*-
"""
Avaliação em tempo real usando Tesseract e modelo treinado
Captura screenshots, extrai texto e avalia probabilidade em tempo real
"""
from __future__ import annotations
import threading
import time
import subprocess
import os
from datetime import datetime
import warnings
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import shutil
import sys
import numpy as np
import re
from collections import deque

# Importar o modelo e a função de parsing robusta do seu módulo 'tensor.py'
from tensor import NextMultiplierModel, parse_totalvalue

# Suprimir warning do urllib3 sobre SSL
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")


class RealTimeEvaluator:
    def __init__(self, interval=0.5, region=None, model_path="next_multiplier_model.keras", lang='por'):
        """
        Args:
            interval: intervalo de captura em segundos
            region: (x, y, w, h) da região a capturar. None = tela inteira
            model_path: caminho do modelo treinado (.keras)
            lang: idioma do Tesseract (ex.: 'por')
        """
        self.interval = float(interval)
        self.region = region  # (x, y, w, h)
        self.running = False
        self.lang = lang

        # Históricos para médias móveis (incluem o valor atual quando disponível)
        self.mult_hist = deque(maxlen=50)
        self.total_hist = deque(maxlen=50)

        # Init Tesseract uma vez
        tess_path = shutil.which("tesseract")
        if not tess_path:
            raise EnvironmentError("tesseract não encontrado. Instale com: brew install tesseract")
        pytesseract.pytesseract.tesseract_cmd = tess_path

        # Carrega o modelo treinado (inclui scaler, threshold e outlier_threshold dos metadados)
        self.model = NextMultiplierModel()
        try:
            self.model.load(model_path)
            print(f"Modelo carregado de {model_path}")
            print(f"Threshold de decisão: {getattr(self.model, 'decision_threshold', 0.5)} | outlier_threshold: {getattr(self.model, 'outlier_threshold', 100.0)}")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            sys.exit(1)

        # Regiões RELATIVAS à captura atual (0,0 = topo-esquerdo da imagem capturada)
        # ⚠️ Ajuste estas coordenadas ao seu layout da janela capturada!
        self.REL_REGIONS = {
            "voou": (1110, 452, 346, 49),          # Texto "VOOU PARA LONGE!"
            "premio_total": (645, 413, 150, 35),   # Campo "Total pago"
            "multiplicadores": (745, 359, 245, 38) # Faixa com multiplicadores (ex.: "1.10x 1.12x ...")
        }

    # ---------- Utilidades de OCR ----------
    def _preprocess(self, img: Image.Image, for_digits=False) -> Image.Image:
        """Upsample, autocontraste e threshold leve para ajudar o OCR."""
        out = img.convert("L")
        out = out.resize((out.width * 2, out.height * 2), resample=Image.LANCZOS)
        out = ImageOps.autocontrast(out)
        out = out.filter(ImageFilter.SHARPEN)
        # threshold simples
        out = out.point(lambda p: 255 if p > 180 else 0)
        return out

    def _ocr(self, img: Image.Image, config: str) -> str:
        try:
            return pytesseract.image_to_string(img, config=config, lang=self.lang).strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    def _crop(self, image: Image.Image, box):
        x, y, w, h = box
        return image.crop((x, y, x + w, y + h))

    def capture_screenshot(self) -> Image.Image | None:
        """
        Captura screenshot usando 'screencapture' (macOS) e retorna PIL Image.
        """
        temp_file = "/tmp/evaluate.png"
        if self.region:
            x, y, w, h = self.region
            cmd = ["screencapture", "-x", "-t", "png", "-R", f"{x},{y},{w},{h}", temp_file]
        else:
            cmd = ["screencapture", "-x", "-t", "png", temp_file]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            image = Image.open(temp_file).convert("RGB")
            image.load()  # carrega para memória
            try:
                os.remove(temp_file)
            except Exception:
                pass
            return image
        except subprocess.CalledProcessError as e:
            print(f"Erro ao capturar screenshot: {e}")
            return None
        except Exception as e:
            print(f"Erro processando screenshot: {e}")
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
            return None

    def get_voou_para_longe(self, image: Image.Image) -> bool:
        box = self.REL_REGIONS["voou"]
        cropped = self._crop(image, box)
        proc = self._preprocess(cropped, for_digits=False)
        # psm 7 para linha única de texto curto
        text = self._ocr(proc, config="--psm 7 --oem 1")
        norm = re.sub(r"[^A-Z0-9]", "", text.upper())
        return "VOOUPARALONGE" in norm  # tolera pequenas variações

    def capture_premio_total(self, image: Image.Image) -> str | None:
        box = self.REL_REGIONS["premio_total"]
        cropped = self._crop(image, box)
        proc = self._preprocess(cropped, for_digits=True)
        # restringe caracteres a dígitos, ponto e vírgula
        text = self._ocr(proc, config="--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789.,")
        return text if text else None

    def extract_multipliers_text(self, image: Image.Image) -> str:
        box = self.REL_REGIONS["multiplicadores"]
        cropped = self._crop(image, box)
        proc = self._preprocess(cropped, for_digits=True)
        text = self._ocr(proc, config="--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789.xX")
        return text

    def parse_multipliers(self, text: str):
        """Extrai até dois multiplicadores (float) de um texto (preferência por número seguido de 'x')."""
        
        #print(f"Texto bruto de multiplicadores: '{text}'")
        if not text:
            return None, None
        matches = re.findall(r"(\d+(?:\.\d+)?)(?=\s*[xX])", text)
        if not matches:
            matches = re.findall(r"(\d+(?:\.\d+)?)", text)
        floats = []
        for m in matches:
            try:
                floats.append(float(m))
            except:
                continue
            if len(floats) >= 2:
                break
        first = floats[0] if len(floats) >= 1 else None
        second = floats[1] if len(floats) >= 2 else None
        #print(f"Multiplicadores extraídos: {first}, {second}")
        return first, second

    def getVoouParaLonge(self, image):
        #region3 = (1110, 452, 346, 49) # Voou para longe
        region = self.REL_REGIONS["voou"]
        try:
            tess_path = shutil.which("tesseract")
            if tess_path:
                pytesseract.pytesseract.tesseract_cmd = tess_path
            else:
                raise EnvironmentError("tesseract is not installed or it's not in your PATH")
            
            x, y, w, h = region
            cropped = image.crop((x, y, x + w, y + h))
            text3 = pytesseract.image_to_string(cropped).strip()
            if text3 == "VOOU PARA LONGE!":
                return True
        except Exception as e:
            print(f"Error extracting text: {e}. Install tesseract (macOS): `brew install tesseract`")
            return False    
    def capturePremioTotal(self, image):
        #region = (645, 413, 145, 33)    # Total pago
        region = self.REL_REGIONS["premio_total"]

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

        #region = (745, 359, 245, 38)      # Multiplicadores
        region = self.REL_REGIONS["multiplicadores"]
        
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
        
    # ---------- Loop principal ----------
    def evaluate_loop(self):
        premio_encontrado: str | None = None  # armazena texto do "Total pago" detectado ao ver "voou"

        primeirovalorNovo= None
        segundoValorNovo = None
        
        primeirovalorAntigo = None
        segundoValorAntigo = None
        
        valorPremioValido = None
        ultimoValorPremio = None

        print("Iniciando loop de avaliação em tempo real...")
        while self.running:
            try:
                image = self.capture_screenshot()
                if image is None:
                    time.sleep(self.interval)
                    continue
                else:
                # 1) Detecta "voou" e captura "Total pago" (texto cru)
                # voou = self.get_voou_para_longe(image)
                # if voou:
                #     premio_text = self.capture_premio_total(image)
                #     if premio_text:
                #         premio_encontrado = premio_text

                # # Extract text
                # ultimoValorPremio,text = self.extract_text2(image)
                # print(f"Voou: {voou} | Premio capturado: {premio_encontrado} | Texto multiplicadores: '{text}'")

                # # 2) Extrai multiplicadores e pega o primeiro (m1)
                # mult_text = self.extract_multipliers_text(image)
                # m1, m2 = self.parse_multipliers(mult_text)
                    ultimoValorPremio,text = self.extract_text2(image)

                    if ultimoValorPremio:
                        valorPremioValido = ultimoValorPremio
            
                    primeirovalorAntigo = primeirovalorNovo
                    segundoValorAntigo = segundoValorNovo
                
                    primeirovalorNovo, segundoValorNovo = self.parse_multipliers(text)
                    if primeirovalorNovo == primeirovalorAntigo and segundoValorNovo == segundoValorAntigo:

                        if valorPremioValido and primeirovalorNovo and not ultimoValorPremio:

                            m1 = primeirovalorNovo
                            premio_encontrado = valorPremioValido
                            # 3) Atualiza históricos quando valores válidos existem
                            if m1 is not None and premio_encontrado:
                                tv_parsed = parse_totalvalue(premio_encontrado)  # usa parser robusto do módulo 'tensor'
                                if not np.isnan(tv_parsed):
                                    self.mult_hist.append(float(m1))
                                    self.total_hist.append(float(tv_parsed))
                            
                            # 4) Predição: quando temos prêmio válido e NÃO estamos no mesmo frame de "voou"
                            if premio_encontrado and (m1 is not None):
                                try:
                                    tv_parsed = parse_totalvalue(premio_encontrado)
                                    if not np.isnan(tv_parsed):
                                        # Usa o modelo para construir as **11 features** (inclui flags zero/outlier)
                                        prob = self.model.predict_prob(
                                            multiplier=float(m1),
                                            totalvalue=float(tv_parsed),
                                            mult_hist=list(self.mult_hist),
                                            total_hist=list(self.total_hist)
                                        )
                                        # Usa o threshold aprendido (salvo em meta.json). Fallback 0.6 se não disponível.
                                        try:
                                            label = self.model.predict_label(
                                                multiplier=float(m1),
                                                totalvalue=float(tv_parsed),
                                                mult_hist=list(self.mult_hist),
                                                total_hist=list(self.total_hist)
                                            )   
                                            threshold = getattr(self.model, "decision_threshold", 0.6)
                                        except Exception:
                                            label = int(prob >= 0.6)
                                            threshold = 0.6

                                        print("--------------------------------")
                                        #print(f"[{datetime.now().isoformat()}]")
                                        print(
                                            f"multiplicador=>{m1:.2f} | total=>{tv_parsed:.2f} | "
                                            f"prob(next>2x) => {prob:.3f}"
                                        )
                                        valorPremioValido = None
                            #if prob >= threshold:
                            #    print("ALERTA: Probabilidade alta detectada!")

                                except Exception as e:
                                    print(f"Erro na avaliação: {e}")

                    # Limpa prêmio para evitar reuso indevido no próximo frame
                                premio_encontrado = None

                # Fecha imagem explicitamente
                image.close()

            except Exception as e:
                print(f"Erro no loop principal: {e}")

            time.sleep(self.interval)

    def start(self):
        if self.running:
            print("Evaluator já está rodando")
            return
        self.running = True
        self.thread = threading.Thread(target=self.evaluate_loop, daemon=True)
        self.thread.start()
        print(f"Real-time evaluator iniciado (intervalo: {self.interval}s)")

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        print("Real-time evaluator parado")


# --------- Execução ---------
def get_chrome_window_bounds():
    """Bounds da janela ativa do Chrome via AppleScript (macOS)."""
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
            raise Exception(f"AppleScript falhou: {result.stderr}")
        bounds_str = result.stdout.strip()
        if not bounds_str:
            raise Exception("Nenhuma janela retornada")
        bounds = list(map(int, bounds_str.split(", ")))
        x1, y1, x2, y2 = bounds
        return (x1, y1, x2 - x1, y2 - y1)
    except Exception as e:
        print(f"Erro obtendo bounds do Chrome: {e}")
        print("Usando fallback (0, 0, 800, 600)")
        return (0, 0, 800, 600)

def main():
    chrome_region = get_chrome_window_bounds()
    evaluator = RealTimeEvaluator(interval=0.0, region=chrome_region, model_path="next_multiplier_model.keras", lang='por')
    evaluator.start()
    try:
        time.sleep(28800)  # 8 horas
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    finally:
        evaluator.stop()

if __name__ == "__main__":
    main()
