import os
import sys
import time
import tempfile
import traceback
import cv2
from cog import BasePredictor, Input, Path as CogPath
from propainter.inference import Propainter, get_device


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Predictor(BasePredictor):
    def setup(self):
        log("=== SETUP BAŞLADI ===")

        log(f"Python: {sys.version}")
        log(f"Çalışma dizini: {os.getcwd()}")

        # Weights kontrolü
        weights_dir = "weights/propainter"
        expected = ["raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth"]
        log(f"Weights dizini: {os.path.abspath(weights_dir)}")
        if os.path.exists(weights_dir):
            for f in expected:
                fpath = os.path.join(weights_dir, f)
                if os.path.exists(fpath):
                    size_mb = os.path.getsize(fpath) / (1024 * 1024)
                    log(f"  OK {f} ({size_mb:.1f} MB)")
                else:
                    log(f"  EKSIK {f} — indirme gerekecek")
        else:
            log(f"  UYARI: {weights_dir} dizini yok — indirme gerekecek")

        try:
            log("Device tespit ediliyor...")
            self.device = get_device()
            log(f"Device: {self.device}")

            log("Propainter modeli yükleniyor...")
            t0 = time.time()
            self.propainter = Propainter(
                propainter_model_dir=weights_dir,
                device=self.device
            )
            log(f"Model yüklendi. Süre: {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"HATA setup() sırasında: {e}")
            log(traceback.format_exc())
            raise

        log("=== SETUP TAMAMLANDI ===")

    def predict(
        self,
        video: CogPath = Input(description="İnpainting yapılacak video (.mp4)"),
        mask: CogPath = Input(description="Maske — tek png veya video (.png / .mp4)"),
        fp16: bool = Input(description="Yarı hassasiyet — daha hızlı, daha az VRAM", default=False),
        neighbor_length: int = Input(description="Lokal komşu sayısı (azalt = daha az VRAM)", default=10, ge=5, le=20),
        subvideo_length: int = Input(description="Alt video uzunluğu (azalt = daha az VRAM)", default=80, ge=20, le=150),
        max_seconds: int = Input(description="Maksimum işlenecek süre (saniye)", default=10, ge=1, le=60),
    ) -> CogPath:
        log("=== PREDICT BAŞLADI ===")
        log(f"Girdi video: {video}")
        log(f"Girdi maske: {mask}")
        log(f"Parametreler: fp16={fp16}, neighbor_length={neighbor_length}, subvideo_length={subvideo_length}, max_seconds={max_seconds}")

        try:
            cap = cv2.VideoCapture(str(video))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            log(f"Video meta: {orig_w}x{orig_h}, {fps:.2f}fps, {frame_count} frame")

            width = (orig_w // 8) * 8
            height = (orig_h // 8) * 8

            duration = frame_count / fps if fps > 0 else 0
            log(f"Video süresi: {duration:.1f}s, işlenecek boyut: {width}x{height}")

            if duration > 60:
                raise ValueError(f"Video çok uzun ({duration:.1f}s). Maksimum 60 saniye.")
            if frame_count < 5:
                raise ValueError("Video çok kısa, en az 5 frame gerekli.")

            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = tmp.name
            tmp.close()
            log(f"Çıktı yolu: {output_path}")

            log("ProPainter forward başlıyor...")
            t0 = time.time()
            self.propainter.forward(
                video=str(video),
                mask=str(mask),
                output_path=output_path,
                video_length=max_seconds,
                width=width,
                height=height,
                fp16=fp16,
                neighbor_length=neighbor_length,
                subvideo_length=subvideo_length,
            )
            log(f"Forward tamamlandı. Süre: {time.time()-t0:.1f}s")

            if not os.path.exists(output_path):
                raise RuntimeError("ProPainter çıktı dosyası oluşturulamadı.")

            out_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            log(f"Çıktı dosyası: {out_size_mb:.1f} MB")
            log("=== PREDICT TAMAMLANDI ===")
            return CogPath(output_path)

        except Exception as e:
            log(f"HATA predict() sırasında: {e}")
            log(traceback.format_exc())
            raise
