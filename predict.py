import os
import tempfile
import subprocess
from pathlib import Path
from cog import BasePredictor, Input, Path as CogPath


class Predictor(BasePredictor):
    def setup(self):
        """Model ağırlıklarını önceden indir (cold start optimizasyonu)"""
        os.makedirs("weights", exist_ok=True)

        weights = {
            "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
            "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
            "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
        }

        for filename, url in weights.items():
            dest = f"weights/{filename}"
            if not os.path.exists(dest):
                print(f"İndiriliyor: {filename}")
                subprocess.run(["wget", "-q", "-O", dest, url], check=True)
            else:
                print(f"Zaten var: {filename}")

    def predict(
        self,
        video: CogPath = Input(description="Maskelenecek video (.mp4)"),
        mask: CogPath = Input(description="Maske görüntüsü veya klasörü (.png)"),
        width: int = Input(description="Çıktı genişliği", default=640),
        height: int = Input(description="Çıktı yüksekliği", default=360),
        fp16: bool = Input(description="Yarı hassasiyet (daha hızlı, daha az VRAM)", default=True),
        neighbor_length: int = Input(description="Lokal komşu uzunluğu (azalt = daha az VRAM)", default=10),
        subvideo_length: int = Input(description="Alt video uzunluğu", default=80),
    ) -> CogPath:
        """Video inpainting çalıştır"""

        output_dir = tempfile.mkdtemp()

        cmd = [
            "python", "inference_propainter.py",
            "--video", str(video),
            "--mask", str(mask),
            "--output", output_dir,
            "--width", str(width),
            "--height", str(height),
            "--neighbor_length", str(neighbor_length),
            "--subvideo_length", str(subvideo_length),
        ]

        if fp16:
            cmd.append("--fp16")

        print("Çalıştırılıyor:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # Çıktı videosunu bul
        output_files = list(Path(output_dir).rglob("*.mp4"))
        if not output_files:
            raise RuntimeError(f"Çıktı bulunamadı! {output_dir} klasörü: {os.listdir(output_dir)}")

        return CogPath(output_files[0])