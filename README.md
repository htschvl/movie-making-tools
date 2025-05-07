# 🎬 Ferramentas Audiovisuais Experimentais

Este repositório reúne **ferramentas para experimentação radical entre som e imagem**, possibilitando projetos inovadores nas fronteiras do audiovisual — como curtas experimentais, videoartes, performances imersivas ou instalações sensoriais.

Cada ferramenta propõe um uso não convencional de dados audiovisuais, promovendo uma **fusão reversível ou simbiótica** entre áudio e vídeo. Ideal para artistas, pesquisadores, programadores criativos e entusiastas do glitch, noise, arte generativa e sonificação.

---

## 📦 Ferramentas incluídas

### 1. `Sound to Video` – 🎨 Mapeamento Binário entre Áudio e Vídeo

Converte `.wav` em `.avi` e vice-versa, tratando os **bytes crus** como pixels ou samples. Foco na reversibilidade e nos glitches estéticos gerados por essa transcodificação direta.

- `wav2avi`: transforma áudio em vídeo, cada frame codifica blocos de bytes do som.
- `avi2wav`: extrai som dos pixels do vídeo como se fossem amostras de áudio.
- Suporte a efeitos visuais e futuros modos artísticos (granular, contextual).
- Ideal para explorações de sinestesia e arte computacional de baixa fidelidade.

📄 [Leia mais](./README%20-%20Sound%20to%20Video.md)

---

### 2. `Video Sonificação` – 🎧 Imagem transformada em Som

Converte vídeos (sem considerar o áudio original) em paisagens sonoras baseadas em análise visual.

- Extrai frames e analisa brilho, cor, movimento e textura.
- Mapeia essas características para propriedades sonoras como volume, frequência, timbre e modulação.
- Cria um `.wav` sincronizado com a duração do vídeo.
- Aplicações: arte sonora, acessibilidade audiovisual, composição generativa.

📄 [Leia mais](./README%20-%20Video%20to%20Sound.md)

---

## ✨ Propósito

Este repositório busca **expandir as formas de criar e perceber audiovisual**, cruzando domínios digitais de forma não convencional e abrindo espaço para:

- Narrativas sensoriais não-lineares
- Obras multimídia experimentais
- Explorações do inconsciente computacional
- Visualizações de dados sinestésicas
- Ferramentas de acessibilidade criativa

---

## 🛠 Requisitos Gerais

- Python 3.6+
- FFmpeg (`ffmpeg`, `ffprobe`)
- Bibliotecas: `numpy`, `scipy`, `Pillow`, `wave`, `argparse`, `subprocess`

---

# Special thanks
Dedicado ao meu amigo Lucca, que me inspirou a criar essas ferramentas e sempre está do meu lado.

