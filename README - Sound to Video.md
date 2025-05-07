# 🎨 Audio ↔ Video Byte Mapping Tool

Este projeto permite converter arquivos `.wav` em `.avi` e vice-versa, tratando os **dados binários crus** como conteúdo visual ou sonoro. Ideal para experimentação artística com audiovisual.

---

## 🔁 `wav2avi`: Converter WAV para AVI (Áudio → Vídeo)

Cria um vídeo onde cada frame é construído diretamente dos bytes do áudio.

### ✅ Uso

```bash
python script.py wav2avi --input INPUT.wav --output OUTPUT.avi [opções]
```

### 🔧 Parâmetros

| Flag         | Descrição                                                                 |
|--------------|---------------------------------------------------------------------------|
| `--input`    | Caminho para o arquivo `.wav` de entrada (obrigatório)                    |
| `--output`   | Caminho para salvar o arquivo `.avi` de saída (obrigatório)               |
| `--fps`      | Frames por segundo (opcional, padrão: 30).                                |
| `--color`    | Modo de mapeamento de cor dos bytes: `default`, `invert`, `threshold`, `log` |
| `--glitch`   | Adiciona glitches visuais aleatórios nos frames (opcional)               |

---

## 🔁 `avi2wav`: Converter AVI para WAV (Vídeo → Áudio)

Reinterpreta os dados de pixel dos frames de vídeo como valores de áudio e os escreve em um `.wav`.

### ✅ Uso

```bash
python script.py avi2wav --input INPUT.avi --output OUTPUT.wav [opções]
```

### 🔧 Parâmetros

| Flag              | Descrição                                                                 |
|-------------------|---------------------------------------------------------------------------|
| `--input`         | Caminho para o arquivo `.avi` de entrada (obrigatório)                    |
| `--output`        | Caminho para salvar o arquivo `.wav` de saída (obrigatório)               |
| `--use-fps`       | Detecta o FPS real do vídeo com `ffprobe` (desativado por padrão)         |
| `--detect-pixfmt` | Detecta dinamicamente o pixel format com `ffprobe` (padrão: rgb24)        |
| `--normalize`     | Normaliza o volume do áudio para usar toda a faixa de 16-bit              |
| `--stereo`        | Salva o áudio em estéreo (canais L/R duplicados)                          |
| `--envelope`      | [Não implementado] Aplicar fade-ins, cortes dinâmicos etc.                |
| `--granular`      | [Não implementado] Aplicar granular synthesis com base nos pixels         |
| `--contextual`    | [Não implementado] Alterar o som com base em média de cores do frame      |

---

## 📝 Notas

- Os arquivos `.avi` usam codec `rawvideo`, preservando fidelidade de bits para reversibilidade total
- WAV deve estar em formato PCM (padrão)
- Efeitos como glitch, granular e contextual são pensados para arte audiovisual, não precisão técnica

---

## 📦 Requisitos

- Python 3.x
- `ffmpeg` e `ffprobe` instalados e disponíveis no PATH
- Bibliotecas Python: `numpy`, `wave`, `subprocess`, `argparse`

---

## 💡 Possibilidades futuras

- Suporte a outros mapeamentos artísticos (como mapa de calor, mapeamento polar)
- Exportação de visualizações intermediárias
- Modo interativo com sliders e pré-visualização ao vivo