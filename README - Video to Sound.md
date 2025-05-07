
# 🎧 VideoSonificação – Transforme Vídeos em Som

Este script converte as **características visuais** de um vídeo (como movimento, cor, textura e brilho) em **áudio sintetizado**, gerando uma paisagem sonora baseada exclusivamente no que se vê no vídeo — ignorando completamente qualquer áudio original.

---

## 🚀 Como Usar

### ⚙️ Requisitos

Certifique-se de ter os seguintes softwares instalados:

- Python 3.6+
- FFmpeg (`ffmpeg`, `ffprobe`)
- Bibliotecas Python:
  - `numpy`, `scipy`, `Pillow`

Você pode instalar os pacotes necessários com:

```bash
pip install numpy scipy pillow
```

---

### 🖥️ Execução via Linha de Comando

Use o script diretamente com argumentos:

```bash
python video_sonificacao.py --input video.mp4
```

#### Parâmetros:

| Parâmetro              | Descrição                                                                 |
|------------------------|--------------------------------------------------------------------------|
| `--input` ou `-i`      | Caminho do vídeo de entrada. **Obrigatório.**                            |
| `--output` ou `-o`     | Caminho do arquivo `.wav` de saída. Padrão: mesmo nome do vídeo.         |
| `--sample-rate` ou `-sr` | Taxa de amostragem do áudio. Padrão: `44100 Hz`.                         |

---

## 🧠 O que o script faz

1. **Extrai frames** do vídeo.
2. **Analisa cada frame**, detectando:
   - Cor média, brilho e saturação.
   - Movimento entre quadros.
   - Texturas e bordas.
3. **Mapeia visual → áudio**:
   - Brilho → volume
   - Cor → tipo de onda e frequência
   - Movimento → modulação e intensidade
4. **Gera um áudio `.wav`** sincronizado com o tempo do vídeo.

---

## 🧾 Exemplo completo

```bash
python video_sonificacao.py -i natureza.mp4 -o som_natureza.wav -sr 48000
```

Este comando irá:
- Ler o vídeo `natureza.mp4`
- Converter as imagens em som
- Salvar o áudio em `som_natureza.wav`
- Usar uma taxa de amostragem de 48kHz

---

## 🛠️ Saída Esperada

- Áudio `.wav` contendo a sonificação.
- Impressão no terminal com:
  - Informações do vídeo (frames, duração, resolução, bitrate)
  - Progresso da conversão
  - Metadados do áudio gerado

---

## 📌 Observações

- Quanto maior a resolução do vídeo, mais detalhado será o áudio.
- O tipo de som muda com base nas cores dominantes e movimento.
- Ideal para projetos de acessibilidade, arte sonora ou visualização alternativa de vídeos.
