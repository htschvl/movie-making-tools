"""
VideoSonificação: Converte informações visuais de vídeo em áudio

Este script transforma um vídeo em áudio baseado exclusivamente nas informações visuais,
ignorando qualquer áudio original. A sonificação é baseada nas características
visuais como movimento, cores, bordas e intensidade dos pixels.

Requer: FFmpeg, Python 3.6+, NumPy, SciPy, Pillow
"""

import os
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import wave
import struct
from scipy.signal import butter, lfilter
from PIL import Image, ImageStat
import threading
import itertools
import sys
import time

def animate_spinner(message="Processando"):
    spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    while not getattr(threading.current_thread(), "stop", False):
        sys.stdout.write(f"\r{message} {next(spinner)} ")
        sys.stdout.flush()
        time.sleep(0.1)



class VideoAnalyzer:
    """Classe para analisar características visuais de vídeos usando FFmpeg e PIL."""
    
    def __init__(self, video_path):
        """Inicializa o analisador com o caminho do vídeo."""
        self.video_path = video_path
        self.video_info = self._get_video_info()
        
    def _get_video_info(self):
        """Extrai informações do vídeo usando FFmpeg."""
        try:
            # Obter duração
            cmd_duration = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                self.video_path
            ]
            duration = float(subprocess.check_output(cmd_duration).decode('utf-8').strip())
            
            # Obter FPS
            cmd_fps = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                self.video_path
            ]
            fps_str = subprocess.check_output(cmd_fps).decode('utf-8').strip()
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den
            else:
                fps = float(fps_str)
            
            # Obter total de frames
            cmd_frames = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                self.video_path
            ]
            try:
                frame_count = int(subprocess.check_output(cmd_frames).decode('utf-8').strip())
            except (ValueError, subprocess.CalledProcessError):
                # Se não conseguir obter o número exato de frames, calcular com base na duração e FPS
                frame_count = int(duration * fps)
            
            # Obter resolução
            cmd_resolution = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                self.video_path
            ]
            resolution = subprocess.check_output(cmd_resolution).decode('utf-8').strip()
            width, height = map(int, resolution.split(','))
            
            # Obter bitrate
            cmd_bitrate = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=bit_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                self.video_path
            ]
            try:
                bitrate = int(subprocess.check_output(cmd_bitrate).decode('utf-8').strip())
            except (ValueError, subprocess.CalledProcessError):
                # Estimar bitrate se não conseguir obter diretamente
                bitrate = width * height * fps * 0.2  # Estimativa grosseira
            
            return {
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'bitrate': bitrate,
                'bitrate_factor': self._calculate_bitrate_factor(bitrate)
            }
        except Exception as e:
            print(f"Erro ao obter informações do vídeo: {e}")
            return None
    
    def _calculate_bitrate_factor(self, bitrate):
        """Calcula um fator de bitrate para modular o som."""
        # Normaliza o bitrate para um fator entre 0.5 e 2.0
        # Assumindo 5Mbps como bitrate médio de referência
        return 0.5 + min(bitrate / 5000000.0, 1.5)
    
    def extract_frames(self, output_dir, fps=None):
        """
        Extrai frames do vídeo usando FFmpeg.
        
        Args:
            output_dir: Diretório para salvar os frames
            fps: Taxa de frames para extração (opcional, usa a do vídeo se não especificado)
            
        Returns:
            Lista de caminhos para os frames extraídos
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Usar o FPS original se não for especificado
        if fps is None:
            fps = self.video_info['fps']
        
        # Comando para extrair frames
        extract_cmd = [
            'ffmpeg',
            '-i', self.video_path,
            '-vsync', '0',
            '-vf', f'fps={fps}',  # Taxa de frames
            f'{output_dir}/frame_%06d.png'
        ]
        
        subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Listar e ordenar os frames extraídos
        frames = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                        if f.startswith('frame_') and f.endswith('.png')])
        
        return frames
    
    def get_frame_data(self, frame_path):
        """
        Analisa as características visuais de um frame.
        
        Args:
            frame_path: Caminho para o arquivo de imagem do frame
            
        Returns:
            Dicionário com dados visuais do frame
        """
        try:
            img = Image.open(frame_path)
            
            # Converter para RGB se for outro modo (como RGBA)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Obter dados estatísticos básicos
            stat = ImageStat.Stat(img)
            
            # Separar os canais
            r, g, b = img.split()
            
            # Converter para HSV para análise de cor
            hsv_img = img.convert('HSV')
            h, s, v = hsv_img.split()
            
            # Estatísticas dos canais
            h_stat = ImageStat.Stat(h)
            s_stat = ImageStat.Stat(s)
            v_stat = ImageStat.Stat(v)
            
            # Dividir a imagem em seções verticais para análise
            width, height = img.size
            n_divisions = 12
            division_width = width // n_divisions
            
            divisions = []
            for i in range(n_divisions):
                # Recortar divisão vertical
                box = (i * division_width, 0, (i + 1) * division_width, height)
                division = img.crop(box)
                
                # Analisar a divisão
                div_stat = ImageStat.Stat(division)
                
                # Extrair características de cada divisão
                divisions.append({
                    'position': i / n_divisions,
                    'mean': div_stat.mean,
                    'stddev': div_stat.stddev,
                    'median': div_stat.median,
                    'extrema': div_stat.extrema
                })
            
            return {
                'size': img.size,
                'mode': img.mode,
                'means': stat.mean,        # [r_mean, g_mean, b_mean]
                'stddevs': stat.stddev,    # Desvios padrão por canal
                'medians': stat.median,    # Medianas por canal
                'extremas': stat.extrema,  # (min, max) por canal
                'hsv': {
                    'h_mean': h_stat.mean[0],
                    's_mean': s_stat.mean[0],
                    'v_mean': v_stat.mean[0]
                },
                'divisions': divisions
            }
        except Exception as e:
            print(f"Erro ao analisar frame {frame_path}: {e}")
            return None
    
    def calculate_motion(self, prev_frame_data, curr_frame_data):
        """
        Calcula a quantidade de movimento entre dois frames consecutivos.
        
        Args:
            prev_frame_data: Dados do frame anterior
            curr_frame_data: Dados do frame atual
            
        Returns:
            Valor entre 0.0 e 1.0 representando a quantidade de movimento
        """
        if not prev_frame_data or not curr_frame_data:
            return 0.0
        
        # Comparar características entre frames
        motion_factors = []
        
        # Diferença nas médias de cor
        if 'means' in prev_frame_data and 'means' in curr_frame_data:
            mean_diff = sum([abs(a - b) for a, b in zip(prev_frame_data['means'], curr_frame_data['means'])])
            mean_diff_norm = mean_diff / (255 * 3)  # Normalizar para [0,1]
            motion_factors.append(mean_diff_norm)
        
        # Diferença nas divisões
        division_diffs = []
        if 'divisions' in prev_frame_data and 'divisions' in curr_frame_data:
            for prev_div, curr_div in zip(prev_frame_data['divisions'], curr_frame_data['divisions']):
                div_diff = sum([abs(a - b) for a, b in zip(prev_div['mean'], curr_div['mean'])])
                division_diffs.append(div_diff / (255 * 3))
        
        if division_diffs:
            motion_factors.append(sum(division_diffs) / len(division_diffs))
        
        # Calcular média ponderada dos fatores
        if motion_factors:
            motion = sum(motion_factors) / len(motion_factors)
            # Aplicar uma curva de resposta para aumentar a sensibilidade
            motion = min(1.0, motion * 2)
            return motion
        
        return 0.0


class AudioGenerator:
    """Classe para gerar áudio a partir de características visuais."""
    
    def __init__(self, sample_rate=44100):
        """Inicializa o gerador com uma taxa de amostragem específica."""
        self.sample_rate = sample_rate
    
    def butter_bandpass(self, lowcut, highcut, order=5):
        """Cria um filtro passa-banda Butterworth."""
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def butter_bandpass_filter(self, data, lowcut, highcut, order=5):
        """Aplica filtro passa-banda aos dados."""
        b, a = self.butter_bandpass(lowcut, highcut, order=order)
        y = lfilter(b, a, data)
        return y
    
    def generate_waveform(self, frequency, amplitude, duration, waveform_type='sine'):
        """
        Gera uma forma de onda com a frequência e amplitude especificadas.
        
        Args:
            frequency: Frequência em Hz
            amplitude: Amplitude entre 0.0 e 1.0
            duration: Duração em segundos
            waveform_type: Tipo de onda ('sine', 'square', 'sawtooth', 'triangle')
            
        Returns:
            Array NumPy com as amostras de áudio
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        if waveform_type == 'sine':
            tone = amplitude * np.sin(2 * np.pi * frequency * t)
        elif waveform_type == 'square':
            tone = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform_type == 'sawtooth':
            tone = amplitude * 2 * (t * frequency - np.floor(0.5 + t * frequency))
        elif waveform_type == 'triangle':
            tone = amplitude * 2 * np.abs(2 * (t * frequency - np.floor(t * frequency)) - 1) - 1
        else:
            tone = amplitude * np.sin(2 * np.pi * frequency * t)
        
        return tone
    
    def frame_to_frequencies(self, frame_data, prev_frame_data=None, min_freq=80, max_freq=10000):
        """
        Converte características visuais em informações de frequência e amplitude.
        
        Args:
            frame_data: Dados analisados do frame
            prev_frame_data: Dados do frame anterior para cálculo de movimento
            min_freq: Frequência mínima em Hz
            max_freq: Frequência máxima em Hz
            
        Returns:
            Tuple com (lista de frequências, amplitude, tipo de onda)
        """
        if not frame_data:
            return [], 0.5, 'sine'
        
        # Calcular movimento se houver frame anterior
        motion = 0.0
        if prev_frame_data:
            motion = self.calculate_motion(prev_frame_data, frame_data)
        
        # Extrair características principais
        r_mean = frame_data['means'][0] / 255.0
        g_mean = frame_data['means'][1] / 255.0
        b_mean = frame_data['means'][2] / 255.0
        brightness = (r_mean + g_mean + b_mean) / 3
        
        # Informações de cor HSV
        hue = frame_data['hsv']['h_mean'] / 255.0  # Normalizado para [0,1]
        saturation = frame_data['hsv']['s_mean'] / 255.0
        value = frame_data['hsv']['v_mean'] / 255.0
        
        # Gerar frequências baseadas nas divisões da imagem
        frequencies = []
        for i, div in enumerate(frame_data['divisions']):
            # Base de frequência para esta divisão
            base_freq = min_freq + (i / len(frame_data['divisions'])) * (max_freq - min_freq)
            
            # Ajuste baseado em características da divisão
            div_r = div['mean'][0] / 255.0
            div_g = div['mean'][1] / 255.0
            div_b = div['mean'][2] / 255.0
            
            # Criar modulação de frequência baseada nas cores
            color_factor = (div_r * 0.4) + (div_g * 0.3) + (div_b * 0.3)
            
            # Desvio padrão como indicador de texturas/detalhes
            detail_factor = sum(div['stddev']) / (255.0 * 3)
            
            # Calcular frequência final para esta divisão
            freq_mod = 0.7 + (0.3 * color_factor) + (0.3 * detail_factor) + (0.4 * motion)
            freq = base_freq * freq_mod
            
            # Garantir que a frequência está dentro dos limites
            freq = max(min_freq, min(max_freq, freq))
            frequencies.append(freq)
        
        # Calcular amplitude baseada no brilho, saturação e movimento
        amplitude = 0.3 + (0.3 * brightness) + (0.2 * saturation) + (0.2 * motion)
        amplitude = min(1.0, amplitude)
        
        # Escolher tipo de onda baseado no matiz da imagem
        waveform = 'sine'
        if hue < 0.15 or hue > 0.85:  # Vermelho/Magenta
            waveform = 'sine'
        elif 0.15 <= hue < 0.4:  # Amarelo/Verde
            waveform = 'triangle'
        elif 0.4 <= hue < 0.7:  # Ciano/Azul
            waveform = 'square' if saturation > 0.5 else 'sawtooth'
        elif 0.7 <= hue < 0.85:  # Roxo
            waveform = 'sawtooth'
        
        return frequencies, amplitude, waveform
    
    def calculate_motion(self, prev_frame_data, curr_frame_data):
        """
        Calcula o movimento entre dois frames consecutivos.
        
        Args:
            prev_frame_data: Dados do frame anterior
            curr_frame_data: Dados do frame atual
            
        Returns:
            Valor entre 0.0 e 1.0
        """
        # Comparar características entre frames
        motion_factors = []
        
        # Diferença nas médias de cor
        mean_diff = sum([abs(a - b) for a, b in zip(prev_frame_data['means'], curr_frame_data['means'])])
        mean_diff_norm = mean_diff / (255 * 3)  # Normalizar para [0,1]
        motion_factors.append(mean_diff_norm)
        
        # Diferença nas divisões
        division_diffs = []
        for prev_div, curr_div in zip(prev_frame_data['divisions'], curr_frame_data['divisions']):
            div_diff = sum([abs(a - b) for a, b in zip(prev_div['mean'], curr_div['mean'])])
            division_diffs.append(div_diff / (255 * 3))
        
        motion_factors.append(sum(division_diffs) / len(division_diffs))
        
        # Calcular média dos fatores e aplicar curva de resposta
        motion = sum(motion_factors) / len(motion_factors)
        motion = min(1.0, motion * 2)  # Aumentar sensibilidade
        return motion
    
    def synthesize_audio_from_frames(self, frame_data_list, duration, bitrate_factor=1.0):
        """
        Sintetiza áudio a partir de uma lista de dados de frames.
        
        Args:
            frame_data_list: Lista de dados analisados para cada frame
            duration: Duração total do áudio em segundos
            bitrate_factor: Fator para modular o som com base no bitrate
            
        Returns:
            Array NumPy com as amostras de áudio
        """
        # Calcular a duração exata de cada frame
        frame_duration = duration / len(frame_data_list)
        samples_per_frame = int(self.sample_rate * frame_duration)
        
        start_time_outside = time.time()
        total_frames = len(frame_data_list)

        # Array para armazenar amostras de áudio
        audio_samples = np.array([], dtype=np.float32)
        
        prev_frame_data = None
        for i, frame_data in enumerate(frame_data_list):
            if frame_data is None:
                # Se os dados do frame estiverem indisponíveis, use silêncio
                frame_audio = np.zeros(samples_per_frame)
            else:
                # Converter dados do frame em frequências
                frequencies, amplitude, waveform = self.frame_to_frequencies(
                    frame_data, 
                    prev_frame_data, 
                    min_freq=80 * bitrate_factor, 
                    max_freq=10000 * bitrate_factor
                )
                
                # Criar uma mistura de tons para este frame
                frame_audio = np.zeros(samples_per_frame)
                for j, freq in enumerate(frequencies):
                    # Ajustar amplitude com base na posição
                    position_factor = 1.0 - (abs(j - len(frequencies)/2) / (len(frequencies)/2)) * 0.5
                    freq_amplitude = (amplitude / len(frequencies)) * position_factor
                    
                    # Gerar tom com a frequência detectada
                    tone = self.generate_waveform(
                        freq, 
                        freq_amplitude, 
                        frame_duration, 
                        waveform
                    )
                    
                    # Aplicar filtro passa-banda para suavizar
                    nyquist = self.sample_rate / 2
                    low = max(1.0, freq * 0.8)
                    high = min(nyquist - 1, freq * 1.2)
                    if low >= high:
                        low = max(1.0, high - 10)
                    filtered_tone = self.butter_bandpass_filter(tone, low, high)

                    
                    # Adicionar à mistura
                    if len(filtered_tone) >= samples_per_frame:
                        frame_audio += filtered_tone[:samples_per_frame]
                    else:
                        # Preencher com zeros se o tom for muito curto
                        padded_tone = np.pad(filtered_tone, (0, samples_per_frame - len(filtered_tone)))
                        frame_audio += padded_tone
                
                # Normalizar para evitar clipping
                if np.max(np.abs(frame_audio)) > 0:
                    frame_audio = frame_audio / np.max(np.abs(frame_audio)) * amplitude
            
            # Adicionar ao array de amostras
            audio_samples = np.append(audio_samples, frame_audio)
            
            # Atualizar frame anterior
            
            start_time = time.time()
            if i % 50 == 0 or i == len(frame_data_list) - 1:
                prev_frame_data = frame_data
                elapsed_time = time.time() - start_time
                frames_done = i + 1
                total_frames = len(frame_data_list)
                percent = (frames_done / total_frames) * 100
                fps = frames_done / elapsed_time if elapsed_time > 0 else 0
                eta = (total_frames - frames_done) / fps if fps > 0 else 0
                print(f"\r  {frames_done}/{total_frames} {percent:.1f}% - ETA: {eta:.1f}s", end='', flush=True)
        

        elapsed_time = time.time() - start_time_outside
        print(f"\n Um total de {total_frames} frames foram sintetizados para áudio em {elapsed_time:.1f} segundos.")
        return audio_samples
        
    

    
    def save_audio(self, audio_samples, output_path, duration=None):
        """
        Salva as amostras de áudio como arquivo WAV.
        
        Args:
            audio_samples: Array NumPy com amostras de áudio
            output_path: Caminho para salvar o arquivo de áudio
            duration: Duração esperada do áudio em segundos (para ajuste)
        """
        # Verificar se o comprimento do áudio corresponde à duração esperada
        if duration:
            expected_samples = int(duration * self.sample_rate)
            if len(audio_samples) != expected_samples:
                print(f"Ajustando comprimento do áudio para {duration:.2f} segundos...")
                
                if len(audio_samples) > expected_samples:
                    # Cortar o excesso
                    audio_samples = audio_samples[:expected_samples]
                else:
                    # Adicionar silêncio no final
                    padding = np.zeros(expected_samples - len(audio_samples))
                    audio_samples = np.append(audio_samples, padding)
        
        # Aplicar fade-in e fade-out suave
        fade_duration = min(0.1, duration / 20) if duration else 0.1
        fade_samples = int(fade_duration * self.sample_rate)
        
        if fade_samples > 0 and len(audio_samples) > fade_samples * 2:
            # Fade-in
            fade_in = np.linspace(0, 1, fade_samples)
            audio_samples[:fade_samples] *= fade_in
            
            # Fade-out
            fade_out = np.linspace(1, 0, fade_samples)
            audio_samples[-fade_samples:] *= fade_out
        
        # Normalizar o áudio para usar toda a faixa dinâmica
        max_amplitude = np.max(np.abs(audio_samples))
        if max_amplitude > 0:
            # Normalizar para 95% do máximo para evitar clipping
            audio_samples = audio_samples / max_amplitude * 0.95
        
        # Salvar como arquivo WAV
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes por amostra (16 bits)
            wav_file.setframerate(self.sample_rate)
            
            # Converter para PCM de 16 bits
            pcm_data = audio_samples * 32767
            pcm_data = pcm_data.astype(np.int16)
            
            # Escrever para o arquivo
            wav_file.writeframes(struct.pack('h' * len(pcm_data), *pcm_data))
        
        print(f"\n Áudio salvo em: {output_path}")


class VideoToAudio:
    """Classe principal para converter vídeo em áudio."""
    
    def __init__(self, video_path, output_path=None, sample_rate=44100):
        """
        Inicializa o conversor de vídeo para áudio.
        
        Args:
            video_path: Caminho para o arquivo de vídeo
            output_path: Caminho para o arquivo de áudio de saída (opcional)
            sample_rate: Taxa de amostragem do áudio em Hz
        """
        self.video_path = video_path
        
        # Definir caminho de saída se não for especificado
        if not output_path:
            self.output_path = os.path.splitext(video_path)[0] + '.wav'
        else:
            self.output_path = output_path
        
        self.sample_rate = sample_rate
        self.analyzer = VideoAnalyzer(video_path)
        self.generator = AudioGenerator(sample_rate)
    
    def convert(self):
        """
        Realiza o processo de conversão completo.
        
        Returns:
            Dicionário com metadados da conversão
        """
        if not self.analyzer.video_info:
            print("Erro: Não foi possível analisar o vídeo.")
            return None
        
        video_info = self.analyzer.video_info
        print("\n╔════════════════════════════════════════╗")
        print("║         📼 Informações do Vídeo        ║")
        print("╠════════════════════════════════════════╣")
        print(f"║ 🧮 Frames     : {video_info['frame_count']:<24}║")
        print(f"║ 🎞️ FPS        : {video_info['fps']::<24.2f}║")
        print(f"║ ⏱️ Duração    : {video_info['duration']:.2f} segundos{'':<10}║")
        print(f"║ 📐 Resolução : {video_info['width']}x{video_info['height']:<17}║")
        print(f"║ 🔊 Bitrate   : {video_info['bitrate']/1_000_000:.2f} Mbps{'':<10}║")
        print("╚════════════════════════════════════════╝\n")

        
        # Criar diretório temporário para os frames
        temp_dir = tempfile.mkdtemp()
        
        try:
            print("Extraindo frames do vídeo... ", end="", flush=True)
            spinner_thread = threading.Thread(target=animate_spinner, args=("Extraindo frames do vídeo",))
            spinner_thread.start()

            start_time = time.time()
            frame_paths = self.analyzer.extract_frames(temp_dir)
            elapsed_time = time.time() - start_time

            spinner_thread.stop = True
            spinner_thread.join()

            if not frame_paths:
                print("\nErro: Nenhum frame foi extraído do vídeo.")
                return None

            total_frames = len(frame_paths)
            print(f"\r Um total de {total_frames} frames foram extraídos em {elapsed_time:.1f} segundos.")

            print(f"Analisando {total_frames} frames...")
            frame_data_list = []
 
            ellapsed = None
            start_time = time.time()
            for i, frame_path in enumerate(frame_paths):
                frame_data = self.analyzer.get_frame_data(frame_path)
                frame_data_list.append(frame_data)

                # Progresso de análise
                elapsed = time.time() - start_time
                ellapsed = elapsed
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_frames - (i + 1)) / fps if fps > 0 else 0
                percent = (i + 1) / total_frames * 100
                print(f"\r  {i+1}/{total_frames} frames. {percent:.1f}% - {fps:.1f} fps - ETA: {eta:.1f}s", end="", flush=True)

            print(f"\r Um total de {total_frames} frames foram analisados em {ellapsed:.1f} segundos.")

            # --------------------------
            #   SINTETIZANDO ÁUDIO
            # --------------------------

            print("Sintetizando áudio... ")

            start_time = time.time()
            audio_samples = self.generator.synthesize_audio_from_frames(
                frame_data_list,
                video_info['duration'],
                video_info['bitrate_factor']
            )
    
            
            # --------------------------
            #      SALVANDO ÁUDIO
            # --------------------------

            print("Salvando arquivo de áudio... ")
            
            start_time = time.time()
            self.generator.save_audio(
                audio_samples,
                self.output_path,
                video_info['duration']
            )
            elapsed_time = time.time() - start_time
            
            return {
                "original_video": self.video_path,
                "audio_output": self.output_path,
                "duration": video_info['duration'],
                "sample_rate": self.sample_rate,
                "total_frames": len(frame_paths),
                "bitrate": video_info['bitrate']
            }
    
            
        
        finally:
            # Limpar arquivos temporários
            print(f"Arquivo de áudio salvo em {self.output_path} em {elapsed_time:.1f} segundos.")
            print("Limpando arquivos temporários...")
            shutil.rmtree(temp_dir)
            print("Conversão concluída.")


def main():
    """Função principal para execução via linha de comando."""
    parser = argparse.ArgumentParser(
        description='Converte um vídeo em áudio baseado nas características visuais'
    )
    parser.add_argument('--input', '-i', required=True, help='Nome do arquivo de vídeo (ex: video.mp4)')
    parser.add_argument('--output', '-o', help='Nome do arquivo de áudio de saída (.wav)')
    parser.add_argument('--sample-rate', '-sr', type=int, default=44100,
                        help='Taxa de amostragem do áudio em Hz (padrão: 44100)')

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output 

    converter = VideoToAudio(
        input_path,
        output_path,
        args.sample_rate
    )

    metadata = converter.convert()
    
    if metadata:
        print("\n╔════════════════════════════════════════╗")
        print("║       📝 Metadados da Conversão        ║")
        print("╠════════════════════════════════════════╣")

        # Mapeamento de ícones e labels personalizados
        label_map = {
            "original_video": "🎥 Arquivo de vídeo",
            "audio_output":   "🎧 Áudio de saída",
            "duration":       "⏱️ Duração",
            "sample_rate":    "🎚️ Sample rate",
            "total_frames":   "🧮 Total de frames",
            "bitrate":        "📡 Bitrate"
        }

        for key, value in metadata.items():
            label = label_map.get(key, key)
            if key == 'bitrate':
                val_str = f"{value / 1_000_000:.2f} Mbps"
            elif key == 'duration':
                val_str = f"{value:.2f} segundos"
            elif key == 'sample_rate':
                val_str = f"{value} Hz"
            else:
                val_str = str(value)
            print(f"║ {label:<17}: {val_str:<23}║")

        print("╚════════════════════════════════════════╝")


if __name__ == "__main__":
    main()