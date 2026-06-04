import torch
import torchaudio
import random

class AdaptadorAcusticoForense:
    """
    Módulo pericial para simular a degradação de canal de smartphones modernos.
    Aplica ruído térmico, eco de sala (RIR) e limitação de banda de codecs de celular.
    """
    def __init__(self, sample_rate=16000, probabilidade=0.7):
        self.sr = sample_rate
        self.p = probabilidade

    def _adicionar_ruido_hardware(self, waveform, min_snr=15, max_snr=35):
        """Simula o ruído de fundo estático/térmico dos circuitos de smartphones (Relação Sinal-Ruído)."""
        snr_db = random.uniform(min_snr, max_snr)
        potencia_sinal = torch.mean(waveform ** 2)
        if potencia_sinal == 0:
            return waveform
        potencia_ruido = potencia_sinal / (10 ** (snr_db / 10))
        ruido = torch.randn_like(waveform) * torch.sqrt(potencia_ruido)
        return waveform + ruido

    def _simular_reverberacao_ambiente(self, waveform, rir_duration=0.04):
        """Simula o eco de paredes de ambientes residenciais através de uma Resposta ao Impulso Quântica."""
        num_samples = int(self.sr * rir_duration)
        if num_samples % 2 == 0:
            num_samples += 1

        t = torch.linspace(0, rir_duration, num_samples)
        decay = torch.exp(-6 * t)
        rir = torch.randn(num_samples) * decay
        rir = rir / torch.norm(rir, p=1) # Normalização de ganho unitário

        # Padding para manter o tamanho exato do sinal original após a convolução
        waveform_padded = torch.nn.functional.pad(waveform, (num_samples // 2, num_samples // 2), mode='constant', value=0)
        rir = rir.view(1, 1, -1)

        # Aplicação do eco via convolução linear 1D
        waveform_reverb = torch.nn.functional.conv1d(waveform_padded.unsqueeze(0), rir).squeeze(0)
        return waveform_reverb

    def _simular_codec_celular(self, waveform):
        """Aplica filtros de biquad para simular a compressão e corte de frequência de microfones comerciais (AAC/AMR-WB)."""
        # Filtro Passa-Alta: Remove sub-graves e ruídos de vento abaixo de 200Hz
        waveform = torchaudio.functional.highpass_biquad(waveform, self.sr, cutoff_freq=200.0)
        # Filtro Passa-Baixa: Limita a banda superior em 7kHz, típica de conexões móveis "HD Voice"
        waveform = torchaudio.functional.lowpass_biquad(waveform, self.sr, cutoff_freq=7000.0)
        return waveform

    def aplicar_degradacao(self, waveform):
        """Executa a esteira de modificação acústica baseada em sorteio probabilístico."""
        if random.random() > self.p:
            return waveform # Preserva a amostra original (estúdio limpo) para garantir variabilidade híbrida

        # Pipeline sequencial de degradação móvel
        waveform = self._simular_reverberacao_ambiente(waveform)
        waveform = self._adicionar_ruido_hardware(waveform)
        waveform = self._simular_codec_celular(waveform)
        return waveform
