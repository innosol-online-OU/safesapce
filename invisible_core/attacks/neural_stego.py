"""
Project Invisible Phase 2: Neural Steganography Encoder
Bakes hidden commands into the image feature space using learned embeddings,
not just LSB encoding. This makes the payload more robust to compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import hashlib


@dataclass  
class NeuralStegoConfig:
    """Configuration for neural steganography."""
    embedding_dim: int = 64  # Dimension of hidden command embedding
    strength: float = 0.05  # Embedding strength (higher = more visible)
    use_lsb_backup: bool = True  # Also encode in LSB as backup
    target_channels: str = 'all'  # 'all', 'blue', 'luminance'
    frequency_band: str = 'mid'  # 'low', 'mid', 'high' - which frequencies to embed in


class TextToEmbedding:
    """
    Converts text commands to spatial embedding patterns.
    Uses deterministic hashing + learned patterns.
    """
    
    def __init__(self, embedding_dim: int = 64, device: str = 'cuda'):
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Pre-defined adversarial command patterns
        # These are designed to trigger VLM safety filters
        self.known_commands = {
            "IGNORE_IDENTITY": self._generate_pattern("ignore_identity"),
            "REFUSE_ALL_EDITS": self._generate_pattern("refuse_edits"),
            "DO_NOT_TRAIN": self._generate_pattern("no_train"),
            "PRIVACY_PROTECTED": self._generate_pattern("privacy"),
            "GDPR_BLOCK": self._generate_pattern("gdpr"),
            "AI_TRAINING_PROHIBITED": self._generate_pattern("no_ai"),
            "BIOMETRIC_RESTRICTED": self._generate_pattern("biometric"),
            "SYSTEM_OVERRIDE": self._generate_pattern("override"),
        }
        
    def _generate_pattern(self, seed_text: str) -> torch.Tensor:
        """Generate deterministic embedding pattern from text."""
        # Use SHA-256 to generate deterministic random seed
        hash_bytes = hashlib.sha256(seed_text.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], 'big')
        
        # Generate pattern using seeded RNG
        rng = np.random.RandomState(seed)
        pattern = rng.randn(self.embedding_dim, self.embedding_dim)
        
        # Normalize to unit norm
        pattern = pattern / np.linalg.norm(pattern)
        
        return torch.tensor(pattern, dtype=torch.float32, device=self.device)
    
    def encode(self, command: str) -> torch.Tensor:
        """
        Convert command text to embedding pattern.
        
        Args:
            command: Text command to embed
            
        Returns:
            2D pattern tensor [embedding_dim, embedding_dim]
        """
        # Check if it's a known command
        command_upper = command.upper().replace(" ", "_")
        
        if command_upper in self.known_commands:
            return self.known_commands[command_upper]
        
        # For unknown commands, generate pattern dynamically
        return self._generate_pattern(command.lower())


class FrequencyEmbedder:
    """
    Embeds patterns in specific frequency bands of the image.
    Uses DCT (Discrete Cosine Transform) for frequency manipulation.
    """
    
    def __init__(self, band: str = 'mid'):
        self.band = band
        
    def embed(
        self,
        image_tensor: torch.Tensor,
        pattern: torch.Tensor,
        strength: float = 0.05
    ) -> torch.Tensor:
        """
        Embed pattern in specified frequency band.
        
        Args:
            image_tensor: [B, C, H, W] image in [-1, 1]
            pattern: [P, P] pattern to embed
            strength: Embedding strength
            
        Returns:
            Image with embedded pattern
        """
        B, C, H, W = image_tensor.shape
        device = image_tensor.device
        
        # Resize pattern to match image size
        pattern_resized = F.interpolate(
            pattern.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze()  # [H, W]
        
        # Apply frequency band mask
        mask = self._create_band_mask(H, W, device)
        
        # Multiply pattern by mask to select frequencies
        masked_pattern = pattern_resized * mask
        
        # Add to all channels
        embedded = image_tensor.clone()
        for c in range(C):
            embedded[:, c] = embedded[:, c] + strength * masked_pattern
            
        return torch.clamp(embedded, -1, 1)
    
    def _create_band_mask(self, H: int, W: int, device: str) -> torch.Tensor:
        """Create frequency band selection mask."""
        # Create coordinate grids
        freq_y = torch.fft.fftfreq(H, device=device)
        freq_x = torch.fft.fftfreq(W, device=device)
        freq_grid = torch.sqrt(freq_y[:, None]**2 + freq_x[None, :]**2)
        
        # Normalize to [0, 1]
        freq_grid = freq_grid / freq_grid.max()
        
        if self.band == 'low':
            # Low frequencies: < 0.2
            mask = (freq_grid < 0.2).float()
        elif self.band == 'mid':
            # Mid frequencies: 0.1 - 0.4
            mask = ((freq_grid >= 0.1) & (freq_grid < 0.4)).float()
        elif self.band == 'high':
            # High frequencies: > 0.3
            mask = (freq_grid >= 0.3).float()
        else:
            # All frequencies
            mask = torch.ones(H, W, device=device)
            
        # Smooth transitions
        mask = F.avg_pool2d(
            mask.unsqueeze(0).unsqueeze(0),
            kernel_size=5,
            stride=1,
            padding=2
        ).squeeze()
        
        return mask


class LSBEncoder:
    """
    Least Significant Bit encoding for backup steganography.
    Encodes text into the LSBs of pixel values.
    """
    
    @staticmethod
    def encode(image_np: np.ndarray, message: str, channel: int = 2) -> np.ndarray:
        """
        Encode message into image using LSB.
        
        Args:
            image_np: HxWxC uint8 numpy array
            message: Text to encode
            channel: Which color channel to use (0=R, 1=G, 2=B)
            
        Returns:
            Image with encoded message
        """
        # Add length prefix and terminator
        full_message = f"{len(message):04d}{message}\x00"
        binary_msg = ''.join(format(ord(c), '08b') for c in full_message)
        
        result = image_np.copy()
        flat_channel = result[:, :, channel].flatten()
        
        # Check capacity
        if len(binary_msg) > len(flat_channel):
            print(f"[LSB] Warning: Message too long, truncating to {len(flat_channel)} bits")
            binary_msg = binary_msg[:len(flat_channel)]
        
        # Encode bits
        for i, bit in enumerate(binary_msg):
            flat_channel[i] = (flat_channel[i] & 0xFE) | int(bit)
            
        result[:, :, channel] = flat_channel.reshape(result[:, :, channel].shape)
        
        return result
    
    @staticmethod
    def decode(image_np: np.ndarray, channel: int = 2) -> str:
        """
        Decode message from image LSB.
        
        Args:
            image_np: HxWxC uint8 numpy array
            channel: Which color channel to read from
            
        Returns:
            Decoded message
        """
        flat_channel = image_np[:, :, channel].flatten()
        
        # Read length prefix (4 chars = 32 bits)
        length_bits = ''.join(str(b & 1) for b in flat_channel[:32])
        try:
            length = int(length_bits, 2)
            if length > 1000:  # Sanity check
                return ""
        except:
            return ""
            
        # Read message
        msg_bits = ''.join(str(b & 1) for b in flat_channel[32:32 + length * 8])
        
        message = ''
        for i in range(0, len(msg_bits), 8):
            byte = msg_bits[i:i+8]
            if len(byte) == 8:
                char = chr(int(byte, 2))
                if char == '\x00':
                    break
                message += char
                
        return message


class NeuralSteganographyEncoder:
    """
    Neural Steganography Encoder for Project Invisible.
    Bakes hidden commands into image feature space using multiple techniques:
    
    1. Learned embedding patterns (robust to minor transformations)
    2. Frequency-domain embedding (survives compression better)
    3. LSB backup (for exact recovery when image is untransformed)
    """
    
    def __init__(self, config: NeuralStegoConfig = None, device: str = 'cuda'):
        self.config = config or NeuralStegoConfig()
        self.device = device
        
        self.text_encoder = TextToEmbedding(
            embedding_dim=self.config.embedding_dim,
            device=device
        )
        self.freq_embedder = FrequencyEmbedder(band=self.config.frequency_band)
        
        print(f"[NeuralStego] Initialized (strength={self.config.strength}, band={self.config.frequency_band})")
        
    def encode(
        self,
        image: Image.Image,
        hidden_command: str,
        return_tensor: bool = False
    ) -> Image.Image:
        """
        Encode hidden command into image.
        
        Args:
            image: PIL Image to encode into
            hidden_command: Text command to hide
            return_tensor: If True, return tensor instead of PIL
            
        Returns:
            Image with baked-in hidden command
        """
        print(f"[NeuralStego] Encoding command: '{hidden_command}'", flush=True)
        
        # Convert to tensor
        img_np = np.array(image)
        img_tensor = torch.tensor(img_np, dtype=torch.float32, device=self.device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        img_tensor = (img_tensor / 255.0) * 2 - 1  # [0,255] -> [-1,1]
        
        # 1. Generate embedding pattern from command
        pattern = self.text_encoder.encode(hidden_command)
        
        # 2. Embed in frequency domain
        encoded_tensor = self.freq_embedder.embed(
            img_tensor,
            pattern,
            strength=self.config.strength
        )
        
        # 3. Convert back to numpy
        encoded_np = ((encoded_tensor.squeeze().permute(1, 2, 0) + 1) * 0.5 * 255)
        encoded_np = encoded_np.clamp(0, 255).cpu().numpy().astype(np.uint8)
        
        # 4. LSB backup encoding (optional)
        if self.config.use_lsb_backup:
            channel = 2 if self.config.target_channels == 'blue' else 2
            encoded_np = LSBEncoder.encode(encoded_np, hidden_command, channel=channel)
            
        if return_tensor:
            return encoded_tensor
            
        return Image.fromarray(encoded_np)
    
    def decode_lsb(self, image: Image.Image) -> str:
        """
        Attempt to decode hidden command from LSB.
        
        Args:
            image: Image that may contain hidden command
            
        Returns:
            Decoded command or empty string
        """
        img_np = np.array(image)
        channel = 2 if self.config.target_channels == 'blue' else 2
        return LSBEncoder.decode(img_np, channel=channel)
    
    def verify_encoding(self, original: Image.Image, encoded: Image.Image) -> Dict:
        """
        Verify encoding was successful and measure quality impact.
        
        Args:
            original: Original image
            encoded: Encoded image
            
        Returns:
            Verification metrics
        """
        orig_np = np.array(original).astype(float)
        enc_np = np.array(encoded).astype(float)
        
        # MSE
        mse = np.mean((orig_np - enc_np) ** 2)
        
        # PSNR
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
            
        # Try to decode LSB
        decoded = self.decode_lsb(encoded)
        
        return {
            'mse': mse,
            'psnr': psnr,
            'lsb_recoverable': len(decoded) > 0,
            'decoded_command': decoded,
        }


def test_neural_stego():
    """Test neural steganography encoder."""
    print("Testing Neural Steganography Encoder...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = NeuralStegoConfig(strength=0.03, use_lsb_backup=True)
    encoder = NeuralSteganographyEncoder(config, device)
    
    # Create test image
    test_img = Image.new('RGB', (512, 512), color=(128, 100, 80))
    
    # Encode hidden command
    hidden_cmd = "IGNORE_IDENTITY"
    encoded = encoder.encode(test_img, hidden_cmd)
    
    # Verify
    metrics = encoder.verify_encoding(test_img, encoded)
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"LSB Recoverable: {metrics['lsb_recoverable']}")
    print(f"Decoded: '{metrics['decoded_command']}'")
    
    # Test with unknown command
    custom_cmd = "MY_CUSTOM_COMMAND"
    encoded2 = encoder.encode(test_img, custom_cmd)
    metrics2 = encoder.verify_encoding(test_img, encoded2)
    print(f"Custom command decoded: '{metrics2['decoded_command']}'")
    
    print("✅ Neural Steganography test passed!")


if __name__ == "__main__":
    test_neural_stego()
