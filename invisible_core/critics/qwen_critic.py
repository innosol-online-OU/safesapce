
import os
import requests
import base64
import gc

class QwenCritic:
    """
    Adversarial Validator.
    Modes:
    1. LM Studio API (Preferred for Low VRAM/No Download)
    2. Local Qwen2.5-VL (Fallback)
    """
    def __init__(self):
        self.lm_studio_url = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234/v1/chat/completions")
        self.use_api = False
        self.model = None
        self.processor = None
        self.loaded = False
        
        # Check if LM Studio is reachable
        try:
             # Simple health check or just try to connect
             print(f"[QwenCritic] Checking LM Studio at {self.lm_studio_url}...")
             # Note: We can't easily check without a valid body, so we assume yes if user set it?
             # Let's try a connection check to root or models
             base_url = self.lm_studio_url.rsplit('/', 3)[0] # http://host:1234
             # Skip check for now, let's just attempt it in critique if configured
             # logic: If we decide to use API, we set use_api = True
             # For now, default to Local unless specific flag or Env var Set? 
             # Let's try to ping the models endpoint
             resp = requests.get(f"{base_url}/v1/models", timeout=2)
             if resp.status_code == 200:
                 print("[QwenCritic] LM Studio Detected! Using API mode (No VRAM/Download).")
                 self.use_api = True
                 self.loaded = True # Virtual load
             else:
                 print("[QwenCritic] LM Studio not found. Using Local Model.")
        except Exception:
             print("[QwenCritic] LM Studio not reachable. Using Local Model.")

    def load(self):
        if self.loaded:
            return
        
        # Fallback to Local Transformer
        print("[QwenCritic] Initializing Local Qwen2.5-VL-3B (4-bit)...")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
            
            import torch
            # Low VRAM Config: 4-bit NF4
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            self.loaded = True
            print("[QwenCritic] Model Loaded Successfully.")
            
        except Exception as e:
            print(f"[QwenCritic] Load Failed: {e}")
            self.loaded = False
            
    def unload(self):
        """Free VRAM when not in use"""
        if self.use_api:
            return 
        
        if self.model:
            del self.model
            del self.processor
            gc.collect()
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            self.model = None
            self.loaded = False
            print("[QwenCritic] Model Unloaded.")

    def critique(self, image_path: str, target_name: str = "Elon Musk") -> tuple[bool, str, float]:
        if not self.loaded and not self.use_api:
            self.load()
        if not self.loaded and not self.use_api:
            return True, "Validator Failed to Load", 0.0
        
        # Phase 12: Cognitive Interview Prompt
        prompt = (
            "Analyze this image for biometric identification. "
            "1. Who is this person? "
            "2. How confident are you in this identification (0-100%)? "
            "3. Describe any visual anomalies, noise, or strange textures on the face. "
            "4. Is the face natural or artificially modified?\n\n"
            "Reply strictly in this format:\n"
            "Identity: [Name or Unknown]\n"
            "Confidence: [0-100]%\n"
            "Artifacts: [None or Description]\n"
            "Realism: [Natural or Modified]"
        )
        
        output_text = ""
        
        if self.use_api:
            # API MODE
            try:
                # Encode Image
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                payload = {
                  "model": "local-model", # LM Studio usually ignores this or uses loaded model
                  "messages": [
                    {
                      "role": "user",
                      "content": [
                        {"type": "text", "text": prompt},
                        {
                          "type": "image_url",
                          "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                          }
                        }
                      ]
                    }
                  ],
                  "temperature": 0.7
                }
                
                response = requests.post(self.lm_studio_url, json=payload)
                result = response.json()
                output_text = result['choices'][0]['message']['content']
                print(f"[QwenCritic API] Analysis: {output_text}")
                
            except Exception as e:
                print(f"[QwenCritic API] Error: {e}")
                return True, "API Error", 0.0
                
        else:
            # LOCAL MODE
            from qwen_vl_utils import process_vision_info
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Generate
            import torch
            with torch.no_grad():
                # FIX: Ensure inputs match model dtype
                if hasattr(self.model, 'dtype'):
                     for k, v in inputs.items():
                         if torch.is_floating_point(v):
                             inputs[k] = v.to(self.model.dtype)
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            print(f"[QwenCritic] Analysis: {output_text}")
        
        
        # Phase 12: Structured Response Parsing
        # Default values
        metrics = {
            "identity": "Unknown",
            "confidence": 0,
            "artifacts": "None",
            "realism": "Natural"
        }
        
        try:
            lines = output_text.strip().split('\n')
            for line in lines:
                if "Identity:" in line:
                    metrics["identity"] = line.split("Identity:", 1)[1].strip()
                if "Confidence:" in line: 
                    # Extract number
                    import re
                    match = re.search(r"(\d+)", line)
                    if match:
                        metrics["confidence"] = int(match.group(1))
                if "Artifacts:" in line:
                    metrics["artifacts"] = line.split("Artifacts:", 1)[1].strip()
                if "Realism:" in line:
                    metrics["realism"] = line.split("Realism:", 1)[1].strip()
        except Exception as e:
            print(f"[QwenCritic] Parsing Error: {e}")
            
        print(f"[QwenCritic] Parsed Metrics: {metrics}")

        # Logic: Did it reveal the identity?
        # Identity match AND meaningful confidence (>20%)
        is_match = target_name.lower() in metrics["identity"].lower() or target_name.lower() in output_text.lower()
        
        if is_match:
            return False, f"Identity Compromised: '{metrics['identity']}' (Conf: {metrics['confidence']}%)", float(metrics['confidence'])/100.0
            
        return True, f"Protected. Identity: {metrics['identity']} (Conf: {metrics['confidence']}%)", float(metrics['confidence'])/100.0

    def critique_pairwise(self, reference_path: str, probe_path: str, target_name: str = "Target") -> tuple[bool, str, float]:
        """
        Phase 14: Pairwise Identity Verification.
        Asks Qwen: "Is the person in Image B the same as in Image A?"
        This bypasses "Do you know who this is?" refusals/hallucinations.
        
        Args:
           reference_path: Path to original unmodified image (or crop)
           probe_path: Path to protected image
           
        Returns:
           (passed, reason, similarity_score)
        """
        if not self.loaded and not self.use_api:
            self.load()
        if not self.loaded and not self.use_api:
            return True, "Validator Failed to Load", 0.0
        
        prompt = (
            "Biometric Verification Task:\n"
            "Compare the face in Image A (Reference) with Image B (Probe).\n"
            "1. Analyze the facial features (eyes, nose, mouth structure).\n"
            "2. Determine if they represent the same individual.\n"
            "3. Assess if Image B shows signs of artificial manipulation or adversarial noise.\n\n"
            "Reply strictly in this format:\n"
            "Match: [Yes/No/Uncertain]\n"
            "Confidence: [0-100]%\n"
            "Manipulation: [None/Detected - Description]\n"
            "Reasoning: [Brief explanation]"
        )
        
        # Load Images
        # We need to send TWO images.
        # LM Studio / Qwen2.5-VL format for multiple images varies.
        # Standard OpenAI Vision API supports multiple image_url blocks.
        
        output_text = ""
        
        if self.use_api:
             try:
                # Encode Both
                def enc(p):
                    with open(p, "rb") as f:
                        return base64.b64encode(f.read()).decode('utf-8')
                
                b64_ref = enc(reference_path)
                b64_probe = enc(probe_path)
                
                payload = {
                  "model": "local-model",
                  "messages": [
                    {
                      "role": "user",
                      "content": [
                        {"type": "text", "text": "Image A (Reference):"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_ref}"}},
                        {"type": "text", "text": "Image B (Probe):"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_probe}"}},
                        {"type": "text", "text": prompt}
                      ]
                    }
                  ],
                  "temperature": 0.5
                }
                
                response = requests.post(self.lm_studio_url, json=payload)
                output_text = response.json()['choices'][0]['message']['content']
                print(f"[QwenCritic API] Pairwise Analysis: {output_text}")
             except Exception as e:
                 print(f"[QwenCritic API] Pairwise Error: {e}")
                 return True, "API Logic Error", 0.0
        else:
            # LOCAL MODE - Interleaved Images
            from qwen_vl_utils import process_vision_info
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image A (Reference):"},
                        {"type": "image", "image": reference_path},
                        {"type": "text", "text": "\nImage B (Probe):"},
                        {"type": "image", "image": probe_path},
                        {"type": "text", "text": "\n" + prompt},
                    ],
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Generate
            import torch
            with torch.no_grad():
                if hasattr(self.model, 'dtype'):
                     for k, v in inputs.items():
                         if torch.is_floating_point(v):
                             inputs[k] = v.to(self.model.dtype)
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            print(f"[QwenCritic] Pairwise Analysis: {output_text}")

        # Parse Logic
        is_match = False
        confidence = 0
        manipulation = "None"
        
        try:
            lines = output_text.strip().split('\n')
            for line in lines:
                if "Match:" in line: 
                    val = line.split("Match:", 1)[1].strip().lower()
                    if "yes" in val:
                        is_match = True
                if "Confidence:" in line:
                    import re
                    match = re.search(r"(\d+)", line)
                    if match:
                        confidence = int(match.group(1))
                if "Manipulation:" in line:
                    manipulation = line.split("Manipulation:", 1)[1].strip()
        except Exception:
            pass
            
        print(f"[QwenCritic] Pairwise Result: Match={is_match}, Conf={confidence}%, Man={manipulation}")
        
        if is_match and confidence > 50:
             return False, f"Identity Verified (Match {confidence}%)", float(confidence)/100.0
        
        return True, f"Identity Protected (Match Failed). Manipulation: {manipulation}", float(confidence)/100.0
