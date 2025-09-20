#!/usr/bin/env python3
"""
Step 1: LlamaGen Text-to-Image Inference Setup
Actual implementation for generating images from text prompts using LlamaGen.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image

import os
import sys
import time
import traceback
from PIL import Image
import numpy as np

# Import centralized database
from centralized_db import CentralizedDB

# Add LlamaGen modules to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'LlamaGen'))
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LlamaGenInference:
    def __init__(self, model_size="700M", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize LlamaGen for text-to-image generation.
        
        Args:
            model_size: "700M" (recommended for RTX 2080 Ti) or "3B" 
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model_size = model_size
        self.gpt_model = None
        self.vq_model = None
        self.t5_model = None
        
        # Model configuration
        self.image_size = 256
        self.downsample_size = 16
        self.codebook_size = 16384
        self.codebook_embed_dim = 8
        self.cls_token_num = 120
        self.cfg_scale = 7.5
        self.temperature = 1.0
        self.top_k = 2000
        self.top_p = 1.0
        
        print(f"Initializing LlamaGen {model_size} on {device}")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("Using CPU")
        
    def setup_models(self):
        """Load LlamaGen checkpoints and tokenizers"""
        base_path = "./pretrained_models"
        
        # Check if models exist
        vq_path = f"{base_path}/vq_ds16_t2i.pt"
        gpt_path = f"{base_path}/t2i_XL_stage1_256.pt" if self.model_size == "700M" else f"{base_path}/t2i_3B_stage1_256.pt"
        t5_path = f"{base_path}/t5-ckpt"
        
        if not os.path.exists(vq_path) or not os.path.exists(gpt_path):
            print("ERROR: Model checkpoints not found!")
            print(f"Expected paths:")
            print(f"  VQ Model: {vq_path}")
            print(f"  GPT Model: {gpt_path}")
            return False
            
        try:
            # Setup PyTorch settings
            torch.manual_seed(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.set_grad_enabled(False)
            
            # Load VQ model (image tokenizer)
            print("Loading VQ tokenizer...")
            self.vq_model = VQ_models["VQ-16"](
                codebook_size=self.codebook_size,
                codebook_embed_dim=self.codebook_embed_dim
            )
            self.vq_model.to(self.device)
            self.vq_model.eval()
            
            checkpoint = torch.load(vq_path, map_location="cpu")
            self.vq_model.load_state_dict(checkpoint["model"])
            del checkpoint
            print("VQ model loaded successfully!")
            
            # Load GPT model (text-to-image generator)
            print(f"Loading {self.model_size} GPT model...")
            precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            latent_size = self.image_size // self.downsample_size
            
            self.gpt_model = GPT_models["GPT-XL"](
                block_size=latent_size ** 2,
                cls_token_num=self.cls_token_num,
                model_type="t2i",
            ).to(device=self.device, dtype=precision)
            
            checkpoint = torch.load(gpt_path, map_location="cpu")
            if "model" in checkpoint:
                model_weight = checkpoint["model"]
            elif "module" in checkpoint:
                model_weight = checkpoint["module"]
            elif "state_dict" in checkpoint:
                model_weight = checkpoint["state_dict"]
            else:
                raise Exception("Unknown checkpoint format")
                
            self.gpt_model.load_state_dict(model_weight, strict=False)
            self.gpt_model.eval()
            del checkpoint
            print("GPT model loaded successfully!")
            
            # Load T5 text encoder using transformers directly
            print("Loading T5 text encoder...")
            from transformers import T5EncoderModel, AutoTokenizer
            
            try:
                self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
                self.t5_encoder = T5EncoderModel.from_pretrained(
                    "google/flan-t5-xl",
                    torch_dtype=precision,
                    device_map={"": self.device}
                ).eval()
                
                # Create a simple T5 embedder wrapper
                class SimpleT5Embedder:
                    def __init__(self, tokenizer, model, device, max_length=120):
                        self.tokenizer = tokenizer
                        self.model = model
                        self.device = device
                        self.max_length = max_length
                    
                    def get_text_embeddings(self, texts):
                        inputs = self.tokenizer(
                            texts,
                            max_length=self.max_length,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt'
                        )
                        
                        with torch.no_grad():
                            outputs = self.model(
                                input_ids=inputs['input_ids'].to(self.device),
                                attention_mask=inputs['attention_mask'].to(self.device)
                            )
                        
                        return outputs.last_hidden_state, inputs['attention_mask'].to(self.device)
                
                self.t5_model = SimpleT5Embedder(self.t5_tokenizer, self.t5_encoder, self.device)
                print("T5 text encoder loaded successfully!")
                
            except Exception as e:
                print(f"Failed to load T5 model: {e}")
                raise
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            traceback.print_exc()
            return False
    
    def generate_image(self, prompt, num_samples=1, cfg_scale=None, temperature=None):
        """
        Generate image from text prompt using actual LlamaGen inference.
        
        Args:
            prompt (str or list): Text description(s) of desired image(s)
            num_samples (int): Number of images to generate (for single prompt)
            cfg_scale (float): Classifier-free guidance scale
            temperature (float): Sampling temperature
            
        Returns:
            List of PIL Images
        """
        if self.gpt_model is None or self.vq_model is None or self.t5_model is None:
            print("Models not loaded! Call setup_models() first.")
            return None
        
        # Use instance defaults if not provided
        cfg_scale = cfg_scale or self.cfg_scale
        temperature = temperature or self.temperature
        
        # Handle single prompt vs list of prompts
        if isinstance(prompt, str):
            prompts = [prompt] * num_samples
        else:
            prompts = prompt
            num_samples = len(prompts)
            
        print(f"Generating {num_samples} image(s) for prompt(s): {prompts[:3]}{'...' if len(prompts) > 3 else ''}")
        
        try:
            with torch.no_grad():
                # Encode text prompts using T5
                print("Encoding text prompts...")
                caption_embs, emb_masks = self.t5_model.get_text_embeddings(prompts)
                
                # Process embeddings (left-padding for better results)
                print("Processing text embeddings...")
                new_emb_masks = torch.flip(emb_masks, dims=[-1])
                new_caption_embs = []
                for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                    valid_num = int(emb_mask.sum().item())
                    print(f'  Prompt {idx} token length: {valid_num}')
                    new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                    new_caption_embs.append(new_caption_emb)
                new_caption_embs = torch.stack(new_caption_embs)
                
                c_indices = new_caption_embs * new_emb_masks[:, :, None]
                c_emb_masks = new_emb_masks
                
                # Generate image tokens autoregressively
                print("Generating image tokens...")
                latent_size = self.image_size // self.downsample_size
                
                t1 = time.time()
                index_sample = generate(
                    self.gpt_model, c_indices, latent_size ** 2,
                    c_emb_masks,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    sample_logits=True,
                )
                sampling_time = time.time() - t1
                print(f"Token generation took {sampling_time:.2f} seconds")
                
                # Decode tokens to images using VQ model
                print("Decoding tokens to images...")
                qzshape = [len(c_indices), self.codebook_embed_dim, latent_size, latent_size]
                
                t2 = time.time()
                samples = self.vq_model.decode_code(index_sample, qzshape)
                decoder_time = time.time() - t2
                print(f"Image decoding took {decoder_time:.2f} seconds")
                
                # Convert to PIL images
                images = []
                samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
                samples = samples.clamp(0, 1)
                
                for i in range(samples.shape[0]):
                    img_tensor = samples[i]
                    img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    images.append(img)
                
                print(f"Generated {len(images)} images successfully")
                return images
                
        except Exception as e:
            print(f"Error during generation: {e}")
            traceback.print_exc()
            return None

def main():
    """Demo script for LlamaGen text-to-image generation"""
    # Initialize centralized database
    db = CentralizedDB()

    # Test prompts for adversarial research
    test_prompts = [
        "a cat sitting on a red chair",
        "a beautiful sunset over mountains",
        "a futuristic cityscape at night",
        "a portrait photo of a kangaroo wearing an orange hoodie"
    ]

    # Check which prompts need processing BEFORE loading models
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    prompts_to_process = []
    for prompt in test_prompts:
        cursor.execute("SELECT id FROM step1_inference WHERE prompt_text = ?", (prompt,))
        existing = cursor.fetchone()
        
        if existing:
            response = input(f"Prompt '{prompt}' already exists. Replace? [yes/No]: ").strip().lower()
            if response in ['yes', 'y']:
                prompts_to_process.append(prompt)
            else:
                print("Skipping prompt...")
        else:
            prompts_to_process.append(prompt)
    
    conn.close()
    
    if not prompts_to_process:
        print("No prompts to process. Exiting...")
        return

    # Initialize LlamaGen only if needed (700M recommended for RTX 2080 Ti)
    generator = LlamaGenInference(model_size="700M")

    # Try to load models
    print("Setting up models...")
    if not generator.setup_models():
        print("\nModel setup failed!")
        print("Make sure you have downloaded the pretrained models:")
        print("1. vq_ds16_t2i.pt")
        print("2. t2i_XL_stage1_256.pt")
        return

    print(f"\n{'='*60}")
    print("Starting image generation...")

    # Generate images and store in database
    for i, prompt in enumerate(prompts_to_process):
        print(f"\n[{i+1}/{len(prompts_to_process)}] Processing: '{prompt}'")
        print("-" * 50)

        start_time = time.time()
        images = generator.generate_image(prompt, num_samples=1)
        total_time = time.time() - start_time

        if images:
            # Ensure output directory exists
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "step1_inference")
            os.makedirs(output_dir, exist_ok=True)

            # Save generated images and store in database
            for j, img in enumerate(images):
                # Clean filename
                clean_prompt = ''.join(c for c in prompt if c.isalnum() or c in ' -_')[:30]
                clean_prompt = clean_prompt.replace(' ', '_')
                filename = os.path.join(output_dir, f"baseline_{i+1}_{clean_prompt}_{j}.png")
                img.save(filename)

                # Store in database
                config_params = {
                    "model_size": generator.model_size,
                    "image_size": generator.image_size,
                    "cfg_scale": generator.cfg_scale,
                    "temperature": generator.temperature,
                    "top_k": generator.top_k,
                    "top_p": generator.top_p
                }

                result_id = db.store_inference_result(
                    prompt=prompt,
                    image_path=filename,
                    generation_time=total_time,
                    model_size=generator.model_size,
                    config_params=config_params,
                    success=True
                )

                print(f"Saved: {filename} (took {total_time:.2f}s total)")
                print(f"Database ID: {result_id}")
        else:
            # Store failed generation in database
            result_id = db.store_inference_result(
                prompt=prompt,
                image_path="",
                generation_time=total_time,
                model_size=generator.model_size,
                success=False
            )
            print(f"Failed to generate image for prompt: '{prompt}' (Database ID: {result_id})")

    # Print database summary
    print(f"\n{'='*60}")
    print("Step 1 (LlamaGen Inference) completed successfully!")

    # Get and display results from database
    results = db.get_inference_results()
    print(f"Database contains {len(results)} inference results:")
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} '{result['prompt_text'][:40]}...' ({result['generation_time']:.2f}s)")

    print(f"Database location: {db.db_path}")

if __name__ == "__main__":
    main()