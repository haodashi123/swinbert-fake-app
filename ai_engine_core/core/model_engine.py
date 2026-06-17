import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import BertTokenizer, BertModel, AutoImageProcessor, SwinModel, AutoTokenizer, AutoModelForSeq2SeqLM
import requests
from io import BytesIO
import random
import os
import re
import json
import nltk
from nltk.corpus import stopwords
import string
import cv2
import pickle
import shap
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import LayerIntegratedGradients

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Helper for stopword filtering
def get_stopwords():
    try:
        stops = set(stopwords.words('english'))
    except:
        stops = set()
    # Add BERT special tokens and punctuation
    stops.update(['[cls]', '[sep]', '[pad]', '[unk]'])
    stops.update(list(string.punctuation))
    return stops

STOPWORDS = get_stopwords()

# ==========================================
# Helper Functions
# ==========================================
def reshape_transform_swin(tensor, height=7, width=7):
    # Handle tuple output from SwinBlock
    if isinstance(tensor, tuple):
        tensor = tensor[0]
        
    token_count = int(tensor.size(1))
    side = int(token_count ** 0.5)
    if side * side != token_count:
        side = height
    result = tensor.transpose(1, 2)
    result = result.reshape(tensor.size(0), -1, side, side)
    return result

# ==========================================
# Dynamic Network Definition
# ==========================================
class DynamicFusionNet(nn.Module):
    def __init__(self, config):
        super(DynamicFusionNet, self).__init__()
        self.config = config
        
        # --- Deep Stream Dimensions ---
        self.deep_input_dim = 0
        if config['use_text']: self.deep_input_dim += 768
        
        # Determine image dimension based on backbone
        if config['use_image']:
            backbone = config.get('visual_backbone', 'swin') # Default to Swin if not set
            if backbone == 'swin':
                self.deep_input_dim += 1024
            elif backbone == 'vit':
                self.deep_input_dim += 768
            else:
                pass

        if config['use_caption']: self.deep_input_dim += 768
        
        # --- Explicit Stream Removed ---
        
        # --- Deep Stream MLP ---
        if self.deep_input_dim > 0:
            self.deep_mlp = nn.Sequential(
                nn.Linear(self.deep_input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.deep_out_dim = 512
        else:
            self.deep_mlp = None
            self.deep_out_dim = 0
            
        # --- Fusion Layer ---
        self.fusion_dim = self.deep_out_dim 
        
        if self.fusion_dim == 0:
            self.classifier = nn.Linear(1, 2) 
        else:
            self.classifier = nn.Linear(self.fusion_dim, 2)
        
    def forward(self, text_emb, img_emb, cap_emb):
        deep_features = []
        
        # 1. Deep Stream Concatenation
        if self.config['use_text']:
            deep_features.append(text_emb)
        if self.config['use_image']:
            deep_features.append(img_emb)
        if self.config['use_caption']:
            deep_features.append(cap_emb)
            
        final_in = None
        
        # Process Deep Stream
        if deep_features and self.deep_mlp:
            deep_in = torch.cat(deep_features, dim=1)
            final_in = self.deep_mlp(deep_in)
        else:
            # Fallback
            final_in = torch.zeros((text_emb.shape[0], self.fusion_dim)).to(text_emb.device)
            
        logits = self.classifier(final_in)
        return logits


class RealTimeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"RealTimeDetector initialized on {self.device}")
        
        # Local model directory in this app version
        self.app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.local_model_dir = os.path.join(self.app_root, "models")
        
        # Load SHAP background data
        bg_path = os.path.join(self.app_root, "core", "shap_background.npy")
        if os.path.exists(bg_path):
            self.background_data = np.load(bg_path)
            print(f"Loaded SHAP background data: {self.background_data.shape}")
        else:
            # Try root if local not found
            root_bg = "core/shap_background.npy"
            if os.path.exists(root_bg):
                self.background_data = np.load(root_bg)
                print(f"Loaded SHAP background data from root: {self.background_data.shape}")
            else:
                print(f"Warning: SHAP background data not found. SHAP analysis will be disabled.")
                self.background_data = None
        
        # Initialize explainer state
        self.shap_explainer = None
        self._fusion_model_cache = {}
        self._image_cache = {}
        self._image_cache_keys = []

    def _get_shap_explainer(self):
        """Lazy initialization of SHAP explainer to ensure models are ready."""
        if self.shap_explainer is None and self.background_data is not None:
            # We need a model for the prediction function. 
            # The requirement implies using the "Ours (Swin)" model for SHAP.
            # We'll ensure it's loaded inside the predict wrapper.
            print("Initializing SHAP KernelExplainer...")
            self.shap_explainer = shap.KernelExplainer(self.predict_proba_fn, self.background_data)
        return self.shap_explainer

    def predict_proba_fn(self, data_numpy):
        """
        Wrapper for SHAP. Input: (N, 1792) numpy array.
        Output: (N,) numpy array of LOGITS for class 1 (Real).
        Switching to Logits avoids saturation issues with Softmax where probabilities
        are extremely close to 0 or 1, making SHAP values tiny and indistinguishable.
        """
        # Ensure base models and fusion model are loaded
        base_models = self._load_base_models()
        
        # Load Swin Fusion Model (Model C)
        config_c = {'use_text': True, 'use_image': True, 'use_caption': False, 'use_explicit': False, 'visual_backbone': 'swin'}
        model_c = self._load_fusion_model("models/model_text_image_swin.pth", config_c)
        model_c.eval()
        
        n_samples = data_numpy.shape[0]
        # Split features
        txt_part = data_numpy[:, :768]
        swin_part = data_numpy[:, 768:]
        
        txt_t = torch.tensor(txt_part, dtype=torch.float32).to(self.device)
        swin_t = torch.tensor(swin_part, dtype=torch.float32).to(self.device)
        cap_t = torch.zeros((n_samples, 768), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = model_c(txt_t, swin_t, cap_t)
            # Return LOGIT for Class 1 directly
            # Shape (N,)
            return logits[:, 1].cpu().numpy()

    @st.cache_resource
    def _load_base_models(_self):
        """
        Loads base feature extractors (BERT, Swin) once.
        """
        print("Loading base models...")
        device = _self.device
        allow_online = str(os.getenv("IGP_MODEL_ALLOW_ONLINE_DOWNLOAD", "0")).strip() in ("1", "true", "True")
        
        # 1. BERT
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=not allow_online)
        try:
            bert_model = BertModel.from_pretrained("bert-base-uncased", use_safetensors=True, low_cpu_mem_usage=True, local_files_only=not allow_online).to(device)
        except:
            print("Warning: Failed to load BERT with safetensors/low_mem. Fallback to default.")
            bert_model = BertModel.from_pretrained("bert-base-uncased", local_files_only=not allow_online).to(device)
        bert_model.eval()
        
        # 2. Swin
        swin_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224", local_files_only=not allow_online)
        try:
            swin_model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224", use_safetensors=True, low_cpu_mem_usage=True, local_files_only=not allow_online).to(device)
        except:
            print("Warning: Failed to load Swin with safetensors/low_mem. Fallback to default.")
            swin_model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224", local_files_only=not allow_online).to(device)
        swin_model.eval()
        
        return {
            'tokenizer': tokenizer,
            'bert': bert_model,
            'swin_processor': swin_processor,
            'swin': swin_model
        }

    @st.cache_resource
    def _load_zh_en_translator(_self):
        device = _self.device
        name = "Helsinki-NLP/opus-mt-zh-en"
        allow_online = str(os.getenv("IGP_TRANSLATION_ALLOW_ONLINE_DOWNLOAD", "0")).strip() in ("1", "true", "True")
        tok = AutoTokenizer.from_pretrained(name, local_files_only=not allow_online)
        model = AutoModelForSeq2SeqLM.from_pretrained(name, local_files_only=not allow_online).to(device)
        model.eval()
        return {"tokenizer": tok, "model": model}

    def _contains_cjk(self, text: str) -> bool:
        if not text:
            return False
        return re.search(r"[\u4e00-\u9fff]", text) is not None

    def _normalize_text(self, text: str, translate_zh: bool):
        if not translate_zh:
            return text, {"language": "raw", "translation_applied": False}
        if not self._contains_cjk(text):
            return text, {"language": "en_or_other", "translation_applied": False}

        try:
            translator = self._load_zh_en_translator()
            tok = translator["tokenizer"]
            model = translator["model"]
            src = text.strip()
            if not src:
                return text, {"language": "raw", "translation_applied": False}

            raw_parts = [p.strip() for p in re.split(r"(?<=[。！？!?；;])\s*|\n+", src) if p and p.strip()]
            if not raw_parts:
                raw_parts = [src]

            chunks = []
            current = ""
            for part in raw_parts:
                if not current:
                    current = part
                    continue
                if len(current) + len(part) + 1 <= 96:
                    current = f"{current} {part}"
                else:
                    chunks.append(current)
                    current = part
            if current:
                chunks.append(current)

            translated_parts = []
            for chunk in chunks:
                inputs = tok([chunk], return_tensors="pt", truncation=True, max_length=256).to(self.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=320, num_beams=4)
                piece = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
                translated_parts.append(piece if piece else chunk)

            translated = " ".join(x for x in translated_parts if x).strip()
            if translated:
                return translated, {
                    "language": "zh_translated",
                    "translation_applied": True,
                    "translated_text": translated,
                    "translation_chunks": len(chunks),
                }
            return text, {"language": "zh_failed", "translation_applied": False, "translation_error": "empty_translation_output"}
        except Exception as e:
            err = str(e).strip().replace("\n", " ")
            if len(err) > 240:
                err = err[:240]
            return text, {"language": "zh_failed", "translation_applied": False, "translation_error": err or "translator_unavailable"}

    def _load_fusion_model(self, model_path, config_override=None):
        """
        Loads a specific FusionNet checkpoint.
        Prioritizes local models/ directory if available.
        """
        # Resolve path
        target_path = model_path
        if not os.path.isabs(model_path):
            # Check local app_v2/models/ folder first
            local_path = os.path.join(self.local_model_dir, os.path.basename(model_path))
            if os.path.exists(local_path):
                target_path = local_path
            elif not os.path.exists(model_path):
                # If neither exists, and it's relative, it will fail in torch.load
                pass

        config = config_override or {}
        cache_key = f"{os.path.abspath(target_path)}::{json.dumps(config, sort_keys=True, ensure_ascii=False)}"
        cached = self._fusion_model_cache.get(cache_key)
        if cached is not None:
            return cached

        if not os.path.exists(target_path):
            # Create a dummy model if file missing (for demonstration/arena robustness)
            print(f"Warning: {target_path} not found. Creating random initialized model.")
            model = DynamicFusionNet(config_override).to(self.device)
            model.eval()
            self._fusion_model_cache[cache_key] = model
            return model
            
        try:
            checkpoint = torch.load(target_path, map_location=self.device)
            # Check if config is in checkpoint, else use override
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                config = checkpoint['config']
                state_dict = checkpoint['state_dict']
            else:
                config = config_override
                state_dict = checkpoint
            
            model = DynamicFusionNet(config).to(self.device)
            # Filter out explicit weights if they exist in checkpoint but not in new model
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            model.eval()
            self._fusion_model_cache[cache_key] = model
            return model
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            fallback = DynamicFusionNet(config_override).to(self.device)
            fallback.eval()
            self._fusion_model_cache[cache_key] = fallback
            return fallback

    def _get_image(self, image_source, fallback_black=True):
        image = None
        if image_source:
            try:
                if isinstance(image_source, str):
                    src = image_source.strip()
                    if src.startswith("http"):
                        cached = self._image_cache.get(src)
                        if cached is not None:
                            return cached.copy()
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        response = requests.get(src, headers=headers, timeout=(2, 4))
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                        self._image_cache[src] = image.copy()
                        self._image_cache_keys.append(src)
                        if len(self._image_cache_keys) > 32:
                            oldest = self._image_cache_keys.pop(0)
                            self._image_cache.pop(oldest, None)
                    else:
                        image = Image.open(src).convert("RGB")
                elif hasattr(image_source, 'read'):
                    if hasattr(image_source, "seek"):
                        image_source.seek(0)
                    if hasattr(image_source, "getvalue"):
                        image = Image.open(BytesIO(image_source.getvalue())).convert("RGB")
                    else:
                        image = Image.open(image_source).convert("RGB")
                else:
                    image = image_source.convert("RGB")
            except Exception as e:
                print(f"Image load error: {e}")
        if image is None and fallback_black:
            image = Image.new('RGB', (224, 224), color='black')
        return image

    def _safe_empty_cache(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _is_cuda_oom(self, error):
        return "out of memory" in str(error).lower()

    def _switch_runtime_device(self, target_device):
        target_device = torch.device(target_device)
        if str(self.device) == str(target_device):
            return

        try:
            base_models = self._load_base_models()
            for key in ("bert", "swin"):
                model = base_models.get(key)
                if model is not None:
                    model.to(target_device)
                    model.eval()
        except Exception as e:
            print(f"Base model device switch failed: {e}")

        try:
            translator = self._load_zh_en_translator()
            model = translator.get("model")
            if model is not None:
                model.to(target_device)
                model.eval()
        except Exception:
            pass

        for model in self._fusion_model_cache.values():
            try:
                model.to(target_device)
                model.eval()
            except Exception as e:
                print(f"Fusion model device switch failed: {e}")

        self.device = target_device
        self._safe_empty_cache()

    def _switch_to_cpu_fallback(self):
        print("CUDA OOM detected. Switching runtime to CPU fallback mode.")
        self._safe_empty_cache()
        self._switch_runtime_device("cpu")

    def _build_activation_heatmap(self, image, swin_processor, swin_model, device):
        if image is None:
            raise ValueError("no_image_source")

        def _render(target_device):
            swin_inputs = swin_processor(images=image, return_tensors="pt").to(target_device)
            with torch.no_grad():
                outputs = swin_model(**swin_inputs, output_hidden_states=True, return_dict=True)

            feats = getattr(outputs, "last_hidden_state", None)
            if feats is None:
                hidden_states = getattr(outputs, "hidden_states", None) or []
                feats = hidden_states[-1] if hidden_states else None
            if feats is None:
                raise ValueError("missing_swin_hidden_states")

            feats = feats[0].detach().float().cpu().numpy()
            token_count = int(feats.shape[0])
            side = int(token_count ** 0.5)
            if side * side != token_count and token_count > 1:
                token_count_wo_cls = token_count - 1
                side_wo_cls = int(token_count_wo_cls ** 0.5)
                if side_wo_cls * side_wo_cls == token_count_wo_cls:
                    feats = feats[1:]
                    side = side_wo_cls
            if side * side != int(feats.shape[0]):
                raise ValueError("invalid_swin_token_shape")

            activation_map = np.linalg.norm(feats, axis=1).reshape(side, side)
            activation_map = np.nan_to_num(activation_map)
            if activation_map.max() > activation_map.min():
                activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
            else:
                activation_map = np.zeros_like(activation_map)

            raw_width, raw_height = image.size
            activation_map = cv2.resize(activation_map, (raw_width, raw_height), interpolation=cv2.INTER_CUBIC)
            activation_map = np.power(np.clip(activation_map, 0, 1), 0.85)
            rgb_img = np.float32(image) / 255
            rgb_img = np.clip(rgb_img, 0, 1)
            visualization = show_cam_on_image(rgb_img, activation_map, use_rgb=True, image_weight=0.55)
            return Image.fromarray(visualization)

        try:
            return _render(device)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or getattr(device, "type", str(device)) != "cuda":
                raise
            self._safe_empty_cache()
            original_device = next(swin_model.parameters()).device
            swin_model.to("cpu")
            try:
                return _render(torch.device("cpu"))
            finally:
                swin_model.to(original_device)
                self._safe_empty_cache()

    def _predict_all_impl(self, text, image_source=None, with_shap=True, translate_zh=True):
        base_models = self._load_base_models()

        image = self._get_image(image_source, fallback_black=True)

        normalized_text, text_meta = self._normalize_text(text, translate_zh=translate_zh)
        inputs = base_models['tokenizer'](normalized_text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            text_emb = base_models['bert'](input_ids, mask).last_hidden_state[:, 0, :]

        swin_inputs = base_models['swin_processor'](images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            swin_out = base_models['swin'](**swin_inputs)
            swin_emb = swin_out.pooler_output

        cap_emb = torch.zeros((1, 768)).to(self.device)

        config_c = {'use_text': True, 'use_image': True, 'use_caption': False, 'use_explicit': False, 'visual_backbone': 'swin'}
        model_c = self._load_fusion_model("models/model_text_image_swin.pth", config_c)
        with torch.no_grad():
            logits_c = model_c(text_emb, swin_emb, torch.zeros_like(cap_emb))
            prob_c = torch.softmax(logits_c, dim=1)[0, 1].item()

        shap_scores = None
        if with_shap:
            try:
                explainer = self._get_shap_explainer()
                if explainer:
                    t_np = text_emb.cpu().numpy()
                    s_np = swin_emb.cpu().numpy()
                    sample_data = np.concatenate([t_np, s_np], axis=1)

                    shap_nsamples = int(os.getenv("IGP_SHAP_NSAMPLES", "120") or 120)
                    shap_nsamples = max(20, min(shap_nsamples, 300))
                    shap_values = explainer.shap_values(sample_data, nsamples=shap_nsamples)

                    if isinstance(shap_values, list):
                        shap_vals = shap_values[0]
                    else:
                        shap_vals = shap_values

                    s_val = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
                    text_score = np.sum(s_val[:768])
                    image_score = np.sum(s_val[768:])

                    total_abs = abs(text_score) + abs(image_score)
                    if total_abs > 1e-9:
                        text_pct = abs(text_score) / total_abs
                        image_pct = abs(image_score) / total_abs
                    else:
                        text_pct = 0.5
                        image_pct = 0.5

                    shap_scores = {
                        "text_score": float(text_score),
                        "image_score": float(image_score),
                        "text_pct": float(text_pct),
                        "image_pct": float(image_pct)
                    }
            except Exception as e:
                print(f"SHAP calculation error: {e}")

        return {
            "text_original": text,
            "text_used": normalized_text,
            "text_meta": text_meta,
            "model_c": {"prob": prob_c, "label": "Real" if prob_c > 0.5 else "Fake"},
            "shap_scores": shap_scores,
            "runtime": {
                "device": str(self.device),
                "shap_enabled": bool(with_shap),
            }
        }

    def predict_all(self, text, image_source=None, with_shap=True, translate_zh=True):
        """
        Runs inference on Swin model (Ours) only.
        """
        try:
            return self._predict_all_impl(text, image_source=image_source, with_shap=with_shap, translate_zh=translate_zh)
        except RuntimeError as e:
            if not self._is_cuda_oom(e) or getattr(self.device, "type", str(self.device)) != "cuda":
                raise
            self._switch_to_cpu_fallback()
            result = self._predict_all_impl(text, image_source=image_source, with_shap=False, translate_zh=translate_zh)
            runtime = result.get("runtime") or {}
            runtime["fallback_reason"] = "cuda_oom"
            result["runtime"] = runtime
            return result

    def explain(self, text, image_source=None, translate_zh=True):
        """
        Generates explanations for Swin (Ours).
        Returns:
            heatmap_swin: PIL Image (Grad-CAM overlay for Swin)
            tokens: List of (word, score) tuples for text attribution
        """
        print("Starting explanation generation...")
        base_models = self._load_base_models()
        device = self.device
        use_amp = (getattr(device, 'type', str(device)) == 'cuda')
        
        config_c = {'use_text': True, 'use_image': True, 'use_caption': False, 'use_explicit': False, 'visual_backbone': 'swin'}
        fusion_model_swin = self._load_fusion_model("models/model_text_image_swin.pth", config_c)
        fusion_model_swin.eval()
        base_models['swin'].eval()
        base_models['bert'].eval()
        
        # --- Prepare Inputs ---
        # 1. Image
        image = self._get_image(image_source, fallback_black=False)
            
        # 2. Text
        normalized_text, _ = self._normalize_text(text, translate_zh=translate_zh)
        inputs = base_models['tokenizer'](normalized_text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        input_ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)

        # 3. Dummy Caption (Disabled)
        cap_emb = torch.zeros((1, 768)).to(device)
        
        # --- Pre-calculate Full Model Prediction (Swin) for Consistent Explanation Target ---
        pred_idx = 0
        try:
            with torch.no_grad():
                text_emb = base_models['bert'](input_ids, mask).last_hidden_state[:, 0, :]
                if image is not None:
                    swin_inputs = base_models['swin_processor'](images=image, return_tensors="pt").to(device)
                    swin_out = base_models['swin'](**swin_inputs)
                    img_emb = swin_out.pooler_output
                else:
                    img_emb = torch.zeros((1, 1024)).to(device)
                logits = fusion_model_swin(text_emb, img_emb, cap_emb)
                pred_idx = torch.argmax(logits, dim=1).item()
                print(f"Explanation Target Class: {pred_idx}")
        except Exception as e:
            print(f"Error calculating target class: {e}")
            pred_idx = 0
        
        # ==========================================
        # 1. Swin Visual Explanation (EigenCAM)
        # ==========================================
        heatmap_swin = None
        cam = None
        try:
            if image is None:
                raise ValueError("no_image_source")
            class VisualWrapperSwin(nn.Module):
                def __init__(self, fusion, swin):
                    super().__init__()
                    self.fusion = fusion
                    self.swin = swin
                def forward(self, x):
                    swin_out = self.swin(x)
                    img_emb = swin_out.pooler_output
                    dummy_text = torch.zeros((x.size(0), 768)).to(device)
                    dummy_cap = torch.zeros((x.size(0), 768)).to(device)
                    logits = self.fusion(dummy_text, img_emb, dummy_cap)
                    return logits

            swin_backbone = base_models['swin']
            wrapper = VisualWrapperSwin(fusion_model_swin, swin_backbone)
            
            try:
                target_layers = [swin_backbone.encoder.layers[-1].blocks[-1]]
            except Exception:
                target_layers = [swin_backbone.layernorm]
            
            cam = EigenCAM(model=wrapper, target_layers=target_layers, reshape_transform=reshape_transform_swin)
            swin_inputs = base_models['swin_processor'](images=image, return_tensors="pt").to(device)
            pixel_values = swin_inputs['pixel_values']
            targets = [ClassifierOutputTarget(pred_idx)]
            
            grayscale_cam = cam(input_tensor=pixel_values, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Robustness: Handle potential NaNs and normalization
            grayscale_cam = np.nan_to_num(grayscale_cam)
            if grayscale_cam.max() > grayscale_cam.min():
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
            
            raw_width, raw_height = image.size
            grayscale_cam = cv2.resize(grayscale_cam, (raw_width, raw_height), interpolation=cv2.INTER_CUBIC)
            grayscale_cam = np.power(grayscale_cam, 0.75)
            
            # Ensure Image is Float32 [0, 1]
            rgb_img = np.float32(image) / 255
            rgb_img = np.clip(rgb_img, 0, 1)
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.45)
            heatmap_swin = Image.fromarray(visualization)
            
        except Exception as e:
            print(f"Grad-CAM (Swin) skipped/failed: {e}")
            self._safe_empty_cache()
            try:
                heatmap_swin = self._build_activation_heatmap(
                    image,
                    base_models['swin_processor'],
                    base_models['swin'],
                    device
                )
                print("Activation fallback heatmap generated.")
            except Exception as e2:
                print(f"Activation heatmap fallback failed: {e2}")
                heatmap_swin = None
        finally:
            try:
                del cam
            except Exception:
                pass
            self._safe_empty_cache()

        # ==========================================
        # 3. Text Explanation (Integrated Gradients)
        # ==========================================
        text_attributions = []
        try:
            class TextWrapper(nn.Module):
                def __init__(self, fusion, bert):
                    super().__init__()
                    self.fusion = fusion
                    self.bert = bert
                def forward(self, input_embeds):
                    outputs = self.bert(inputs_embeds=input_embeds)
                    text_emb = outputs.last_hidden_state[:, 0, :]
                    dummy_img = torch.zeros((input_embeds.size(0), 1024)).to(device)
                    dummy_cap = torch.zeros((input_embeds.size(0), 768)).to(device)
                    logits = self.fusion(text_emb, dummy_img, dummy_cap)
                    return logits

            bert_backbone = base_models['bert']
            wrapper_txt = TextWrapper(fusion_model_swin, bert_backbone)
            embeddings = bert_backbone.embeddings(input_ids)
            lig = LayerIntegratedGradients(wrapper_txt, bert_backbone.embeddings)
            attributions, delta = lig.attribute(inputs=embeddings, target=pred_idx, n_steps=10, return_convergence_delta=True)
            attributions = attributions.sum(dim=2).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            attr_scores = attributions.cpu().detach().numpy()
            tokens_raw = base_models['tokenizer'].convert_ids_to_tokens(input_ids[0])
            text_attributions = self._aggregate_tokens(tokens_raw, attr_scores)
                    
        except Exception as e:
            print(f"IG failed (Text): {e}")
            text_attributions = [("Error", 0.0)]
            
        # ==========================================
        # 4. Final Aggregation
        # ==========================================
        # Return full tuples (word, score) for the UI visualization
        tokens = [pair[0] for pair in text_attributions]
        
        return heatmap_swin, text_attributions

    def _aggregate_tokens(self, tokens, scores):
        """
        Aggregates sub-word tokens back to words and filters stopwords.
        """
        aggregated = []
        current_word = ""
        current_score = 0.0
        
        for token, score in zip(tokens, scores):
            # BERT subword token handling (##)
            if token.startswith("##"):
                current_word += token[2:]
                current_score += score # Sum scores for subwords
            else:
                if current_word:
                    aggregated.append((current_word, current_score))
                current_word = token
                current_score = score
                
        if current_word:
            aggregated.append((current_word, current_score))
            
        # --- Stopword Filtering & Renormalization ---
        filtered = []
        scores_only = []
        special_tokens = {'[cls]', '[sep]', '[pad]', '[unk]'}
        
        for word, score in aggregated:
            word_lower = word.lower()
            core_word = str(word).replace("##", "")
            has_text_char = re.search(r"[A-Za-z0-9\u4e00-\u9fff]", core_word) is not None
            
            # Filter out BERT special tokens
            if word_lower in special_tokens:
                continue

            # Check if stopword or purely punctuation
            if word_lower in STOPWORDS or (not has_text_char) or all(char in string.punctuation for char in core_word):
                filtered.append((word, 0.0)) # Force zero
            else:
                punct_like = sum(1 for ch in core_word if not (ch.isalnum() or ("\u4e00" <= ch <= "\u9fff")))
                ratio = punct_like / max(len(core_word), 1)
                adj_score = float(score) * (0.25 if ratio >= 0.5 else 1.0)
                filtered.append((word, adj_score))
                scores_only.append(abs(adj_score))
                
        # Re-normalize only non-zero scores to stretch contrast
        if scores_only and max(scores_only) > 0:
            max_s = float(max(scores_only))
            min_s = float(min(scores_only))
            range_s = max_s - min_s if max_s > min_s else 1.0
            
            final_aggregated = []
            for word, score in filtered:
                if score == 0.0:
                    final_aggregated.append((word, 0.0))
                else:
                    # Min-Max Scaling to [0.1, 1.0] to ensure visibility
                    norm_score = 0.1 + 0.9 * ((float(abs(score)) - min_s) / range_s)
                    # Restore sign if needed, though usually absolute importance is shown
                    final_aggregated.append((word, float(norm_score)))
            return final_aggregated
        else:
            # Ensure float types even if no normalization
            return [(w, float(s)) for w, s in filtered]

    @staticmethod
    def simulate_prediction():
        return {
            "model_c": {"prob": 0.55, "label": "Real"},
            "shap_scores": {"text_score": 0.1, "image_score": 0.1, "text_pct": 0.5, "image_pct": 0.5},
        }
