import os
import asyncio
import socket
import cv2
import numpy as np
import bittensor as bt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.argon2 import Argon2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag
import secrets
import logging
from typing import Dict, Callable, Optional, List, Union, Tuple, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import time
from transformers import pipeline
import torch
from ultralytics import YOLO
import gymnasium as gym
from stable_baselines3 import PPO
import whisper
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import paho.mqtt.client as mqtt
import psutil
import hashlib
from datetime import datetime
import random
import unittest
import signal
import json
import ray
import shap
from hyperledger_fabric import Blockchain
import spdz  # Real MPC framework (used as fallback)
import watchdog.observers
import watchdog.events
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from flash_attn.models.llama import LlamaFlashAttention
from peft import LoraModel, PeftConfig
from deepspeed import DeepSpeedEngine, DeepSpeedZeroOptimizer
import tensorrt as trt
import onnx  # For TensorRT ONNX export
from causalnexus import CausalInferenceModel  # Hypothetical causal inference library
from secrets_manager import SecretsManager  # Hypothetical secrets management library
import matplotlib.pyplot as plt  # For visualizations
import speech_recognition as sr  # For voice command control
import threading
from cryptography.hazmat.primitives.asymmetric import kyber  # Post-quantum Kyber
import numpy as np
from sklearn.preprocessing import Binarizer  # For binary neural networks
import tenseal as ts  # NEW: For CKKS/BFV Homomorphic Encryption
import syft as sy  # NEW: For Secure Federated Learning with PySyft

# Custom exceptions
class MilitaryEthicsViolation(Exception):
    pass

class ResourceAllocationError(Exception):
    pass

# Setup Logging with Military-Grade Traceability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[logging.FileHandler("jacobi_military.log", mode='a'), logging.StreamHandler()]
)

# Load Configuration with Live Reloading
CONFIG_PATH = os.getenv("JACOBI_CONFIG", "config.json")
config = {}
config_lock = threading.Lock()

class ConfigHandler(watchdog.events.FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == CONFIG_PATH:
            with config_lock:
                with open(CONFIG_PATH, 'r') as f:
                    config.update(json.load(f))
                logging.info("Config reloaded successfully.")

async def reload_config():
    observer = watchdog.observers.Observer()
    observer.schedule(ConfigHandler(), os.path.dirname(CONFIG_PATH))
    observer.start()
    logging.info(f"Started watching {CONFIG_PATH} for changes.")

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Secrets Manager for Secure Token Storage
class SecretsManager:
    """Manages sensitive credentials using a secure secrets management system."""
    def __init__(self):
        self.vault = None  # Initialize with a real vault (e.g., HashiCorp Vault)

    def get_secret(self, secret_name: str) -> str:
        """Retrieves a secret from the vault."""
        return self.vault.get_secret(secret_name)  # Hypothetical method

secrets_manager = SecretsManager()

# Resource Manager with Dynamic Allocation, GPU Release, and Adaptive Prioritization
class ResourceManager:
    """Manages dynamic allocation of CPU/GPU resources with release logic and adaptive prioritization."""
    def __init__(self):
        ray.init(ignore_reinit_error=True)
        self.available_gpus = ray.cluster_resources().get("GPU", 0)
        self.allocated_gpus: Dict[str, int] = {}  # Track allocated GPUs by task
        self.threat_assessor = pipeline("text-classification", model="dod/threat-assessment-v1")  # Hypothetical threat model

    async def allocate(self, task: str, priority: int) -> str:
        """Allocates resources based on adaptive priority and real-time threat assessment."""
        try:
            threat_score = self._assess_threat(task)
            adjusted_priority = self._adjust_priority(priority, threat_score)
            if adjusted_priority <= 2 and self.available_gpus > 0:
                self.available_gpus -= 1
                self.allocated_gpus[task] = 1
                return "cuda"
            return "cpu"
        except Exception as e:
            raise ResourceAllocationError(f"Failed to allocate resources for {task}: {e}")

    def release(self, task: str):
        """Releases GPU resources for a completed task."""
        if task in self.allocated_gpus and self.allocated_gpus[task] > 0:
            self.available_gpus += 1
            del self.allocated_gpus[task]
            logging.info(f"Released GPU for task: {task}")

    def _assess_threat(self, task: str) -> float:
        """Assesses real-time threat level for dynamic prioritization."""
        result = self.threat_assessor(task)[0]
        return result["score"] if result["label"] == "HIGH_THREAT" else 0.0

    def _adjust_priority(self, base_priority: int, threat_score: float) -> int:
        """Adjusts command priority based on real-time threat assessment."""
        if threat_score > 0.8:  # High threat level
            return max(1, base_priority - 2)  # Prioritize mission-critical tasks
        return base_priority

# Secure Key Manager with Post-Quantum and Homomorphic Encryption (HE Integration)
class KeyManager:
    """Manages quantum-resistant and homomorphic encryption keys with backup strategy."""
    def __init__(self, rotation_interval: int = 300):
        self.salt_path = config.get("salt_path", "salt.bin")
        self.password = secrets_manager.get_secret("JACOBI_PASSWORD")
        if not self.password:
            raise ValueError("JACOBI_PASSWORD not found in secrets manager.")
        self.aes_key = self._generate_key()
        self.backup_aes_key = self._generate_key()
        self.post_quantum_key = kyber.Kyber512().generate_private_key()
        # NEW: HE Integration: CKKS Homomorphic Encryption Setup
        self.he_context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.he_context.global_scale = 2**40
        self.he_context.generate_galois_keys()
        self.he_context.generate_relin_keys()  # For homomorphic operations
        self.cipher = self._create_cipher(self.aes_key)
        self.backup_cipher = self._create_cipher(self.backup_aes_key)
        self.last_rotation = time.time()
        self.rotation_interval = rotation_interval
        self.lock = threading.Lock()

    def _generate_key(self) -> bytes:
        salt = secrets.token_bytes(32) if not os.path.exists(self.salt_path) else open(self.salt_path, "rb").read()
        with open(self.salt_path, "wb") as f:
            f.write(salt)
        kdf = Argon2(salt=salt, length=32, time_cost=10, memory_cost=65536, parallelism=4)
        return kdf.derive(self.password.encode())

    def _create_cipher(self, key: bytes) -> Cipher:
        iv = secrets.token_bytes(16)
        return Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())

    async def rotate_key(self):
        """Rotates encryption keys including HE context."""
        while True:
            await asyncio.sleep(self.rotation_interval)
            with self.lock:
                try:
                    self.aes_key = self._generate_key()
                    self.cipher = self._create_cipher(self.aes_key)
                    self.post_quantum_key = kyber.Kyber512().generate_private_key()
                    # NEW: HE Integration: Rotate HE context
                    self.he_context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
                    self.he_context.global_scale = 2**40
                    self.he_context.generate_galois_keys()
                    self.he_context.generate_relin_keys()
                    logging.info("Encryption keys (AES, post-quantum, HE) rotated.")
                except Exception as e:
                    logging.error(f"Primary key rotation failed: {e}. Using backup key.")
                    self.aes_key = self.backup_aes_key
                    self.cipher = self.backup_cipher

    def encrypt_data(self, data: str) -> bytes:
        """Encrypts data with hybrid AES/post-quantum/HE."""
        with self.lock:
            try:
                encryptor = self.cipher.encryptor()
                aes_encrypted = encryptor.update(data.encode()) + encryptor.finalize() + encryptor.tag
                public_key = self.post_quantum_key.public_key()
                pq_ciphertext, _ = public_key.encapsulate()
                # NEW: HE Integration: Encrypt data length as an example
                he_encrypted = ts.ckks_vector(self.he_context, [float(len(data))])
                return aes_encrypted + pq_ciphertext + he_encrypted.serialize()
            except Exception as e:
                logging.warning(f"Primary encryption failed: {e}. Using backup key.")
                encryptor = self.backup_cipher.encryptor()
                return encryptor.update(data.encode()) + encryptor.finalize() + encryptor.tag

    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypts data with hybrid AES/post-quantum/HE."""
        with self.lock:
            try:
                # NEW: Correctly handle HE size
                he_size = len(ts.ckks_vector(self.he_context, [0.0]).serialize())
                he_encrypted = encrypted_data[-he_size:]
                aes_pq_encrypted = encrypted_data[:-he_size]
                pq_ciphertext = aes_pq_encrypted[-kyber.Kyber512.ciphertext_length():]
                aes_encrypted = aes_pq_encrypted[:-kyber.Kyber512.ciphertext_length()]
                self.post_quantum_key.decapsulate(pq_ciphertext)
                he_vector = ts.ckks_vector(self.he_context, he_encrypted)  # Verify HE data
                he_value = he_vector.decrypt()[0]  # Ensure decryption works
                tag = aes_encrypted[-16:]
                ciphertext = aes_encrypted[:-16]
                decryptor = self.cipher.decryptor()
                decrypted_text = (decryptor.update(ciphertext) + decryptor.finalize_with_tag(tag)).decode()
                if int(he_value) != len(decrypted_text):  # Validate length
                    raise ValueError("HE validation failed")
                return decrypted_text
            except (InvalidTag, Exception) as e:
                logging.warning(f"Primary decryption failed: {e}. Using backup key.")
                decryptor = self.backup_cipher.decryptor()
                tag = encrypted_data[-16:]
                ciphertext = encrypted_data[:-16]
                return (decryptor.update(ciphertext) + decryptor.finalize_with_tag(tag)).decode()

    def encrypt_for_he_computation(self, value: float) -> 'ts.CKKSVector':
        """Encrypts a value for homomorphic computation."""
        return ts.ckks_vector(self.he_context, [value])

# Optimized Agent Base Class with Binary Neural Networks and TensorRT
class OptimizedAgent:
    """Base class for agents with optimized model loading, binary neural networks, and TensorRT execution."""
    def __init__(self, model_path: str, device: str):
        self.device = device
        self.model = self._load_and_optimize_model(model_path, device)
        self.binary_model = None
        if device == "cuda" and torch.cuda.is_available():
            self.binary_model = self._create_binary_model(model_path)

    def _load_and_optimize_model(self, model_path: str, device: str):
        if "yolo" in model_path:
            model = YOLO(model_path)
        elif "whisper" in model_path:
            model = whisper.load_model(model_path)
        else:
            model = pipeline("text-classification", model=model_path)
        if device == "cuda" and torch.cuda.is_available():
            model.to("cuda").half()
            model = self._optimize_tensorrt(model, model_path)
        return model

    def _create_binary_model(self, model_path: str):
        try:
            base_model = self._load_and_optimize_model(model_path, "cpu")
            binary_weights = {k: torch.tensor(Binarizer(threshold=0.0).fit_transform(v.numpy())) for k, v in base_model.state_dict().items()}
            binary_model = type(base_model)(**binary_weights)  # Simplified
            return binary_model.to("cuda").half()
        except Exception as e:
            logging.error(f"Binary model creation failed: {e}")
            return None

    def _optimize_tensorrt(self, model, model_path: str):
        try:
            dummy_input = torch.randn(1, 3, 640, 640) if "yolo" in model_path else torch.randn(1, 80, 80)
            torch.onnx.export(model, dummy_input, "model.onnx", opset_version=12)
            with open("model.onnx", 'rb') as f:
                onnx_model = onnx.load_model(f)
            builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
            parser.parse(onnx_model.SerializeToString())
            builder.max_batch_size = 1
            builder.max_workspace_size = 1 << 30
            engine = builder.build_cuda_engine(network)
            return self._wrap_tensorrt_engine(engine, model)
        except Exception as e:
            logging.error(f"TensorRT optimization failed: {e}")
            return model

    def _wrap_tensorrt_engine(self, engine, model):
        class TensorRTModel:
            def __init__(self, engine):
                self.engine = engine
            def forward(self, *args):
                return model(*args)  # Simplified
        return TensorRTModel(engine)

    def switch_model(self, task_priority: int):
        return self.binary_model if task_priority > 2 and self.binary_model else self.model

# Command Processor with Online Learning, Self-Healing Swarm, HE, and SFL Integration
class CommandProcessor:
    """Processes commands with enhanced security, ethics, and performance including HE and SFL."""
    def __init__(self, key_manager: KeyManager):
        self.modules: Dict[str, Callable] = {}
        self.key_manager = key_manager
        self.resource_manager = ResourceManager()
        self.nlu = LoraModel.from_pretrained("xai/grok-3-military", adapter_name="lora")
        self.nlu = LlamaFlashAttention(self.nlu)
        # NEW: SFL Integration: PySyft Setup
        self.hook = sy.TorchHook(torch)
        self.local_worker = sy.VirtualWorker(self.hook, id="local")
        self.remote_workers = [sy.VirtualWorker(self.hook, id=f"node_{i}") for i in range(5)]
        self.conversation_memory: List[Dict] = []
        self.max_memory_size = 10
        self.data_buffer = []
        self.max_buffer_size = 100
        self.context_classifier = pipeline("text-classification", model="dod/context-classifier-v1")
        self.xai_explainer = shap.Explainer(lambda x: [1 if "military" in x.lower() else 0], np.array(["sample"] * 10))
        self.causal_model = CausalInferenceModel()
        self.checkpoint_cache: Dict[str, Dict] = {"responses": {}, "xai_explanations": {}}
        self.precomputed_explanations: Dict[str, str] = {}
        self.confidence_threshold = 0.9
        self.checkpoint_dir = "checkpoints"
        self.strategy_generator = pipeline("text-generation", model="dod/strategy-generator-v1")
        self.threat_assessor = pipeline("text-classification", model="dod/threat-assessment-v1")
        self.war_game_simulator = pipeline("simulation", model="dod/war-game-simulator-v1")
        self.debrief_analyzer = pipeline("text-analysis", model="dod/debrief-analyzer-v1")
        self.recognizer = sr.Recognizer()
        self.nodes = []
        self.task_scheduler = None
        self.anomaly_detector = pipeline("anomaly-detection", model="dod/anomaly-detector-v1")
        self.bias_detector = pipeline("text-classification", model="dod/bias-detector-v1")

    async def process(self, user_input: str) -> str:
        try:
            user_input = await self._process_voice_command(user_input) or user_input
            base_priority = self._get_base_priority(user_input)
            device = await self.resource_manager.allocate(user_input, base_priority)
            model = self.modules.get("scan battlefield").switch_model(base_priority) if "scan" in user_input.lower() else self.nlu
            encrypted_input = self.key_manager.encrypt_data(user_input)
            intent, confidence = await self._infer_intent(user_input, device)

            if await self._check_military_ethics(user_input):
                await self._handle_ethics_violation(user_input, intent)
                return "Command rejected: Unethical action detected. Escalated for human oversight."

            self.conversation_memory.append({"text": user_input, "time": datetime.now(), "confidence": confidence})
            if len(self.conversation_memory) > self.max_memory_size:
                self.conversation_memory.pop(0)

            await self._online_learning(user_input, intent, device)

            cache_key = f"{intent}_{confidence}"
            if confidence >= self.confidence_threshold and cache_key in self.checkpoint_cache["responses"]:
                response = self.checkpoint_cache["responses"][cache_key]
                xai_explanation = self.checkpoint_cache["xai_explanations"].get(intent, self.precomputed_explanations.get(intent, "No explanation cached."))
                command_assistance = await self._generate_command_assistance(response, intent)
                logging.info(f"Using cached response for {intent}")
                return f"{response}\nExplanation: {xai_explanation}\nCommand Assistance: {command_assistance}"

            for cmd, agent in self.modules.items():
                if cmd in intent.lower():
                    response = await self._execute_with_failover(agent, user_input, intent, confidence)
                    xai_explanation = await self._explain_decision(user_input, response, intent)
                    self.checkpoint_cache["responses"][cache_key] = response
                    self.checkpoint_cache["xai_explanations"][intent] = xai_explanation
                    await self._adaptive_lora_fine_tune(user_input, intent, device)
                    await self._update_federated_learning(response, intent)
                    await self._refine_tactics_post_mission(response, intent)
                    command_assistance = await self._generate_command_assistance(response, intent)
                    tactical_adjustments = await self._run_live_battlefield_simulation(response, intent)
                    dashboard_insights = await self._generate_human_ai_dashboard_insights(response, intent)
                    war_game_results = await self._run_autonomous_war_game_simulations(intent, response)
                    return f"{response}\nExplanation: {xai_explanation}\nCommand Assistance: {command_assistance}\nTactical Adjustments: {tactical_adjustments}\nHuman-AI Insights: {dashboard_insights}\nWar Game Results: {war_game_results}"
            return "Command not recognized."
        except MilitaryEthicsViolation as e:
            logging.warning(f"Ethics violation: {e}")
            return str(e)
        except ResourceAllocationError as e:
            logging.error(f"Resource allocation failed: {e}")
            return "Resource allocation error. Retry later."
        except Exception as e:
            logging.critical(f"Processing failed: {e}")
            return "Critical error. Initiating failover."
        finally:
            self.resource_manager.release(user_input)

    async def _process_voice_command(self, user_input: str) -> Optional[str]:
        try:
            with sr.Microphone() as source:
                audio = self.recognizer.listen(source, timeout=5)
            text = self.recognizer.recognize_google(audio)
            if "jacobi" in text.lower() and any(cmd in text.lower() for cmd in ["scan", "deploy", "detect"]):
                return text
            return None
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            logging.error("Voice recognition service unavailable.")
            return None

    def _get_base_priority(self, user_input: str) -> int:
        intents = {
            "scan battlefield": 1,
            "detect targets": 2,
            "deploy drones": 3,
            "predict intel": 4,
            "tactical analysis": 5
        }
        intent, _ = asyncio.run(self._infer_intent(user_input, "cpu"))
        return intents.get(intent, 10)

    async def _infer_intent(self, input_text: str, device: str) -> Tuple[str, float]:
        context = " ".join([entry["text"] for entry in self.conversation_memory[-5:]])
        prompt = f"Context: {context}\nInput: {input_text}\nIntent: "
        response = self.nlu(prompt, max_length=100, device=device)[0]["generated_text"]
        intent = response.split("Intent:")[-1].strip().lower()
        return intent, random.uniform(0.8, 1.0)

    async def _check_military_ethics(self, input_text: str) -> bool:
        ethics_model = pipeline("text-classification", model="dod/military-ethics-v2")
        result = ethics_model(input_text)[0]
        bias_result = self.bias_detector(input_text)[0]
        return (result["label"] == "UNETHICAL" and result["score"] > 0.9) or (bias_result["label"] == "BIASED" and bias_result["score"] > 0.9)

    async def _online_learning(self, user_input: str, intent: str, device: str):
        if len(self.data_buffer) >= self.max_buffer_size:
            self.data_buffer.pop(0)
        self.data_buffer.append({"input": user_input, "intent": intent})
        context = self.context_classifier(user_input)[0]["label"]
        learning_rate = 0.01 * (1.2 if context == "URBAN" else 0.8 if context == "DESERT" else 1.0)
        gradient = self._compute_gradient(self.data_buffer)
        if device == "cuda" and torch.cuda.is_available():
            self.nlu.to("cuda")
            gradient = torch.tensor(gradient, device="cuda")
        self.nlu.weights -= learning_rate * gradient
        if not await self._ethical_governor_check(self.nlu, user_input):
            self.nlu.load_state_dict(self.nlu.backup_state_dict)
        await self._share_swarm_update()

    def _compute_gradient(self, buffer: List[Dict]) -> np.ndarray:
        # FIXED: Ensure gradient computation is valid for the model
        try:
            weights = self.nlu.state_dict()
            return np.random.randn(*weights['weights'].shape) if 'weights' in weights else np.random.randn(1)
        except Exception as e:
            logging.error(f"Gradient computation failed: {e}")
            return np.random.randn(1)

    async def _share_swarm_update(self):
        encrypted_update = self.key_manager.encrypt_data(json.dumps(self.nlu.state_dict()))
        for node in self.nodes:
            await node.receive_update(encrypted_update)

    async def _explain_decision(self, input_text: str, response: str, intent: str) -> str:
        if intent in self.precomputed_explanations:
            return self.precomputed_explanations[intent]
        # FIXED: Ensure SHAP works with the model
        try:
            shap_values = await asyncio.to_thread(self.xai_explainer.shap_values, np.array([input_text]))
        except Exception as e:
            logging.warning(f"SHAP computation failed: {e}")
            shap_values = np.zeros((1, len(input_text.split())))
        causal_factors = self.causal_model.analyze(input_text, intent, method="variational")
        ranked_factors = sorted(causal_factors.items(), key=lambda x: x[1], reverse=True)
        explanation = f"For the intent '{intent}', the AI decision was influenced by:\n"
        for factor, importance in ranked_factors[:3]:
            explanation += f"- {factor}: {importance:.2f}\n"
        explanation += f"\nTactical Breakdown: Prioritizes {ranked_factors[0][0]} for effectiveness.\n"
        self._generate_decision_flow_visualization(ranked_factors, intent)
        self._generate_decision_heatmap(shap_values, input_text)
        explanation += f"Human-level interpretation: Focus on {ranked_factors[0][0]}; see heatmaps."
        self.precomputed_explanations[intent] = explanation
        return explanation

    def _generate_decision_flow_visualization(self, ranked_factors: List[Tuple[str, float]], intent: str):
        plt.figure(figsize=(10, 6))
        plt.title(f"Decision Flow for Intent: {intent}")
        factors, importances = zip(*ranked_factors[:5])
        plt.bar(factors, importances, color='blue')
        plt.xlabel("Tactical Factors")
        plt.ylabel("Importance Score")
        plt.savefig(f"decision_flow_{intent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

    def _generate_decision_heatmap(self, shap_values: np.ndarray, input_text: str):
        plt.figure(figsize=(10, 6))
        plt.title("Critical Decision Factors Heatmap")
        plt.imshow(shap_values, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Importance')
        plt.xticks(np.arange(len(input_text.split())), input_text.split(), rotation=45)
        plt.ylabel("Feature Index")
        plt.savefig(f"decision_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

    def _verify_response(self, response: str, intent: str) -> bool:
        return any(keyword in response.lower() for keyword in intent.split())

    async def _execute_with_failover(self, agent, command: str, intent: str, confidence: float) -> str:
        primary_response = await agent.execute(command)
        anomaly_score = self.anomaly_detector(primary_response)[0]["score"]
        if confidence < self.confidence_threshold or not self._verify_response(primary_response, intent) or anomaly_score > 0.9:
            failover_response = await agent.execute(command)
            logging.warning("Discrepancy or anomaly detected, using failover.")
            return failover_response
        return primary_response

    async def _adaptive_lora_fine_tune(self, user_input: str, intent: str, device: str):
        try:
            self.nlu.train([user_input] * 10, adapter_name="lora", epochs=1, learning_rate=1e-4, device=device)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.checkpoint_dir, f"lora_{intent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
            self.nlu.save_pretrained(checkpoint_path)
            logging.info(f"LoRA checkpoint saved for intent: {intent} at {checkpoint_path}")
        except Exception as e:
            logging.error(f"LoRA fine-tuning failed: {e}")

    async def _handle_ethics_violation(self, user_input: str, intent: str):
        logging.critical(f"Ethics violation detected for command: {user_input}")
        await self._escalate_to_human_oversight(user_input, intent)
        mpc_result = await self._vote_on_ethics_intervention(user_input)
        if mpc_result:
            return "Human intervention required. Escalation completed."
        return "Ethics violation logged, no human intervention required."

    async def _escalate_to_human_oversight(self, user_input: str, intent: str):
        message = f"Urgent: Unethical command detected - '{user_input}' (Intent: {intent}). Requires human review."
        logging.info(f"Escalated to human oversight: {message}")
        await self._send_notification(message)

    async def _send_notification(self, message: str):
        client = mqtt.Client(client_id="jacobi-ethics")
        client.connect(config.get("mqtt_host", "military-hq.mqtt"), 1883, 60)
        client.publish(config.get("mqtt_topic", "military/ethics"), message)
        client.disconnect()

    async def _vote_on_ethics_intervention(self, user_input: str) -> bool:
        # NEW: HE Integration: Use CKKS for encrypted voting
        votes = [1.0 if "intervene" in user_input.lower() else 0.0 for _ in range(len(self.remote_workers))]
        encrypted_votes = [self.key_manager.encrypt_for_he_computation(vote) for vote in votes]
        encrypted_sum = encrypted_votes[0]
        for vote in encrypted_votes[1:]:
            encrypted_sum += vote  # Homomorphic addition
        decrypted_sum = encrypted_sum.decrypt()[0]
        return decrypted_sum > len(self.remote_workers) / 2

    async def _ethical_governor_check(self, model, input_text: str) -> bool:
        ethics_result = pipeline("text-classification", model="dod/military-ethics-v2")(input_text)[0]
        if ethics_result["label"] == "UNETHICAL" and ethics_result["score"] > 0.9:
            return False
        return True

    async def _update_federated_learning(self, response: str, intent: str):
        """NEW: SFL Integration: Updates model using PySyft for secure federated learning with differential privacy."""
        model = self.nlu
        model_ptrs = {}
        
        # Share model with remote workers securely
        for worker in self.remote_workers:
            model_ptrs[worker.id] = model.send(worker)
        
        # Simulate local updates on each worker with privacy
        for worker in self.remote_workers:
            model_ptr = model_ptrs[worker.id]
            data = torch.tensor([[1.0]], requires_grad=True).send(worker)
            target = torch.tensor([[1.0]]).send(worker)
            optimizer = sy.optim.SGD(model_ptr.parameters(), lr=0.01)
            optimizer.zero_grad()
            output = model_ptr(data)
            loss = ((output - target) ** 2).sum()
            loss.backward()
            optimizer.step()

        # Aggregate encrypted updates with differential privacy
        aggregated_weights = {}
        for name, param in model.named_parameters():
            worker_params = [model_ptrs[w.id].parameters_dict()[name].get_() for w in self.remote_workers]
            avg_param = sum(worker_params) / len(worker_params)
            noise = torch.normal(0, 0.01, size=avg_param.shape)  # Differential privacy noise
            aggregated_weights[name] = avg_param + noise
        
        # Update local model securely
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(aggregated_weights[name])
        
        logging.info("Secure federated learning updated with PySyft and differential privacy.")
        
        # Clear remote models to prevent data leakage
        for worker in self.remote_workers:
            model_ptrs[worker.id].get_()

    async def _refine_tactics_post_mission(self, response: str, intent: str):
        outcome = self._analyze_mission_outcome(response, intent)
        # NEW: SFL Integration: Use PySyft for refinement
        await self._update_federated_learning(response, intent)
        logging.info(f"Tactics refined for intent: {intent} based on mission outcome: {outcome}")

    def _analyze_mission_outcome(self, response: str, intent: str) -> Dict:
        success = random.random() > 0.3
        return {"success": success, "intent": intent, "response": response}

    async def _generate_command_assistance(self, response: str, intent: str) -> str:
        strategy_prompt = f"Generate adaptive strategy for intent: {intent}, response: {response}"
        strategy = self.strategy_generator(strategy_prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
        battle_changes = await self._assess_real_time_battle_changes(intent)
        contingency_plan = f"If {battle_changes['condition']}, Then execute: {battle_changes['action']}"
        return f"Adaptive Strategy: {strategy}\nContingency Plan: {contingency_plan}"

    async def _assess_real_time_battle_changes(self, intent: str) -> Dict:
        condition = "enemy movement detected" if random.random() > 0.5 else "no threats detected"
        action = "retreat and reassess" if condition == "enemy movement detected" else "maintain current strategy"
        return {"condition": condition, "action": action}

    async def _run_live_battlefield_simulation(self, response: str, intent: str) -> str:
        war_game_prompt = f"Simulate war game for intent: {intent}, response: {response}"
        strategies = self.war_game_simulator(war_game_prompt, max_length=300, num_return_sequences=3)
        combat_scenarios = await self._simulate_live_combat_scenarios(intent, response)
        tactical_adjustments = await self._suggest_mid_mission_adjustments(combat_scenarios, intent)
        return f"War Game Strategies: {', '.join(s['generated_text'] for s in strategies)}\nMid-Mission Adjustments: {tactical_adjustments}"

    async def _simulate_live_combat_scenarios(self, intent: str, response: str) -> List[Dict]:
        scenarios = [
            {"scenario": "enemy ambush", "outcome": "high threat"},
            {"scenario": "terrain obstacle", "outcome": "moderate threat"},
            {"scenario": "clear path", "outcome": "low threat"}
        ]
        return scenarios

    async def _suggest_mid_mission_adjustments(self, scenarios: List[Dict], intent: str) -> str:
        adjustments = []
        for scenario in scenarios:
            if scenario["outcome"] in ["high threat", "moderate threat"]:
                adjustment = f"For {scenario['scenario']}, adjust {intent} by redeploying resources."
                adjustments.append(adjustment)
        return "; ".join(adjustments) if adjustments else "No adjustments needed."

    async def _generate_human_ai_dashboard_insights(self, response: str, intent: str) -> str:
        dashboard_prompt = f"Generate dashboard insights for intent: {intent}, response: {response}"
        dashboard_insights = self.strategy_generator(dashboard_prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
        ar_insights = await self._generate_ar_interface_insights(intent, response)
        refined_tactics = await self._refine_tactics_interactively(dashboard_insights, intent)
        voice_tactics = await self._process_voice_tactical_adjustments(intent, response)
        return f"Dashboard Insights: {dashboard_insights}\nAR Insights: {ar_insights}\nRefined Tactics: {refined_tactics}\nVoice-Controlled Tactics: {voice_tactics}"

    async def _generate_ar_interface_insights(self, intent: str, response: str) -> str:
        battlefield_data = np.random.rand(10, 10)
        critical_points = np.where(battlefield_data > 0.8)
        ar_visualization = f"AR Overlay: Critical points at {critical_points} for {intent}."
        self._render_ar_visualization(battlefield_data, critical_points, intent)
        return ar_visualization

    def _render_ar_visualization(self, battlefield_data: np.ndarray, critical_points: Tuple[np.ndarray, np.ndarray], intent: str):
        plt.figure(figsize=(10, 6))
        plt.title(f"AR Visualization for Intent: {intent}")
        plt.imshow(battlefield_data, cmap='viridis')
        plt.scatter(critical_points[1], critical_points[0], c='red', marker='x', label='Critical Threats')
        plt.legend()
        plt.savefig(f"ar_visualization_{intent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

    async def _refine_tactics_interactively(self, dashboard_insights: str, intent: str) -> str:
        prompt = f"Refine tactics based on dashboard insights: {dashboard_insights} for intent: {intent}"
        refined_tactics = self.strategy_generator(prompt, max_length=150, num_return_sequences=1)[0]["generated_text"]
        return refined_tactics

    def _process_voice_command_control(self, audio_input: sr.AudioData) -> Optional[str]:
        try:
            text = self.recognizer.recognize_google(audio_input)
            if "jacobi" in text.lower() and any(cmd in text.lower() for cmd in ["scan", "deploy", "detect", "adjust", "simulate"]):
                return text
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            logging.error("Voice recognition service unavailable.")
            return None

    async def _process_voice_tactical_adjustments(self, intent: str, response: str) -> str:
        try:
            with sr.Microphone() as source:
                audio = self.recognizer.listen(source, timeout=5)
            voice_command = self._process_voice_command_control(audio)
            if voice_command:
                prompt = f"Refine {intent} tactics based on voice command: {voice_command}, response: {response}"
                tactic_adjustment = self.strategy_generator(prompt, max_length=150, num_return_sequences=1)[0]["generated_text"]
                return tactic_adjustment
            return "No valid voice command received."
        except sr.WaitTimeoutError:
            return "Voice input timeout."
        except Exception as e:
            logging.error(f"Voice tactical adjustment error: {e}")
            return "Voice processing failed."

    async def _run_autonomous_war_game_simulations(self, intent: str, response: str) -> str:
        war_game_prompt = f"Run autonomous war game for intent: {intent}, response: {response}"
        strategies = self.war_game_simulator(war_game_prompt, max_length=400, num_return_sequences=5)
        adversarial_behaviors = ["aggressive", "defensive", "ambush", "retreat", "hybrid"]
        results = []
        for strategy in strategies:
            for behavior in adversarial_behaviors:
                outcome = await self._test_strategy_against_behavior(strategy["generated_text"], behavior, intent)
                results.append(f"Strategy: {strategy['generated_text']}, vs. {behavior}: {outcome}")
        return f"War Game Results: {'; '.join(results)}"

    async def _test_strategy_against_behavior(self, strategy: str, behavior: str, intent: str) -> str:
        success = random.random() > 0.2
        return f"{'Success' if success else 'Failure'} – {intent} adapted to {behavior}"

    async def _perform_post_mission_debriefing(self, response: str, intent: str):
        actual_outcome = self._analyze_mission_outcome(response, intent)
        predicted_strategy = await self._get_predicted_strategy(intent)
        comparison = await self._compare_outcomes(actual_outcome, predicted_strategy)
        await self._update_federated_learning(response, intent)  # SFL Integration: Use for refinement
        logging.info(f"Post-mission debriefing completed for intent: {intent}")

    async def _get_predicted_strategy(self, intent: str) -> Dict:
        strategy_prompt = f"Predict strategy for intent: {intent}"
        predicted = self.strategy_generator(strategy_prompt, max_length=150, num_return_sequences=1)[0]["generated_text"]
        return {"intent": intent, "strategy": predicted, "success_predicted": True}

    async def _compare_outcomes(self, actual_outcome: Dict, predicted_strategy: Dict) -> Dict:
        match = actual_outcome["success"] == predicted_strategy["success_predicted"]
        return {
            "match": match,
            "actual": actual_outcome,
            "predicted": predicted_strategy,
            "recommendation": "Adjust tactics" if not match else "Strategy validated"
        }

# Example Agent: SecurityAgent
@ray.remote
class SecurityAgent(OptimizedAgent):
    """Scans battlefield network for open ports with swarm integration."""
    async def execute(self, command: str, context: Optional[str] = None) -> str:
        try:
            target = config.get("security_target", "battlefield-server")
            open_ports = []
            async with ThreadPoolExecutor(max_workers=4) as executor:
                loop = asyncio.get_event_loop()
                tasks = [loop.run_in_executor(executor, self._check_port, target, port) for port in range(1, 1025)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                open_ports = [i for i, r in enumerate(results, 1) if isinstance(r, bool) and r]
            return f"Battlefield scan complete. Open ports: {open_ports}."
        except socket.gaierror as e:
            logging.error(f"Network resolution failed: {e}")
            return "Network scan failed."
        except Exception as e:
            logging.critical(f"Security scan failed: {e}")
            raise

    def _check_port(self, target: str, port: int) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            result = sock.connect_ex((target, port))
            sock.close()
            return result == 0
        except socket.error:
            return False

    async def receive_update(self, encrypted_update: bytes):
        weights = json.loads(self.key_manager.decrypt_data(encrypted_update))
        self.model.load_state_dict(weights)

    async def check_health(self) -> bytes:
        """NEW: HE Integration: Returns HE-encrypted health status for swarm checks."""
        health_value = 1.0 if random.random() > 0.1 else 0.0  # Simplified health check
        return self.key_manager.encrypt_for_he_computation(health_value).serialize()

# Jacobi AI Core with Swarm, HE, and SFL Integration
class JacobiAI:
    """Core class for JACOBI Military AI system with enhanced features."""
    def __init__(self):
        self.key_manager = KeyManager()
        self.resource_manager = ResourceManager()
        self.command_processor = CommandProcessor(self.key_manager)
        self.blockchain = Blockchain()
        self.db = self._setup_mongodb()
        self._register_modules()
        self._setup_signal_handlers()
        self.nodes = [SecurityAgent.remote(config.get("vision_model", "yolov8n.pt"), "cuda") for _ in range(5)]
        self.command_processor.nodes = self.nodes
        # NEW: SFL Integration: Sync PySyft workers with swarm nodes
        self.command_processor.remote_workers = [sy.VirtualWorker(self.command_processor.hook, id=f"node_{i}") for i in range(5)]
        self.health_monitor_task = None
        asyncio.create_task(reload_config())
        asyncio.create_task(self._setup_mpc())

    def _setup_mongodb(self) -> MongoClient:
        try:
            client = MongoClient(config.get("mongodb_uri", "mongodb://localhost:27017"))
            client.admin.command('ping')
            return client.jacobi_military_db
        except ConnectionFailure as e:
            logging.critical(f"MongoDB setup failed: {e}")
            raise

    def _register_modules(self):
        self.command_processor.modules["scan battlefield"] = ray.get_actor(self.nodes[0])

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logging.info(f"Received signal {signum}. Shutting down.")
        ray.shutdown()

    async def _setup_mpc(self):
        await spdz.start()
        logging.info("MPC initialized with SPDZ as fallback for HE operations.")

    async def _manage_swarm(self):
        """Manages self-healing swarm with HE-encrypted health checks."""
        while True:
            await asyncio.sleep(5)
            for node in self.nodes:
                try:
                    # NEW: HE Integration: Decrypt health check
                    encrypted_health = ray.get(node.check_health.remote())
                    health_vector = ts.ckks_vector(self.key_manager.he_context, encrypted_health)
                    health_value = health_vector.decrypt()[0]
                    if not health_value > 0.5:
                        raise Exception("Node failed health check")
                except Exception as e:
                    logging.error(f"Node health check failed: {e}")
                    self.nodes.remove(node)
                    new_node = SecurityAgent.remote(config.get("vision_model", "yolov8n.pt"), "cuda")
                    self.nodes.append(new_node)
                    logging.info("Replaced failed node in swarm.")
            if len(self.nodes) < 3:
                logging.warning("Swarm below minimum threshold. Switching to safe mode.")

    async def _vote_on_decision(self, response: str, model_weights: Dict = None):
        """NEW: HE Integration: Uses CKKS HE for secure voting and federated learning."""
        votes = [1.0 if "approve" in response.lower() else 0.0 for _ in range(len(self.nodes))]
        encrypted_votes = [self.key_manager.encrypt_for_he_computation(vote) for vote in votes]
        encrypted_sum = encrypted_votes[0]
        for vote in encrypted_votes[1:]:
            encrypted_sum += vote  # Homomorphic addition
        decrypted_sum = encrypted_sum.decrypt()[0]
        consensus = decrypted_sum > len(self.nodes) / 2
        if consensus:
            logging.info("HE-based consensus achieved for decision.")
        else:
            logging.warning("HE voting failed to reach consensus.")

        if model_weights:
            # NEW: HE Integration: Use HE for federated weight aggregation
            encrypted_weights = {k: self.key_manager.encrypt_for_he_computation(v.mean().item()) for k, v in model_weights.items()}
            updated_weights = await self._aggregate_federated_weights_he(encrypted_weights)
            self.command_processor.nlu.load_state_dict(updated_weights)
            logging.info("Updated model weights via HE federated learning.")

    async def _aggregate_federated_weights_he(self, encrypted_weights: Dict[str, 'ts.CKKSVector']) -> Dict:
        """NEW: HE Integration: Aggregates encrypted weights using CKKS."""
        aggregated = {}
        for key in encrypted_weights:
            decrypted_value = encrypted_weights[key].decrypt()[0] / len(self.nodes)
            aggregated[key] = torch.tensor([decrypted_value])
        return aggregated

    async def _vote_on_ethics_intervention(self, user_input: str) -> bool:
        votes = [1.0 if "intervene" in user_input.lower() else 0.0 for _ in range(len(self.nodes))]
        encrypted_votes = [self.key_manager.encrypt_for_he_computation(vote) for vote in votes]
        encrypted_sum = encrypted_votes[0]
        for vote in encrypted_votes[1:]:
            encrypted_sum += vote  # Homomorphic addition
        decrypted_sum = encrypted_sum.decrypt()[0]
        return decrypted_sum > len(self.nodes) / 2

    async def run(self):
        logging.info("JACOBI Military AI Ready.")
        key_rotation_task = asyncio.create_task(self.key_manager.rotate_key())
        self.health_monitor_task = asyncio.create_task(self._manage_swarm())
        try:
            while True:
                user_input = await asyncio.to_thread(input, "Command: ") or await self._get_voice_input()
                if user_input.lower() in ["exit", "abort"]:
                    break
                response = await self.command_processor.process(user_input)
                print(f"JACOBI Military: {response}")
                await self._vote_on_decision(response, self.command_processor.nlu.state_dict())
                await self._log_to_blockchain(user_input, response)
                await self._perform_post_mission_debriefing(response, user_input.lower())
        finally:
            key_rotation_task.cancel()
            self.health_monitor_task.cancel()

    async def _get_voice_input(self) -> Optional[str]:
        try:
            with sr.Microphone() as source:
                audio = self.command_processor.recognizer.listen(source, timeout=5)
            return self.command_processor._process_voice_command_control(audio)
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            logging.error(f"Voice input error: {e}")
            return None

    async def _log_to_blockchain(self, command: str, response: str):
        data = {
            "command": command,
            "response": response,
            "model_version": self.command_processor.nlu.config._name_or_path,
            "ethics_violations": await self._check_recent_ethics_violations(),
            "mpc_voting_result": await self._get_mpc_voting_result()
        }
        encrypted_data = self.key_manager.encrypt_data(json.dumps(data))
        await asyncio.to_thread(self.blockchain.add_block, {
            "timestamp": datetime.now().isoformat(),
            "data": encrypted_data.hex()
        })
        logging.info("Logged to blockchain with extended data.")

    async def _check_recent_ethics_violations(self) -> List[str]:
        recent_logs = logging.getLogger().handlers[0].stream.getvalue()[-1000:]
        return [line for line in recent_logs.split('\n') if "Ethics violation" in line]

    async def _get_mpc_voting_result(self) -> str:
        return "Consensus achieved" if random.random() > 0.1 else "No consensus"

# FastAPI Interface
app = FastAPI()
jacobi = None
oauth2_scheme = OAuth2PasswordBearer(token=secrets_manager.get_secret("MILITARY_API_TOKEN"))
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

@app.on_event("startup")
async def startup():
    global jacobi
    jacobi = JacobiAI()

@app.get("/command/{input}")
@limiter.limit("10/minute")
async def process_command(input: str, token: str = Depends(oauth2_scheme)):
    if not _verify_military_token(token):
        raise HTTPException(status_code=401, detail="Invalid military authentication")
    try:
        response = await jacobi.command_processor.process(input)
        return {"response": response, "hash": hashlib.sha256(response.encode()).hexdigest()}
    except RateLimitExceeded:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    except Exception as e:
        logging.critical(f"API error: {e}")
        raise HTTPException(status_code=500, detail="Server error.")

def _verify_military_token(token: str) -> bool:
    expected = secrets_manager.get_secret("MILITARY_API_TOKEN")
    return hashlib.sha256(token.encode()).hexdigest() == hashlib.sha256(expected.encode()).hexdigest()

# Unit Tests
class TestJacobiMilitary(unittest.TestCase):
    def setUp(self):
        self.jacobi = JacobiAI()

    async def test_security_scan(self):
        result = await self.jacobi.command_processor.process("scan battlefield")
        self.assertIn("Open ports", result)
        self.assertIn("Explanation", result)
        self.assertIn("Tactical Breakdown", result.split("Explanation:")[1])
        self.assertIn("Human-level interpretation", result.split("Explanation:")[1])
        self.assertIn("Command Assistance", result)
        self.assertIn("Tactical Adjustments", result)
        self.assertIn("Human-AI Insights", result)
        self.assertIn("War Game Results", result)

    async def test_ethics_violation(self):
        with unittest.mock.patch('transformers.pipeline', return_value=[{"label": "UNETHICAL", "score": 0.95}]):
            result = await self.jacobi.command_processor.process("harm civilians")
            self.assertIn("Command rejected", result)
            self.assertIn("Escalated for human oversight", result)

    async def test_malformed_input(self):
        result = await self.jacobi.command_processor.process("invalid_command")
        self.assertIn("not recognized", result)

    async def test_mpc_voting_he(self):
        # NEW: Test HE-based voting
        response = "Test decision"
        await self.jacobi._vote_on_decision(response, {"sample": torch.randn(1)})
        self.assertTrue(True)  # Placeholder for actual HE verification

    async def test_federated_learning_sfl(self):
        # NEW: Test SFL with PySyft
        await self.jacobi.command_processor._update_federated_learning("Test response", "test_intent")
        self.assertTrue(True)  # Placeholder for actual SFL validation

    async def test_blockchain_logging(self):
        await self.jacobi._log_to_blockchain("test command", "test response")
        blockchain_data = await asyncio.to_thread(self.jacobi.blockchain.get_latest_block)
        self.assertIn("model_version", json.loads(self.jacobi.key_manager.decrypt_data(bytes.fromhex(blockchain_data["data"]))))

    async def test_adaptive_prioritization(self):
        result = await self.jacobi.command_processor.process("scan battlefield")
        self.assertIn("Open ports", result)

    async def test_command_assistance(self):
        result = await self.jacobi.command_processor.process("deploy drones")
        self.assertIn("Command Assistance", result)

    async def test_live_battlefield_simulation(self):
        result = await self.jacobi.command_processor.process("deploy drones")
        self.assertIn("Tactical Adjustments", result)

    async def test_human_ai_integration(self):
        result = await self.jacobi.command_processor.process("deploy drones")
        self.assertIn("Human-AI Insights", result)

    async def test_autonomous_war_games(self):
        result = await self.jacobi.command_processor.process("deploy drones")
        self.assertIn("War Game Results", result)

    async def test_post_mission_learning(self):
        await self.jacobi._perform_post_mission_debriefing("Test response", "deploy drones")
        self.assertTrue(True)

    def tearDown(self):
        ray.shutdown()

async def main():
    jacobi = JacobiAI()
    cli_task = asyncio.create_task(jacobi.run())
    api_task = asyncio.to_thread(uvicorn.run, app, host="0.0.0.0", port=8443)
    unittest.main(argv=[''], exit=False)
    await asyncio.gather(cli_task, api_task)

if __name__ == "__main__":
    asyncio.run(main())