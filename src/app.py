#!/usr/bin/env python3

import os
import time
import requests
import argparse
from typing import Any, Literal, ClassVar
from dotenv import load_dotenv
import logging
from dataclasses import dataclass
from prettytable import PrettyTable
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


load_dotenv()

Second = float
GB = float

PROVIDER = Literal["huggingface", "vast"]


@dataclass
class vLLMConfig:
    model: str
    gpu_memory_utilization: float = 0.8
    max_num_batched_tokens: int = 2048
    download_dir: str = "/workspace/models"
    host: str = "127.0.0.1"
    port: int = 18000
    model_name_key: ClassVar[str] = "VLLM_MODEL"
    model_args_key: ClassVar[str] = "VLLM_ARGS"

    def as_args(self) -> list[str]:
        return [
            f"--gpu-memory-utilization={self.gpu_memory_utilization}",
            f"--max-num-batched-tokens={self.max_num_batched_tokens}",
            f"--download-dir={self.download_dir}",
            f"--host={self.host}",
            f"--port={self.port}",
        ]


@dataclass
class VastAIvLLMTemplate:
    # --image vastai/vllm:v0.8.1-cuda-12.4-pytorch-2.6.0-py312 --env '-p 1111:1111 -p 8080:8080 -p 8000:8000 -p 8265:8265 -e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e JUPYTER_DIR=/ -e DATA_DIRECTORY=/workspace/ -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8000:18000:/docs:vLLM API|localhost:8265:28265:/:Ray Dashboard|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal" -e VLLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B -e VLLM_ARGS="--max-model-len 8192 --enforce-eager --download-dir /workspace/models --host 127.0.0.1 --port 18000" -e RAY_ARGS="--head --port 6379 --dashboard-host 127.0.0.1 --dashboard-port 28265" -e RAY_ADDRESS=127.0.0.1:6379 -e USE_ALL_GPUS=true' --onstart-cmd 'entrypoint.sh' --disk 32 --ssh --direct

    vllm_config: vLLMConfig

    image: str = "vastai/vllm:v0.8.1-cuda-12.4-pytorch-2.6.0-py312"
    port_forwards: tuple[tuple[int, int], ...] = (
        (1111, 1111),
        (8080, 8080),
        (8000, 8000),
        (8265, 8265),
    )
    env_vars: tuple[tuple[str, str], ...] = (
        ("OPEN_BUTTON_PORT", "1111"),
        ("OPEN_BUTTON_TOKEN", "1"),
        ("JUPYTER_DIR", "/"),
        ("DATA_DIRECTORY", "/workspace/"),
        (
            "PORTAL_CONFIG",
            "localhost:1111:11111:/:Instance Portal|localhost:8265:28265:/:Ray Dashboard|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal",
        ),
        ("RAY_ARGS", "--head --port 6379 --dashboard-host 127.0.0.1 --dashboard-port 28265"),
        ("RAY_ADDRESS", "127.0.0.1:6379"),
        ("USE_ALL_GPUS", "true"),
    )

    onstart_cmd: str = "entrypoint.sh"
    disk: int = 32
    ssh: bool = True
    direct: bool = True

    def __post_init__(self):
        # If we have portal config than add the vllm bits to it
        # -- localhost:8000:18000:/docs:vLLM API

        # Check if we have a tuple with the key being the "PORTAL_CONFIG"
        portal_config = next((x for x in self.env_vars if x[0] == "PORTAL_CONFIG"), None)
        if portal_config:
            portal_config_values = portal_config[1].split("|")
            portal_config_values.append(f"{self.vllm_config.host}:{self.vllm_config.port}:/docs:vLLM")
            portal_config_values_str = "|".join(portal_config_values)

            # replace
            self.env_vars = tuple(
                (x if x[0] != "PORTAL_CONFIG" else ("PORTAL_CONFIG", portal_config_values_str)) for x in self.env_vars
            )


@dataclass
class Config:
    gpu_memory_buffer_factor: float = 1.2  # 20% buffer for GPU memory
    hf_api_token: str | None = os.getenv("HF_API_TOKEN")
    vast_api_token: str | None = os.getenv("VAST_API_TOKEN")
    hf_api_base_url: str = "https://huggingface.co/api/models"
    vast_api_base_url: str = "https://console.vast.ai/api/v0"
    download_speeds: ClassVar[dict[PROVIDER, float]] = {
        "huggingface": 0.125,  # 1 Gbps default
    }


config = Config()


@dataclass
class InputConfig:
    num_examples: int
    max_input_len: int
    max_output_len: int


@dataclass
class HFModelInfo:
    name: str
    parameters: dict[str, int]
    used_storage: GB
    embedding_dimension: int
    num_layers: int

    @property
    def total_parameters(self) -> int:
        return sum(self.parameters.values())

    @property
    def total_size_gb(self) -> GB:
        """Calculate the total size of the model in GB"""

        total_size = 0
        for key, value in self.parameters.items():
            if key == "BF16" or key == "FP16":
                total_size += value * 2
            elif key == "FP32":
                total_size += value * 4
            elif key == "FP64":
                total_size += value * 8
            else:
                # Default to 2 bytes for unrecognized types
                total_size += value * 2
                logger.warning(f"Unknown parameter type: {key}")
        return total_size / (1024**3)

    @property
    def quant_bits(self) -> int:
        """Find the most relevant quantization factor from the parameters"""

        QUANT_FACTORS = {
            "BF16": 16,
            "FP16": 16,
            "FP32": 32,
            "FP64": 64,
            "INT8": 8,
            "INT4": 4,
        }

        # Find the one that contributes the most to the weighted sum
        max_factor = max(self.parameters, key=lambda x: self.parameters[x] * QUANT_FACTORS.get(x, 16))
        return QUANT_FACTORS.get(max_factor, 16)

    def kv_cache_size(self, input: InputConfig, batch: int = 1) -> GB:
        """Calculate the KV cache size based on the model parameters"""

        # Explonation: (K + V) * embedd_dim * layers * batch * seq_len * quant
        return (
            2
            * self.embedding_dimension
            * self.num_layers
            * batch
            * (input.max_input_len + input.max_output_len)
            * self.quant_bits
            / 8
        ) / (1024**3)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HFModelInfo":
        return cls(
            name=data["id"],
            parameters=data["safetensors"]["parameters"],
            used_storage=data.get("usedStorage", 0),
            embedding_dimension=data.get("hidden_size", 4096),
            num_layers=data.get("num_hidden_layers", 32),
        )

    @classmethod
    def from_hf_model(cls, model_name: str, token: str) -> "HFModelInfo":
        # Make API request to HF API
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"https://huggingface.co/api/models/{model_name}", headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch model info: {response.text}")

        data = response.json()
        return cls.from_dict(data)


@dataclass
class VastAiGPUOffer:
    # Core identifiers
    offer_id: int
    machine_id: int

    # GPU specifications
    gpu_name: str
    gpu_ram: GB
    gpu_mem_bw: float  # Memory bandwidth in GB/s
    gpu_arch: str
    num_gpus: int

    # Performance metrics
    dlperf: float  # Deep learning performance score
    total_tflops: float  # in TFLOPS
    tflops_per_dphtotal: float
    dlperf_per_dphtotal: float

    # Pricing
    dph_total: float  # Total dollars per hour including storage costs

    # Connection specifications
    pcie_gen: float
    pcie_lanes: int
    pcie_bw: float  # PCIe bandwidth in GB/s

    # System specifications
    cpu_cores_effective: float
    cpu_ram: GB  # in MB
    disk_space: GB  # in GB

    # Network specifications
    inet_up: float  # Upload speed in GB/s
    inet_down: float  # Download speed in GB/s

    # Status indicators
    verified: bool
    rentable: bool
    rented: bool
    reliability: float

    # Location
    geolocation: str

    @property
    def id(self) -> str:
        return f"{self.machine_id}-{self.gpu_name}-{self.num_gpus}"

    @property
    def pci_speed(self) -> float:
        """Calculate PCIe transfer speed in GB/s based on generation and lanes"""
        return 0.25 * (2 ** (self.pcie_gen - 1)) * self.pcie_lanes

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VastAiGPUOffer":
        return cls(
            offer_id=data["id"],
            machine_id=data["machine_id"],
            gpu_name=data["gpu_name"],
            gpu_ram=data["gpu_ram"] / 1024,  # Convert to GB
            gpu_mem_bw=data["gpu_mem_bw"],  # Convert to GB/s
            gpu_arch=data["gpu_arch"],
            num_gpus=data["num_gpus"],
            dlperf=data["dlperf"],
            total_tflops=data["total_flops"],
            tflops_per_dphtotal=data["flops_per_dphtotal"],
            dlperf_per_dphtotal=data["dlperf_per_dphtotal"],
            dph_total=data["dph_total"],
            pcie_gen=data["pci_gen"],
            pcie_lanes=data["gpu_lanes"],
            pcie_bw=data["pcie_bw"],
            cpu_cores_effective=data["cpu_cores_effective"],
            cpu_ram=data["cpu_ram"] / 1024,  # Convert to GB
            disk_space=data["disk_space"] / 1024,  # Convert to GB
            inet_up=data["inet_up"] / 1024,  # Convert to GB/s
            inet_down=data["inet_down"] / 1024,  # Convert to GB/s
            verified=data["verification"] == "verified",
            rentable=data["rentable"],
            rented=data["rented"],
            reliability=data["reliability"],
            geolocation=data["geolocation"],
        )

    @classmethod
    def from_task(cls, model_info: HFModelInfo, input_config: InputConfig, n: int = 10) -> list["VastAiGPUOffer"]:

        # For now we assume that we use kv cache estimation for 1 batch
        kv_cache_size: GB = model_info.kv_cache_size(input_config)
        logger.info(f"KV cache size for batch of 1: {kv_cache_size:.2f} GB")

        minimum_vram = kv_cache_size * config.gpu_memory_buffer_factor

        query_common = {
            "verified": {"eq": True},  # Only verified sellers
            "external": {"eq": False},  # Only internal sellers
            "rentable": {"eq": True},  # Only rentable GPUs
            "rented": {"eq": False},  # Only available GPUs
            "type": "on-demand",  # TODO: in the future we could do the "bid" for spot instances
            "disable_bundling": True,  # Disable offer bundling
            "gpu_arch": {"eq": "nvidia"},
            "num_gpus": {"eq": 1},  # Only single GPU instances
            "gpu_ram": {"gte": minimum_vram * 1024},  # minimum VRAM in MB
            "limit": n,
        }

        query_flops_dph = {
            "order": [
                ["flops_per_dphtotal", "desc"],
                ["dph_total", "asc"],
                ["gpu_ram", "desc"],
            ],
        }

        query_dlperf_dph = {
            "order": [
                ["flops_per_dphtotal", "desc"],
                ["dph_total", "asc"],
                ["gpu_ram", "desc"],
            ],
        }

        query_mem = {
            "order": [
                ["gpu_ram", "desc"],
                ["flops_per_dphtotal", "desc"],
                ["dph_total", "asc"],
            ],
        }

        query_flops_total = {
            "order": [
                ["total_flops", "desc"],
                ["gpu_ram", "desc"],
                ["dph_total", "asc"],
            ],
        }

        query_dlperf_total = {
            "order": [
                ["dlperf", "desc"],
                ["gpu_ram", "desc"],
                ["dph_total", "asc"],
            ],
        }

        query_price = {
            "order": [
                ["dph_total", "asc"],
                ["gpu_ram", "desc"],
                ["total_flops", "desc"],
            ],
        }

        query_options = [
            query_flops_dph,
            query_dlperf_dph,
            query_mem,
            query_flops_total,
            query_dlperf_total,
            query_price,
        ]

        results: dict[int, dict] = {}
        for query in query_options:
            q = {**query_common, **query}

            # Exponential backoff for rate limiting
            max_retries = 5
            base_delay = 1.0  # seconds
            for attempt in range(max_retries):
                response = requests.put(
                    url=f"{config.vast_api_base_url}/search/asks/",
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {config.vast_api_token}",
                        "Content-Type": "application/json",
                    },
                    json={"q": q},
                )

                if response.status_code != 429:  # Not rate limited
                    break

                delay = base_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
            else:
                logger.error("Max retries exceeded for rate limiting")
                raise Exception("Failed to search GPUs: Rate limit exceeded")

            if response.status_code != 200:
                logger.error(f"Failed to search GPUs: {response.text}")
                raise Exception(f"Failed to search GPUs: {response.text}")

            gpu_offers = [VastAiGPUOffer.from_dict(offer) for offer in response.json().get("offers", [])]

            results.update({offer.id: offer for offer in gpu_offers})

        logger.info(
            f"Found {len(results)}/{n * len(query_options)} unique offers"
        )

        return list(results.values())


@dataclass
class Offer:
    gpu_offer: VastAiGPUOffer
    model_info: HFModelInfo
    input_config: InputConfig

    @property
    def total_cost(self) -> float:
        """Estimate total cost for inference in dollars"""
        return self.total_time * self.gpu_offer.dph_total / 3600

    @property
    def total_time(self) -> Second:
        """Estimate total time for inference in seconds"""

        # Considering setup for model download and upload
        # then every generation with the batch size

        rounds = (self.input_config.num_examples + self.batch_size - 1) // self.batch_size

        total_time: Second = 0
        total_time += self.model_download_time_estimate
        total_time += self.upload_time_estimate
        total_time += self.ttft * rounds
        total_time += self.tpot * rounds * self.input_config.max_output_len

        return total_time

    @property
    def model_download_time_estimate(self) -> Second:
        """Estimate model download time in seconds from huggingface"""
        return min(self.gpu_offer.inet_down, config.download_speeds["huggingface"]) * self.model_info.total_size_gb

    @property
    def upload_time_estimate(self) -> Second:
        """Estimate time to upload model to GPU memory in seconds"""
        return self.model_info.total_size_gb / self.gpu_offer.pci_speed

    @property
    def ttft(self) -> Second:
        """Estimate Time to First Token in seconds

        For now we assume that the calculation of the kvcache is memory bound
        """
        flops = self.gpu_offer.total_tflops * 1e12  # Convert to actual FLOPS from TFLOPS
        gpu_mbw = self.gpu_offer.gpu_mem_bw * 1e9  # Convert to bytes/sec
        q = self.model_info.quant_bits / 8  # Bytes per parameter
        n = self.model_info.total_parameters

        # Prefill calculations
        prefill_compute = q * n * self.input_config.max_input_len
        prefill_memory = q * n

        # TTFT calculation
        compute_time = prefill_compute / flops
        memory_time = prefill_memory / gpu_mbw

        return compute_time + memory_time

    @property
    def tpot(self) -> Second:
        """Estimate Time Per Output Token in seconds"""
        flops = self.gpu_offer.total_tflops * 1e12
        gpu_mbw = self.gpu_offer.gpu_mem_bw * 1e9
        q = self.model_info.quant_bits / 8
        n = self.model_info.total_parameters

        # Decode calculations
        decode_compute = q * n * 1
        decode_memory = q * n

        # TPOT calculation
        compute_time = decode_compute / flops
        memory_time = decode_memory / gpu_mbw

        return compute_time + memory_time

    @property
    def batch_size(self) -> int:
        """We will compute the optimal batch size based on the KV cache size"""

        kv_cache_size = self.model_info.kv_cache_size(self.input_config)
        logger.info(f"KV cache size: {kv_cache_size:.2f} GB")

        if (kv_cache_size + self.model_info.total_size_gb) * config.gpu_memory_buffer_factor > self.gpu_offer.gpu_ram:
            logger.error("Model does not fit in GPU memory")
            raise Exception("Model does not fit in GPU memory")

        # How many kv-caches can we fit in the GPU memory
        vram_free = self.gpu_offer.gpu_ram - self.model_info.total_size_gb * config.gpu_memory_buffer_factor
        max_batch_size = int(vram_free / kv_cache_size)

        logger.info(f"Maximum possible batch size: {max_batch_size}")

        # We will return the best batch size for the input
        return min(max_batch_size, self.input_config.num_examples)

    @classmethod
    def from_task(cls, model_info: HFModelInfo, input_config: InputConfig, gpu_offer: VastAiGPUOffer) -> "Offer":

        return cls(
            gpu_offer=gpu_offer,
            model_info=model_info,
            input_config=input_config,
        )


class OfferManager:
    def __init__(self, model_info: HFModelInfo, input_config: InputConfig):
        self.model_info = model_info
        self.input_config = input_config
        self.offers: list[Offer] = []

    def search_best_offers(self, n: int = 10) -> list[Offer]:
        """Search for the best GPU offers based on the model and input requirements"""
        logger.info(
            f"Searching for best GPU offers for {self.model_info.total_parameters:,} parameter model"
        )
        logger.info(
            f"Input config: {self.input_config.num_examples} examples, {self.input_config.max_input_len} input tokens, {self.input_config.max_output_len} output tokens"
        )

        # Get GPU offers from VastAI
        gpu_offers = VastAiGPUOffer.from_task(self.model_info, self.input_config, n)

        # Create Offer objects for each GPU offer
        self.offers = [Offer.from_task(self.model_info, self.input_config, gpu_offer) for gpu_offer in gpu_offers]

        # Sort by total cost (cheapest first)
        self.offers.sort(key=lambda x: x.total_cost)

        return self.offers

    def display_offers(self, limit: int = 10) -> None:
        """Display the best offers in a nice terminal table"""
        if not self.offers:
            logger.warning("No offers to display. Run search_best_offers first.")
            return

        table = PrettyTable()
        table.field_names = ["Rank", "Offer ID", "GPU", "VRAM (GB)", "$/hr", "Est. Cost ($)", "Est. Time", "Batch Size"]

        for i, offer in enumerate(self.offers[:limit], 1):
            # Format time as minutes:seconds
            total_mins = int(offer.total_time // 60)
            total_secs = int(offer.total_time % 60)
            time_fmt = f"{total_mins}m {total_secs}s"

            table.add_row(
                [
                    i,
                    offer.gpu_offer.offer_id,
                    offer.gpu_offer.gpu_name,
                    f"{offer.gpu_offer.gpu_ram:.1f}",
                    f"{offer.gpu_offer.dph_total:.4f}",
                    f"{offer.total_cost:.4f}",
                    time_fmt,
                    offer.batch_size,
                ]
            )

        print(table)

        # Print summary for the best offer
        best = self.offers[0]
        print("\nBest offer details:")
        print(
            f"  GPU: {best.gpu_offer.gpu_name} with {best.gpu_offer.gpu_ram:.1f} GB VRAM"
        )
        print(
            f"  Performance: {best.gpu_offer.total_tflops:.1f} TFLOPS, {best.gpu_offer.gpu_mem_bw:.1f} GB/s memory bandwidth"
        )
        print(
            f"  Network: ↓ {best.gpu_offer.inet_down:.2f} GB/s, ↑ {best.gpu_offer.inet_up:.2f} GB/s"
        )
        print(
            f"  Estimated download time: {best.model_download_time_estimate:.1f}s"
        )
        print(
            f"  Estimated TTFT: {best.ttft:.3f}s, TPOT: {best.tpot*1000:.2f}ms"
        )
        print(
            f"  Cost: ${best.gpu_offer.dph_total:.4f}/hr, Est. total: ${best.total_cost:.4f}"
        )

    def accept_offer(self, offer: Offer, template: VastAIvLLMTemplate) -> None:
        """Accept the offer and start the instance"""
        # url = "https://console.vast.ai/api/v0/asks/{id}/"
        #
        # payload = json.dumps({
        #    "body": {
        #       "disk": 0,
        #       "extra_env": {},
        #       "target_state": "string<running | stopped>"
        #    }
        # })
        # headers = {
        #    'Accept': 'application/json',
        #    'Content-Type': 'application/json'
        # }
        #

        template.direct

        accpet_url = f"{config.vast_api_base_url}/asks/{offer.gpu_offer.offer_id}/"
        payload = {
            "body": {
                "image": template.image,
                "disk": 0,
                "extra_env": {k: v for k, v in template.env_vars},
                "target_state": "running",
                "runtype": "ssh",
                "onstart": template.onstart_cmd,

            }
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.vast_api_token}",
        }


        # Convert the request to a curl command

        # Format the payload for curl
        payload_str = json.dumps(payload)

        # Build the curl command
        curl_cmd = f"curl -X PUT {accpet_url}"

        # Add headers
        for key, value in headers.items():
            curl_cmd += f" -H '{key}: {value}'"

        # Add payload
        curl_cmd += f" -d '{payload_str}'"

        # Print the curl command
        print(f"Curl command to accept offer:\n{curl_cmd}")

        # Actually make the request
        response = requests.put(
            url=accpet_url,
            headers=headers,
            json=payload,
        )

        if response.status_code != 200:
            logger.error(f"Failed to accept offer: {response.text}")
            raise Exception(f"Failed to accept offer: {response.text}")

        data = response.json()
        if not data.get("success", False):
            logger.error(f"Failed to accept offer: {data}")
            raise Exception(f"Failed to accept offer: {data}")

        contract_id = data.get("new_contract")
        logger.info(f"Instance created successfully with contract ID: {contract_id}")

        # // Instance created successfully 
        # {
        #   "success": true,
        #   "new_contract": 1234568
        # }





def main():
    # Load environment variables from .env file

    parser = argparse.ArgumentParser(description="Find the best GPU for running a Hugging Face model")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("HF_MODEL_NAME") or "meta-llama/CodeLlama-7b-Instruct-hf",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face API token (overrides environment variable)",
    )
    parser.add_argument(
        "--vast-token",
        type=str,
        help="Vast.ai API token (overrides environment variable)",
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=1024,
        help="Maximum number of input tokens",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=1024,
        help="Maximum number of output tokens to generate",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100 * 10 * 20,
        help="Number of examples to process in parallel (affects batch size calculations)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Number of top offers to display",
    )

    args = parser.parse_args()

    # Update config with CLI arguments if provided
    if args.hf_token:
        config.hf_api_token = args.hf_token
    if args.vast_token:
        config.vast_api_token = args.vast_token

    if not config.hf_api_token:
        raise ValueError("Hugging Face API token is required. Set HF_API_TOKEN in .env file or use --hf-token")

    if not config.vast_api_token:
        raise ValueError("Vast.ai API token is required. Set VAST_API_TOKEN in .env file or use --vast-token")

    # Create input configuration
    input_config = InputConfig(
        num_examples=args.num_examples,
        max_input_len=args.input_tokens,
        max_output_len=args.output_tokens,
    )

    print(f"Input config: {input_config}")

    # Get model information from Hugging Face
    logger.info(f"Fetching model information for {args.model}")
    model_info = HFModelInfo.from_hf_model(args.model, config.hf_api_token)

    # Create offer manager and search for best offers
    offer_manager = OfferManager(model_info, input_config)
    offer_manager.search_best_offers()

    # Display the best offers
    offer_manager.display_offers(limit=args.limit)

    model = offer_manager.offers[0].model_info.name

    vllm_config = vLLMConfig(model=model)
    template = VastAIvLLMTemplate(vllm_config)

    print(f"Do you want to accept the best offer? (y/n)")
    accept = input().strip().lower()
    if accept == "y":
        offer_manager.accept_offer(offer_manager.offers[0], template)
    else:
        print("Offer not accepted.")


if __name__ == "__main__":
    main()
