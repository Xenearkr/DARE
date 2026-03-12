# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import socket
from typing import Any, List, Optional

import numpy as np
import torch
import torch.distributed as dist


def broadcast_pyobj(
    data: List[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = False,
):
    """from https://github.com/sgl-project/sglang/blob/844e2f227ab0cce6ef818a719170ce37b9eb1e1b/python/sglang/srt/utils.py#L905

    Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu_device else "cpu"
    )

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)

            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            ).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data


def get_ip() -> str:
    """Get the local IP address for distributed communication.

    This function tries multiple methods to get a usable IP address:
    1. Check environment variables (MASTER_ADDR, COORDINATOR_ADDR, HOST_IP)
    2. Connect to a remote address and get local socket address
    3. Get hostname and resolve it

    Returns:
        str: The IP address to use for distributed communication
    """
    # Check environment variables first
    for env_var in ["MASTER_ADDR", "COORDINATOR_ADDR", "HOST_IP", "HOSTNAME"]:
        addr = os.getenv(env_var)
        if addr and addr not in ["localhost", "127.0.0.1", "0.0.0.0"]:
            return addr

    # Try to get IP by connecting to a remote address
    try:
        # Try connecting to a well-known external address
        # This doesn't actually send data, just determines the local interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Google's DNS server - doesn't actually connect, just determines interface
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            return ip
    except Exception:
        pass

    # Fallback to hostname resolution
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip not in ["127.0.0.1", "0.0.0.0"]:
            return ip
    except Exception:
        pass

    # Last resort - use localhost (may not work for multi-node)
    return "127.0.0.1"


def get_open_port() -> int:
    """Get an available port for distributed communication.

    This function tries multiple methods to get an available port:
    1. Check SGLANG_PORT environment variable
    2. Find an available port by binding to port 0 (OS assigns free port)
    3. Increment from SGLANG_PORT if already in use

    Returns:
        int: An available port number
    """
    port = os.getenv("SGLANG_PORT")
    if port is not None:
        port = int(port)
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use

    # Try to get any available port (OS assigns one)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # Try IPv6 as fallback
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def is_ipv6(ip: str) -> bool:
    """Check if the given IP address is IPv6.

    Args:
        ip: IP address string

    Returns:
        True if IPv6, False otherwise
    """
    return ":" in ip
