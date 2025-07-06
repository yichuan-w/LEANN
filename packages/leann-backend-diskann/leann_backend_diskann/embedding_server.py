#!/usr/bin/env python3
"""
Embedding server for leann-backend-diskann - Fixed ZMQ REQ-REP pattern
"""

import pickle
import argparse
import threading
import time
import json
from typing import Dict, Any, Optional, Union

from transformers import AutoTokenizer, AutoModel
import os
from contextlib import contextmanager
import zmq
import numpy as np

RED = "\033[91m"
RESET = "\033[0m"

# --- New Passage Loader from HNSW backend ---
class SimplePassageLoader:
    """
    Simple passage loader that replaces config.py dependencies
    """
    def __init__(self, passages_data: Optional[Dict[str, Any]] = None):
        self.passages_data = passages_data or {}
    
    def __getitem__(self, passage_id: Union[str, int]) -> Dict[str, str]:
        """Get passage by ID"""
        str_id = str(passage_id)
        if str_id in self.passages_data:
            return {"text": self.passages_data[str_id]}
        else:
            # Return empty text for missing passages
            return {"text": ""}
    
    def __len__(self) -> int:
        return len(self.passages_data)

def load_passages_from_file(passages_file: str) -> SimplePassageLoader:
    """
    Load passages from a JSONL file with label map support
    Expected format: {"id": "passage_id", "text": "passage_text", "metadata": {...}} (one per line)
    """
    from pathlib import Path
    import pickle
    
    if not os.path.exists(passages_file):
        raise FileNotFoundError(f"Passages file {passages_file} not found.")
    
    if not passages_file.endswith('.jsonl'):
        raise ValueError(f"Expected .jsonl file format, got: {passages_file}")
    
    # Load label map (int -> string_id)
    passages_dir = Path(passages_file).parent
    label_map_file = passages_dir / "leann.labels.map"
    
    label_map = {}
    if label_map_file.exists():
        with open(label_map_file, 'rb') as f:
            label_map = pickle.load(f)
        print(f"Loaded label map with {len(label_map)} entries")
    else:
        raise FileNotFoundError(f"Label map file not found: {label_map_file}")
    
    # Load passages by string ID
    string_id_passages = {}
    with open(passages_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                passage = json.loads(line)
                string_id_passages[passage['id']] = passage['text']
    
    # Create int ID -> text mapping using label map
    passages_data = {}
    for int_id, string_id in label_map.items():
        if string_id in string_id_passages:
            passages_data[str(int_id)] = string_id_passages[string_id]
        else:
            print(f"WARNING: String ID {string_id} from label map not found in passages")
    
    print(f"Loaded {len(passages_data)} passages from JSONL file {passages_file} using label map")
    return SimplePassageLoader(passages_data)

def create_embedding_server_thread(
    zmq_port=5555,
    model_name="sentence-transformers/all-mpnet-base-v2",
    max_batch_size=128,
    passages_file: Optional[str] = None,
):
    """
    在当前线程中创建并运行 embedding server
    这个函数设计为在单独的线程中调用
    """
    print(f"INFO: Initializing embedding server thread on port {zmq_port}")
    
    try:
        # 检查端口是否已被占用
        import socket
        def check_port(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0

        if check_port(zmq_port):
            print(f"{RED}Port {zmq_port} is already in use{RESET}")
            return

        # 初始化模型
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        import torch

        # 选择设备
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device = torch.device("cuda")
            print("INFO: Using CUDA device")
        elif mps_available:
            device = torch.device("mps")
            print("INFO: Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("INFO: Using CPU device")
        
        # 加载模型
        print(f"INFO: Loading model {model_name}")
        model = AutoModel.from_pretrained(model_name).to(device).eval()

        # 优化模型
        if cuda_available or mps_available:
            try:
                model = model.half()
                model = torch.compile(model)
                print(f"INFO: Using FP16 precision with model: {model_name}")
            except Exception as e:
                print(f"WARNING: Model optimization failed: {e}")

        # Load passages from file if provided
        if passages_file and os.path.exists(passages_file):
            passages = load_passages_from_file(passages_file)
        else:
            print("WARNING: No passages file provided or file not found. Using an empty passage loader.")
            passages = SimplePassageLoader()

        print(f"INFO: Loaded {len(passages)} passages.")

        class DeviceTimer:
            """设备计时器"""
            def __init__(self, name="", device=device):
                self.name = name
                self.device = device
                self.start_time = 0
                self.end_time = 0
                
                if cuda_available:
                    self.start_event = torch.cuda.Event(enable_timing=True)
                    self.end_event = torch.cuda.Event(enable_timing=True)
                else:
                    self.start_event = None
                    self.end_event = None

            @contextmanager
            def timing(self):
                self.start()
                yield
                self.end()

            def start(self):
                if cuda_available:
                    torch.cuda.synchronize()
                    self.start_event.record()
                else:
                    if self.device.type == "mps":
                        torch.mps.synchronize()
                    self.start_time = time.time()

            def end(self):
                if cuda_available:
                    self.end_event.record()
                    torch.cuda.synchronize()
                else:
                    if self.device.type == "mps":
                        torch.mps.synchronize()
                    self.end_time = time.time()

            def elapsed_time(self):
                if cuda_available:
                    return self.start_event.elapsed_time(self.end_event) / 1000.0
                else:
                    return self.end_time - self.start_time

            def print_elapsed(self):
                print(f"Time taken for {self.name}: {self.elapsed_time():.6f} seconds")

        def process_batch(texts_batch, ids_batch, missing_ids):
            """处理文本批次"""
            batch_size = len(texts_batch)
            print(f"INFO: Processing batch of size {batch_size}")

            tokenize_timer = DeviceTimer("tokenization (batch)", device)
            to_device_timer = DeviceTimer("transfer to device (batch)", device)
            embed_timer = DeviceTimer("embedding (batch)", device)
            pool_timer = DeviceTimer("mean pooling (batch)", device)

            with tokenize_timer.timing():
                encoded_batch = tokenizer.batch_encode_plus(
                    texts_batch,
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            tokenize_timer.print_elapsed()

            seq_length = encoded_batch["input_ids"].size(1)
            print(f"Batch size: {batch_size}, Sequence length: {seq_length}")

            with to_device_timer.timing():
                enc = {k: v.to(device) for k, v in encoded_batch.items()}
            to_device_timer.print_elapsed()

            with torch.no_grad():
                with embed_timer.timing():
                    out = model(enc["input_ids"], enc["attention_mask"])
                embed_timer.print_elapsed()

                with pool_timer.timing():
                    hidden_states = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
                    mask_expanded = enc["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                pool_timer.print_elapsed()

            return batch_embeddings.cpu().numpy()

        # ZMQ server 主循环 - 修改为REP套接字
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)  # 改为REP套接字
        socket.bind(f"tcp://127.0.0.1:{zmq_port}")
        print(f"INFO: ZMQ ROUTER server listening on port {zmq_port}")

        # 设置超时
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒接收超时
        socket.setsockopt(zmq.SNDTIMEO, 300000)  # 300秒发送超时

        from . import embedding_pb2

        print(f"INFO: Embedding server ready to serve requests")

        while True:
            try:
                parts = socket.recv_multipart()

                # --- 恢复稳健的消息格式判断 ---
                # 必须检查 parts 的长度，避免 IndexError
                if len(parts) >= 3:
                    identity = parts[0]
                    # empty = parts[1]  # 中间的空帧我们通常不关心
                    message = parts[2]
                elif len(parts) == 2:
                    # 也能处理没有空帧的情况
                    identity = parts[0]
                    message = parts[1]
                else:
                    # 如果收到格式错误的消息，打印警告并忽略它，而不是崩溃
                    print(f"WARNING: Received unexpected message format with {len(parts)} parts. Ignoring.")
                    continue
                print(f"INFO: Received ZMQ request from client {identity.hex()[:8]}, size {len(message)} bytes")

                e2e_start = time.time()
                lookup_timer = DeviceTimer("text lookup", device)

                # 解析请求
                req_proto = embedding_pb2.NodeEmbeddingRequest()
                req_proto.ParseFromString(message)
                node_ids = req_proto.node_ids
                print(f"INFO: Request for {len(node_ids)} node embeddings: {list(node_ids)}")

                # 添加调试信息
                if len(node_ids) > 0:
                    print(f"DEBUG: Node ID range: {min(node_ids)} to {max(node_ids)}")
                
                # 查找文本
                texts = []
                missing_ids = []
                with lookup_timer.timing():
                    for nid in node_ids:
                        txtinfo = passages[nid]
                        txt = txtinfo["text"]
                        if txt:
                            texts.append(txt)
                        else:
                            # 如果文本为空，我们仍然需要一个占位符来进行批处理，
                            # 但将其ID记录为缺失
                            texts.append("") 
                            missing_ids.append(nid)
                lookup_timer.print_elapsed()

                if missing_ids:
                    print(f"WARNING: Missing passages for IDs: {missing_ids}")

                # 处理批次
                total_size = len(texts)
                print(f"INFO: Total batch size: {total_size}, max_batch_size: {max_batch_size}")
                
                all_embeddings = []
                
                if total_size > max_batch_size:
                    print(f"INFO: Splitting batch of size {total_size} into chunks of {max_batch_size}")
                    for i in range(0, total_size, max_batch_size):
                        end_idx = min(i + max_batch_size, total_size)
                        print(f"INFO: Processing chunk {i//max_batch_size + 1}/{(total_size + max_batch_size - 1)//max_batch_size}: items {i} to {end_idx-1}")
                        
                        chunk_texts = texts[i:end_idx]
                        chunk_ids = node_ids[i:end_idx]
                        
                        embeddings_chunk = process_batch(chunk_texts, chunk_ids, missing_ids)
                        all_embeddings.append(embeddings_chunk)
                        
                        if cuda_available:
                            torch.cuda.empty_cache()
                        elif device.type == "mps":
                            torch.mps.empty_cache()
                            
                    hidden = np.vstack(all_embeddings)
                    print(f"INFO: Combined embeddings shape: {hidden.shape}")
                else:
                    hidden = process_batch(texts, node_ids, missing_ids)

                # 序列化响应
                ser_start = time.time()

                resp_proto = embedding_pb2.NodeEmbeddingResponse()
                hidden_contiguous = np.ascontiguousarray(hidden, dtype=np.float32)
                resp_proto.embeddings_data = hidden_contiguous.tobytes()
                resp_proto.dimensions.append(hidden_contiguous.shape[0])
                resp_proto.dimensions.append(hidden_contiguous.shape[1])
                resp_proto.missing_ids.extend(missing_ids)

                response_data = resp_proto.SerializeToString()
                
                # REP 套接字发送单个响应
                socket.send_multipart([identity, b'', response_data])

                ser_end = time.time()

                print(f"INFO: Serialize time: {ser_end - ser_start:.6f} seconds")

                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                e2e_end = time.time()
                print(f"INFO: ZMQ E2E time: {e2e_end - e2e_start:.6f} seconds")

            except zmq.Again:
                print("INFO: ZMQ socket timeout, continuing to listen")
                # REP套接字不需要重新创建，只需要继续监听
                continue
            except Exception as e:
                print(f"ERROR: Error in ZMQ server: {e}")
                try:
                    # 发送空响应以维持REQ-REP状态
                    empty_resp = embedding_pb2.NodeEmbeddingResponse()
                    socket.send(empty_resp.SerializeToString())
                except:
                    # 如果发送失败，重新创建socket
                    socket.close()
                    socket = context.socket(zmq.REP)
                    socket.bind(f"tcp://127.0.0.1:{zmq_port}")
                    socket.setsockopt(zmq.RCVTIMEO, 5000)
                    socket.setsockopt(zmq.SNDTIMEO, 300000)
                    print("INFO: ZMQ socket recreated after error")

    except Exception as e:
        print(f"ERROR: Failed to start embedding server: {e}")
        raise


# 保持原有的 create_embedding_server 函数不变，只添加线程化版本
def create_embedding_server(
    domain="demo",
    load_passages=True,
    load_embeddings=False,
    use_fp16=True,
    use_int8=False,
    use_cuda_graphs=False,
    zmq_port=5555,
    max_batch_size=128,
    lazy_load_passages=False,
    model_name="sentence-transformers/all-mpnet-base-v2",
    passages_file: Optional[str] = None,
):
    """
    原有的 create_embedding_server 函数保持不变
    这个是阻塞版本，用于直接运行
    """
    create_embedding_server_thread(zmq_port, model_name, max_batch_size, passages_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding service")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZMQ port to run on")
    parser.add_argument("--domain", type=str, default="demo", help="Domain name")
    parser.add_argument("--passages-file", type=str, help="JSON file containing passage ID to text mapping")
    parser.add_argument("--load-passages", action="store_true", default=True)
    parser.add_argument("--load-embeddings", action="store_true", default=False)
    parser.add_argument("--use-fp16", action="store_true", default=False)
    parser.add_argument("--use-int8", action="store_true", default=False)
    parser.add_argument("--use-cuda-graphs", action="store_true", default=False)
    parser.add_argument("--max-batch-size", type=int, default=128, help="Maximum batch size before splitting")
    parser.add_argument("--lazy-load-passages", action="store_true", default=True)
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-mpnet-base-v2", 
                        help="Embedding model name")
    args = parser.parse_args()

    create_embedding_server(
        domain=args.domain,
        load_passages=args.load_passages,
        load_embeddings=args.load_embeddings,
        use_fp16=args.use_fp16,
        use_int8=args.use_int8,
        use_cuda_graphs=args.use_cuda_graphs,
        zmq_port=args.zmq_port,
        max_batch_size=args.max_batch_size,
        lazy_load_passages=args.lazy_load_passages,
        model_name=args.model_name,
        passages_file=args.passages_file,
    )