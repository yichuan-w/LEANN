import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import json
from contriever.src.contriever import load_retriever

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_BLOCKTIME"] = "0"

torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def embed_queries(queries, model, tokenizer, model_name_or_path, per_gpu_batch_size=64):
    """Embed queries using the model with batching"""
    model = model.half()
    model.eval()
    embeddings = []
    batch_question = []

    with torch.no_grad():
        for k, query in tqdm(enumerate(queries), desc="Encoding queries"):
            batch_question.append(query)

            # Process when batch is full or at the end
            if len(batch_question) == per_gpu_batch_size or k == len(queries) - 1:
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=512,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                output = model(**encoded_batch)

                # Contriever typically uses output.last_hidden_state pooling or something specialized
                # if "contriever" not in model_name_or_path:
                #     output = output.last_hidden_state[:, 0, :]

                embeddings.append(output.cpu())
                batch_question = []  # Reset batch

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print(f"Query embeddings shape: {embeddings.shape}")
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Debug embedding tool")
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-Qwen2-1.5B-instruct", 
                        help="Model name for embedding (default: facebook/contriever-msmarco)")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="Batch size for encoding (default: 32)")
    parser.add_argument("--input-file", type=str, 
                        help="Input file with queries (JSON lines format with 'query' field)")
    parser.add_argument("--output-file", type=str, default="embeddings.npy", 
                        help="Output file to save embeddings (default: embeddings.npy)")
    parser.add_argument("--text", type=str, nargs="+", 
                        help="Direct text input to embed (can provide multiple)")
    parser.add_argument("--save-text", action="store_true", 
                        help="Save the input text alongside embeddings")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading query encoder: {args.model}")
    query_encoder, query_tokenizer, _ = load_retriever(args.model)
    query_encoder = query_encoder.to(device)
    query_encoder.eval()
    
    # Get queries
    queries = []
    
    # From file if provided
    if args.input_file:
        print(f"Loading queries from: {args.input_file}")
        with open(args.input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                queries.append(data["query"])
    
    # From command line if provided
    if args.text:
        print(f"Using {len(args.text)} queries from command line")
        queries.extend(args.text)
    
    # If no queries, use some examples
    if not queries:
        print("No queries provided, using example queries")
        queries = [
            "Were there any variances detected for hour 6 on 3/9/01?"
        ]
    
    print(f"Embedding {len(queries)} queries")
    for i, q in enumerate(queries[:5]):  # Print first 5 queries
        print(f"Query {i+1}: {q}")
    if len(queries) > 5:
        print(f"... and {len(queries)-5} more")
    
    # Encode queries
    embeddings = embed_queries(
        queries, query_encoder, query_tokenizer, args.model, per_gpu_batch_size=args.batch_size
    )


    passages = [
        "Start Date: 3/9/01; HourAhead hour: 6; No ancillary schedules awarded. Variances detected. Variances detected in Generation schedule. Variances detected in Energy Import/Export schedule. LOG MESSAGES: PARSING FILE -->> O:\\Portland\\WestDesk\\California Scheduling\\ISO Final Schedules\\2001030906.txt ---- Generation Schedule ---- $$$ Variance found in table tblGEN_SCHEDULE. Details: (Hour: 6 / Preferred: 20.00 / Final: 19.80) TRANS_TYPE: FINAL SC_ID: TOSCO MKT_TYPE: 2 TRANS_DATE: 3/9/01 UNIT_ID: UNCHEM_1_UNIT $$$ Variance found in table tblGEN_SCHEDULE. Details: (Hour: 6 / Preferred: 29.00 / Final: 28.20) TRANS_TYPE: FINAL SC_ID: ARCO MKT_TYPE: 2 TRANS_DATE: 3/9/01 UNIT_ID: CARBGN_6_UNIT 1 $$$ Variance found in table tblGEN_SCHEDULE. Details: (Hour: 6 / Preferred: 45.00 / Final: 43.80) TRANS_TYPE: FINAL SC_ID: DELANO MKT_TYPE: 2 TRANS_DATE: 3/9/01 UNIT_ID: PANDOL_6_UNIT $$$ Variance found in table tblGEN_SCHEDULE. Details: (Hour: 6 / Preferred: 13.00 / Final: 12.60) TRANS_TYPE: FINAL SC_ID: Wheelabrat MKT_TYPE: 2 TRANS_DATE: 3/9/01 UNIT_ID: MARTEL_2_AMFOR ---- Energy Import/Export Schedule ---- $$$ Variance found in table tblINTCHG_IMPEXP. Details: (Hour: 6 / Preferred: 62.00 / Final: 60.40) TRANS_TYPE: FINAL SC_ID: ECTstCA MKT_TYPE: 2 TRANS_DATE: 3/9/01 TIE_POINT: PVERDE_5_DEVERS INTERCHG_ID: EPMI_CISO_5001 ENGY_TYPE: FIRM $$$ Variance found in table tblINTCHG_IMPEXP. Details: (Hour: 6 / Preferred: 63.00 / Final: 61.23) TRANS_TYPE: FINAL SC_ID: ECTstSW MKT_TYPE: 2 TRANS_DATE: 3/9/01 TIE_POINT: PVERDE_5_DEVERS INTERCHG_ID: EPMI_CISO_5000 ENGY_TYPE: FIRM $$$ Variance found in table tblINTCHG_IMPEXP. Details: (Hour: 6 / Preferred: 17.00 / Final: 11.00) TRANS_TYPE: FINAL SC_ID: ECTRT MKT_TYPE: 2 TRANS_DATE: 3/9/01 TIE_POINT: SYLMAR_2_NOB INTERCHG_ID: EPMI_CISO_LUCKY ENGY_TYPE: NFRM",
        "Start Date: 3/30/01; HourAhead hour: 15; No ancillary schedules awarded. Variances detected. Variances detected in Generation schedule. LOG MESSAGES: PARSING FILE -->> O:\\Portland\\WestDesk\\California Scheduling\\ISO Final Schedules\\2001033015.txt ---- Generation Schedule ---- $$$ Variance found in table tblGEN_SCHEDULE. Details: (Hour: 15 / Preferred: 0.00 / Final: 0.00) TRANS_TYPE: FINAL SC_ID: ARCO MKT_TYPE: 2 TRANS_DATE: 3/30/01 UNIT_ID: CARBGN_6_UNIT 1 $$$ Variance found in table tblGEN_SCHEDULE. Details: (Hour: 15 / Preferred: 45.00 / Final: 44.00) TRANS_TYPE: FINAL SC_ID: DELANO MKT_TYPE: 2 TRANS_DATE: 3/30/01 UNIT_ID: PANDOL_6_UNIT"
    ]

    # Embed passages
    passage_embeddings = embed_queries(passages, query_encoder, query_tokenizer, args.model, per_gpu_batch_size=args.batch_size)


    # distance with passages 0 and query
    distance_0 = np.linalg.norm(embeddings[0] - passage_embeddings[0])
    print(f"Distance between query 0 and passage 0: {distance_0}")

    # distance with passages 1 and query
    distance_1 = np.linalg.norm(embeddings[0] - passage_embeddings[1])
    print(f"Distance between query 0 and passage 1: {distance_1}")

    # print which one is closer
    if distance_0 < distance_1:
        print("Query 0 is closer to passage 0")
    else:
        print("Query 0 is closer to passage 1")
    

    
    print("Done!")

if __name__ == "__main__":
    main()
