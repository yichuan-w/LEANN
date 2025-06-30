# Recompute Embeddings Saved

```console
python ./demo/main.py --mode serve --engine sglang --load-indices diskann --port 8082 --domain rpj_wiki --lazy --recompute --dedup --use-partition
python ./demo/embedding_server.py --domain rpj_wiki
python ./demo/test_serve.py --port 8082  --nprobe 80  --re --dedup
```

Result:
```
Evaluation Results for nprobe = 80:
Final Recall Rate: 0.9333
Average total latency: 2.427s
Average search time: 2.414s
```

其中，use-partition也可以不加，也可以跑。不加的效果如下：
```
Results for nprobe = 80:
Final Recall Rate: 0.9333
Average total latency: 2.434s
Average search time: 2.421s
```

# Recompute Embeddings + Loading from disk

Remove `--dedup --use-partition`

```console
python ./demo/main.py --mode serve --engine sglang --load-indices diskann --port 8082 --domain rpj_wiki --lazy --recompute
python ./demo/embedding_server.py --domain rpj_wiki
python ./demo/test_serve.py --port 8082  --nprobe 80  --re
```

Result:
```
Evaluation Results for nprobe = 80:
Evaluation Results for nprobe = 80:
Average F1 Score: 0.5708
Average Exact Match Score: 0.4500
Average Recall Rate: 0.9333
Average total latency: 3.709s
Average search time: 3.696s
```
