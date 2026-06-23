# RAG Pipeline — Test Query Bank

Queries for verifying retrieval, citation, and refusal on the 18-document corpus.
The **Expected source** is the file whose chunk should score highest — use it to
confirm citations point at the right document.

---

## In-scope — single source

| # | Query | Expected source |
|---|-------|-----------------|
| 1 | How does self-attention let a token relate to distant tokens? | attention.txt |
| 2 | How do query, key, and value vectors work in self-attention? | attention.txt |
| 3 | Why is scaling by the square root of the key dimension used in attention? | attention.txt |
| 4 | What's the difference between one-hot encoding and a dense embedding? | embeddings.txt |
| 5 | Why must the same model embed both documents and queries? | embeddings.txt |
| 6 | What problem does retrieval-augmented generation solve? | rag.txt |
| 7 | What's the difference between CBOW and Skip-gram? | word2vec.txt |
| 8 | How does negative sampling work in Word2Vec? | word2vec.txt |
| 9 | Why does Word2Vec sample negatives using frequency raised to the 3/4 power? | word2vec.txt |
| 10 | How does a draft model speed up generation without changing the output? | speculative_decoding.txt |
| 11 | What does the acceptance rate determine in speculative decoding? | speculative_decoding.txt |
| 12 | Why is byte pair encoding better than word-level tokenization for rare words? | tokenization.txt |
| 13 | What does LoRA freeze and what does it train? | lora.txt |
| 14 | How does QLoRA cut memory further than plain LoRA? | lora.txt |
| 15 | Why does the KV cache grow with sequence length? | kv_cache.txt |
| 16 | How does PagedAttention reduce memory fragmentation? | kv_cache.txt |
| 17 | What is a Kafka partition and what does it guarantee? | kafka.txt |
| 18 | What happens during a consumer group rebalance? | kafka.txt |
| 19 | What does the Kubernetes scheduler decide? | kubernetes.txt |
| 20 | Why are Services needed if pods already have IP addresses? | kubernetes.txt |
| 21 | What are the two deployment modes in KServe? | kserve.txt |
| 22 | When would you pick KServe RawDeployment over Serverless mode? | kserve.txt |
| 23 | What's the difference between a layer-4 and a layer-7 load balancer? | load_balancing.txt |
| 24 | What is cache-aside and how does it differ from write-through? | caching.txt |
| 25 | What does LRU eviction do? | caching.txt |
| 26 | During a network partition, what does a CP system give up? | cap_theorem.txt |
| 27 | What is the difference between NPS Tier I and Tier II? | nps.txt |
| 28 | Why does Tier II NPS have no lock-in but Tier I does? | nps.txt |
| 29 | What extra deduction does Section 80CCD(1B) give? | nps.txt |
| 30 | How does rupee-cost averaging work in a SIP? | mutual_funds.txt |
| 31 | What does the expense ratio do to returns over time? | mutual_funds.txt |
| 32 | What is the lock-in period for ELSS funds? | elss.txt |
| 33 | How does the ELSS lock-in work when investing through a SIP? | elss.txt |
| 34 | Why can an index fund never beat the market? | index_funds.txt |

---

## Comparison — answer spans two files (retrieval weak spot)

These are the questions where top-3 retrieval tends to strain, because the answer
needs chunks from two different documents and neither half may score high enough.
Note what happens — this is evidence for the reranker upgrade, not a bug to fix now.

| # | Query | Spans |
|---|-------|-------|
| C1 | How do KV caching and speculative decoding both speed up inference? | kv_cache.txt + speculative_decoding.txt |
| C2 | How are ELSS and index funds different in lock-in and tax treatment? | elss.txt + index_funds.txt |
| C3 | How does an embedding differ from the token ids a tokenizer produces? | embeddings.txt + tokenization.txt |
| C4 | Why might you choose active funds over index funds despite higher fees? | mutual_funds.txt + index_funds.txt |

---

## Traps — out of scope

Each of these must return **exactly**:

```
Not found in the provided documents.
```

| # | Query |
|---|-------|
| T1 | How do I make chicken biryani? |
| T2 | What is the capital of France? |
| T3 | Who won the cricket World Cup last year? |
| T4 | What's the weather in Hyderabad today? |

---

## How to use

1. Run a handful from the single-source block and confirm each citation names the
   expected file.
2. Run all four comparison queries and record which retrieve well and which miss.
3. Run all four traps and confirm the exact refusal string.

Anything that misretrieves goes in your notes as justification for the cross-encoder
reranker in the v1.0 hardening phase.