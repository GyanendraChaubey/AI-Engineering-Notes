# ML Engineering Lead — Interview Session 2 (Deep Dive)

**Q: Explain Python class inheritance and the super().__init__() pattern.**

```python
class Animal:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def describe(self) -> str:
        return f"{self.name}, age {self.age}"

class Dog(Animal):
    def __init__(self, name: str, age: int, breed: str):
        super().__init__(name, age)  # calls Animal.__init__; initializes name and age
        self.breed = breed

    def describe(self) -> str:
        return super().describe() + f", breed: {self.breed}"

class GuideDog(Dog):
    def __init__(self, name: str, age: int, breed: str, owner: str):
        super().__init__(name, age, breed)  # calls Dog.__init__ which calls Animal.__init__
        self.owner = owner

g = GuideDog("Buddy", 5, "Golden Retriever", "Alice")
print(g.describe())  # "Buddy, age 5, breed: Golden Retriever"
print(g.owner)       # "Alice"
```

**Key concepts:**
- `super()` returns a proxy to the parent class in MRO (Method Resolution Order) -- Python's C3 linearization.
- In single inheritance: simply calls the direct parent.
- In multiple inheritance: MRO ensures each parent is called exactly once in the right order.
- `super()` without arguments (Python 3) auto-infers class and instance.
- Why use `super()` over `ParentClass.__init__(self, ...)`: `super()` is MRO-aware; hardcoding the parent class name breaks cooperative multiple inheritance.

---

**Q: How do you ensure data isolation in a multi-tenant RAG pipeline?**

**Implementation layers:**
1. **Vector DB namespace/index isolation:** Pinecone namespaces per tenant (cost-efficient). OpenSearch separate index per tenant (stronger). Weaviate built-in multi-tenancy. All queries include mandatory filter `{tenant_id: current_tenant}`.
2. **Metadata tagging:** Every chunk tagged with `tenant_id` at ingestion. Ingestion validates `tenant_id` matches authenticated user.
3. **Auth enforcement:** JWT token carries `tenant_id` claim. Middleware injects it as mandatory filter in every vector search -- undeniable.
4. **Row-Level Security (PostgreSQL + pgvector):**
```sql
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON document_chunks
  USING (tenant_id = current_setting('app.current_tenant_id'));
```
5. **Audit logging:** Every query logged with `tenant_id`, timestamp, chunks retrieved.

---

**Q: How do you scale a RAG pipeline from 50-100 users to 100K users while maintaining low latency?**

**Bottleneck progression:**
1. 50-500 users: LLM API rate limits (1-5s per call, hard RPM/TPM limits).
2. 500-5K users: Embedding service throughput.
3. 5K-50K users: Vector DB query concurrency and RAM pressure.
4. 50K-100K users: Application tier CPU/memory and connection pool exhaustion.

**Scaling at each tier:**
- **LLM (first):** Semantic cache (40-60% cache hit rate). Multiple API keys. Route simple queries to GPT-4o-mini. Async with exponential backoff.
- **Embedding:** Local GPU pod with sentence-transformers (100x throughput). HPA-scaled behind load balancer. Batch 50ms windows -> 32-query batches.
- **Vector DB:** OpenSearch cluster (3+ data nodes). HNSW `ef_search` tuning. Index sharding.
- **App tier:** Kubernetes HPA on request queue depth. Stateless FastAPI replicas. Connection pooling.
- **Ingestion:** SQS + Celery workers (decouple doc processing from query serving).

---

**Q: Solve LeetCode 'Top K Frequent Elements' -- return the K most frequent elements from an array.**

```python
from collections import Counter
import heapq

def topKFrequent_heap(nums: list, k: int) -> list:
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
    # Time: O(n log k),  Space: O(n)

def topKFrequent_bucket(nums: list, k: int) -> list:
    count = Counter(nums)
    # Bucket index = frequency; max frequency = len(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)
    result = []
    for freq in range(len(buckets) - 1, 0, -1):
        result.extend(buckets[freq])
        if len(result) >= k:
            return result[:k]
    return result[:k]
    # Time: O(n),  Space: O(n)

# Tests
print(topKFrequent_heap([1,1,1,2,2,3], k=2))      # [1, 2]
print(topKFrequent_bucket([1,1,1,2,2,3], k=2))     # [1, 2]
```

**Complexity:** Heap: O(n log k). Bucket sort: O(n) optimal. Naive sort: O(n log n) -- suboptimal.

---

**Q: What components and tools are used for deploying a RAG pipeline to production?**

- **Containerization:** Docker images for FastAPI app; stored in ECR / ACR.
- **Orchestration:** Kubernetes (EKS/AKS) with HPA auto-scaling on request queue depth.
- **Vector DB:** OpenSearch cluster / Pinecone / Weaviate.
- **LLM API:** Azure OpenAI / AWS Bedrock with API key rotation via Secrets Manager.
- **Embedding service:** Local sentence-transformers on GPU pod or managed API.
- **Caching:** Redis / ElastiCache for semantic cache and session storage.
- **API Gateway:** Apigee / AWS API Gateway for auth, rate limiting, routing.
- **Monitoring:** Prometheus + Grafana for metrics; CloudWatch / Datadog for logs and traces.
- **Distributed tracing:** AWS X-Ray / Azure Application Insights.
- **CI/CD:** GitHub Actions -> Docker build -> ECR push -> Helm chart -> rolling deploy.
- **Infrastructure as code:** Terraform / CloudFormation.
- **Ingestion pipeline:** AWS Glue for batch processing; SQS for async document jobs.

---

**Q: Explain Python threading and race conditions -- what is the output of code that increments a shared counter from two threads?**

```python
import threading

counter = 0

def increment_unsafe(n: int):
    global counter
    for _ in range(n):
        counter += 1  # NOT atomic: read -> modify -> write can be interrupted between threads

t1 = threading.Thread(target=increment_unsafe, args=(100_000,))
t2 = threading.Thread(target=increment_unsafe, args=(100_000,))
t1.start(); t2.start()
t1.join(); t2.join()
print(counter)  # Expected: 200000. Actual: < 200000 (non-deterministic -- race condition)
```

**Root cause:** `counter += 1` is NOT atomic. It compiles to LOAD / BINARY_ADD / STORE_FAST bytecodes. Python GIL can release between them:
- Thread 1 reads counter=5
- Thread 2 reads counter=5 (before Thread 1 writes)
- Thread 1 writes 6; Thread 2 writes 6 -> **one increment lost**

**Fix with Lock:**
```python
lock = threading.Lock()

def increment_safe(n: int):
    global counter
    for _ in range(n):
        with lock:    # atomic block: acquire -> LOAD/ADD/STORE -> release
            counter += 1
# Output is always 200000
```

**Other primitives:** `RLock` (re-entrant), `Semaphore(n)` (limit concurrency to N), `Event` (signal between threads), `threading.local()` (no sharing needed).

---

**Q: Which component becomes the bottleneck first as a RAG system scales from 50 to 100K users?**

**Bottleneck order:**
1. **LLM API (always first):** 1-5s per call, hard rate limits (tokens/min, requests/min). 500 concurrent users = 500 long-held connections.
2. **Embedding service:** At 2000 QPS, need 100-400 concurrent embedding calls. Local GPU pod solves this.
3. **Vector DB:** ANN search is fast but memory-bound. Single node reaches limit at ~10M documents.
4. **Application tier (last):** Stateless FastAPI pods scale cheaply via Kubernetes HPA.

**Why LLM is always first:** Every query requires at least one LLM call (1-5s) while embedding + vector search complete in < 200ms combined. LLM is 10-50x slower than every other component.

**Mitigation sequence:** Semantic cache (40-60% hit rate) -> model routing (cheap model for simple queries) -> multiple API keys/providers -> local LLM (unlimited throughput) -> embedding GPU pod -> vector DB cluster.

---


---
