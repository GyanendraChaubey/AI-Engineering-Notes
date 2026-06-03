# Case Studies

> Applying everything together in realistic system design scenarios.

---

## Case Study 1: LLM Chat Assistant with Dynamic Context Based on Query

**Problem:** Build a production chat assistant for a large e-commerce platform that handles customer queries about orders, products, policies, and general questions — using the right information source for each query type.

### System Design

```
User Message
     │
[Intent Classifier] ──── routes to ────►  Chitchat Handler (no retrieval needed)
     │                                 ►  Order Lookup (structured DB query)
     │                                 ►  Policy RAG (vector retrieval)
     │                                 ►  Product Search (hybrid retrieval)
     │
[Context Builder] ──── assembles prompt with appropriate context
     │
[LLM (GPT-4o)] ──── generates grounded response
     │
[Safety Filter] ──── checks output before returning
     │
[Response + Citations]
```

### Implementation

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import json

class QueryIntent(Enum):
    CHITCHAT = "chitchat"
    ORDER_STATUS = "order_status"
    POLICY = "policy"
    PRODUCT = "product"
    UNKNOWN = "unknown"

@dataclass
class Context:
    intent: QueryIntent
    data: str
    sources: list[str]

class SmartChatAssistant:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.order_db = OrderDatabase()
        self.policy_rag = PolicyRAG()
        self.product_search = ProductSearch()
        self.llm = LLMClient(model="gpt-4o", temperature=0.3)
    
    def respond(self, user_message: str, user_id: str) -> dict:
        # Step 1: Classify intent
        intent = self.classifier.classify(user_message)
        
        # Step 2: Fetch appropriate context
        context = self._get_context(intent, user_message, user_id)
        
        # Step 3: Build prompt
        prompt = self._build_prompt(user_message, context)
        
        # Step 4: Generate
        response = self.llm.chat(prompt)
        
        return {
            "response": response,
            "sources": context.sources,
            "intent": intent.value
        }
    
    def _get_context(self, intent: QueryIntent, query: str, user_id: str) -> Context:
        if intent == QueryIntent.ORDER_STATUS:
            # Structured lookup — exact, no hallucination possible
            order_data = self.order_db.get_recent_orders(user_id)
            return Context(intent, json.dumps(order_data), ["Order Database"])
        
        elif intent == QueryIntent.POLICY:
            # RAG over policy documents
            chunks = self.policy_rag.retrieve(query, k=3)
            context_text = "\n---\n".join(c.text for c in chunks)
            sources = [c.metadata["doc_title"] for c in chunks]
            return Context(intent, context_text, sources)
        
        elif intent == QueryIntent.PRODUCT:
            # Hybrid search: semantic + keyword
            results = self.product_search.hybrid_search(query, k=5)
            context_text = format_products(results)
            return Context(intent, context_text, ["Product Catalog"])
        
        else:
            return Context(intent, "", [])  # No retrieval for chitchat
    
    def _build_prompt(self, query: str, context: Context) -> str:
        base_system = """You are a helpful customer service assistant for ShopCo.
        Answer based on the provided context. If you don't have enough information,
        say so honestly. Never make up order details, prices, or policies."""
        
        if context.data:
            return f"""{base_system}

Context ({context.intent.value}):
{context.data}

Customer question: {query}"""
        else:
            return f"{base_system}\n\nCustomer: {query}"
```

### Intent Classifier

```python
class IntentClassifier:
    def __init__(self):
        self.llm = LLMClient(model="gpt-4o-mini", temperature=0)
    
    def classify(self, message: str) -> QueryIntent:
        prompt = f"""Classify this customer message into one category:
- order_status: asking about their order, tracking, delivery
- policy: asking about returns, refunds, shipping policy, terms
- product: asking about products, availability, specs, prices
- chitchat: greetings, thanks, general conversation

Message: "{message}"
Return only the category name."""
        
        result = self.llm.complete(prompt).strip().lower()
        try:
            return QueryIntent(result)
        except ValueError:
            return QueryIntent.UNKNOWN
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Intent classification first | Different intents need different data sources |
| Structured DB for orders | 100% accuracy needed; no hallucination tolerable |
| RAG for policies | Documents change; don't bake into model weights |
| Hybrid search for products | Exact product names (BM25) + semantic (dense) |
| Low temperature (0.3) | Customer service needs consistent, factual tone |
| Always return sources | Builds trust; enables human verification |

### Evaluation Metrics
- **Task completion rate** — did the user get their question answered?
- **Hallucination rate** — factual errors per 1000 queries
- **Retrieval precision** — were retrieved chunks relevant?
- **Response latency** — p95 < 3 seconds
- **CSAT score** — customer satisfaction rating

---

## Case Study 2: Prompting Techniques in Practice

**Problem:** A legal tech company wants to extract structured information from contracts — parties involved, key dates, payment terms, and obligations — at high accuracy.

### Naive Approach (fails)

```python
# BAD: Too open-ended, inconsistent output format
prompt = f"Extract important information from this contract: {contract_text}"
# Output: varies wildly between runs, hard to parse programmatically
```

### Step-by-Step Improvement

**Iteration 1: Define output schema**
```python
prompt = f"""Extract the following from this contract and return as JSON:
- parties (list of party names and their roles)
- effective_date
- expiration_date
- payment_terms
- key_obligations (list, max 5)

Contract:
{contract_text}

Return valid JSON only, no other text."""
```

**Iteration 2: Add Chain-of-Thought for complex clauses**
```python
prompt = f"""You are a legal analyst extracting structured data from contracts.

First, read the contract carefully and identify:
1. All named parties and their roles
2. All dates mentioned and their significance
3. All payment-related clauses
4. The key obligations of each party

Then, based on your analysis, return a JSON object with:
{{
  "parties": [{{"name": "...", "role": "...", "address": "..."}}],
  "effective_date": "YYYY-MM-DD or null",
  "expiration_date": "YYYY-MM-DD or null",
  "payment_terms": {{
    "amount": "...", 
    "frequency": "...", 
    "due_date": "...",
    "late_penalty": "..."
  }},
  "obligations": [{{"party": "...", "obligation": "...", "deadline": "..."}}],
  "governing_law": "..."
}}

Contract:
{contract_text}

Think through each field carefully before outputting the JSON."""
```

**Iteration 3: Add validation and self-correction**
```python
def extract_with_validation(contract_text: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        response = llm(extraction_prompt.format(contract=contract_text))
        
        try:
            data = json.loads(response)
            
            # Validate required fields
            assert "parties" in data and len(data["parties"]) >= 2
            assert "effective_date" in data
            
            # Validate date format
            for date_field in ["effective_date", "expiration_date"]:
                if data.get(date_field):
                    datetime.strptime(data[date_field], "%Y-%m-%d")
            
            return data
            
        except (json.JSONDecodeError, AssertionError, ValueError) as e:
            if attempt < max_retries - 1:
                # Self-correction: tell the model what went wrong
                correction_prompt = f"""
Your previous extraction had an error: {str(e)}
Previous output: {response}

Please fix the error and return corrected valid JSON.
Contract: {contract_text}
"""
                response = llm(correction_prompt)
    
    raise ValueError("Could not extract valid data after retries")
```

**Iteration 4: Few-shot examples for edge cases**
```python
few_shot_examples = """
EXAMPLE 1:
Contract snippet: "This Agreement is entered into as of January 1, 2024 between 
ACME Corp ("Buyer") and XYZ Ltd ("Seller")..."
Extracted: {"parties": [{"name": "ACME Corp", "role": "Buyer"}, {"name": "XYZ Ltd", "role": "Seller"}], 
             "effective_date": "2024-01-01"}

EXAMPLE 2 (amendment — no new effective date):
Contract snippet: "This Amendment No. 2 hereby modifies the Agreement dated March 15, 2023..."
Extracted: {"parties": [...], "effective_date": null, 
             "notes": "This is an amendment to existing agreement dated 2023-03-15"}
"""
```

### Results Comparison

| Approach | JSON parse success | Field accuracy | Time per contract |
|---|---|---|---|
| Naive | 60% | 70% | 2s |
| Schema + CoT | 90% | 88% | 3s |
| + Validation/retry | 98% | 88% | 4s avg |
| + Few-shot | 98% | 94% | 4s avg |

### Lessons Learned

1. **Schema definition is non-negotiable** for structured extraction — free-form output is unparseable at scale
2. **CoT dramatically improves complex clause interpretation** — the model "thinks" before committing to values
3. **Validation + retry is cheap insurance** — most failures self-correct on second attempt
4. **Few-shot examples handle edge cases** that instructions alone miss
5. **Evaluation drives improvement** — without the accuracy table above, you'd be flying blind

---

*Back to [Home](../../README.md)*
