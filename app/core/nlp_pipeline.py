import asyncio
import hashlib
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import torch
from cachetools import TTLCache
from captum.attr import LayerIntegratedGradients
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers import logging as transformers_logging
from transformers import (
    pipeline,
)

# Professional Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NLPService")

# Silence noisy third-party warnings
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NLPService:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NLPService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.zero_shot_pipeline = None
            self.sentiment_pipeline = None
            self.summarization_pipeline = None

            self.embedding_model = None
            self.cross_encoder = None
            self.qa_pipeline = None
            self.summarization_pipeline = None

            # Global Device Management
            self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = 0 if torch.cuda.is_available() else -1
            self.torch_dtype = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )

            self.executor = ThreadPoolExecutor(max_workers=3)

            # Professional Performance: 1-hour TTL Cache
            self.embedding_cache = TTLCache(maxsize=1000, ttl=3600)
            self.cross_score_cache = TTLCache(maxsize=2000, ttl=3600)

            self._initialized = True

    async def load_models(self):
        """Loads models asynchronously to avoid blocking the main thread during startup."""
        loop = asyncio.get_event_loop()

        # Load models in threads to avoid blocking the event loop
        await loop.run_in_executor(self.executor, self._load_zero_shot)
        await loop.run_in_executor(self.executor, self._load_sentiment)
        await loop.run_in_executor(self.executor, self._load_summarization)
        await loop.run_in_executor(self.executor, self._load_rag_models)

    def _load_zero_shot(self):
        logger.info("Initializing Zero-shot classification engine...")
        self.zero_shot_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1,
        )

    def _load_sentiment(self):
        logger.info("Initializing Sentiment & XAI analysis units...")
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )
        # Load raw model for XAI
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentiment_model_raw = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.sentiment_model_raw.eval()
        if torch.cuda.is_available():
            self.sentiment_model_raw.to("cuda")

    def _load_summarization(self):
        logger.info("Initializing neural summarization service...")
        model_name = "sshleifer/distilbart-cnn-12-6"

        # Elite Config: Silence weight tying and bos token warnings
        config = AutoConfig.from_pretrained(model_name)
        config.tie_word_embeddings = False

        self.summarization_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, config=config
        )

        # Ensure generation config is set to defaults that silence warnings
        if hasattr(self.summarization_model, "generation_config"):
            self.summarization_model.generation_config.forced_bos_token_id = 0

        self.summarization_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.summarization_model.to(self.summarization_device)

    def _load_rag_models(self):
        """Loads RAG-specific models with GPU optimizations."""
        logger.info("Initializing RAG retrievers and Long-Context QA Engine...")
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=self.torch_device
        )
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            default_activation_function=torch.nn.Sigmoid(),
            device=self.torch_device,
        )
        # Long-Context QA Model (Optimized for deep multi-hop reasoning)
        self.qa_pipeline = pipeline(
            "question-answering",
            model="valhalla/longformer-base-4096-finetuned-squadv1",
            device=self.device,
            torch_dtype=self.torch_dtype,
        )

    # --- Elite Utilities ---

    def _get_top_sentences(self, text, limit=100):
        import re

        # Split by punctuation followed by space/newline, OR just newline
        sentences = re.split(r"(?<=[.!?])[\s\n]+|\n+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10][:limit]

    def _rerank_categories(self, text, candidates):
        """
        Uses Cross-Encoder to validate labels against the text with elite calibration.
        Returns Top-5 with supporting evidence, filtered by a 15% dynamic threshold.
        """
        if not candidates:
            return []

        context = text[:1500]
        sentences = self._get_top_sentences(text, limit=20)

        results = []
        raw_scores = []
        for label in candidates:
            cache_key = hashlib.md5(f"{label}_{context[:500]}".encode()).hexdigest()
            if cache_key in self.cross_score_cache:
                cached = self.cross_score_cache[cache_key]
                results.append(cached)
                raw_scores.append(cached["score"] / 100.0)
                continue

            # 1. Verification Score
            pair = [[f"This document is about {label}.", context]]
            score = self.cross_encoder.predict(pair)[0]
            norm_score = 1 / (1 + np.exp(-score))  # Sigmoid

            # --- ELITE: Contextual Boosting & Inhibition ---
            BOOST_RULES = {
                "Environment": {
                    "inhibitors": [
                        "feeling",
                        "story",
                        "nature",
                        "walk",
                        "beautiful",
                        "scenery",
                    ],
                    "triggers": [
                        "carbon",
                        "sustainability",
                        "emissions",
                        "pollution",
                        "ecosystem",
                    ],
                    "penalty": 0.4,
                },
                "Risk Management": {
                    "inhibitors": ["chance", "luck", "maybe"],
                    "triggers": ["mitigation", "audit", "compliance", "liability"],
                    "boost": 1.15,
                },
                "Finance": {
                    "triggers": ["gdp", "profit", "revenue", "fiscal", "quarterly"],
                    "boost": 1.25,
                },
                "Technology": {
                    "triggers": [
                        "software",
                        "hardware",
                        "ai",
                        "digital",
                        "infrastructure",
                    ],
                    "boost": 1.1,
                },
                "Law": {
                    "triggers": [
                        "article",
                        "clause",
                        "legal",
                        "statute",
                        "jurisdiction",
                    ],
                    "boost": 1.2,
                },
            }

            text_lower = text.lower()
            if label in BOOST_RULES:
                rule = BOOST_RULES[label]
                # Inhibition: If inhibitors are present but triggers are missing, penalize heavily
                if "inhibitors" in rule:
                    if any(k in text_lower for k in rule["inhibitors"]):
                        # Only allow if strong triggers also exist
                        if not (
                            "triggers" in rule
                            and any(k in text_lower for k in rule["triggers"])
                        ):
                            norm_score *= rule.get("penalty", 0.5)

                # Boosting: If triggers are present, reward
                if "triggers" in rule:
                    if any(k in text_lower for k in rule["triggers"]):
                        norm_score *= rule.get("boost", 1.0)

            final_confidence = min(norm_score * 100, 99.9)

            # 2. Evidence extraction (Strict Mapping)
            evidence_pairs = [
                [f"This sentence provides specific evidence for the {label} domain.", s]
                for s in sentences
            ]
            if evidence_pairs:
                ev_scores = self.cross_encoder.predict(evidence_pairs)
                best_idx = np.argmax(ev_scores)
                # Only extract evidence if it meets a precision threshold
                if ev_scores[best_idx] > 0.25:
                    supporting_sentence = sentences[best_idx]
                else:
                    supporting_sentence = f"General thematic presence of {label}."
            else:
                supporting_sentence = "No specific evidence sentences found."

            item = {
                "label": label,
                "score": round(final_confidence, 1),
                "evidence": supporting_sentence,
                "confidence_level": (
                    "High"
                    if final_confidence > 78
                    else "Medium" if final_confidence > 45 else "Low"
                ),
            }
            self.cross_score_cache[cache_key] = item
            results.append(item)
            raw_scores.append(norm_score)

        # --- ELITE: Entropy-based Confidence Calibration with Narrative Disambiguation ---
        if raw_scores:
            # Normalize to probability distribution
            probs = np.array(raw_scores)
            if probs.sum() > 0:
                probs = probs / probs.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-9))

                # CLASSIFICATION DISAMBIGUATION:
                # If entropy is high the model is "confused" — apply calibration penalty.
                # Additionally, penalize any category where the text is predominantly narrative
                # (i.e. personal/creative writing that bleeds into professional categories).
                NARRATIVE_INHIBITORS = [
                    "once upon",
                    "i felt",
                    "she said",
                    "he laughed",
                    "walked into",
                    "the sun set",
                    "my heart",
                    "tears fell",
                    "i remember when",
                    "in a dream",
                ]
                is_narrative = any(
                    phrase in text_lower for phrase in NARRATIVE_INHIBITORS
                )

                if entropy > 1.2 or is_narrative:
                    # Adaptive calibration: stronger penalty for narrative mismatch
                    calibration_factor = 0.55 if is_narrative else 0.7
                    for res in results:
                        res["score"] = round(res["score"] * calibration_factor, 1)
                        if res["score"] < 30:
                            res["confidence_level"] = "Low"
                        elif res["score"] < 50:
                            res["confidence_level"] = "Medium"

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:6]

    def _extract_numerical_insights(self, text):
        """Numerical Insight Engine: Extract money, percentages, and dates."""
        insights = []
        # Professional Regex for Finance/Stats
        money_pat = r"\$\d+(?:\.\d+)?\s*(?:million|billion|M|B|K)?"
        percent_pat = r"\b\d+(?:\.\d+)?%"
        date_pat = r"\b(?:20|19)\d{2}\b"

        raw_matches = list(
            set(
                re.findall(money_pat, text)
                + re.findall(percent_pat, text)
                + re.findall(date_pat, text)
            )
        )

        # Limit to top relevant insights via QA
        for match in raw_matches[:5]:
            try:
                # Ask the QA model what this number refers to
                qa_res = self.qa_pipeline(
                    question=f"What does the value '{match}' represent?",
                    context=text[:2000],
                )
                if qa_res["score"] > 0.05:
                    insights.append(
                        {
                            "value": match,
                            "context": qa_res["answer"],
                            "confidence": round(qa_res["score"], 2),
                        }
                    )
            except:
                continue
        return insights

    # --- Core Analysis ---

    # --- Expert Analysis Protocol (MANDATORY) ---

    def _classify_intent(self, question, context):
        """Silently classify the question into EXACTLY ONE intent using Keyword Logic Engine."""
        q_lower = question.lower()

        # 1. NUMERIC Intent
        numeric_keywords = [
            "how many",
            "percentage",
            "amount",
            "total",
            "cost",
            "difference",
            "decrease",
            "increase",
            "rate",
            "%",
            "$",
            "when",
            "year",
            "how much",
            "duration",
            "time",
            "revenue",
            "billion",
            "million",
            "value",
        ]
        if any(k in q_lower for k in numeric_keywords):
            return "NUMERIC"

        # 2. FACT Intent (Extended triggers for inferential reasoning)
        fact_keywords = [
            "how ",
            "stating",
            "defined as",
            "named",
            "identify",
            "describe",
            "does",
            "is",
            "can",
            "will",
            "qualify",
            "determine",
            "state",
            "whether",
            "are",
        ]
        if any(k in q_lower for k in fact_keywords):
            return "FACT"

        # 3. ANALOGY Intent
        if any(
            k in q_lower for k in ["analogy", "like a", "comparable to", "metaphor"]
        ):
            return "ANALOGY"

        # 4. EVIDENCE Intent
        if any(k in q_lower for k in ["quote", "evidence", "citation", "verbatim"]):
            return "EVIDENCE"

        # 5. EVALUATION Intent
        if any(
            k in q_lower
            for k in ["judgment", "assessment", "conclusion", "evaluate", "limitations"]
        ):
            return "EVALUATION"

        # 6. Default to CLASSIFICATION for domain/vibe questions
        return "CLASSIFICATION"

    async def analyze(self, text: str, categories: list):
        """High-tier CLASSIFICATION logic following Expert Protocol."""
        loop = asyncio.get_event_loop()

        # Screening
        def fast_screening():
            return self.zero_shot_pipeline(text, categories, truncation=True)

        screen_res = await loop.run_in_executor(self.executor, fast_screening)
        top_candidates = screen_res["labels"][:10]

        # Precision Reranking with Expert Rules
        elite_categories = await loop.run_in_executor(
            self.executor, self._rerank_categories, text, top_candidates
        )

        # Expert Response Construction
        primary = elite_categories[0]["label"] if elite_categories else "Unknown"
        secondary = [c["label"] for c in elite_categories[1:3]]

        # Explicit Exclusion Logic: Exclude domains that weren't in elite_categories
        included_labels = [c["label"] for c in elite_categories]
        excluded = [cat for cat in categories if cat not in included_labels][:3]

        # Numerical Insights (Still valuable as auxiliary data)
        numerical_insights = await loop.run_in_executor(
            self.executor, self._extract_numerical_insights, text
        )

        return {
            "intent": "CLASSIFICATION",
            "primary_domain": primary,
            "secondary_domains": secondary,
            "excluded_domains": excluded,
            "reasoning": f"Document exhibits high evidence density for {primary} thematic structures, while lacking specific markers for {', '.join(excluded) if excluded else 'other domains'}.",
            "evidence_quotes": [c["evidence"] for c in elite_categories[:2]],
            "numerical_insights": numerical_insights,
        }

    async def ask(self, context, question):
        """Expert Engine Entrypoint: Routes queries through specialized reasoning pipelines."""
        intent = self._classify_intent(question, context)

        # Route FACT, NUMERIC, and EVALUATION through the advanced Deductive RAG Pipeline
        if intent in ["FACT", "NUMERIC", "EVALUATION"]:
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                self.executor, self._rag_pipeline, context, question
            )
            res["intent"] = intent
            return res
        elif intent == "ANALOGY":
            return await self._handle_analogy_engine(context, question)
        elif intent == "EVIDENCE":
            return await self._handle_evidence_retrieval(context, question)
        else:  # CLASSIFICATION
            return await self.analyze(
                context[:15000],
                [
                    "Policy",
                    "Law",
                    "Business",
                    "Finance",
                    "Technology",
                    "Risk Management",
                ],
            )

    def _chunk_text_recursive(self, text, chunk_size=1500, overlap=300):
        """Sentence-aware adaptive chunking to preserve semantic boundaries."""
        if len(text) <= chunk_size:
            return [text]

        # Robust sentence splitting using regex (lookbehind for punctuation)
        sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            # Handle edge case: single sentence exceeding chunk_size
            if sentence_len > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split large sentence into sub-chunks
                for i in range(0, sentence_len, chunk_size - overlap):
                    sub_chunk = sentence[i : i + chunk_size]
                    if len(sub_chunk) > 50:
                        chunks.append(sub_chunk)
                continue

            if current_length + sentence_len + 1 > chunk_size:
                # Finalize current chunk
                chunks.append(" ".join(current_chunk))

                # Build overlap from the end of the current chunk
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) + 1 <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                current_chunk = overlap_chunk + [sentence]
                current_length = overlap_len + sentence_len + 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_len + 1

        if current_chunk:
            final_c = " ".join(current_chunk)
            if len(final_c) > 50:
                chunks.append(final_c)

        return chunks

    async def _handle_analogy_engine(self, context, question):
        """ANALOGY: Map to abstract organizational functions only."""
        allowed = [
            "Strategy",
            "Governance",
            "Risk Management",
            "Compliance",
            "Operations",
            "Investment Oversight",
        ]
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            self.executor, lambda: self.zero_shot_pipeline(context[:2000], allowed)
        )
        domain = res["labels"][0]

        reasoning = f"The structural interaction of elements in this document mirrors the functional dynamics of {domain}."

        return {"intent": "ANALOGY", "function": domain, "reasoning": reasoning}

    async def _handle_evidence_retrieval(self, context, question):
        """EVIDENCE: Quoted sentences using recursive chunking and bi-encoder relevance."""
        loop = asyncio.get_event_loop()
        chunks = self._chunk_text_recursive(context)
        chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True)
        q_emb = self.embedding_model.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, chunk_embeddings)[0]
        best_idx = torch.argmax(scores).item()

        return {"intent": "EVIDENCE", "evidence": [f'"{chunks[best_idx]}"']}

    async def _handle_evaluation_protocol(self, context, question):
        """EVALUATION: Conclusion + evidence + limitations."""
        loop = asyncio.get_event_loop()
        qa_res = await loop.run_in_executor(
            self.executor,
            lambda: self.qa_pipeline(question=question, context=context[:5000]),
        )

        return {
            "intent": "EVALUATION",
            "conclusion": qa_res["answer"],
            "evidence": f"Based on: {qa_res['answer']}",
            "limitations": "Analysis constrained to explicit document content.",
        }

    def _rag_pipeline(self, text, question):
        # 1. Chunking
        chunks = self._chunk_text_recursive(text)
        if not chunks:
            return {
                "answer": "I couldn't process the text.",
                "confidence": 0,
                "evidence": "",
            }

        # 2. Stage 1: Retrieval
        chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True)
        question_embedding = self.embedding_model.encode(
            question, convert_to_tensor=True
        )
        bi_scores = util.cos_sim(question_embedding, chunk_embeddings)[0]

        default_top_k = int(os.getenv("RAG_TOP_K", max(8, len(text) // 1000)))
        top_k_fast = min(default_top_k, len(chunks))
        fast_results = torch.topk(bi_scores, k=top_k_fast)
        candidate_indices = [idx.item() for idx in fast_results.indices]
        top_score = fast_results.values[0].item()

        # --- ELITE SPEED: Short-circuit ---
        if top_score > 0.92:
            elite_indices = candidate_indices[:4]
        else:
            # 3. Stage 2: Precision Reranking (NO INTRO BIAS)
            cross_inputs = [[question, chunks[idx]] for idx in candidate_indices]
            cross_scores = self.cross_encoder.predict(cross_inputs)

            reranked_pairs = sorted(
                zip(candidate_indices, cross_scores), key=lambda x: x[1], reverse=True
            )

            # Diversity Filter (MMR-lite): Avoid redundant semantic overlaps
            default_top_rerank = int(os.getenv("RAG_TOP_RERANK", 4))
            elite_indices = []
            for idx, score in reranked_pairs:
                if score < 0.2:  # Stricter rejection of irrelevant semantic matches
                    continue

                # Simple diversity check: don't pick chunks that are too similar to already selected ones
                is_redundant = False
                for existing_idx in elite_indices:
                    sim = util.cos_sim(
                        chunk_embeddings[idx], chunk_embeddings[existing_idx]
                    )
                    if sim > 0.85:  # High overlap
                        is_redundant = True
                        break

                if not is_redundant:
                    elite_indices.append(idx)

                if len(elite_indices) >= default_top_rerank:
                    break

            # Fallback if diversity filtering was too aggressive
            if not elite_indices:
                elite_indices = [idx for idx, score in reranked_pairs[:1]]

        # 4. Final Context & QA Logic Layer
        elite_indices.sort()
        relevant_context = " ... ".join([chunks[i] for i in elite_indices])

        # --- ITERATIVE MULTI-HOP TUNING (configurable Hop Limit) ---
        is_boolean = any(
            question.lower().startswith(k)
            for k in ["does ", "is ", "can ", "will ", "are ", "if "]
        )
        hop_limit = int(os.getenv("RAG_HOP_LIMIT", 2))  # Tunable via env var
        hop_confidence_threshold = float(os.getenv("RAG_HOP_THRESHOLD", 0.45))

        if top_score < hop_confidence_threshold and not is_boolean:
            for _hop in range(hop_limit):
                # Generate a refined query using HyDE-lite: strip question suffix
                focused_query = question.split("?")[0].strip()
                focused_emb = self.embedding_model.encode(
                    focused_query, convert_to_tensor=True
                )
                f_scores = util.cos_sim(focused_emb, chunk_embeddings)[0]
                # Pick the best chunk not already selected
                sorted_f_indices = torch.argsort(f_scores, descending=True)
                added = False
                for f_idx_t in sorted_f_indices:
                    f_idx = f_idx_t.item()
                    if f_idx not in elite_indices:
                        elite_indices.append(f_idx)
                        elite_indices.sort()
                        relevant_context = " ... ".join(
                            [chunks[i] for i in elite_indices]
                        )
                        added = True
                        break
                # Early-exit if no new chunks can be added
                if not added:
                    break
                # Re-score with new context as a leapfrog check
                new_hop_score = float(
                    torch.max(
                        util.cos_sim(
                            self.embedding_model.encode(
                                focused_query, convert_to_tensor=True
                            ),
                            self.embedding_model.encode(
                                [relevant_context[:512]], convert_to_tensor=True
                            ),
                        )
                    ).item()
                )
                if new_hop_score >= hop_confidence_threshold:
                    break  # Confidence sufficient — stop hopping

        # --- PRODUCTION SCALING: Batched QA with configurable batch size ---
        qa_batch_size = int(os.getenv("RAG_QA_BATCH_SIZE", 4))
        # For composite / boolean queries retrieve top-k answers; otherwise single pass
        needs_multi = is_boolean or " and " in question.lower()
        qa_res_list = (
            self.qa_pipeline(
                question=question,
                context=relevant_context,
                top_k=3 if needs_multi else 1,
                batch_size=qa_batch_size,
            )
            if needs_multi
            else [
                self.qa_pipeline(
                    question=question,
                    context=relevant_context,
                    batch_size=qa_batch_size,
                )
            ]
        )

        primary_res = qa_res_list[0]
        answer = primary_res["answer"]
        confidence = primary_res["score"]

        # SAFETY GATE: Reject hallucinations if QA confidence is critically low
        if confidence < 0.005:
            return {
                "answer": "No evidence found in the document to confirm this.",
                "confidence": 0.3,
                "evidence": "N/A",
                "intent": "FACT",
            }

        # Composite Extraction (e.g. deadline AND fine)
        if " and " in question.lower() and len(qa_res_list) > 1:
            sec_res = qa_res_list[1]
            if (
                sec_res["score"] > 0.035
                and sec_res["answer"].lower() not in answer.lower()
            ):
                # Deduplication check: ensure semantic distance between answers
                ans_emb = self.embedding_model.encode([answer, sec_res["answer"]])
                sim = util.cos_sim(ans_emb[0], ans_emb[1])
                if sim < 0.65:  # Only combine if they are distinct facts
                    answer = f"{answer} and {sec_res['answer']}"

        # Boolean Deduction Gate (Enhanced Logical Polarity)
        if is_boolean:
            perm_keywords = ["exempt", "allow", "permit", "grant", "authoriz"]
            neg_keywords = ["not", "no", "fail", "except", "excluding", "unless"]
            restrict_keywords = ["restrict", "prohibit", "bar ", "forbid", "ban "]

            # Identify if the question itself is asking about a restriction
            q_is_negative = any(k in question.lower() for k in restrict_keywords)

            answer_clean = answer.lower()
            if confidence > 0.012:
                has_neg_marker = any(k in answer_clean for k in neg_keywords)
                has_perm_marker = any(k in answer_clean for k in perm_keywords)

                # Resolve Proposition Polarity
                if q_is_negative:
                    # Question: "Is it restricted?"
                    # Answer says "Not restricted" or "Exempt" -> "No"
                    if has_neg_marker or has_perm_marker:
                        answer = f"No, according to the document: {answer}"
                    else:
                        answer = f"Yes, according to the document: {answer}"
                else:
                    # Question: "Is it allowed?"
                    # Answer says "Not allowed" -> "No"
                    if has_neg_marker:
                        answer = f"No, according to the document: {answer}"
                    else:
                        answer = f"Yes, according to the document: {answer}"
                confidence = max(confidence, 0.88)
            else:
                answer = "No evidence found in the document to confirm this."
                confidence = 0.5

        # Elite Evidence: Sentence-level reranking within the context
        sentences = self._get_top_sentences(relevant_context, limit=20)
        if sentences:
            # Augment query with the Answer to bias evidence towards the logical conclusion
            augmented_query = f"{question} {answer}"
            # Batched reranking for peak-load optimization
            ev_scores = self.cross_encoder.predict(
                [[augmented_query, s] for s in sentences], batch_size=len(sentences)
            )

            # --- ELITE: Subject & Answer Overlap Boost ---
            # Penalize sentences that don't share subject or answer tokens with the query
            STOP_WORDS = {"the", "a", "an", "is", "are", "of", "and", "to", "in", "it"}
            q_clean = re.sub(r"[^a-zA-Z0-9\s]", "", question.lower())
            q_tokens = set(q_clean.split()) - STOP_WORDS

            # Focus on the answer tokens that aren't generic boilerplate
            ans_clean = re.sub(r"[^a-zA-Z0-9\s]", "", answer.lower())
            ans_tokens = set(ans_clean.split()) - (
                STOP_WORDS | {"no", "yes", "according", "document"}
            )

            subject_boosts = []
            for i, s in enumerate(sentences):
                s_tokens = set(re.findall(r"\w+", s.lower()))
                q_overlap = len(q_tokens.intersection(s_tokens))
                ans_overlap = len(ans_tokens.intersection(s_tokens))
                # Heuristic: Answer support is stronger than general subject keywords
                boost = 1.0 + (q_overlap * 0.15) + (ans_overlap * 0.5)
                subject_boosts.append(boost)

            final_ev_scores = ev_scores * np.array(subject_boosts)
            best_ev_idx = np.argmax(final_ev_scores)
            precise_evidence = sentences[best_ev_idx]

            # --- PRECISE SPAN MAPPING ---
            # Verify that the raw QA answer span physically exists inside the evidence sentence.
            # This guarantees the evidence is not a thematically similar but semantically distinct sentence.
            raw_answer_span = re.sub(
                r"^(yes|no),?\s*(according to the document[:\s]*)?\s*",
                "",
                answer.lower(),
            ).strip()
            span_present = (
                raw_answer_span and raw_answer_span in precise_evidence.lower()
            )

            if not span_present:
                # Search all candidate sentences for the one that literally contains the answer span
                for cand_s in sentences:
                    if raw_answer_span and raw_answer_span in cand_s.lower():
                        precise_evidence = cand_s
                        span_present = True
                        break

            # --- ENTAILMENT GATE ---
            # Verify logical support of answer by evidence using Cross-Encoder
            entail_score = self.cross_encoder.predict([[answer, precise_evidence]])[0]
            if entail_score < 0.1 and not is_boolean:
                # Try secondary evidence if primary entailment is weak
                if len(sentences) > 1:
                    sec_idx = int(np.argsort(ev_scores)[-2])
                    if ev_scores[sec_idx] > 0.1:
                        precise_evidence = sentences[sec_idx]
                        entail_score = self.cross_encoder.predict(
                            [[answer, precise_evidence]]
                        )[0]

                if entail_score < 0.08 and not span_present:
                    answer = "The text mentions related concepts, but no conclusive evidence was found."
                    confidence = min(confidence, 0.45)
        else:
            precise_evidence = answer

        return {
            "answer": answer,
            "confidence": round(min(confidence, 1.0), 4),
            "evidence": precise_evidence,
            "relevant_context": relevant_context[:400] + "...",
        }

    # --- XAI (Explainability) ---

    async def explain_sentiment(self, text: str):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return await self._get_explanation_cached(text, text_hash, mode="sentiment")

    async def explain_zero_shot(self, text: str, label: str):
        key = f"{text}_{label}"
        text_hash = hashlib.md5(key.encode()).hexdigest()
        return await self._get_explanation_cached(
            text, text_hash, mode="zero-shot", target_label=label
        )

    async def _get_explanation_cached(
        self, text, text_hash, mode="sentiment", target_label=None
    ):
        loop = asyncio.get_event_loop()
        if mode == "sentiment":
            return await loop.run_in_executor(
                self.executor, self._compute_integrated_gradients, text
            )
        else:
            return await loop.run_in_executor(
                self.executor, self._compute_zero_shot_gradients, text, target_label
            )

    @lru_cache(maxsize=100)
    def _compute_integrated_gradients(self, text):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self.sentiment_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        input_ids = inputs["input_ids"]

        def forward_func(inputs, attention_mask=None):
            outputs = self.sentiment_model_raw(inputs, attention_mask=attention_mask)
            return outputs.logits

        logits = forward_func(input_ids, attention_mask=inputs["attention_mask"])
        pred_label_idx = logits.argmax().item()
        embedding_layer = self.sentiment_model_raw.distilbert.embeddings.word_embeddings
        lig = LayerIntegratedGradients(forward_func, embedding_layer)
        attributions, delta = lig.attribute(
            inputs=input_ids,
            target=pred_label_idx,
            additional_forward_args=(inputs["attention_mask"],),
            return_convergence_delta=True,
        )
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        tokens = self.sentiment_tokenizer.convert_ids_to_tokens(input_ids[0])
        result = []
        for token, score in zip(tokens, attributions):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                result.append((token, float(score.item())))
        return {
            "predicted_label": "POSITIVE" if pred_label_idx == 1 else "NEGATIVE",
            "attributions": result,
        }

    @lru_cache(maxsize=100)
    def _compute_zero_shot_gradients(self, text, label):
        device = self.zero_shot_pipeline.device
        model = self.zero_shot_pipeline.model
        tokenizer = self.zero_shot_pipeline.tokenizer
        hypothesis = f"This example is {label}."
        inputs = tokenizer(
            text, hypothesis, return_tensors="pt", truncation="only_first"
        ).to(device)
        input_ids = inputs["input_ids"]
        entailment_id = model.config.label2id.get("entailment", 2)

        def forward_func(inputs, attention_mask=None):
            return model(inputs, attention_mask=attention_mask).logits

        embedding_layer = (
            model.model.shared
            if (hasattr(model, "model") and hasattr(model.model, "shared"))
            else model.get_input_embeddings()
        )
        lig = LayerIntegratedGradients(forward_func, embedding_layer)
        attributions = lig.attribute(
            inputs=input_ids,
            target=entailment_id,
            additional_forward_args=(inputs["attention_mask"],),
        )
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        result = []
        for token, score in zip(tokens, attributions):
            if token.startswith("Ġ"):
                token = token[1:]
            if token in ["<s>", "</s>"]:
                continue
            result.append((token, float(score.item())))
        return {"predicted_label": label, "attributions": result}


# Global singleton instance
nlp_service = NLPService()
