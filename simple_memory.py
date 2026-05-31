# simple_memory.py – улучшенная память: дедупликация, важность, временные метки, ассоциации, эфемерность, консолидация, временные срезы, поиск повторных вопросов, батчинг эмбеддингов, кеш, TTL, разделение persona/state, трекер настроения Киры, инициатива, режим молчания, FAISS IDMap, сохранение эмбеддингов, асинхронная запись
import asyncio
import hashlib
import json
import logging
import os
import pickle
import re
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple, Any, Union
import numpy as np

# Опциональные библиотеки
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from groq import Groq, RateLimitError, APITimeoutError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimpleMemory")

# LRU кеш эмбеддингов
class LRUCache:
    def __init__(self, maxsize=5000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
    
    def __contains__(self, key):
        return key in self.cache
    
    def __getitem__(self, key):
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
    
    def get(self, key, default=None):
        if key in self.cache:
            return self[key]
        return default

_embedding_cache = LRUCache(maxsize=5000)

def get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

@dataclass
class Fact:
    """Один атомарный факт в памяти"""
    text: str
    category: str
    trust: float = 0.5
    importance: float = 0.5
    base_importance: float = 0.5
    source: str = "user"
    timestamp: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    usefulness: float = 0.5
    ttl_seconds: Optional[int] = None
    embedding: Optional[List[float]] = None
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(f"{self.text}{self.timestamp.isoformat()}".encode()).hexdigest()[:12]
        if self.base_importance == 0.5 and self.importance != 0.5:
            self.base_importance = self.importance

    def is_expired(self, now: datetime) -> bool:
        if self.ttl_seconds is None:
            return False
        return (now - self.timestamp).total_seconds() > self.ttl_seconds

    def update_dynamic_importance(self, now: datetime):
        age_days = (now - self.timestamp).total_seconds() / 86400.0
        hours_since_last_use = (now - self.last_used).total_seconds() / 3600.0 if self.last_used else age_days * 24
        usage_bonus = min(0.3, self.access_count / 50.0)
        age_penalty = min(0.4, age_days / 30.0)
        idle_penalty = min(0.3, hours_since_last_use / 48.0)
        new_importance = self.base_importance + usage_bonus - age_penalty - idle_penalty
        self.importance = max(0.1, min(1.0, new_importance))

    def update_usefulness(self, recency_bonus: float, max_access: int):
        freq_score = min(1.0, self.access_count / max(1, max_access)) * 0.5
        recency_score = recency_bonus * 0.3
        trust_score = self.trust * 0.2
        self.usefulness = min(1.0, freq_score + recency_score + trust_score)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = d['timestamp'].isoformat()
        d['last_used'] = d['last_used'].isoformat()
        if self.embedding is not None:
            d['embedding'] = self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding
        else:
            d['embedding'] = None
        d['ttl_seconds'] = self.ttl_seconds
        return d

    @classmethod
    def from_dict(cls, data: Dict):
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'last_used' in data and data['last_used']:
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        else:
            data['last_used'] = data['timestamp']
        if data.get('embedding') is not None:
            data['embedding'] = np.array(data['embedding'], dtype=np.float32)
        else:
            data['embedding'] = None
        if 'base_importance' not in data:
            data['base_importance'] = data.get('importance', 0.5)
        data['ttl_seconds'] = data.get('ttl_seconds')
        return cls(**data)


class SimpleMemoryManager:
    def __init__(self, groq_client: Groq, max_facts: int = 2000, embedding_model: str = "intfloat/multilingual-e5-small", timezone_offset: int = 0):
        self.client = groq_client
        self.max_facts = max_facts
        self.embedding_model_name = embedding_model
        self.timezone_offset = timezone_offset

        self.facts: Dict[str, Fact] = {}
        self.category_index: Dict[str, List[str]] = defaultdict(list)
        self.associations: Dict[str, Set[str]] = defaultdict(set)
        self.working_memory: List[Tuple[datetime, str]] = []
        self.max_working_memory = 40
        self.short_term_buffer: List[str] = []
        self.max_short_term = 50

        # Эмбеддеры и индексы
        self.encoder = None
        self.faiss_index = None
        self.faiss_id_to_fact_id: Dict[int, str] = {}
        self.embedding_dim = 384  # для multilingual-e5-small
        self.bm25_index = None
        self.bm25_docs: List[str] = []

        self._init_embedder()
        self._init_faiss()
        self._init_bm25()

        self.max_access_seen = 0

        # Состояние Киры
        self.kira_mood: float = 0.5
        self.consecutive_short_replies: int = 0
        self.short_reply_threshold: int = 3
        self.laconic_mode: bool = False
        self.last_user_message_time: Optional[datetime] = None
        self.ids_of_unnoticed_important_facts: Set[str] = set()
        
        # Для анти-спама инициативы
        self.last_initiative_time: Optional[datetime] = None
        self.pending_initiative_fact: Optional[str] = None
        
        # Для отслеживания асинхронных FAISS задач
        self._pending_faiss_tasks: List[asyncio.Task] = []
        
        # Автосохранение
        self.auto_save_path: Optional[str] = None
        self.auto_save_interval: int = 300
        self._save_task: Optional[asyncio.Task] = None

        self.stats = {
            'memories_created': 0,
            'memories_forgotten': 0,
            'recalls_performed': 0,
            'avg_recall_time_ms': 0,
            'last_consolidation': None
        }

    def _now(self) -> datetime:
        """Возвращает текущее время с учётом часового пояса"""
        return datetime.now() + timedelta(hours=self.timezone_offset)

    # ---------- Инициализация ----------
    def _init_embedder(self):
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer(self.embedding_model_name)
                self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
                logger.info(f"Эмбеддер {self.embedding_model_name} загружен, размерность: {self.embedding_dim}")
            except Exception as e:
                logger.error(f"Ошибка загрузки эмбеддера: {e}")
                self.encoder = None
        else:
            logger.warning("sentence-transformers не установлен")

    def _init_faiss(self):
        if FAISS_AVAILABLE and self.encoder is not None:
            base_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index = faiss.IndexIDMap(base_index)
            self.faiss_id_to_fact_id = {}
            logger.info(f"FAISS IndexIDMap инициализирован (размерность {self.embedding_dim})")
        else:
            self.faiss_index = None

    def _init_bm25(self):
        if BM25_AVAILABLE:
            self.bm25_index = None
            self.bm25_docs = []
            logger.info("BM25 готов")
        else:
            self.bm25_index = None

    def _update_bm25(self):
        if not BM25_AVAILABLE:
            return
        docs = [fact.text for fact in self.facts.values()]
        if docs != self.bm25_docs:
            self.bm25_docs = docs
            tokenized = [doc.lower().split() for doc in docs]
            self.bm25_index = BM25Okapi(tokenized) if docs else None

    # ---------- Эмбеддинги с LRU кешем ----------
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        if self.encoder is None:
            return None
        h = get_text_hash(text)
        if h in _embedding_cache:
            return _embedding_cache[h]
        try:
            emb = self.encoder.encode(text, normalize_embeddings=True)
            _embedding_cache[h] = emb
            return emb
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга: {e}")
            return None

    def _get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        if self.encoder is None:
            return [None] * len(texts)
        results = []
        to_encode = []
        indices = []
        for i, t in enumerate(texts):
            h = get_text_hash(t)
            if h in _embedding_cache:
                results.append((i, _embedding_cache[h]))
            else:
                to_encode.append(t)
                indices.append(i)
        if to_encode:
            try:
                embs = self.encoder.encode(to_encode, normalize_embeddings=True)
                for idx, emb in zip(indices, embs):
                    h = get_text_hash(to_encode[indices.index(idx)])
                    _embedding_cache[h] = emb
                    results.append((idx, emb))
            except Exception as e:
                logger.error(f"Ошибка пакетного кодирования: {e}")
                for idx in indices:
                    results.append((idx, None))
        ordered = [None] * len(texts)
        for i, emb in results:
            ordered[i] = emb
        return ordered

    # ---------- FAISS операции (IDMap) ----------
    def _add_to_faiss(self, fact_id: str, embedding: np.ndarray):
        if self.faiss_index is None:
            return
        vec = embedding.reshape(1, -1).astype(np.float32)
        try:
            int_id = int(fact_id[:8], 16)
        except:
            int_id = hash(fact_id) % (2**31)
        self.faiss_index.add_with_ids(vec, np.array([int_id], dtype=np.int64))
        self.faiss_id_to_fact_id[int_id] = fact_id

    async def _async_add_to_faiss(self, fact_id: str, embedding: np.ndarray):
        task = asyncio.create_task(asyncio.to_thread(self._add_to_faiss, fact_id, embedding))
        self._pending_faiss_tasks.append(task)
        self._pending_faiss_tasks = [t for t in self._pending_faiss_tasks if not t.done()]

    async def flush_faiss_tasks(self):
        """Ожидает завершения всех асинхронных операций FAISS"""
        if self._pending_faiss_tasks:
            await asyncio.gather(*self._pending_faiss_tasks, return_exceptions=True)
            self._pending_faiss_tasks.clear()

    def _remove_from_faiss(self, fact_id: str):
        if self.faiss_index is None:
            return
        try:
            int_id = int(fact_id[:8], 16)
        except:
            int_id = hash(fact_id) % (2**31)
        try:
            self.faiss_index.remove_ids(np.array([int_id], dtype=np.int64))
            if int_id in self.faiss_id_to_fact_id:
                del self.faiss_id_to_fact_id[int_id]
        except Exception as e:
            logger.warning(f"Не удалось удалить {fact_id} из FAISS: {e}")

    def _semantic_search(self, query_emb: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        query = query_emb.reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_index.search(query, min(k, self.faiss_index.ntotal))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx in self.faiss_id_to_fact_id:
                fact_id = self.faiss_id_to_fact_id[idx]
                results.append((fact_id, float(score)))
        return results

    def _bm25_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        if self.bm25_index is None or not self.bm25_docs:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in indexed[:k]:
            if idx < len(self.bm25_docs):
                for fid, fact in self.facts.items():
                    if fact.text == self.bm25_docs[idx]:
                        results.append((fid, float(score)))
                        break
        return results

    # ---------- JSON extraction ----------
    def _extract_json(self, text: str) -> Optional[Union[List, Dict]]:
        start = -1
        for i, ch in enumerate(text):
            if ch in '[{':
                start = i
                break
        if start == -1:
            return None
        stack = []
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if ch in '[{':
                stack.append(ch)
            elif ch == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                if not stack:
                    end = i
                    break
            elif ch == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                if not stack:
                    end = i
                    break
        if end == -1:
            return None
        json_str = text[start:end+1]
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Не удалось распарсить JSON: {e}\nСтрока: {json_str[:200]}")
            return None

    # ---------- Groq вызов с retry ----------
    async def _call_groq_with_retry(self, messages, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(messages=messages, **kwargs)
                )
            except (RateLimitError, APITimeoutError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Ошибка Groq, retry через {wait:.1f}с: {e}")
                await asyncio.sleep(wait)
            except Exception as e:
                raise
        raise RuntimeError("Max retries exceeded")

    # ---------- Извлечение фактов через LLM ----------
    async def extract_facts_from_dialogue(self, messages: List[Dict[str, str]], existing_fact_texts: List[str] = None) -> List[Dict]:
        system_prompt = """Ты — система извлечения фактов из диалога. Из последних сообщений выдели НОВЫЕ факты о пользователе (User) и о девушке Кире (Kira). Факты должны быть краткими утверждениями в третьем лице.

Категория: начинается с user., kira., system., general.
Для Киры используй:
- kira.persona — постоянные черты, биография, привычки (возраст, город, кот, характер)
- kira.state — временное состояние (настроение, желания, самочувствие)

importance (0..1):
- 0.9-1.0: критичные факты (имя, возраст, место жительства, важные правила)
- 0.7-0.8: важные предпочтения, привычки, работа
- 0.5-0.6: обычная информация, хобби
- 0.3-0.4: эфемерные факты (приветствия, текущее настроение)
- 0.1-0.2: очень временные (пользователь сказал "ок", "спасибо")

trust (0..1):
- 0.9-1.0: прямая информация
- 0.7-0.8: из контекста
- 0.5-0.6: предположение

Верни JSON массив. Пример:
[{"text": "Пользователь любит чёрный кофе", "category": "user.preference", "trust": 0.85, "importance": 0.7},
 {"text": "Кире 19 лет", "category": "kira.persona", "trust": 0.95, "importance": 0.95}]

Если новых фактов нет, верни [].
"""
        existing_block = ""
        if existing_fact_texts:
            existing_block = "\nУже есть факты (не извлекай заново):\n" + "\n".join(f"- {t}" for t in existing_fact_texts[:30])

        user_content = f"{existing_block}\n\nДиалог:\n"
        for msg in messages[-6:]:
            role = "User" if msg["role"] == "user" else "Kira"
            user_content += f"{role}: {msg['content']}\n"

        try:
            response = await self._call_groq_with_retry(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=500,
                top_p=0.9
            )
            if not response.choices:
                logger.warning("Groq вернул пустой ответ")
                return []
            content = response.choices[0].message.content
            parsed = self._extract_json(content)
            if isinstance(parsed, list):
                facts = parsed
            elif isinstance(parsed, dict):
                facts = [parsed]
            else:
                facts = []

            valid = []
            for f in facts:
                if "text" in f and len(f["text"]) > 3:
                    category = f.get("category", "general.unknown")
                    if category.startswith("kira.") and not category.startswith("kira.state"):
                        temporal_markers = ["устал", "устала", "хочу", "чувствую", "настроение", "сегодня", "сейчас", "не хочу", "грустно", "весело", "сплю"]
                        if any(marker in f["text"].lower() for marker in temporal_markers):
                            category = "kira.state"
                        else:
                            category = "kira.persona"
                    elif not any(category.startswith(p) for p in ["user.", "kira.", "system.", "general."]):
                        category = "general." + category
                    f["category"] = category
                    f.setdefault("trust", 0.5)
                    f.setdefault("importance", 0.5)
                    valid.append(f)
            return valid
        except Exception as e:
            logger.error(f"Ошибка извлечения фактов: {e}")
            return []

    def _is_duplicate(self, text: str, category: str, embedding: np.ndarray, threshold: float = 0.85) -> Tuple[bool, Optional[str]]:
        if embedding is None:
            return False, None
        results = self._semantic_search(embedding, k=3)
        for fid, score in results:
            if score < threshold:
                continue
            fact = self.facts.get(fid)
            if fact and fact.category == category and abs(len(fact.text) - len(text)) / max(len(fact.text), len(text)) < 0.3:
                return True, fid
        return False, None

    def _cleanup_expired_ephemeral(self):
        now = self._now()
        to_delete = []
        for fid, fact in self.facts.items():
            if fact.is_expired(now):
                to_delete.append(fid)
        for fid in to_delete:
            self.forget(fid)
            self.stats['memories_forgotten'] += 1
            logger.info(f"Удалён эфемерный факт {fid} по TTL")

    async def remember_facts(self, dialogue: List[Dict[str, str]], assistant_response: str = None) -> List[str]:
        if not dialogue:
            return []
        existing_texts = [fact.text for fact in self.facts.values()]
        new_facts = await self.extract_facts_from_dialogue(dialogue, existing_texts)
        if not new_facts:
            return []

        saved_ids = []
        new_fact_ids = []

        texts_to_embed = [f["text"] for f in new_facts]
        embeddings = self._get_embeddings_batch(texts_to_embed)

        for fact_dict, emb in zip(new_facts, embeddings):
            text = fact_dict["text"]
            category = fact_dict.get("category", "general")
            trust = fact_dict.get("trust", 0.5)
            importance = fact_dict.get("importance", 0.5)

            if assistant_response and any(w in assistant_response.lower() for w in ["да", "точно", "запомнила", "верно"]):
                trust = min(0.95, trust + 0.1)

            is_dup, dup_id = self._is_duplicate(text, category, emb, threshold=0.85) if emb is not None else (False, None)
            if is_dup and dup_id:
                existing = self.facts[dup_id]
                if trust > existing.trust or importance > existing.importance:
                    self.forget(dup_id)
                    logger.info(f"Дубликат '{existing.text}' заменён на '{text}' (trust={trust})")
                else:
                    logger.info(f"Дубликат игнорирован: '{text}' уже есть как '{existing.text}'")
                    continue

            ttl_seconds = None
            if importance < 0.4 or "ephemeral" in category:
                ttl_seconds = 3600 * 2

            fact = Fact(
                text=text,
                category=category,
                trust=trust,
                base_importance=importance,
                importance=importance,
                source="assistant" if "Кира" in text or "kira" in category else "user",
                ttl_seconds=ttl_seconds,
                embedding=emb
            )
            if emb is not None:
                await self._async_add_to_faiss(fact.id, emb)

            self.facts[fact.id] = fact
            self.category_index[category].append(fact.id)
            saved_ids.append(fact.id)
            new_fact_ids.append(fact.id)
            self.stats['memories_created'] += 1
            logger.info(f"Запомнен [{fact.id}]: {text[:50]}... (cat={category}, trust={trust}, imp={importance})")

            if importance > 0.8 and category.startswith("user."):
                self.ids_of_unnoticed_important_facts.add(fact.id)

            if len(self.facts) > self.max_facts:
                self._forget_least_useful(int(self.max_facts * 0.1))

        # Создаём ассоциации
        for i in range(len(new_fact_ids)):
            for j in range(i+1, len(new_fact_ids)):
                self.associations.setdefault(new_fact_ids[i], set()).add(new_fact_ids[j])
                self.associations.setdefault(new_fact_ids[j], set()).add(new_fact_ids[i])

        self._cleanup_expired_ephemeral()
        await self._async_update_bm25()
        return saved_ids

    async def _async_update_bm25(self):
        await asyncio.to_thread(self._update_bm25)

    def _forget_least_useful(self, count: int):
        now = self._now()
        scored = []
        for fid, fact in self.facts.items():
            if fact.category.startswith("system."):
                continue
            age_days = (now - fact.timestamp).total_seconds() / 86400.0
            age_factor = min(1.0, age_days / 365.0) * 0.3 + 0.7
            forget_score = fact.importance * fact.trust * fact.usefulness * age_factor
            scored.append((fid, forget_score))
        scored.sort(key=lambda x: x[1])
        to_forget = [fid for fid, _ in scored[:count]]
        for fid in to_forget:
            self.forget(fid)

    def forget(self, fact_id: str) -> bool:
        if fact_id not in self.facts:
            return False
        fact = self.facts[fact_id]
        if fact.embedding is not None:
            self._remove_from_faiss(fact_id)
        if fact.category in self.category_index:
            self.category_index[fact.category] = [fid for fid in self.category_index[fact.category] if fid != fact_id]
        if fact_id in self.associations:
            for assoc_id in self.associations[fact_id]:
                self.associations[assoc_id].discard(fact_id)
            del self.associations[fact_id]
        del self.facts[fact_id]
        self.stats['memories_forgotten'] += 1
        self._update_bm25()
        return True

    def get_time_ago(self, timestamp: datetime) -> str:
        delta = self._now() - timestamp
        seconds = delta.total_seconds()
        if seconds < 60:
            return "только что"
        if seconds < 3600:
            minutes = int(seconds // 60)
            return f"{minutes} мин. назад"
        if seconds < 86400:
            hours = int(seconds // 3600)
            return f"{hours} ч. назад"
        days = delta.days
        if days == 1:
            return "вчера"
        if days == 2:
            return "позавчера"
        if days < 7:
            return f"{days} дн. назад"
        if days < 14:
            return "неделю назад"
        if days < 21:
            return "2 недели назад"
        if days < 28:
            return "3 недели назад"
        if days < 35:
            return "месяц назад"
        if days < 60:
            return "больше месяца назад"
        if days < 90:
            return "2 месяца назад"
        if days < 180:
            return "несколько месяцев назад"
        if days < 365:
            months = days // 30
            return f"{months} мес. назад"
        years = days // 365
        return f"{years} год{ 'а' if years % 10 == 1 and years != 11 else 'ов'} назад"

    def parse_time_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        now = self._now()
        query_lower = query.lower()
        if "за последний час" in query_lower or "последний час" in query_lower:
            return (now - timedelta(hours=1), now)
        if "за последние 3 часа" in query_lower:
            return (now - timedelta(hours=3), now)
        if "вчера" in query_lower:
            yesterday = now - timedelta(days=1)
            return (yesterday.replace(hour=0, minute=0, second=0), yesterday.replace(hour=23, minute=59, second=59))
        if "на прошлой неделе" in query_lower or "за прошлую неделю" in query_lower:
            return (now - timedelta(days=7), now)
        if "за последние 2 недели" in query_lower:
            return (now - timedelta(days=14), now)
        if "месяц назад" in query_lower:
            return (now - timedelta(days=30), now)
        return None

    async def find_similar_recent_messages(self, query: str, max_age_hours: float = 1.0, similarity_threshold: float = 0.7) -> List[Tuple[datetime, str, float]]:
        if not self.working_memory or self.encoder is None:
            return []
        now = self._now()
        cutoff = now - timedelta(hours=max_age_hours)
        user_messages = []
        for ts, msg in self.working_memory:
            if ts < cutoff:
                continue
            if msg.startswith("User:"):
                content = msg[5:].strip() if msg.startswith("User:") else msg
                user_messages.append((ts, content))
        if not user_messages:
            return []
        q_emb = self._get_embedding(query)
        if q_emb is None:
            return []
        results = []
        for ts, msg_text in user_messages:
            msg_emb = self._get_embedding(msg_text)
            if msg_emb is None:
                continue
            sim = float(np.dot(q_emb, msg_emb))
            if sim >= similarity_threshold:
                results.append((ts, msg_text, sim))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    async def filter_facts_for_query(self, query: str, candidate_facts: List[Fact]) -> List[Fact]:
        if not candidate_facts:
            return []
        
        # Сначала семантическая сортировка
        q_emb = self._get_embedding(query)
        if q_emb is not None:
            for fact in candidate_facts:
                if fact.embedding is not None:
                    fact._sem_score = float(np.dot(q_emb, fact.embedding))
                else:
                    fact._sem_score = 0.0
            candidate_facts.sort(key=lambda f: getattr(f, '_sem_score', 0), reverse=True)
        
        # Берём топ-15 для LLM
        top_candidates = candidate_facts[:15]
        
        facts_list = []
        for f in top_candidates:
            time_ago = self.get_time_ago(f.timestamp)
            facts_list.append({
                "id": f.id,
                "text": f.text,
                "category": f.category,
                "time": time_ago,
                "trust": f.trust,
                "importance": f.importance
            })
        
        facts_json = json.dumps(facts_list, ensure_ascii=False, indent=2)
        system_prompt = """Ты — помощник для отбора фактов из памяти. Твоя задача — выбрать из предоставленного списка фактов только те, которые ДЕЙСТВИТЕЛЬНО необходимы для ответа на вопрос пользователя.

Правила:
- Оставь только факты, напрямую связанные с вопросом.
- Факты о личности пользователя (имя, возраст) оставляй только если вопрос о пользователе.
- Факты о Кире (возраст, город, кот) оставляй только если вопрос о ней или её жизни.
- Системные правила (system.*) никогда не включай — они уже в промпте.
- Если ни один факт не нужен, верни пустой массив.
- Будь строг: лучше вернуть пустой массив, чем дать лишние факты.

Верни ТОЛЬКО JSON-массив идентификаторов фактов (id), например: ["id1", "id2"].
Никаких пояснений.
"""
        user_content = f"""Вопрос пользователя: {query}

Список фактов (каждый с id, текстом, категорией):
{facts_json}

Выбери id фактов, которые помогут ответить на вопрос."""
        try:
            response = await self._call_groq_with_retry(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=500,
                top_p=0.9
            )
            if not response.choices:
                logger.warning("Groq вернул пустой ответ")
                return candidate_facts[:3] if candidate_facts else []
            content = response.choices[0].message.content
            parsed = self._extract_json(content)
            if isinstance(parsed, list):
                selected_ids = set(parsed)
                logger.info(f"Фильтрация: на входе {len(candidate_facts)}, на выходе {len(selected_ids)}")
                result = [f for f in candidate_facts if f.id in selected_ids]
                if not result and candidate_facts:
                    logger.info("LLM не выбрал факты, возвращаю топ-3 по семантике")
                    return candidate_facts[:3]
                return result
            else:
                logger.warning(f"Не удалось распарсить фильтрацию: {content[:200]}")
                return candidate_facts[:3] if candidate_facts else []
        except Exception as e:
            logger.error(f"Ошибка фильтрации фактов: {e}")
            return candidate_facts[:3] if candidate_facts else []

    async def is_working_memory_relevant(self, query: str, working_memory_messages: List[Tuple[datetime, str]]) -> bool:
        if not working_memory_messages:
            return False
        lines = []
        for ts, msg in working_memory_messages[-6:]:
            time_ago = self.get_time_ago(ts)
            lines.append(f"  [{time_ago}] {msg}")
        wm_block = "\n".join(lines)
        prompt = f"""Вопрос пользователя: «{query}»

Последние сообщения диалога:
{wm_block}

Нужен ли контекст этих сообщений, чтобы корректно ответить на вопрос?
Ответь одним словом: YES или NO."""
        try:
            response = await self._call_groq_with_retry(
                model="qwen/qwen3-32b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            if not response.choices:
                return False
            answer = response.choices[0].message.content.strip().upper()
            return answer.startswith("YES")
        except Exception as e:
            logger.error(f"Ошибка is_working_memory_relevant: {e}")
            return False

    def _update_kira_mood(self, user_msg: str, kira_reply: str = ""):
        positive_words = ["хорошо", "отлично", "круто", "замечательно", "радость", "весело", "спасибо", "❤️", "😊", "🔥"]
        negative_words = ["плохо", "ужасно", "грустно", "тоскливо", "бесит", "надоело", "сволочь", "дурак", "😡", "💀"]
        mood_change = 0.0
        for w in positive_words:
            if w in user_msg.lower():
                mood_change += 0.05
                break
        for w in negative_words:
            if w in user_msg.lower():
                mood_change -= 0.08
                break
        if kira_reply:
            for w in positive_words:
                if w in kira_reply.lower():
                    mood_change += 0.03
                    break
            for w in negative_words:
                if w in kira_reply.lower():
                    mood_change -= 0.05
                    break
        self.kira_mood = max(0.0, min(1.0, self.kira_mood + mood_change))
        self.kira_mood = self.kira_mood * 0.95 + 0.5 * 0.05

    def get_temperature(self) -> float:
        base = 0.85
        mood_factor = (self.kira_mood - 0.5) * 0.3
        if self.laconic_mode:
            mood_factor -= 0.2
        return max(0.2, min(1.2, base + mood_factor))

    def update_laconic_mode(self, user_msg: str):
        if len(user_msg.strip()) < 10:
            self.consecutive_short_replies += 1
        else:
            self.consecutive_short_replies = 0
        if self.consecutive_short_replies >= self.short_reply_threshold:
            self.laconic_mode = True
        else:
            if len(user_msg.strip()) > 20:
                self.laconic_mode = False

    async def check_for_initiative(self, last_user_msg_time: datetime) -> Optional[str]:
        # Ждём ответа на предыдущий вопрос
        if self.pending_initiative_fact:
            return None
        
        if not self.ids_of_unnoticed_important_facts:
            return None
        
        # Не чаще раза в 5 минут
        if self.last_initiative_time and (self._now() - self.last_initiative_time).total_seconds() < 300:
            return None
        
        if self.last_user_message_time and (self._now() - self.last_user_message_time).total_seconds() < 60:
            return None
        
        best_fact = None
        best_importance = 0
        for fid in list(self.ids_of_unnoticed_important_facts):
            fact = self.facts.get(fid)
            if fact and fact.importance > best_importance:
                best_importance = fact.importance
                best_fact = fact
        
        if best_fact and best_importance > 0.8:
            self.ids_of_unnoticed_important_facts.discard(best_fact.id)
            self.last_initiative_time = self._now()
            self.pending_initiative_fact = best_fact.id
            return f"Кстати, {best_fact.text.lower()}. Расскажешь подробнее?"
        return None

    async def recall_facts(self, query: str, limit: int = 8, time_range: Optional[Tuple[datetime, datetime]] = None, use_associations: bool = True) -> List[Tuple[Fact, str]]:
        start_time = datetime.now()
        self.stats['recalls_performed'] += 1
        now = self._now()

        candidate_ids = set()
        q_emb = self._get_embedding(query)
        if q_emb is not None:
            sem_results = self._semantic_search(q_emb, k=limit*3)
            for fid, score in sem_results:
                if score > 0.5:
                    candidate_ids.add(fid)

        bm25_results = self._bm25_search(query, k=limit*2)
        for fid, score in bm25_results:
            candidate_ids.add(fid)

        query_lower = query.lower()
        for fid, fact in self.facts.items():
            if query_lower in fact.text.lower():
                candidate_ids.add(fid)

        if use_associations:
            assoc_ids = set()
            for fid in list(candidate_ids):
                if fid in self.associations:
                    assoc_ids.update(self.associations[fid])
            candidate_ids.update(assoc_ids)

        candidates = [self.facts[fid] for fid in candidate_ids if fid in self.facts]
        if time_range:
            since, until = time_range
            candidates = [f for f in candidates if since <= f.timestamp <= until]

        self.max_access_seen = max(self.max_access_seen, max((f.access_count for f in candidates), default=0))
        scored = []
        for fact in candidates:
            fact.update_dynamic_importance(now)
            age_hours = (now - fact.timestamp).total_seconds() / 3600.0
            recency_bonus = 1.0 / (1.0 + age_hours / 24.0)
            fact.update_usefulness(recency_bonus, self.max_access_seen)

            sem_score = 0.0
            if q_emb is not None and fact.embedding is not None:
                sem_score = float(np.dot(q_emb, fact.embedding))
            bm25_score = sum(1 for w in query_lower.split() if w in fact.text.lower()) / max(1, len(query_lower.split()))
            score = (sem_score * 0.4) + (bm25_score * 0.2) + (fact.trust * 0.15) + (fact.importance * 0.15) + (fact.usefulness * 0.1)
            scored.append((fact, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        result = []
        for fact, _ in scored[:limit]:
            fact.access_count += 1
            fact.last_used = now
            time_ago = self.get_time_ago(fact.timestamp)
            result.append((fact, time_ago))

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.stats['avg_recall_time_ms'] = self.stats['avg_recall_time_ms'] * 0.95 + elapsed_ms * 0.05
        return result

    def add_to_working_memory(self, message: str):
        now = self._now()
        self.working_memory.append((now, message))
        if len(self.working_memory) > self.max_working_memory:
            self.working_memory.pop(0)
        self.short_term_buffer.append(message)
        if len(self.short_term_buffer) > self.max_short_term:
            self.short_term_buffer.pop(0)

    def get_working_memory_context(self, n: int = 8) -> str:
        if not self.working_memory:
            return ""
        recent = self.working_memory[-n:]
        lines = []
        for ts, msg in recent:
            time_ago = self.get_time_ago(ts)
            lines.append(f"{msg} ({time_ago})")
        return "\n".join(lines)

    def get_user_facts(self, limit: int = 10) -> List[Fact]:
        facts = [f for f in self.facts.values() if f.category.startswith("user.")]
        facts.sort(key=lambda f: (f.importance, f.trust), reverse=True)
        return facts[:limit]

    def get_kira_facts(self, limit: int = 10, persona_only: bool = False) -> List[Fact]:
        facts = [f for f in self.facts.values() if f.category.startswith("kira.")]
        if persona_only:
            facts = [f for f in facts if f.category == "kira.persona"]
        facts.sort(key=lambda f: (f.importance, f.trust), reverse=True)
        return facts[:limit]

    def get_all_facts_sorted(self, limit: int = 50) -> List[Fact]:
        facts = list(self.facts.values())
        facts.sort(key=lambda f: f.timestamp, reverse=True)
        return facts[:limit]

    def get_statistics(self) -> Dict:
        assoc_pairs = set()
        for src, targets in self.associations.items():
            for tgt in targets:
                if src < tgt:
                    assoc_pairs.add((src, tgt))
        stats = self.stats.copy()
        stats.update({
            'total_facts': len(self.facts),
            'by_category': {cat: len(ids) for cat, ids in self.category_index.items()},
            'working_memory_size': len(self.working_memory),
            'short_term_size': len(self.short_term_buffer),
            'faiss_size': self.faiss_index.ntotal if self.faiss_index else 0,
            'associations_count': len(assoc_pairs),
            'kira_mood': self.kira_mood,
            'laconic_mode': self.laconic_mode
        })
        return stats

    def trust_fact(self, fact_id: str, increase: bool = True) -> bool:
        if fact_id in self.facts:
            delta = 0.2 if increase else -0.2
            self.facts[fact_id].trust = max(0.0, min(1.0, self.facts[fact_id].trust + delta))
            return True
        return False

    def set_importance(self, fact_id: str, importance: float) -> bool:
        if fact_id in self.facts:
            self.facts[fact_id].base_importance = max(0.0, min(1.0, importance))
            self.facts[fact_id].importance = self.facts[fact_id].base_importance
            return True
        return False

    def get_conflicts(self) -> List[Dict]:
        conflicts = []
        for cat, ids in self.category_index.items():
            if len(ids) > 1:
                facts_in_cat = [self.facts[fid] for fid in ids if fid in self.facts]
                if len(set(f.text for f in facts_in_cat)) > 1:
                    conflicts.append({
                        "category": cat,
                        "facts": [{"id": f.id, "text": f.text, "trust": f.trust} for f in facts_in_cat]
                    })
        return conflicts

    async def semantic_consolidation(self):
        if len(self.facts) < 10:
            return
        all_facts = [f for f in self.facts.values() if not f.category.startswith("system.")]
        if len(all_facts) < 2:
            return
        groups = []
        used = set()
        for i, f1 in enumerate(all_facts):
            if f1.id in used:
                continue
            group = [f1]
            used.add(f1.id)
            for j, f2 in enumerate(all_facts[i+1:], i+1):
                if f2.id in used:
                    continue
                if f1.embedding is not None and f2.embedding is not None:
                    sim = float(np.dot(f1.embedding, f2.embedding))
                    if sim > 0.85:
                        group.append(f2)
                        used.add(f2.id)
            if len(group) > 1:
                groups.append(group)
        if not groups:
            return
        for group in groups:
            # Сохраняем все ассоциации группы
            all_associations = set()
            for f in group:
                all_associations.update(self.associations.get(f.id, set()))
            all_associations.difference_update(f.id for f in group)
            
            texts = [f.text for f in group]
            prompt = f"""Объедини следующие похожие факты в один или два факта, сохранив всю важную информацию. Используй русский язык. Верни JSON массив с полями "text", "trust" (среднее), "importance" (среднее).
Факты:
{chr(10).join(f'- {t}' for t in texts)}
"""
            try:
                response = await self._call_groq_with_retry(
                    model="qwen/qwen3-32b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300
                )
                if not response.choices:
                    continue
                content = response.choices[0].message.content
                new_facts = self._extract_json(content)
                if isinstance(new_facts, list) and new_facts:
                    for f in group:
                        self.forget(f.id)
                    for nf in new_facts:
                        if "text" in nf:
                            new_ids = await self.remember_facts([{"role": "assistant", "content": f"Консолидированный факт: {nf['text']}"}])
                            if new_ids:
                                # Переносим ассоциации
                                for assoc_id in all_associations:
                                    self.associations.setdefault(new_ids[0], set()).add(assoc_id)
                                    self.associations.setdefault(assoc_id, set()).add(new_ids[0])
                    logger.info(f"Семантическая консолидация: объединено {len(group)} фактов -> {len(new_facts)}")
            except Exception as e:
                logger.error(f"Ошибка семантической консолидации: {e}")

    async def periodic_consolidation(self):
        while True:
            await asyncio.sleep(86400)
            try:
                await self.semantic_consolidation()
                for category, fact_ids in list(self.category_index.items()):
                    if len(fact_ids) <= 2:
                        continue
                    facts_to_consolidate = [self.facts[fid] for fid in fact_ids if fid in self.facts]
                    if len(facts_to_consolidate) < 2:
                        continue
                    texts = [f.text for f in facts_to_consolidate]
                    prompt = f"""Обобщи факты в категории "{category}" в один-два факта. Удали устаревшие, объедини похожие. Верни JSON массив с полями "text", "trust", "importance".
Факты:
{chr(10).join(f'- {t}' for t in texts)}
"""
                    try:
                        response = await self._call_groq_with_retry(
                            model="qwen/qwen3-32b",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=300
                        )
                        if not response.choices:
                            continue
                        content = response.choices[0].message.content
                        new_facts = self._extract_json(content)
                        if not isinstance(new_facts, list):
                            new_facts = []
                        if new_facts:
                            for fid in fact_ids:
                                self.forget(fid)
                            for nf in new_facts:
                                if "text" in nf:
                                    await self.remember_facts([{"role": "assistant", "content": f"Обобщённый факт: {nf['text']}"}])
                            logger.info(f"Категориальная консолидация {category}: {len(facts_to_consolidate)} -> {len(new_facts)} фактов")
                    except Exception as e:
                        logger.error(f"Ошибка консолидации {category}: {e}")
                self.stats['last_consolidation'] = self._now()
            except Exception as e:
                logger.error(f"Ошибка в периодической консолидации: {e}")

    async def warmup_memory(self):
        """Прогрев памяти: очистка, обновление важности, консолидация, прогрев кеша, восстановление FAISS."""
        logger.info("Прогрев памяти...")
        # 1. Удаляем устаревшие эфемерные факты
        self._cleanup_expired_ephemeral()
        # 2. Обновляем динамическую важность
        now = self._now()
        for fact in self.facts.values():
            fact.update_dynamic_importance(now)
        # 3. Консолидация, если много фактов
        if len(self.facts) > 50:
            logger.info("Запуск семантической консолидации...")
            await self.semantic_consolidation()
        # 4. Если FAISS пуст, перестраиваем его
        if self.faiss_index is not None and self.faiss_index.ntotal == 0 and self.facts:
            logger.info("FAISS пуст, перестраиваю индекс...")
            self.faiss_index.reset()
            self.faiss_id_to_fact_id.clear()
            for fid, fact in self.facts.items():
                if fact.embedding is not None:
                    # Проверяем размерность
                    if len(fact.embedding) != self.embedding_dim:
                        logger.warning(f"Размерность эмбеддинга {len(fact.embedding)} не совпадает с {self.embedding_dim}, пересоздаю")
                        fact.embedding = self._get_embedding(fact.text)
                        if fact.embedding is None:
                            continue
                    self._add_to_faiss(fid, fact.embedding)
                else:
                    emb = self._get_embedding(fact.text)
                    if emb is not None:
                        fact.embedding = emb
                        self._add_to_faiss(fid, emb)
            logger.info(f"Перестроен FAISS, добавлено {self.faiss_index.ntotal} векторов")
        # 5. Прогрев кеша эмбеддингов
        logger.info("Прогрев кеша эмбеддингов...")
        texts = [fact.text for fact in self.facts.values()]
        if texts:
            self._get_embeddings_batch(texts)
        # 6. Обновляем BM25
        self._update_bm25()
        # 7. Логируем конфликты
        conflicts = self.get_conflicts()
        if conflicts:
            logger.warning(f"Обнаружены противоречия в памяти: {len(conflicts)} категорий")
            for conf in conflicts[:3]:
                sample = ', '.join(f['text'][:30] for f in conf['facts'][:3])
                logger.warning(f"  {conf['category']}: {sample}")
        logger.info(f"Прогрев завершён. Всего фактов: {len(self.facts)}, FAISS размер: {self.faiss_index.ntotal if self.faiss_index else 0}")

    async def start_auto_save(self, path: str, interval: int = 300):
        """Запускает фоновое автосохранение"""
        self.auto_save_path = path
        self.auto_save_interval = interval
        self._save_task = asyncio.create_task(self._auto_save_loop())

    async def _auto_save_loop(self):
        """Фоновый цикл автосохранения"""
        while True:
            await asyncio.sleep(self.auto_save_interval)
            if self.auto_save_path:
                await self.flush_faiss_tasks()
                await asyncio.to_thread(self.save, self.auto_save_path)
                logger.debug("Автосохранение памяти выполнено")

    def save(self, filepath: str):
        """Синхронное сохранение (для вызова из потока)"""
        data = {
            'facts': {fid: fact.to_dict() for fid, fact in self.facts.items()},
            'category_index': dict(self.category_index),
            'associations': {k: list(v) for k, v in self.associations.items()},
            'stats': self.stats,
            'working_memory': [(ts.isoformat(), msg) for ts, msg in self.working_memory],
            'short_term_buffer': self.short_term_buffer,
            'max_access_seen': self.max_access_seen,
            'kira_mood': self.kira_mood,
            'ids_of_unnoticed_important_facts': list(self.ids_of_unnoticed_important_facts)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Сохранено {len(self.facts)} фактов в {filepath}")

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.facts = {}
        for fid, fact_dict in data['facts'].items():
            fact = Fact.from_dict(fact_dict)
            self.facts[fid] = fact
            if fact.embedding is not None:
                # Проверяем размерность эмбеддинга
                if len(fact.embedding) != self.embedding_dim:
                    logger.warning(f"Размерность эмбеддинга {len(fact.embedding)} не совпадает с {self.embedding_dim}, пересоздаю")
                    fact.embedding = self._get_embedding(fact.text)
                    if fact.embedding is None:
                        continue
                self._add_to_faiss(fid, fact.embedding)
            else:
                emb = self._get_embedding(fact.text)
                if emb is not None:
                    fact.embedding = emb
                    self._add_to_faiss(fid, emb)
        self.category_index = defaultdict(list, data.get('category_index', {}))
        self.associations = defaultdict(set, {k: set(v) for k, v in data.get('associations', {}).items()})
        self.stats = data.get('stats', self.stats)
        wm_data = data.get('working_memory', [])
        if wm_data and isinstance(wm_data[0], str):
            now = self._now()
            self.working_memory = [(now - timedelta(seconds=i*5), msg) for i, msg in enumerate(reversed(wm_data))]
            self.working_memory.reverse()
        else:
            self.working_memory = [(datetime.fromisoformat(ts), msg) for ts, msg in wm_data] if wm_data else []
        self.short_term_buffer = data.get('short_term_buffer', [])
        self.max_access_seen = data.get('max_access_seen', 0)
        self.kira_mood = data.get('kira_mood', 0.5)
        self.ids_of_unnoticed_important_facts = set(data.get('ids_of_unnoticed_important_facts', []))
        self._update_bm25()
        logger.info(f"Загружено {len(self.facts)} фактов, FAISS размер: {self.faiss_index.ntotal if self.faiss_index else 0}")

    def export_to_json(self, filepath: str):
        data = {
            "facts": [
                {
                    "id": fid,
                    "text": f.text,
                    "category": f.category,
                    "trust": f.trust,
                    "importance": f.importance,
                    "base_importance": f.base_importance,
                    "timestamp": f.timestamp.isoformat(),
                    "last_used": f.last_used.isoformat(),
                    "access_count": f.access_count,
                    "usefulness": f.usefulness,
                    "ttl_seconds": f.ttl_seconds
                }
                for fid, f in self.facts.items()
            ],
            "associations": {k: list(v) for k, v in self.associations.items()},
            "working_memory": [(ts.isoformat(), msg) for ts, msg in self.working_memory]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def import_from_json(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for fact_data in data.get("facts", []):
            if fact_data["id"] in self.facts:
                continue
            fact = Fact(
                text=fact_data["text"],
                category=fact_data["category"],
                trust=fact_data["trust"],
                base_importance=fact_data.get("base_importance", fact_data.get("importance", 0.5)),
                importance=fact_data.get("importance", 0.5),
                timestamp=datetime.fromisoformat(fact_data["timestamp"]),
                last_used=datetime.fromisoformat(fact_data["last_used"]) if "last_used" in fact_data else datetime.fromisoformat(fact_data["timestamp"]),
                access_count=fact_data.get("access_count", 0),
                usefulness=fact_data.get("usefulness", 0.5),
                ttl_seconds=fact_data.get("ttl_seconds"),
                id=fact_data["id"]
            )
            emb = self._get_embedding(fact.text)
            if emb is not None:
                fact.embedding = emb
                self._add_to_faiss(fact.id, emb)
            self.facts[fact.id] = fact
            self.category_index[fact.category].append(fact.id)
        for src, targets in data.get("associations", {}).items():
            self.associations[src] = set(targets)
        wm_data = data.get("working_memory", [])
        self.working_memory = [(datetime.fromisoformat(ts), msg) for ts, msg in wm_data] if wm_data else []
        self._update_bm25()
        logger.info(f"Импортировано {len(data.get('facts', []))} фактов")