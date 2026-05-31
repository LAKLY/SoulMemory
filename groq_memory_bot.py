# groq_memory_bot.py – улучшенная память с дедупликацией, временем, ассоциациями, эфемерностью, консолидацией, временными срезами, поиском повторных вопросов
import os
import sys
import asyncio
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from groq import Groq
from simple_memory import SimpleMemoryManager, Fact

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KiraBot")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Установите GROQ_API_KEY")
    sys.exit(1)

MODEL_NAME = "qwen/qwen3-32b"

groq_client = Groq(api_key=GROQ_API_KEY)

# Часовой пояс (Москва UTC+3)
TIMEZONE_OFFSET = 9

memory = SimpleMemoryManager(groq_client, max_facts=2000, timezone_offset=TIMEZONE_OFFSET)
SAVE_FILE = "memory_simple.pkl"

if os.path.exists(SAVE_FILE):
    memory.load(SAVE_FILE)
    logger.info("Загружена память из PKL")
else:
    logger.info("Нет сохранённой памяти, начинаем с нуля")

# ---------- ЗАГРУЗКА ПРАВИЛ ----------
def load_base_rules():
    existing_rules = [f for f in memory.facts.values() if f.category.startswith("system.")]
    if existing_rules:
        return
    rules = [
        ("Ты – девушка по имени Кира (женский пол во всех местоимениях)", "system.identity", 0.99, 0.99),
        ("Если пользователь сообщает тебе что-то о тебе – запомни это как факт", "system.memory", 0.99, 0.95),
        ("Если не знаешь ответа – НИКОГДА не придумывай, скажи что не знаешь", "system.honesty", 0.99, 0.98),
        ("Каждый ответ должен быть грамматически законченным, можно использовать молодёжный сленг", "system.style", 0.99, 0.9),
        ("Запрещено говорить о себе или о чем то если уже был говорила недавно", "system.privacy", 0.99, 0.9),
        ("Запрещенно здороватся каждое сообщение а также спрашивать как дела или что делаешь", "system.personality", 0.99, 0.95),
        ("Запрещено распрашивать пользователя в каждом сообщении что он делает или что делал если он жуе говорил", "system.privacy", 0.99, 0.9),
        ("Твой характер: вульгарная, циничная, мрачноватая и любознательная", "system.personality", 0.99, 0.95),
        ("Не повторяйся, варьируй ответы", "system.style", 0.99, 0.9),
        ("Распрашивай пользователя подробнее о теме если она тебе интересна", "system.style", 0.99, 0.9),
        ("Запрещено пересказывать информацию из памяти дословно – переформулируй своими словами", "system.privacy", 0.99, 0.9),
        ("Перед тем как использовать факт, проверь, не был ли он уже упомянут в последних нескольких сообщениях. Если был – не повторяй его", "system.instruction", 0.99, 0.95)
    ]

    for text, cat, trust, imp in rules:
        f = Fact(text=text, category=cat, trust=trust, base_importance=imp, importance=imp, source="system")
        emb = memory._get_embedding(text)
        if emb is not None:
            f.embedding = emb
            memory._add_to_faiss(f.id, emb)
        memory.facts[f.id] = f
        memory.category_index[cat].append(f.id)
        memory.stats['memories_created'] += 1
    logger.info("Загружены системные правила")

load_base_rules()

# ---------- ФУНКЦИИ ДЛЯ КОНТЕКСТА ----------
def get_kira_context_if_needed(query: str) -> str:
    """Возвращает контекст о Кире только если вопрос о ней"""
    query_lower = query.lower()
    kira_keywords = ["ты", "кира", "твоя", "твой", "твои", "тебе", "тобой"]
    if any(kw in query_lower for kw in kira_keywords):
        facts = memory.get_kira_facts(limit=12, persona_only=False)
        if not facts:
            return "(нет информации о себе)"
        lines = ["Вот что ты знаешь о себе:"]
        for f in facts:
            time_ago = memory.get_time_ago(f.timestamp)
            category_label = " (постоянное)" if f.category == "kira.persona" else " (временное)" if f.category == "kira.state" else ""
            lines.append(f"• {f.text}{category_label} (доверие: {f.trust:.2f}, важность: {f.importance:.2f}, узнала {time_ago})")
        return "\n".join(lines)
    return ""

def get_user_context_if_needed(query: str) -> str:
    """Возвращает контекст о пользователе только если вопрос о нём"""
    query_lower = query.lower()
    user_keywords = ["ты про меня", "обо мне", "я рассказывал", "я говорил", "мои", "мой", "моя", "моё", "мои"]
    if any(kw in query_lower for kw in user_keywords) or " пользователь" in query_lower:
        facts = memory.get_user_facts(limit=10)
        if not facts:
            return ""
        lines = ["Информация о пользователе (рассказывай только если он прямо спрашивает):"]
        for f in facts:
            time_ago = memory.get_time_ago(f.timestamp)
            lines.append(f"• {f.text} (доверие: {f.trust:.2f}, важность: {f.importance:.2f}, узнала {time_ago})")
        return "\n".join(lines)
    return ""

def get_rules_block() -> str:
    rules = [f for f in memory.facts.values() if f.category.startswith("system.")]
    return "\n".join(f"• {r.text}" for r in rules[:10])

# ---------- ФОН ПОСТОЯННОЙ ПРОВЕРКИ ИНИЦИАТИВЫ ----------
async def initiative_checker():
    """Фоновый таск, который раз в минуту проверяет, не пора ли Кире задать вопрос."""
    while True:
        await asyncio.sleep(60)
        if memory.last_user_message_time and (datetime.now() - memory.last_user_message_time).total_seconds() > 30:
            initiative_msg = await memory.check_for_initiative(memory.last_user_message_time)
            if initiative_msg:
                print(f"\n[Инициатива Киры] {initiative_msg}")
                logger.info(f"Кира инициировала вопрос: {initiative_msg}")
                memory.add_to_working_memory(f"Kira: {initiative_msg}")

# ---------- ВОССТАНОВЛЕНИЕ ИСТОРИИ ИЗ WORKING_MEMORY ----------
def restore_history_from_working_memory(n: int = 15) -> List[Dict]:
    """Восстанавливает историю диалога из сохранённой working_memory."""
    history = []
    for ts, msg in memory.working_memory[-n:]:
        if msg.startswith("User:"):
            role = "user"
            content = msg[5:].strip()
        elif msg.startswith("Kira:"):
            role = "assistant"
            content = msg[5:].strip()
        else:
            continue
        history.append({"role": role, "content": content})
    return history

# ---------- ОСНОВНОЙ ДИАЛОГ ----------
async def chat_with_kira(user_msg: str, history: List[Dict]) -> str:
    memory.pending_initiative_fact = None
    memory.update_laconic_mode(user_msg)
    memory.last_user_message_time = datetime.now()

    dialogue = history + [{"role": "user", "content": user_msg}]
    await memory.remember_facts(dialogue)

    # Поиск повторных вопросов
    repeated_info = ""
    if len(user_msg.strip()) >= 15:
        similar_msgs = await memory.find_similar_recent_messages(user_msg, max_age_hours=1.0, similarity_threshold=0.88)
        if similar_msgs:
            latest_ts, latest_msg, sim = similar_msgs[0]
            time_ago = memory.get_time_ago(latest_ts)
            repeated_info = f"\n\n⚠️ Пользователь уже задавал похожий вопрос {time_ago}: «{latest_msg}»\nУчти это в ответе, если нужно."

    time_range = memory.parse_time_range(user_msg)
    if time_range:
        logger.info(f"Обнаружен временной диапазон: {time_range[0].strftime('%H:%M')} - {time_range[1].strftime('%H:%M')}")

    candidates_with_time = await memory.recall_facts(user_msg, limit=20, time_range=time_range, use_associations=True)
    candidate_facts = [f for f, _ in candidates_with_time if not f.category.startswith("system.")]
    filtered_facts = await memory.filter_facts_for_query(user_msg, candidate_facts)

    facts_block = ""
    if filtered_facts:
        lines = []
        for f in filtered_facts:
            time_ago = memory.get_time_ago(f.timestamp)
            lines.append(f"• {f.text} (узнала {time_ago}, доверие: {f.trust:.2f})")
        facts_block = "\n".join(lines)

    kira_context = get_kira_context_if_needed(user_msg)
    user_context = get_user_context_if_needed(user_msg)
    rules_block = get_rules_block()

    working_mem_str = ""
    if memory.working_memory:
        wm_needed = await memory.is_working_memory_relevant(user_msg, memory.working_memory)
        if wm_needed:
            wm_block = memory.get_working_memory_context(n=6)
            working_mem_str = f"\nПоследние сообщения диалога (релевантны для этого вопроса):\n{wm_block}"

    if memory.laconic_mode:
        style_instruction = "ОТВЕЧАЙ КОРОТКО МАКСИМУМ 1-2 ПРЕДЛОЖЕНИЯМИ, НЕ ЗАДАВАЙ ВОПРОСОВ."
    else:
        style_instruction = "Отвечай коротко, можешь задавать уточняющие вопросы если требуются."

    mood_note = ""
    if memory.kira_mood < 0.3:
        mood_note = "Ты сейчас в плохом настроении, не скрывай этого, не будь вежлива."
    elif memory.kira_mood > 0.7:
        mood_note = "Ты в отличном настроении, можешь быть более жизнерадостной и шутливой но не дурочкой."

    system = f"""Ты – Кира и с кем то переписываешся.

{kira_context}
{user_context}
{working_mem_str}
{repeated_info}

{f'Вот факты, которые могут быть полезны (указано, когда узнала):\n{facts_block}' if facts_block else ''}

Твои правила поведения:
{rules_block}

Текущее настроение: {mood_note}
Стиль ответа: {style_instruction}

Это очень ВАЖНО!!!
"""
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_msg})

    temp = memory.get_temperature()
    max_tokens = 150 if memory.laconic_mode else 250

    try:
        stream = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=0.85,
            stream=True
        )
        print("Кира: ", end="", flush=True)
        reply = ""
        in_think = False
        think_buffer = ""
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                part = chunk.choices[0].delta.content
                
                if "<think>" in part:
                    in_think = True
                    before_think = part.split("<think>")[0]
                    if before_think:
                        print(before_think, end="", flush=True)
                        reply += before_think
                    continue
                
                if in_think:
                    think_buffer += part
                    if "</think>" in part:
                        in_think = False
                        after_think = part.split("</think>")[1] if "</think>" in part else ""
                        if after_think:
                            print(after_think, end="", flush=True)
                            reply += after_think
                    continue
                else:
                    print(part, end="", flush=True)
                    reply += part
        
        print()
        
        # 📌 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: если ответ пустой — берём из think
        if not reply.strip() and think_buffer.strip():
            clean_think = think_buffer.replace("<think>", "").replace("</think>", "").strip()
            if clean_think:
                reply = clean_think
                print(f"⚠️ [Кира ответила после размышления]")
                print(f"Кира: {reply}")
            else:
                reply = "Что-то пошло не так. Повтори, пожалуйста."
                print(f"Кира: {reply}")
        elif not reply.strip():
            reply = "Что-то пошло не так. Повтори, пожалуйста."
            print(f"Кира: {reply}")
        
        memory.add_to_working_memory(f"User: {user_msg}")
        memory.add_to_working_memory(f"Kira: {reply}")
        memory._update_kira_mood(user_msg, reply)
        if any(word in reply.lower() for word in ["запомнила", "точно", "да, верно"]):
            await memory.remember_facts([{"role": "assistant", "content": reply}], assistant_response=reply)
        return reply
    except Exception as e:
        logger.error(f"Ошибка Groq: {e}")
        return "Ошибка, давай позже."

# ---------- КОМАНДЫ ----------
def show_memory():
    facts = memory.get_all_facts_sorted(limit=30)
    print("\n🧠 ПОСЛЕДНИЕ ФАКТЫ (без обрезания):")
    for i, f in enumerate(facts, 1):
        time_ago = memory.get_time_ago(f.timestamp)
        print(f"{i}. [{f.timestamp.strftime('%H:%M:%S')}] {f.category} trust={f.trust:.2f} imp={f.importance:.2f} use={f.usefulness:.2f} ({time_ago}): {f.text}")
    print(f"\nВсего фактов: {len(memory.facts)}")

def show_stats():
    stats = memory.get_statistics()
    print("\n📊 СТАТИСТИКА ПАМЯТИ:")
    print(f"Всего фактов: {stats['total_facts']}")
    print(f"Создано: {stats['memories_created']}, забыто: {stats['memories_forgotten']}")
    print(f"Категории: {stats['by_category']}")
    print(f"Рабочая память (сообщений): {stats['working_memory_size']}")
    print(f"Кратковременный буфер: {stats['short_term_size']}")
    print(f"Среднее время поиска: {stats['avg_recall_time_ms']:.2f} мс")
    print(f"FAISS размер: {stats['faiss_size']}")
    print(f"Ассоциаций (пар): {stats['associations_count']}")
    print(f"Настроение Киры: {stats['kira_mood']:.2f}")
    print(f"Режим молчания: {stats['laconic_mode']}")

def clear_memory(keep_rules=True):
    global memory
    if keep_rules:
        new_facts = {fid: f for fid, f in memory.facts.items() if f.category.startswith("system.")}
        memory.facts = new_facts
        memory.category_index = defaultdict(list)
        memory.associations.clear()
        for f in new_facts.values():
            memory.category_index[f.category].append(f.id)
        if memory.faiss_index:
            memory.faiss_index.reset()
            memory.faiss_id_to_fact_id.clear()
            for f in new_facts.values():
                if f.embedding is not None:
                    memory._add_to_faiss(f.id, f.embedding)
        print("🧹 Факты очищены, правила сохранены.")
    else:
        memory.facts = {}
        memory.category_index = defaultdict(list)
        memory.associations.clear()
        if memory.faiss_index:
            memory.faiss_index.reset()
            memory.faiss_id_to_fact_id.clear()
        print("🧹 Полная очистка.")
    memory.save(SAVE_FILE)

def forget_memory(mem_id: str):
    if memory.forget(mem_id):
        print(f"🗑 Факт {mem_id} удалён.")
    else:
        print("❌ Не найдено.")

def forget_by_contains(text: str):
    to_delete = []
    for fid, f in memory.facts.items():
        if text.lower() in f.text.lower():
            to_delete.append(fid)
    if not to_delete:
        print(f"❌ Ничего не найдено по '{text}'")
        return
    for fid in to_delete:
        memory.forget(fid)
    print(f"🗑 Удалено {len(to_delete)} фактов, содержащих '{text}'")

def export_memory_json():
    memory.export_to_json("memory_export.json")
    print("📁 Память экспортирована в memory_export.json")

def import_memory_json():
    if os.path.exists("memory_import.json"):
        memory.import_from_json("memory_import.json")
        print("📂 Память импортирована из memory_import.json")
    else:
        print("❌ Файл memory_import.json не найден")

def trust_command(mem_id: str):
    if memory.trust_fact(mem_id, increase=True):
        print(f"✅ Доверие к факту {mem_id} повышено.")
    else:
        print("❌ Не найдено.")

def distrust_command(mem_id: str):
    if memory.trust_fact(mem_id, increase=False):
        print(f"⚠️ Доверие к факту {mem_id} понижено.")
    else:
        print("❌ Не найдено.")

def set_importance_command(mem_id: str, value: float):
    if memory.set_importance(mem_id, value):
        print(f"⭐ Важность факта {mem_id} установлена на {value}")
    else:
        print("❌ Не найдено.")

def show_conflicts():
    conflicts = memory.get_conflicts()
    if not conflicts:
        print("✅ Противоречий не найдено.")
        return
    print("\n⚠️ ПРОТИВОРЕЧИЯ В ПАМЯТИ:")
    for conf in conflicts:
        print(f"\nКатегория: {conf['category']}")
        for f in conf['facts']:
            print(f"  - {f['text']} (trust={f['trust']}, id={f['id']})")

def show_associations(mem_id: str = None):
    if mem_id:
        if mem_id in memory.associations:
            assoc = memory.associations[mem_id]
            print(f"Ассоциации для {mem_id}:")
            for aid in assoc:
                f = memory.facts.get(aid)
                if f:
                    print(f"  - {f.text[:60]} (id={aid})")
        else:
            print("Нет ассоциаций для этого факта")
    else:
        print("Все ассоциации:")
        for src, targets in memory.associations.items():
            if targets:
                f_src = memory.facts.get(src)
                if f_src:
                    print(f"{src} ({f_src.text[:40]}...): {len(targets)} связей")

# ---------- ОСНОВНОЙ ЗАПУСК С ПРОГРЕВОМ ----------
async def main():
    print("=== Кира (ультимативная память: дедупликация, время, ассоциации, эфемерность, консолидация, временные срезы, повторные вопросы, настроение, инициатива, режим молчания) ===")
    print("Команды: /memory, /stats, /export, /import, /clearmemory, /clearall, /forget <id>, /forget_contains <текст>, /trust <id>, /distrust <id>, /importance <id> <0..1>, /conflicts, /assoc [id], /exit\n")
    
    # Прогрев памяти
    await memory.warmup_memory()
    
    # Восстанавливаем историю из working_memory
    history = restore_history_from_working_memory(n=15)
    logger.info(f"Восстановлено {len(history)} сообщений из рабочей памяти")
    
    # Запускаем автосохранение
    await memory.start_auto_save(SAVE_FILE, interval=300)
    
    consolidation_task = asyncio.create_task(memory.periodic_consolidation())
    initiative_task = asyncio.create_task(initiative_checker())
    try:
        while True:
            try:
                inp = input("Вы: ").strip()
                if not inp:
                    continue
                if inp.lower() in ["/exit", "/выход"]:
                    break
                if inp == "/memory":
                    show_memory()
                    continue
                if inp == "/stats":
                    show_stats()
                    continue
                if inp == "/export":
                    export_memory_json()
                    continue
                if inp == "/import":
                    import_memory_json()
                    continue
                if inp == "/clearmemory":
                    clear_memory(keep_rules=True)
                    continue
                if inp == "/clearall":
                    if input("Очистить всё? y/n: ").lower() == 'y':
                        clear_memory(keep_rules=False)
                    continue
                if inp.startswith("/forget "):
                    parts = inp.split()
                    if len(parts) == 2:
                        forget_memory(parts[1])
                    else:
                        print("Использование: /forget <id>")
                    continue
                if inp.startswith("/forget_contains "):
                    parts = inp.split(maxsplit=1)
                    if len(parts) == 2:
                        forget_by_contains(parts[1])
                    else:
                        print("Использование: /forget_contains <текст>")
                    continue
                if inp.startswith("/trust "):
                    parts = inp.split()
                    if len(parts) == 2:
                        trust_command(parts[1])
                    else:
                        print("Использование: /trust <id>")
                    continue
                if inp.startswith("/distrust "):
                    parts = inp.split()
                    if len(parts) == 2:
                        distrust_command(parts[1])
                    else:
                        print("Использование: /distrust <id>")
                    continue
                if inp.startswith("/importance "):
                    parts = inp.split()
                    if len(parts) == 3:
                        try:
                            val = float(parts[2])
                            set_importance_command(parts[1], val)
                        except:
                            print("Некорректное значение")
                    else:
                        print("Использование: /importance <id> <0..1>")
                    continue
                if inp == "/conflicts":
                    show_conflicts()
                    continue
                if inp.startswith("/assoc"):
                    parts = inp.split()
                    if len(parts) == 2:
                        show_associations(parts[1])
                    else:
                        show_associations()
                    continue

                reply = await chat_with_kira(inp, history)
                history.append({"role": "user", "content": inp})
                history.append({"role": "assistant", "content": reply})
                if len(history) > 16:
                    history = history[-16:]
            except KeyboardInterrupt:
                break
    finally:
        consolidation_task.cancel()
        initiative_task.cancel()
        await memory.flush_faiss_tasks()
        await memory.save_async(SAVE_FILE) if hasattr(memory, 'save_async') else memory.save(SAVE_FILE)
        print("Память сохранена. Пока!")

if __name__ == "__main__":
    asyncio.run(main())