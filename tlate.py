#!/usr/bin/env python3
import os, sys, json, time, curses, threading, re, requests, concurrent.futures
from collections import deque

NUM_WORKERS = 15

def get_api_keys():
    keys = {}
    auth_path = os.path.expanduser("~/.local/share/opencode/auth.json")
    if os.path.exists(auth_path):
        try:
            with open(auth_path, "r") as f:
                data = json.load(f)
                if "nvidia" in data and "key" in data["nvidia"]: keys["nvidia"] = data["nvidia"]["key"]
                if "openrouter" in data and "key" in data["openrouter"]: keys["openrouter"] = data["openrouter"]["key"]
                if "google" in data and "key" in data["google"]: keys["google"] = data["google"]["key"]
                if "mistral" in data and "key" in data["mistral"]: keys["mistral"] = data["mistral"]["key"]
                if "codestral" in data and "key" in data["codestral"]: keys["codestral"] = data["codestral"]["key"]
                if "cerebras" in data and "key" in data["cerebras"]: keys["cerebras"] = data["cerebras"]["key"]
                if "groq" in data and "key" in data["groq"]: keys["groq"] = data["groq"]["key"]
                if "cohere" in data and "key" in data["cohere"]: keys["cohere"] = data["cohere"]["key"]
        except:
            pass
    # Fallback to pure env variables if missing from config
    if os.environ.get("NVIDIA_API_KEY"): keys["nvidia"] = os.environ.get("NVIDIA_API_KEY")
    if os.environ.get("OPENROUTER_API_KEY"): keys["openrouter"] = os.environ.get("OPENROUTER_API_KEY")
    if os.environ.get("GOOGLE_API_KEY"): keys["google"] = os.environ.get("GOOGLE_API_KEY")
    if os.environ.get("MISTRAL_API_KEY"): keys["mistral"] = os.environ.get("MISTRAL_API_KEY")
    if os.environ.get("CODESTRAL_API_KEY"): keys["codestral"] = os.environ.get("CODESTRAL_API_KEY")
    if os.environ.get("CEREBRAS_API_KEY"): keys["cerebras"] = os.environ.get("CEREBRAS_API_KEY")
    if os.environ.get("GROQ_API_KEY"): keys["groq"] = os.environ.get("GROQ_API_KEY")
    if os.environ.get("COHERE_API_KEY"): keys["cohere"] = os.environ.get("COHERE_API_KEY")
    return keys

def parse_model_stats(model_id):
    model_lower = model_id.lower()
    
    params_match = re.search(r'(\d+(?:\.\d+)?(?:x\d+)?)[bBtTmM]', model_id)
    params_str = params_match.group(0).upper() if params_match else "?"
    if "glm-5" in model_lower or "glm5" in model_lower: params_str = "744B"
    elif "nemo" in model_lower and params_str == "?": params_str = "12B"
    elif "jamba" in model_lower and "large" in model_lower: params_str = "398B"
    elif "jamba" in model_lower and "mini" in model_lower: params_str = "52B"
    
    # Official sizes not explicitly in the URL
    if params_str == "?":
        if "mistral-large" in model_lower or "pixtral-large" in model_lower: params_str = "123B"
        elif "command-r-plus" in model_lower or "command-r+" in model_lower: params_str = "104B"
        elif "command-r" in model_lower: params_str = "35B"
        elif "grok" in model_lower: params_str = "314B"
        elif "codestral" in model_lower: params_str = "22B"
        elif "ministral-3b" in model_lower: params_str = "3B"
        elif "ministral-8b" in model_lower: params_str = "8B"
        elif "deepseek-v3" in model_lower or "deepseek-r1" in model_lower:
            if "distill" not in model_lower: params_str = "671B"

    # Estimates (Tilde prefix differentiates them)
    if params_str == "?":
        if "gpt-4o-mini" in model_lower: params_str = "~100B"
        elif "gpt-4" in model_lower: params_str = "~1.8T"
        elif "claude-3-opus" in model_lower: params_str = "~2T"
        elif "claude-3-5-sonnet" in model_lower or "claude-3-sonnet" in model_lower: params_str = "~1T"
        elif "claude-3-5-haiku" in model_lower or "claude-3-haiku" in model_lower: params_str = "~100B"
        elif "gemini-1.5-flash-8b" in model_lower: params_str = "8B"
        elif "gemini" in model_lower and "flash" in model_lower: params_str = "~50B"
        elif "gemini" in model_lower and "pro" in model_lower: params_str = "~1.5T"
        elif "mistral-small" in model_lower: params_str = "~24B"
        elif "mistral-medium" in model_lower: params_str = "~80B"
    
    ctx = "?"
    if "llama-3.1" in model_lower or "llama-3.2" in model_lower or "llama-3.3" in model_lower: ctx = "128K"
    elif "llama-3" in model_lower: ctx = "8K"
    elif "qwen2.5" in model_lower or "qwen3" in model_lower or "qwq" in model_lower: ctx = "128K"
    elif "mixtral" in model_lower: ctx = "32K"
    elif "mistral-large" in model_lower or "mistral-small" in model_lower or "mistral-nemo" in model_lower: ctx = "128K"
    elif "glm" in model_lower: ctx = "128K"
    if "glm5" in model_lower or "glm-5" in model_lower: ctx = "200K"
    if "claude-3" in model_lower: ctx = "200K"
    if "gpt-4" in model_lower: ctx = "128K"
    
    if ctx == "?": ctx = "8K" # default optimistic
    
    val = 0
    if params_str != "?":
        try:
            num_part = params_str[:-1].replace("+", "").replace("~", "")
            if "X" in num_part:
                parts = num_part.split("X")
                val = float(parts[0]) * float(parts[1])
            else:
                val = float(num_part)
            if "T" in params_str:
                val *= 1000
        except: pass
    
    return params_str, ctx, val

def format_model_name(model_id):
    name = model_id.split('/')[-1]
    name = name.replace('-', ' ').replace('_', ' ')
    if len(name) > 22: name = name[:19] + "..."
    return name.title()

def get_model_modality(model_id):
    m = model_id.lower()
    if "gemini" in m and ("1.5" in m or "flash" in m or "pro" in m): return "T,I,V"
    if "gpt-4o" in m: return "T,I"
    if "claude-3" in m: return "T,I"
    if any(x in m for x in ["vision", "vl", "pixtral", "llava", "paligemma"]): return "T,I"
    return "T"

def get_models(keys):
    models = []
    
    def fetch_nvidia():
        res = []
        if "nvidia" in keys:
            try:
                headers = {"Authorization": f"Bearer {keys['nvidia']}"}
                r = requests.get("https://integrate.api.nvidia.com/v1/models", headers=headers, timeout=15)
                if r.status_code == 200:
                    for m in r.json()["data"]:
                        p_str, ctx, val = parse_model_stats(m["id"])
                        res.append({
                            "id": m["id"], "name": format_model_name(m["id"]), 
                            "params": p_str, "context": ctx, "score": val, 
                            "lat": float('inf'), "stat": "Pending...", "prov": "nvidia",
                            "desc": "Official NVIDIA NIM Endpoint for this model."
                        })
            except Exception: pass
        return res

    def fetch_openrouter():
        res = []
        if "openrouter" in keys:
            try:
                headers = {"Authorization": f"Bearer {keys['openrouter']}"}
                r = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=15)
                if r.status_code == 200:
                    for m in r.json()["data"]:
                        pricing = m.get("pricing") or {}
                        try:
                            if any(float(v or 0) > 0 for v in pricing.values()): continue
                        except: pass
                        p_str, _, val = parse_model_stats(m["id"])
                        ctx_len = m.get("context_length", 8192)
                        ctx = f"{ctx_len//1000}K" if ctx_len >= 1000 else str(ctx_len)
                        name = m.get("name", m["id"].split('/')[-1])
                        if len(name) > 22: name = name[:19] + "..."
                        desc = m.get("description", "")
                        if not desc: desc = "OpenRouter hosted API endpoint."
                        res.append({
                            "id": m["id"], "name": name, "params": p_str, 
                            "context": ctx, "score": val, "lat": float('inf'), 
                            "stat": "Pending...", "prov": "openrouter",
                            "desc": desc
                        })
            except Exception: pass
        return res

    def fetch_google():
        res = []
        if "google" in keys:
            try:
                r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={keys['google']}", timeout=15)
                if r.status_code == 200:
                    for m in r.json().get("models", []):
                        name_id = m["name"].replace("models/", "")
                        
                        # Filter out non-LLM noise to preserve the 15 RPM rate limit
                        ni = name_id.lower()
                        if any(x in ni for x in ["imagen", "veo", "embedding", "audio", "tts", "aqa", "bison", "gecko", "vision"]):
                            continue
                            
                        p_str, ctx, val = parse_model_stats(name_id)
                        ctx_lim = m.get("inputTokenLimit", 8192)
                        ctx = f"{int(ctx_lim/1000)}K" if ctx_lim >= 1000 else str(ctx_lim)
                        desc = m.get("description", "Official Google Gemini API endpoint.")
                        if "flash" in name_id.lower():
                            desc += " | [FREE LIMITS]: 15 RPM, 1M TPM, 1,500 RPD."
                        elif "pro" in name_id.lower():
                            desc += " | [FREE LIMITS]: 2 RPM, 32K TPM, 50 RPD."
                        res.append({
                            "id": name_id, "name": m.get("displayName", name_id), 
                            "params": p_str, "context": ctx, "score": val, 
                            "lat": float('inf'), "stat": "Pending...", "prov": "google",
                            "desc": desc
                        })
            except Exception: pass
        return res

    def fetch_mistral():
        res = []
        if "mistral" in keys:
            try:
                headers = {"Authorization": f"Bearer {keys['mistral']}"}
                r = requests.get("https://api.mistral.ai/v1/models", headers=headers, timeout=15)
                if r.status_code == 200:
                    for m in r.json().get("data", []):
                        model_id = m["id"]
                        if any(x in model_id.lower() for x in ["embed", "moderation", "ocr", "audio", "transcribe"]):
                            continue
                        p_str, ctx, val = parse_model_stats(model_id)
                        ctx_lim = m.get("max_context_length", 32768)
                        ctx = f"{int(ctx_lim/1000)}K" if ctx_lim >= 1000 else str(ctx_lim)
                        desc = m.get("description", "Mistral AI platform model.")
                        desc += " | [FREE LIMITS]: 1 RPS, 500K TPM."
                        res.append({
                            "id": model_id, "name": format_model_name(model_id), 
                            "params": p_str, "context": ctx, "score": val, 
                            "lat": float('inf'), "stat": "Pending...", "prov": "mistral",
                            "desc": desc
                        })
            except Exception: pass
        return res

    def fetch_codestral():
        res = []
        if "codestral" in keys:
            try:
                desc = "Mistral Codestral specialized coding API endpoint. | [FREE LIMITS]: 1 RPS."
                res.append({
                    "id": "codestral-latest", "name": "Codestral Latest", 
                    "params": "22B", "context": "32K", "score": 22.0, 
                    "lat": float('inf'), "stat": "Pending...", "prov": "codestral",
                    "desc": desc
                })
                res.append({
                    "id": "codestral-mamba-latest", "name": "Codestral Mamba Latest", 
                    "params": "7B", "context": "256K", "score": 7.0, 
                    "lat": float('inf'), "stat": "Pending...", "prov": "codestral",
                    "desc": desc
                })
            except Exception: pass
        return res

    def fetch_cerebras():
        res = []
        if "cerebras" in keys:
            try:
                headers = {"Authorization": f"Bearer {keys['cerebras']}"}
                r = requests.get("https://api.cerebras.ai/v1/models", headers=headers, timeout=15)
                if r.status_code == 200:
                    for m in r.json().get("data", []):
                        model_id = m["id"]
                        p_str, ctx, val = parse_model_stats(model_id)
                        desc = "Official Cerebras Wafer-Scale Inference Endpoint. | [FREE LIMITS]: 30 RPM, 60K TPM."
                        res.append({
                            "id": model_id, "name": format_model_name(model_id), 
                            "params": p_str, "context": ctx, "score": val, 
                            "lat": float('inf'), "stat": "Pending...", "prov": "cerebras",
                            "desc": desc
                        })
            except Exception: pass
        return res

    def fetch_groq():
        res = []
        if "groq" in keys:
            try:
                headers = {"Authorization": f"Bearer {keys['groq']}"}
                r = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=15)
                if r.status_code == 200:
                    for m in r.json().get("data", []):
                        model_id = m["id"]
                        if "whisper" in model_id.lower() or "guard" in model_id.lower(): continue
                        p_str, ctx, val = parse_model_stats(model_id)
                        desc = "Official Groq LPU Ultra-fast Inference Engine. | [FREE LIMITS]: 30 RPM, 14,400 RPD."
                        res.append({
                            "id": model_id, "name": format_model_name(model_id), 
                            "params": p_str, "context": ctx, "score": val, 
                            "lat": float('inf'), "stat": "Pending...", "prov": "groq",
                            "desc": desc
                        })
            except Exception: pass
        return res

    def fetch_cohere():
        res = []
        if "cohere" in keys:
            try:
                headers = {"Authorization": f"Bearer {keys['cohere']}"}
                r = requests.get("https://api.cohere.com/v1/models", headers=headers, timeout=15)
                if r.status_code == 200:
                    for m in r.json().get("models", []):
                        model_id = m.get("name", "")
                        if not model_id or "embed" in model_id.lower() or "rerank" in model_id.lower(): continue
                        p_str, ctx, val = parse_model_stats(model_id)
                        desc = m.get("description", "Cohere Native API.")
                        desc += " | [FREE LIMITS]: Trial Key = 40 RPM, 1K Calls/Month."
                        res.append({
                            "id": model_id, "name": format_model_name(model_id), 
                            "params": p_str, "context": ctx, "score": val, 
                            "lat": float('inf'), "stat": "Pending...", "prov": "cohere", "desc": desc
                        })
            except Exception: pass
        return res
        
    tasks = [
        fetch_nvidia, fetch_openrouter, fetch_google, fetch_mistral, 
        fetch_codestral, fetch_cerebras, fetch_groq, fetch_cohere
    ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            models.extend(future.result())
            
    return models

def set_model_in_opencode(model_id, prov):
    full_model_id = f"{prov}/{model_id}"
    paths = ["~/.config/opencode/opencode.json", "~/.opencode/oh-my-opencode.json"]
    for path in paths:
        p = os.path.expanduser(path)
        if os.path.exists(p):
            with open(p, "r") as f: content = f.read()
            content = re.sub(r'("model":\s*)"[^"]+"', rf'\1"{full_model_id}"', content)
            with open(p, "w") as f: f.write(content)

LATENCIES = {} 
STATUS = {}
LAST_PING = {}

# Define how many seconds to wait before re-pinging a model based on provider limits
PROVIDER_INTERVALS = {
    "google": 3600,       # Ping at most once per hour to save ultra-low RPD (50)
    "cohere": 3600,       # Ping at most once per hour to save monthly Trial limit (1k/mo)
    "cerebras": 120,      # Generous RPM, wait 2 mins
    "groq": 120,          # High RPM, wait 2 mins
    "codestral": 120,
    "mistral": 120,
    "nvidia": 60,         # Very high limits
    "openrouter": 60,
}

CACHE_FILE = os.path.expanduser("~/.local/share/tetralatency/tlate_cache.json")

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
            STATUS.update(data.get("status", {}))
            LAST_PING.update(data.get("last_ping", {}))
            lat_data = data.get("latencies", {})
            for k, v in lat_data.items():
                if k not in LATENCIES: LATENCIES[k] = deque(maxlen=5)
                for lat in v: LATENCIES[k].append(lat)
        except Exception: pass

def save_cache():
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        lat_data = {k: list(v) for k, v in LATENCIES.items()}
        data = {
            "status": STATUS,
            "last_ping": LAST_PING,
            "latencies": lat_data
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f)
    except Exception: pass

def measure_loop(models, keys):
    def ping(m):
        m_id = m["id"]
        prov = m["prov"]
        start = time.time()
        
        api_key = keys[prov]
        
        if prov == "google":
            try:
                payload = {"contents": [{"parts":[{"text": "Hi"}]}]}
                res = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m_id}:generateContent?key={api_key}", json=payload, timeout=15)
                lat = (time.time() - start) * 1000
                if res.status_code == 200: return m_id, lat, "OK"
                else: return m_id, None, f"Err {res.status_code}"
            except requests.exceptions.Timeout:
                return m_id, None, "Timeout"
            except requests.exceptions.RequestException as e:
                return m_id, None, "Net Error"
            except Exception:
                return m_id, None, "Error"
        else:
            if prov == "nvidia": pt = "https://integrate.api.nvidia.com/v1/chat/completions"
            elif prov == "mistral": pt = "https://api.mistral.ai/v1/chat/completions"
            elif prov == "codestral": pt = "https://codestral.mistral.ai/v1/chat/completions"
            elif prov == "cerebras": pt = "https://api.cerebras.ai/v1/chat/completions"
            elif prov == "groq": pt = "https://api.groq.com/openai/v1/chat/completions"
            elif prov == "cohere": pt = "https://api.cohere.com/v2/chat"
            else: pt = "https://openrouter.ai/api/v1/chat/completions"
            
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            if prov == "openrouter": headers["HTTP-Referer"] = "https://github.com/nim-latency"
                
            payload = {"model": m_id, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
            try:
                res = requests.post(pt, headers=headers, json=payload, timeout=15)
                lat = (time.time() - start) * 1000
                if res.status_code == 200:
                    return m_id, lat, "OK"
                elif res.status_code in (404, 401, 403, 400, 429, 502):
                    return m_id, None, "Fail"
                else: return m_id, None, f"Err {res.status_code}"
            except requests.exceptions.Timeout:
                return m_id, None, "Timeout"
            except requests.exceptions.RequestException as e:
                return m_id, None, "Net Error"
            except Exception:
                return m_id, None, "Error"
    
    load_cache()
    while True:
        now = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for m in models:
                m_id = m["id"]
                prov = m["prov"]
                
                # Smart Rate Limiting: Check if we are allowed to ping this model again
                last = LAST_PING.get(m_id, 0)
                interval = PROVIDER_INTERVALS.get(prov, 60)
                
                if now - last >= interval:
                    LAST_PING[m_id] = now
                    executor.submit(ping_and_update, ping, m)
                    time.sleep(0.04) 
                    
        time.sleep(1) # Prevent hot-looping when everything is rate limited
        save_cache()

def ping_and_update(fn, m):
    m_id, lat, stat = fn(m)
    STATUS[m_id] = stat
    if lat is not None:
        if m_id not in LATENCIES: LATENCIES[m_id] = deque(maxlen=5)
        LATENCIES[m_id].append(lat)

def get_gauge(latency):
    if latency == float('inf'): return " " * 12
    # 0-2000ms range
    blocks = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    width = 12
    ratio = min(1.0, latency / 2000.0)
    filled_full = int(ratio * width)
    rem = (ratio * width) - filled_full
    rem_char = blocks[int(rem * 8)] if filled_full < width else ""
    empty_space = width - filled_full - (1 if rem_char else 0)
    bar = ("█" * filled_full) + rem_char + (" " * empty_space)
    return bar

def main(stdscr, models, keys):
    curses.curs_set(0)
    stdscr.timeout(100)
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)     # Headers / borders
    curses.init_pair(2, curses.COLOR_GREEN, -1)    # Low Latency / Success
    curses.init_pair(3, curses.COLOR_RED, -1)      # High Latency / Error
    curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_WHITE) # Selected
    curses.init_pair(5, curses.COLOR_YELLOW, -1)   # Medium Latency / Score
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # High power
    curses.init_pair(7, curses.COLOR_BLUE, -1)     # Muted text
    
    selected_idx = 0
    scroll_offset = 0
    sort_col = "Latency"
    sort_desc = False
    
    t = threading.Thread(target=measure_loop, args=(models, keys), daemon=True)
    t.start()

    status_msg = "Up/Down: Navigate | Enter: Set Opencode Model | Click Column: Sort | ESC: Quit"

    id_w, pv_w, name_w, p_w, ctx_w, mod_w, lat_w, bar_w = 33, 6, 18, 10, 10, 6, 10, 14

    search_query = ""

    f_prv_opts = ["ALL", "NV", "OR", "GG", "MS", "CS", "CB", "CO", "GQ"]
    f_par_opts = ["ALL", "<10B", "10B-30B", "30B-70B", ">70B"]
    f_ctx_opts = ["ALL", ">=32K", ">=128K"]
    f_lat_opts = ["ALL", "<300ms", "<800ms"]
    f_err_opts = ["Hide", "Show"]

    f_prv_idx, f_par_idx, f_ctx_idx, f_lat_idx, f_err_idx = 0, 0, 0, 0, 0

    while True:
        height, width = stdscr.getmaxyx()
        
        if height < 15 or width < 60:
            stdscr.erase()
            msg = "Terminal too small"
            try: stdscr.addstr(height//2, (width-len(msg))//2, msg, curses.A_BOLD)
            except: pass
            stdscr.refresh()
            key = stdscr.getch()
            if key in (3, 27, ord('q'), ord('Q')): break
            continue
        
        # Merge stats
        active_models = []
        error_models = []
        best_lat = float('inf')
        tested_count = 0
        
        for m in models:
            m_id = m["id"]
            lats = LATENCIES.get(m_id, [])
            avg_lat = sum(lats)/len(lats) if lats else float('inf')
            stat = STATUS.get(m_id, "Pending...")
            
            model_dict = {
                "id": m_id, "name": m["name"], "params": m["params"], 
                "context": m["context"], "score": m["score"], 
                "lat": avg_lat, "stat": stat, "prov": m["prov"],
                "desc": m["desc"], "mod": get_model_modality(m_id)
            }
            
            if search_query:
                s_lower = search_query.lower()
                if s_lower not in m_id.lower() and s_lower not in m["name"].lower():
                    continue

            # Check Filters
            skip = False
            
            p_opt = f_prv_opts[f_prv_idx]
            if p_opt == "NV" and m["prov"] != "nvidia": skip = True
            elif p_opt == "OR" and m["prov"] != "openrouter": skip = True
            elif p_opt == "GG" and m["prov"] != "google": skip = True
            elif p_opt == "MS" and m["prov"] != "mistral": skip = True
            elif p_opt == "CS" and m["prov"] != "codestral": skip = True
            elif p_opt == "CB" and m["prov"] != "cerebras": skip = True
            elif p_opt == "CO" and m["prov"] != "cohere": skip = True
            elif p_opt == "GQ" and m["prov"] != "groq": skip = True
                
            par_opt = f_par_opts[f_par_idx]
            if par_opt != "ALL" and m["score"] > 0:
                s = m["score"]
                if par_opt == "<10B" and s >= 10: skip = True
                elif par_opt == "10B-30B" and (s < 10 or s > 30): skip = True
                elif par_opt == "30B-70B" and (s < 30 or s > 70): skip = True
                elif par_opt == ">70B" and s <= 70: skip = True
                
            ctx_opt = f_ctx_opts[f_ctx_idx]
            if ctx_opt != "ALL" and "K" in m["context"]:
                c_val = float(m["context"][:-1])
                if ctx_opt == ">=32K" and c_val < 32: skip = True
                elif ctx_opt == ">=128K" and c_val < 128: skip = True
                
            lat_opt = f_lat_opts[f_lat_idx]
            if lat_opt != "ALL" and avg_lat != float('inf'):
                if lat_opt == "<300ms" and avg_lat >= 300: skip = True
                elif lat_opt == "<800ms" and avg_lat >= 800: skip = True
                
            err_opt = f_err_opts[f_err_idx]
            if err_opt == "Hide" and stat not in ["OK", "Pending...", "Timeout"]:
                skip = True
                
            if skip: continue
            
            if stat == "OK":
                tested_count += 1
                if avg_lat < best_lat: best_lat = avg_lat
                active_models.append(model_dict)
            elif stat == "Pending...":
                active_models.append(model_dict)
            else:
                error_models.append(model_dict)
            
        def s_key(x):
            if sort_col == "Latency": return x["lat"]
            if sort_col == "Params": return x["score"]
            if sort_col == "Context": 
                k = x["context"]
                return float(k[:-1]) if "K" in k else 0
            if sort_col == "Name": return x["name"]
            if sort_col == "Prv": return x["prov"]
            if sort_col == "Inp": return x["mod"]
            return x["id"]
            
        active_models.sort(key=s_key, reverse=sort_desc)
        error_models.sort(key=lambda x: x["id"])
        
        all_models = active_models + error_models
        
        # Handle input
        key = stdscr.getch()
        if key in (3, 27): break  # 3=Ctrl+C, 27=ESC
        elif key == curses.KEY_UP: selected_idx = max(0, selected_idx - 1)
        elif key == curses.KEY_DOWN: selected_idx = min(len(all_models) - 1, selected_idx + 1)
        elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
            search_query = search_query[:-1]
            selected_idx = 0
            scroll_offset = 0
        elif 32 <= key <= 126:
            search_query += chr(key)
            selected_idx = 0
            scroll_offset = 0
        elif key in (10, 13):
            if all_models and selected_idx < len(all_models):
                sel_id = all_models[selected_idx]["id"]
                sel_prov = all_models[selected_idx]["prov"]
                set_model_in_opencode(sel_id, sel_prov)
                status_msg = f"SUCCESS! [{sel_prov}/{sel_id}] Applied to Opencode"
        elif key == curses.KEY_MOUSE:
            try:
                _, mx, my, _, bstate = curses.getmouse()
                if bstate & curses.BUTTON1_CLICKED:
                    if my == 3: # Filter toggles
                        s_prv = f" [ Prv: {f_prv_opts[f_prv_idx]} ] "
                        s_par = f" [ Params: {f_par_opts[f_par_idx]} ] "
                        s_ctx = f" [ Ctx: {f_ctx_opts[f_ctx_idx]} ] "
                        s_lat = f" [ Lat: {f_lat_opts[f_lat_idx]} ] "
                        s_err = f" [ Err: {f_err_opts[f_err_idx]} ] "
                        lx = 10
                        if lx <= mx < lx + len(s_prv): f_prv_idx = (f_prv_idx+1)%len(f_prv_opts)
                        lx += len(s_prv) + 1
                        if lx <= mx < lx + len(s_par): f_par_idx = (f_par_idx+1)%len(f_par_opts)
                        lx += len(s_par) + 1
                        if lx <= mx < lx + len(s_ctx): f_ctx_idx = (f_ctx_idx+1)%len(f_ctx_opts)
                        lx += len(s_ctx) + 1
                        if lx <= mx < lx + len(s_lat): f_lat_idx = (f_lat_idx+1)%len(f_lat_opts)
                        lx += len(s_lat) + 1
                        if lx <= mx < lx + len(s_err): f_err_idx = (f_err_idx+1)%len(f_err_opts)
                        selected_idx = 0; scroll_offset = 0
                    elif my == 5: # Header row offset changed to 5 because of search box & filter
                        new_sort = sort_col
                        cum_w = 2 
                        if mx < cum_w + id_w: new_sort = "ID"
                        elif mx < cum_w + id_w + pv_w + 1: new_sort = "Prv"
                        elif mx < cum_w + id_w + pv_w + name_w + 4: new_sort = "Name"
                        elif mx < cum_w + id_w + pv_w + name_w + p_w + 7: new_sort = "Params"
                        elif mx < cum_w + id_w + pv_w + name_w + p_w + ctx_w + 10: new_sort = "Context"
                        elif mx < cum_w + id_w + pv_w + name_w + p_w + ctx_w + mod_w + 13: new_sort = "Inp"
                        elif mx < cum_w + id_w + pv_w + name_w + p_w + ctx_w + mod_w + lat_w + 16: new_sort = "Latency"
                        else: new_sort = "Status"
                        
                        if sort_col == new_sort: sort_desc = not sort_desc
                        else: sort_col = new_sort; sort_desc = (new_sort in ["Params", "Context", "Inp"])
                    elif my >= 7 and my < height - 2:
                        clicked_idx = scroll_offset + (my - 7)
                        if clicked_idx < len(all_models): selected_idx = clicked_idx
            except: pass
        
        # Calculate reserved height for info panel
        info_height = 8 
        list_max_height = height - 9 - info_height 
        if list_max_height < 5: list_max_height = 5
        
        if selected_idx < scroll_offset: scroll_offset = selected_idx
        elif selected_idx >= scroll_offset + list_max_height: scroll_offset = selected_idx - list_max_height + 1
            
        stdscr.erase()
        
        # Draw Dashboard Header
        stdscr.addstr(0, 0, " TETRALATENCY GLOBAL INFERENCE MATRIX ".center(width)[:width], curses.color_pair(1) | curses.A_REVERSE | curses.A_BOLD)
        
        top_stats = f" Total Avail: {len(models)} | Tested & OK: {tested_count} | Fastest Ping: {best_lat if best_lat != float('inf') else 0:.1f}ms "
        stdscr.addstr(1, 0, top_stats.center(width)[:width], curses.color_pair(2) | curses.A_BOLD)
        
        # Draw Search Box
        search_prompt = f" Search: {search_query}"
        stdscr.addstr(2, 0, search_prompt.ljust(width)[:width], curses.color_pair(5))
        
        # Draw Filters
        stdscr.addstr(3, 1, "Filters: ", curses.color_pair(5))
        s_prv = f" [ Prv: {f_prv_opts[f_prv_idx]} ] "
        s_par = f" [ Params: {f_par_opts[f_par_idx]} ] "
        s_ctx = f" [ Ctx: {f_ctx_opts[f_ctx_idx]} ] "
        s_lat = f" [ Lat: {f_lat_opts[f_lat_idx]} ] "
        s_err = f" [ Err: {f_err_opts[f_err_idx]} ] "
        
        lx = 10
        stdscr.addstr(3, lx, s_prv, curses.color_pair(4) | curses.A_BOLD)
        lx += len(s_prv) + 1
        stdscr.addstr(3, lx, s_par, curses.color_pair(4) | curses.A_BOLD)
        lx += len(s_par) + 1
        stdscr.addstr(3, lx, s_ctx, curses.color_pair(4) | curses.A_BOLD)
        lx += len(s_ctx) + 1
        stdscr.addstr(3, lx, s_lat, curses.color_pair(4) | curses.A_BOLD)
        lx += len(s_lat) + 1
        stdscr.addstr(3, lx, s_err, curses.color_pair(4) | curses.A_BOLD)
        
        # Draw Borders
        stdscr.hline(4, 0, curses.ACS_HLINE, width, curses.color_pair(1))
        
        def f_head(col):
            arr = "↓" if sort_desc else "↑"
            text = f"{col} {arr}" if sort_col == col else col
            w = {"ID": id_w, "Prv": pv_w, "Name": name_w, "Params": p_w, "Context": ctx_w, "Inp": mod_w, "Latency": lat_w, "Status": bar_w}.get(col, id_w)
            return text.center(w)

        # Header Columns
        H_ID = f_head("ID").ljust(id_w)[:id_w]
        H_PV = f_head("Prv").center(pv_w)[:pv_w]
        H_NM = f_head("Name").ljust(name_w)[:name_w]
        H_P  = f_head("Params").center(p_w)[:p_w]
        H_C  = f_head("Context").center(ctx_w)[:ctx_w]
        H_M  = f_head("Inp").center(mod_w)[:mod_w]
        H_L  = f_head("Latency").rjust(lat_w)[:lat_w]
        H_B  = f_head("Status").center(bar_w)[:bar_w]
        
        header_str = f" {H_ID} │{H_PV}│ {H_NM} │ {H_P} │ {H_C} │ {H_M} │ {H_L} │ {H_B}"
        stdscr.addstr(5, 0, header_str.ljust(width)[:width], curses.color_pair(5) | curses.A_BOLD)
        
        stdscr.hline(6, 0, curses.ACS_HLINE, width, curses.color_pair(1))
        
        # Draw Rows
        for i in range(min(list_max_height, len(all_models) - scroll_offset)):
            idx = scroll_offset + i
            if idx >= len(all_models): break
            item = all_models[idx]
            
            # Formatting
            lat = item['lat']
            lat_str = f"{lat:.0f}ms" if lat != float('inf') else "---"
            
            # If OK print the latency bar graph, if error print the error text
            bar = f"[{get_gauge(lat)}]" if item['stat'] == "OK" else item['stat'][:bar_w].ljust(bar_w)
            
            p_str = item["params"].center(p_w)[:p_w]
            c_str = item["context"].center(ctx_w)[:ctx_w]
            m_str = item["mod"].center(mod_w)[:mod_w]
            
            px = item["prov"]
            p_code = "NV" if px == "nvidia" else ("GG" if px == "google" else ("MS" if px == "mistral" else ("CS" if px == "codestral" else ("CB" if px == "cerebras" else ("CO" if px == "cohere" else ("GQ" if px == "groq" else "OR"))))))
            pv_str = p_code.center(pv_w)[:pv_w]
            
            row_str = f" {item['id'].ljust(id_w)[:id_w]} │{pv_str}│ {item['name'].ljust(name_w)[:name_w]} │ {p_str} │ {c_str} │ {m_str} │ {lat_str.rjust(lat_w)[:lat_w]} │ {bar}"
            row_str = row_str.ljust(width)[:width]

            if idx == selected_idx:
                try: stdscr.addstr(7+i, 0, row_str, curses.color_pair(4) | curses.A_BOLD)
                except curses.error: pass
            else:
                # Base Color
                if item['stat'] not in ["OK", "Pending..."]: c = curses.color_pair(7) # Dark grey for errors
                elif lat == float('inf'): c = curses.color_pair(7)
                elif lat < 300: c = curses.color_pair(2)
                elif lat < 800: c = curses.color_pair(5)
                else: c = curses.color_pair(3)
                
                try: stdscr.addstr(7+i, 0, row_str, c)
                except curses.error: pass
                
                # Apply secondary color for params
                p_c = None
                if "~" in item["params"]:
                    p_c = curses.color_pair(7) | curses.A_BOLD # Muted Blue for estimates
                elif item["score"] > 300: # Super huge models
                    p_c = curses.color_pair(6) | curses.A_BOLD
                elif item["score"] > 60: # Medium models
                    p_c = curses.color_pair(5) | curses.A_BOLD
                    
                if p_c:
                    try: stdscr.chgat(7+i, 3+id_w+3+pv_w+2+name_w, p_w, p_c)
                    except: pass
                    
                # Colorize the speed bar blocks to make it dynamic
                if item['stat'] == "OK":
                    bar_c = curses.color_pair(2) if lat < 300 else (curses.color_pair(5) if lat < 800 else curses.color_pair(3))
                    bar_start = 20 + id_w + pv_w + name_w + p_w + ctx_w + mod_w + lat_w
                    try: stdscr.chgat(7+i, bar_start, bar_w, bar_c | curses.A_BOLD)
                    except: pass
                    
        # Draw Info Panel
        info_y = height - info_height - 1
        try:
            stdscr.hline(info_y, 0, curses.ACS_HLINE, width, curses.color_pair(1))
        except: pass
        
        if all_models and selected_idx < len(all_models):
            sel_m = all_models[selected_idx]
            
            p_raw = sel_m["prov"]
            
            p_map = {
                "nvidia": "NVIDIA NIM", "openrouter": "OPENROUTER", 
                "mistral": "MISTRAL CLOUD", "codestral": "CODESTRAL API",
                "google": "GOOGLE API", "cerebras": "CEREBRAS",
                "cohere": "COHERE", "groq": "GROQ LPU"
            }
            provider_str = p_map.get(p_raw, "PROVIDER")
            
            status_str = "HEALTHY" if sel_m["stat"] == "OK" else sel_m["stat"].upper()
            status_color = curses.color_pair(2) if sel_m["stat"] == "OK" else curses.color_pair(3)
            
            # Panel Title
            panel_title = f" {provider_str} SYNOPSIS: {sel_m['id']} "
            try: stdscr.addstr(info_y, 2, panel_title, curses.color_pair(1) | curses.A_BOLD)
            except: pass
            
            def get_rpg_bar(ratio, width=14):
                filled = int(ratio * width)
                return "█" * filled + "▒" * (width - filled)

            p_val = sel_m["params"] if sel_m["params"] != "?" else "Unknown"
            score = sel_m["score"]
            int_ratio = min(1.0, (score / 405.0) ** 0.5) if score > 0 else 0.1
            
            try: c_val = float(sel_m["context"][:-1]) if "K" in sel_m["context"] else 8.0
            except: c_val = 8.0
            per_ratio = min(1.0, (c_val / 200.0) ** 0.5)
            
            lat = sel_m["lat"]
            agi_ratio = 1.0 - min(1.0, lat / 2500.0) if lat != float('inf') else 0.0
            
            intel_bar = get_rpg_bar(int_ratio)
            percep_bar = get_rpg_bar(per_ratio)
            agi_bar = get_rpg_bar(agi_ratio)
            
            lat_str = f"{lat:.0f}ms" if lat != float('inf') else "---"
            
            try: stdscr.addstr(info_y+2, 2, f" PARAMS  [{intel_bar}] {p_val.ljust(6)}", curses.color_pair(6) | curses.A_BOLD)
            except: pass
            try: stdscr.addstr(info_y+3, 2, f" CONTEXT [{percep_bar}] {sel_m['context'].ljust(6)}", curses.color_pair(5) | curses.A_BOLD)
            except: pass
            try: stdscr.addstr(info_y+4, 2, f" LATENCY [{agi_bar}] {lat_str.ljust(6)}", curses.color_pair(2) | curses.A_BOLD)
            except: pass
            
            try: 
                stdscr.addstr(info_y+6, 2, f" STATE:  [", curses.color_pair(7))
                stdscr.addstr(info_y+6, 12, f"{status_str}", status_color | curses.A_BOLD)
                stdscr.addstr(info_y+6, 12+len(status_str), f"]", curses.color_pair(7))
            except: pass

            # Description on the right
            desc = sel_m["desc"].replace('\n', ' ')
            max_desc_len = width - 42 
            if max_desc_len > 10:
                # Basic text wrap
                words = desc.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_desc_len:
                        current_line += (word + " ")
                    else:
                        lines.append(current_line)
                        current_line = word + " "
                if current_line: lines.append(current_line)
                
                for d_idx, d_line in enumerate(lines[:5]): # max 5 lines
                    try: stdscr.addstr(info_y+2+d_idx, 40, d_line.strip(), curses.color_pair(7))
                    except: pass

        # Protect against drawing the exact bottom right character which scrolls the terminal
        try:
            stdscr.addstr(height-1, 0, status_msg.ljust(width-1)[:width-1], curses.color_pair(1) | curses.A_BOLD)
        except curses.error:
            pass
        stdscr.refresh()

if __name__ == "__main__":
    k = get_api_keys()
    if not k:
        print("No API Keys found in config or env.")
        sys.exit(1)
        
    m = get_models(k)
    if not m:
        print("Error fetching models from APIs.")
        sys.exit(1)

    os.environ.setdefault("ESCDELAY", "25")
    try:
        curses.wrapper(main, m, k)
    except KeyboardInterrupt:
        pass
    
    # We do a hard exit to prevent the ThreadPoolExecutor's atexit handler 
    # from hanging the terminal while it waits for 8-second API timeouts to finish.
    os._exit(0)
