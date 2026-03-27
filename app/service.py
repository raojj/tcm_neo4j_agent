import json
import logging
from typing import Any, Dict, List, Set, Tuple
import time
from threading import Lock

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from app.config import settings
from app.database import (
    get_relationship_types_from_mysql,
    get_syndrome_introductions_by_names,
    get_syndrome_name_candidates,
)
from app.models import ParseResp
from app.tools import generate_dynamic_tools

logger = logging.getLogger(__name__)

_llm_lock = Lock()
_llm_instance: ChatOpenAI | None = None

_agent_lock = Lock()
_agent_cache: Dict[str, Any] = {"ts": 0.0, "agent": None, "label_tool_map": None}


def _get_llm() -> ChatOpenAI:
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    with _llm_lock:
        if _llm_instance is None:
            _llm_instance = ChatOpenAI(
                base_url=settings.OPENAI_BASE_URL,
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL,
                temperature=0,
            )
    return _llm_instance


def _get_agent_and_label_map():
    ttl = settings.AGENT_CACHE_TTL_SECONDS
    if ttl > 0:
        with _agent_lock:
            cached = _agent_cache
            if cached.get("agent") and (time.time() - float(cached.get("ts", 0.0)) < ttl):
                return cached["agent"], cached["label_tool_map"], cached.get("tools") or []

    tools, label_tool_map = generate_dynamic_tools()
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(tools)
    agent = create_react_agent(llm_with_tools, tools)

    if ttl > 0:
        with _agent_lock:
            _agent_cache["agent"] = agent
            _agent_cache["label_tool_map"] = label_tool_map
            _agent_cache["tools"] = tools
            _agent_cache["ts"] = time.time()

    return agent, label_tool_map, tools


# -----------------------------
# Utils
# -----------------------------
def get_last_ai_content(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and (m.content and str(m.content).strip()):
            return str(m.content)
    return ""


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def extract_json_object_with_top_keys(text: str, expected_top_keys: Set[str]) -> str:
    """
    Extract a JSON object whose top-level keys exactly match expected_top_keys.
    Picks the last matching candidate.
    """
    if not text or not str(text).strip():
        raise ValueError("Empty model output")

    t = _strip_code_fence(str(text))

    def iter_json_object_spans(s: str):
        n = len(s)
        i = 0
        in_str = False
        esc = False

        while i < n:
            ch = s[i]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            else:
                if ch == '"':
                    in_str = True
                    i += 1
                    continue

            if ch == "{":
                start = i
                depth = 0
                j = i
                in_str2 = False
                esc2 = False
                while j < n:
                    c = s[j]
                    if in_str2:
                        if esc2:
                            esc2 = False
                        elif c == "\\":
                            esc2 = True
                        elif c == '"':
                            in_str2 = False
                        j += 1
                        continue
                    else:
                        if c == '"':
                            in_str2 = True
                            j += 1
                            continue

                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            yield (start, j + 1)
                            break
                    j += 1

                i = start + 1
                continue

            i += 1

    def is_target_schema(obj: Any) -> bool:
        return isinstance(obj, dict) and set(obj.keys()) == expected_top_keys

    key_tokens = [f'"{k}"' for k in expected_top_keys]

    candidates: List[str] = []
    for a, b in iter_json_object_spans(t):
        chunk = t[a:b].strip()
        if any(tok not in chunk for tok in key_tokens):
            continue
        try:
            parsed = json.loads(chunk)
        except Exception:
            continue
        if is_target_schema(parsed):
            candidates.append(chunk)

    if not candidates:
        preview = t[:400].replace("\n", "\\n")
        raise ValueError(
            f"No valid JSON object found for keys={sorted(expected_top_keys)}. Preview: {preview!r}"
        )

    return candidates[-1]


def _dedupe_dict_list(items: List[Dict[str, str]], key_fields: Tuple[str, ...]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for it in items:
        k = tuple((it.get(f) or "").strip() for f in key_fields)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# -----------------------------
# Syndrome recommend (MySQL + LLM)
# -----------------------------
def _extract_json_top_keys(text: str, expected_top_keys: Set[str]) -> Dict[str, Any]:
    js = extract_json_object_with_top_keys(text, expected_top_keys)
    return json.loads(js)


def recommend_syndromes_from_mysql(symptom_text: str, candidate_limit: int | None = None) -> List[str]:
    """
    根据用户症状描述，从 MySQL `originaldata_syndrome` 推荐最相关的5个证型名称。
    流程：
      1) 从表中拉取 name 候选集（candidate_limit）
      2) LLM 从候选 name 中挑 10 个最相关
      3) 取这10个的 introduction
      4) LLM 结合 introduction 复排出最相关 5 个
    """
    text = (symptom_text or "").strip()
    if not text:
        return []

    # 1) candidates from mysql
    limit = candidate_limit if candidate_limit is not None else settings.SYNDROME_CANDIDATE_LIMIT_DEFAULT
    candidates = get_syndrome_name_candidates(limit=limit)
    if not candidates:
        return []

    # LLM
    llm = _get_llm()

    # 2) select top10 names
    sys1 = (
        "你是中医辨证推荐助手。\n"
        "给定用户的症状描述与候选证型名称列表，请从中挑选最相关的10个证型。\n"
        "强约束：只能从候选列表中选择；必须输出严格 JSON 纯文本；顶层键只能是 syndromes。\n"
        "syndromes 必须是长度为10的字符串数组（若候选不足10则输出不超过候选数）。\n"
        "禁止输出任何解释文字、禁止 markdown、禁止代码块。\n"
    )
    user1 = (
        f"【用户症状描述】\n{text}\n\n"
        f"【候选证型名称列表】\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
        "请输出 JSON："
    )
    resp1 = llm.invoke([SystemMessage(content=sys1), HumanMessage(content=user1)])
    data1 = _extract_json_top_keys(str(resp1.content or ""), {"syndromes"})
    top10 = data1.get("syndromes") or []
    if not isinstance(top10, list):
        top10 = []
    top10_clean: List[str] = []
    seen = set()
    cand_set = set(candidates)
    for x in top10:
        n = (str(x or "")).strip()
        if not n or n in seen:
            continue
        if n not in cand_set:
            continue
        seen.add(n)
        top10_clean.append(n)
        if len(top10_clean) >= 10:
            break
    if not top10_clean:
        return []

    # 3) fetch introductions for top10
    intro_rows = get_syndrome_introductions_by_names(top10_clean)
    intro_map = {r["name"]: (r.get("introduction") or "") for r in intro_rows if r.get("name")}
    packed = [{"name": n, "introduction": intro_map.get(n, "")} for n in top10_clean]

    # 4) rerank top5 with introductions
    sys2 = (
        "你是中医辨证推荐助手。\n"
        "给定用户的症状描述，以及 10 个候选证型（含 introduction），请选出最相关的5个证型。\n"
        "强约束：只能从给定10个候选里选择；必须输出严格 JSON 纯文本；顶层键只能是 top5。\n"
        "top5 必须是长度为5的字符串数组（若候选不足5则输出不超过候选数）。\n"
        "禁止输出任何解释文字、禁止 markdown、禁止代码块。\n"
    )
    user2 = (
        f"【用户症状描述】\n{text}\n\n"
        f"【候选证型(含introduction)】\n{json.dumps(packed, ensure_ascii=False)}\n\n"
        "请输出 JSON："
    )
    resp2 = llm.invoke([SystemMessage(content=sys2), HumanMessage(content=user2)])
    data2 = _extract_json_top_keys(str(resp2.content or ""), {"top5"})
    top5 = data2.get("top5") or []
    if not isinstance(top5, list):
        top5 = []

    top10_set = set(top10_clean)
    out: List[str] = []
    seen2 = set()
    for x in top5:
        n = (str(x or "")).strip()
        if not n or n in seen2:
            continue
        if n not in top10_set:
            continue
        seen2.add(n)
        out.append(n)
        if len(out) >= 5:
            break

    # 兜底：LLM 输出不够时，按 top10 补齐
    if len(out) < min(5, len(top10_clean)):
        for n in top10_clean:
            if n not in seen2:
                out.append(n)
                seen2.add(n)
            if len(out) >= min(5, len(top10_clean)):
                break

    return out


# -----------------------------
# Prompts
# -----------------------------
def build_system_prompt_step1() -> str:
    return (
        "你是一个中医知识图谱构建助手。\n"
        "任务：判断用户文本是否出现“方剂治疗某一症状型证型”，必须明确出现症状型证型，如头晕型血虚证。\n"
        "如果出现：\n"
        "1) 你必须抽取：方剂（必须）、证型（必须）、症状列表（必须，可多个，先最小拆分+归一化）。\n"
        "2) 你必须构造表型节点名：方剂名-证型名。\n"
        "3) 你必须调用 INDICATION 节点全集工具检查该表型节点是否已存在：\n"
        "   - 若已存在：needCreate=false\n"
        "   - 若不存在：needCreate=true\n"
        "注意：表型节点的命名只能是方剂-证型，不允许出现症状\n"
        "如果用户输入没有出现症状型证型：needIndication=false，needCreate=false。\n\n"
        "输出必须是 JSON 纯文本，顶层字段只能是：\n"
        "needIndication, needCreate, indicationName, prescriptionName, syndromeName, symptomNames\n"
        "其中 symptomNames 为归一化后的症状名数组。\n"
        "禁止输出任何解释文字、禁止 markdown、禁止代码块。\n"
    )


def build_user_prompt_step1(label_tool_map: List[Dict[str, Any]], user_text: str) -> str:
    return (
        "【可用节点类型(label)与工具】\n"
        "当你需要检查某类节点是否存在时，必须调用对应工具获取该 label 的全部节点进行匹配。\n"
        f"{json.dumps(label_tool_map, ensure_ascii=False)}\n\n"
        f"【用户文本】\n{user_text}\n\n"
        "请输出 JSON："
    )


def build_system_prompt_step2() -> str:
    return (
        "你是一个中医知识图谱构建助手，提取用户输入的证型、疾病、方剂、中医疾病、症状、病理、中药、表型等节点。\n\n"
        "【强制归一化与拆分规则（必须遵守）】\n"
        "1) 凡是症状、体征、脉象、舌象、感觉描述等信息，必须进行归一化处理，不允许跳过。\n"
        "2) 所有复合描述必须先拆分为最小原子单元，再分别进行归一化。\n"
        "   - 复合脉象：脉濡细 → 脉濡、脉细\n"
        "   - 并列症状：腹胀腹痛 → 腹胀、腹痛\n"
        "3) 归一化遵循“中医医学语义等价原则”，不要求字符串一致。\n"
        "4) 归一化后的最小症状/体征是唯一允许用于实体对齐、节点输出的形式。\n"
        "   - 未经归一化的原始文本描述，严禁直接作为节点输出。\n\n"
        "任务：从文本中抽取涉及的实体类别(label)，对每个涉及 label 调用工具获取该类节点全集进行语义匹配。\n"
        "有些症状和疾病会混淆，因此遇到症状/疾病时两类都要检查。\n"
        "除 INDICATION 外，任何节点都必须来自工具返回全集的 props.name（未命中就不要输出）。\n\n"
        "额外要求：如果给定了 indicationName 且该 INDICATION 在图中已存在，你必须把它作为节点输出（nodeType=INDICATION）。\n\n"
        "输出必须是 JSON 纯文本，且顶层只能有 nodes。\n"
        "nodes 每项只能包含 name、nodeType（英文）。\n"
        "禁止输出任何解释文字、禁止 markdown、禁止代码块。\n"
    )


def build_user_prompt_step2(
    label_tool_map: List[Dict[str, Any]],
    user_text: str,
    indication_name: str,
    indication_exists: bool,
) -> str:
    return (
        "【可用节点类型(label)与工具】\n"
        "需要某类节点全集用于匹配/归一化时，必须先调用对应工具获取该 label 的全部节点。\n"
        "工具返回包含 nodes 列表，每项含 props.name（若存在）。\n"
        "如果不确定实体属于哪个 label，允许最多并行调用 3 个可能的工具后再做对齐。\n"
        f"{json.dumps(label_tool_map, ensure_ascii=False)}\n\n"
        f"【用户文本】\n{user_text}\n\n"
        f"【表型提示】indicationName={indication_name!r}, indicationExists={indication_exists}\n"
        "如果 indicationExists=true，你必须确保 nodes 中包含该 INDICATION。\n\n"
        "请输出 JSON："
    )


def build_system_prompt_step3_extra(relationship_type_field: str = "type_name") -> str:
    """
    Optional: let model add other relations beyond phenotype-mandatory ones.
    If you don't need it, you can disable in code with ENABLE_EXTRA_RELATIONS=False.
    """
    return (
        "你是一个中医知识图谱构建助手。\n"
        "任务：在给定 nodes/newNodes 和用户文本的前提下，补充其它可能的关系（不包含表型三关系），关系不能重复。\n"
        f"relationshipType 必须输出 {relationship_type_field}。\n"
        "如果无法确定关系类型，跳过，不要猜。\n"
        "关系端点允许来自 nodes 或 newNodes（但 newNodes 只会是 INDICATION）。\n"
        "输出必须是 JSON 纯文本，且顶层只能有 relationships。\n"
        "relationships 每项只能包含 sourceNodeName、targetNodeName、relationshipType。\n"
        "禁止输出任何解释文字、禁止 markdown、禁止代码块。\n"
    )


def build_user_prompt_step3_extra(
    user_text: str,
    nodes: List[Dict[str, Any]],
    new_nodes: List[Dict[str, Any]],
    rel_compact: List[Dict[str, str]],
    relationship_step_first: List[Dict[str, Any]],
    relationship_type_field: str = "type_name",
) -> str:
    return (
        f"【用户文本】\n{user_text}\n\n"
        f"【nodes】\n{json.dumps(nodes, ensure_ascii=False)}\n\n"
        f"【newNodes】\n{json.dumps(new_nodes, ensure_ascii=False)}\n\n"
        f"【已存在的关系】relationshipType 输出字段：{relationship_step_first}\n"
        f"【可用关系类型】relationshipType 输出字段：{relationship_type_field}\n"
        f"{json.dumps(rel_compact, ensure_ascii=False)}\n\n"
        "请输出 JSON："
    )


# -----------------------------
# Main
# -----------------------------
def parse_text_with_langchain(text: str, limit: int = 120) -> ParseResp:
    """
    Pipeline:
      Step1: 判断是否需要表型 + 构造表型名 + 检查表型是否已存在（工具）
      Step2: 节点抽取/归一化/工具对齐；若表型已存在则必须放入 nodes
      Step3: 若存在表型语义（无论新建/已存在），强制生成三类表型关系：
             方剂 -TREATMENT_RULE-> 表型
             表型 -PHENOTYPIC_SYNDROME-> 证型
             表型 -PHENOTYPIC_SYMPTOM-> 症状(可多个)
            （可选）再让模型补充其它关系
    """
    # 1) relationship types (from mysql)
    rel_types = get_relationship_types_from_mysql()
    logger.info(f"Got {len(rel_types)} relationship types.")

    rel_compact: List[Dict[str, str]] = []
    rel_type_names: Set[str] = set()
    for r in rel_types:
        tn = (r.get("type_name") or r.get("typeName") or r.get("type") or "").strip()
        dn = (r.get("display_name") or r.get("displayName") or "").strip()
        desc = (r.get("description") or "").strip()
        if tn:
            rel_compact.append({"type_name": tn, "display_name": dn, "description": desc})
            rel_type_names.add(tn)

    # Required relation type names (your requirement)
    REQUIRED_REL_TYPES = {"TREATMENT_RULE", "PHENOTYPIC_SYNDROME", "PHENOTYPIC_SYMPTOM"}
    missing_required = [x for x in REQUIRED_REL_TYPES if x not in rel_type_names]
    if missing_required:
        # 不阻塞，但会在日志里提示；如果你的库里名字不同，请在这里改映射
        logger.warning(f"Missing required relationship types in DB: {missing_required}")

    # 2) tools + agent
    agent, label_tool_map, tools = _get_agent_and_label_map()
    logger.info(f"Generated {len(tools)} tools from database.")
    llm = _get_llm()

    # -----------------
    # Step1 (with tools): phenotype decision + existence check
    # -----------------
    def run_step1() -> Dict[str, Any]:
        msgs = [
            SystemMessage(content=build_system_prompt_step1()),
            HumanMessage(content=build_user_prompt_step1(label_tool_map, text)),
        ]
        result = agent.invoke({"messages": msgs})
        out = get_last_ai_content(result["messages"])
        js = extract_json_object_with_top_keys(
            out,
            {
                "needIndication",
                "needCreate",
                "indicationName",
                "prescriptionName",
                "syndromeName",
                "symptomNames",
            },
        )
        data = json.loads(js)

        # sanitize
        need_ind = bool(data.get("needIndication"))
        need_create = bool(data.get("needCreate"))
        indication_name = (data.get("indicationName") or "").strip()
        rx = (data.get("prescriptionName") or "").strip()
        syn = (data.get("syndromeName") or "").strip()
        syms = data.get("symptomNames") or []
        if not isinstance(syms, list):
            syms = []
        syms = [str(x).strip() for x in syms if str(x).strip()]

        # If no phenotype actually needed, normalize fields
        if not need_ind:
            return {
                "needIndication": False,
                "needCreate": False,
                "indicationName": "",
                "prescriptionName": rx,
                "syndromeName": syn,
                "symptomNames": [],
            }

        # If phenotype needed but indication_name is empty, force consistent (avoid downstream crash)
        if need_ind and not indication_name and rx and syn:
            indication_name = f"{rx}-{syn}"

        return {
            "needIndication": need_ind,
            "needCreate": need_create,
            "indicationName": indication_name,
            "prescriptionName": rx,
            "syndromeName": syn,
            "symptomNames": syms,
        }

    # -----------------
    # Step2 (with tools): nodes matching; include INDICATION in nodes if exists
    # -----------------
    def run_step2(indication_name: str, indication_exists: bool) -> List[Dict[str, str]]:
        msgs = [
            SystemMessage(content=build_system_prompt_step2()),
            HumanMessage(content=build_user_prompt_step2(label_tool_map, text, indication_name, indication_exists)),
        ]
        result = agent.invoke({"messages": msgs})
        out = get_last_ai_content(result["messages"])
        js = extract_json_object_with_top_keys(out, {"nodes"})
        data = json.loads(js)

        nodes = data.get("nodes") or []
        if not isinstance(nodes, list):
            nodes = []

        sanitized: List[Dict[str, str]] = []
        for n in nodes:
            name = (n.get("name") or "").strip()
            node_type = (n.get("nodeType") or "").strip()
            if name and node_type:
                sanitized.append({"name": name, "nodeType": node_type})

        # de-dupe
        return _dedupe_dict_list(sanitized, ("name", "nodeType"))

    # ---- Run Step1 first (we need its output to decide create/exist) ----
    step1 = run_step1()
    need_indication = step1["needIndication"]
    need_create = step1["needCreate"]
    indication_name = step1["indicationName"]
    prescription_name_hint = step1["prescriptionName"]
    syndrome_name_hint = step1["syndromeName"]
    symptom_names_hint = step1["symptomNames"]

    # Step1 says: needCreate indicates NOT exists; exists = needIndication and (not needCreate)
    indication_exists = bool(need_indication and (not need_create) and indication_name)

    # ---- Step2: nodes ----
    nodes = run_step2(indication_name, indication_exists)

    # If indication exists, ensure it is present in nodes (safety net)
    if indication_exists and indication_name:
        if not any(n["name"] == indication_name and n["nodeType"] == "INDICATION" for n in nodes):
            nodes.append({"name": indication_name, "nodeType": "INDICATION"})
            nodes = _dedupe_dict_list(nodes, ("name", "nodeType"))

    # ---- newNodes: only if need_indication and need_create ----
    new_nodes: List[Dict[str, str]] = []
    if need_indication and need_create and indication_name:
        new_nodes = [{"name": indication_name, "nodeType": "INDICATION"}]

    # -----------------
    # Step3: enforce phenotype relations in code
    # -----------------
    relationships: List[Dict[str, str]] = []

    # Helper: check presence for endpoints (nodes + newNodes)
    all_node_names = {n["name"] for n in nodes if n.get("name")}
    all_newnode_names = {n["name"] for n in new_nodes if n.get("name")}
    all_endpoints = all_node_names | all_newnode_names

    def _endpoint_ok(name: str) -> bool:
        return name in all_endpoints

    # We must build 3 relations if phenotype semantics exists
    if need_indication and indication_name:
        # Find best-matched prescription/syndrome/symptoms from nodes using hints
        # We only link if endpoints exist (strict)
        rx_name = None
        syn_name = None
        sym_names: List[str] = []

        # Prescription: prefer exact hint in nodes
        if prescription_name_hint and prescription_name_hint in all_node_names:
            rx_name = prescription_name_hint
        else:
            # fallback: any TCM_PRESCRIPTION (if only one)
            rx_candidates = [n["name"] for n in nodes if n.get("nodeType") == "TCM_PRESCRIPTION"]
            if len(rx_candidates) == 1:
                rx_name = rx_candidates[0]

        # Syndrome: prefer exact hint
        if syndrome_name_hint and syndrome_name_hint in all_node_names:
            syn_name = syndrome_name_hint
        else:
            syn_candidates = [n["name"] for n in nodes if n.get("nodeType") == "TCM_SYNDROME"]
            if len(syn_candidates) == 1:
                syn_name = syn_candidates[0]

        # Symptoms: prefer hint list intersection with nodes
        if symptom_names_hint:
            sym_names = [s for s in symptom_names_hint if s in all_node_names]
        if not sym_names:
            # fallback: all SYMPTOM nodes
            sym_names = [n["name"] for n in nodes if n.get("nodeType") == "SYMPTOM"]

        # Mandatory relations (only add when endpoints are available)
        # 1) 方剂 -TREATMENT_RULE-> 表型
        if rx_name and _endpoint_ok(indication_name):
            relationships.append(
                {"sourceNodeName": rx_name, "targetNodeName": indication_name, "relationshipType": "TREATMENT_RULE"}
            )

        # 2) 表型 -PHENOTYPIC_SYNDROME-> 证型
        if syn_name and _endpoint_ok(indication_name):
            relationships.append(
                {"sourceNodeName": indication_name, "targetNodeName": syn_name, "relationshipType": "PHENOTYPIC_SYNDROME"}
            )

        # 3) 表型 -PHENOTYPIC_SYMPTOM-> 症状(可多个)
        for sname in sym_names:
            if _endpoint_ok(indication_name) and _endpoint_ok(sname):
                relationships.append(
                    {"sourceNodeName": indication_name, "targetNodeName": sname, "relationshipType": "PHENOTYPIC_SYMPTOM"}
                )

    # de-dupe phenotype relations
    relationships = _dedupe_dict_list(relationships, ("sourceNodeName", "targetNodeName", "relationshipType"))

    # -----------------
    # Optional: let model add extra relations (non-mandatory)
    # -----------------
    ENABLE_EXTRA_RELATIONS = settings.ENABLE_EXTRA_RELATIONS

    if ENABLE_EXTRA_RELATIONS:
        try:
            msgs3 = [
                SystemMessage(content=build_system_prompt_step3_extra("type_name")),
                HumanMessage(content=build_user_prompt_step3_extra(text, nodes, new_nodes, rel_compact, relationships, "type_name")),
            ]
            resp3 = llm.invoke(msgs3)
            out3 = str(resp3.content or "")
            js3 = extract_json_object_with_top_keys(out3, {"relationships"})
            data3 = json.loads(js3)
            extra_rels = data3.get("relationships") or []
            if not isinstance(extra_rels, list):
                extra_rels = []

            # sanitize: endpoints must exist in nodes/newNodes
            sanitized_extra: List[Dict[str, str]] = []
            for r in extra_rels:
                s = (r.get("sourceNodeName") or "").strip()
                t = (r.get("targetNodeName") or "").strip()
                rt = (r.get("relationshipType") or "").strip()
                if not (s and t and rt):
                    continue
                if s not in all_endpoints or t not in all_endpoints:
                    continue
                sanitized_extra.append({"sourceNodeName": s, "targetNodeName": t, "relationshipType": rt})

            # merge + dedupe
            relationships = _dedupe_dict_list(
                relationships + sanitized_extra,
                ("sourceNodeName", "targetNodeName", "relationshipType"),
            )
        except Exception as e:
            logger.error(f"Error in extra relationship generation: {e}")
            # 不影响主流程

    # -----------------
    # Final
    # -----------------
    result = ParseResp(
        newNodes=new_nodes,
        nodes=nodes,
        relationships=relationships,
    )
    return result
