# -*- coding: utf-8 -*-
"""
AI 面试题库问答助手 —— 基于真实面经的 RAG 问答应用
线上部署版(Render / 本地通用)

与 Colab 版的区别:
1. 去掉了所有 Colab 专属代码(!pip / google.colab.drive / userdata / files)
2. API Key 改为从环境变量读取(在 Render 后台配置 ARK_API_KEY)
3. 面经从项目内的 ./面经数据/ 文件夹读取
4. 去掉了开发期的评测代码(LLM-as-a-Judge 评测是离线质检工具,不参与线上问答)
5. 修正了函数定义顺序(查询理解 必须在 面试助手 之前定义)
6. 启动方式改为监听 0.0.0.0 指定端口(部署平台要求)
"""

import os
import re
import requests
import gradio as gr
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# ============================================================
# 一、初始化大模型客户端(火山引擎方舟 · DeepSeek)
# ============================================================
# API Key 从环境变量读取,不写死在代码里(安全)。
# 部署时在 Render 后台 Environment 里配置 ARK_API_KEY。
API_KEY = os.environ.get("ARK_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "未找到 ARK_API_KEY 环境变量。请在 Render 后台的 Environment 中添加 "
        "ARK_API_KEY,值为你的火山引擎 API Key。"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url="https://ark.cn-beijing.volces.com/api/v3",  # 火山引擎方舟地址
)

# DeepSeek 推理接入点 ID(ep- 开头,不是密码,可公开)
接入点ID = "ep-m-20260702163039-m4qs7"

# 向量化(Embedding)接入点 ID —— 走火山引擎云端 API,不在本机加载大模型。
# 【重要】这里用的是「多模态」Embedding 接入点(doubao-embedding-vision 系列),
#         因为控制台当前只能申请到 vision 版本。它的调用接口与纯文本 Embedding
#         不同(专用路径 + input 需包成 {"type":"text","text":...} 格式),
#         下面的 方舟Embedding 类已做适配。填你自己的 ep- 开头接入点 ID(不是密码)。
向量化接入点ID = "在这里填入你的_Embedding_接入点ID"

# 火山引擎方舟基础地址(与文本生成同一个 base_url)
方舟基础地址 = "https://ark.cn-beijing.volces.com/api/v3"


# ------------------------------------------------------------
# 云端向量化封装:调用火山引擎方舟「多模态」Embedding API
# 为什么这样做:线上部署(Render 免费版仅 512MB 内存)加载本地
# BAAI/bge 模型会内存溢出被杀掉。改成云端 API 后本机几乎不占内存。
#
# 注意:doubao-embedding-vision 系列是多模态模型,不能用标准的
# /embeddings 路径(会报 "does not support this api"),必须用专用的
# /embeddings/multimodal 路径,且 input 要包成 [{"type":"text","text":...}]。
# 这里我们只喂纯文本,所以每条文本包一个 text 片段、逐条向量化。
# ------------------------------------------------------------
class 方舟Embedding(Embeddings):
    """让 LangChain/FAISS 通过火山引擎方舟「多模态 Embedding」API 完成向量化。"""

    def __init__(self, api_key, 基础地址, 接入点):
        self.api_key = api_key
        self.接口地址 = 基础地址.rstrip("/") + "/embeddings/multimodal"
        self.接入点 = 接入点
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _单条向量(self, 文本):
        """把一条文本发给多模态接口,取回它的向量。"""
        payload = {
            "model": self.接入点,
            "input": [{"type": "text", "text": 文本}],
        }
        resp = requests.post(self.接口地址, headers=self.headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # 多模态接口:整个 input(可含多片段)融合成一个向量,放在 data["data"]["embedding"]
        emb = data["data"]["embedding"]
        # 兼容极少数返回成 [[...]] 嵌套一层的情况
        if emb and isinstance(emb[0], list):
            emb = emb[0]
        return emb

    def embed_documents(self, texts):
        # 建库时:逐条向量化(多模态接口按"一次一条内容"设计)
        return [self._单条向量(t) for t in texts]

    def embed_query(self, text):
        # 查询时:单条向量化
        return self._单条向量(text)

# ============================================================
# 二、读取面经 → 切分 → 向量化 → 建 FAISS 索引
# ============================================================
面经文件夹 = os.path.join(os.path.dirname(__file__), "面经数据")


def 载入并构建知识库():
    """读取 ./面经数据/ 下所有 txt,按"一个 Q+A"切块,向量化后建成 FAISS 向量库。"""
    if not os.path.isdir(面经文件夹):
        raise RuntimeError(f"未找到面经文件夹:{面经文件夹}")

    txt_files = [
        os.path.join(面经文件夹, f)
        for f in os.listdir(面经文件夹)
        if f.endswith(".txt")
    ]
    if not txt_files:
        raise RuntimeError("面经文件夹里没有任何 .txt 文件,无法构建知识库。")

    print(f"📁 找到 {len(txt_files)} 份面经文件,开始切分...")

    # 切块:按"一个 Q + 对应 A"切成独立知识块,保证检索单元完整
    chunks = []
    for path in txt_files:
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

        # 提取文件头部(公司/岗位/轮次)作为每块标签,方便溯源
        header = content.split("Q1")[0].strip() if "Q1" in content else ""

        # 用正则把每一对 Q&A 抓出来:从 Qn 到下一个 Qn 之前
        qa_pairs = re.findall(r"(Q\d+[:：].*?)(?=Q\d+[:：]|$)", content, re.DOTALL)
        for qa in qa_pairs:
            qa = qa.strip()
            if qa:
                chunk_text = f"【来源:{filename}】\n{header}\n\n{qa}"
                chunks.append(chunk_text)

    print(f"✅ 切分完成,共 {len(chunks)} 个知识块。正在通过云端 API 向量化...")

    # 向量化:调用火山引擎方舟多模态 Embedding API(云端算,本机不占内存)
    embeddings = 方舟Embedding(API_KEY, 方舟基础地址, 向量化接入点ID)

    print("✅ 向量化完成,正在建立向量库...")
    vs = FAISS.from_texts(chunks, embeddings)
    print(f"🎉 知识库构建完成,已收录 {len(chunks)} 个知识块。")
    return vs


# 应用启动时构建一次(全局共享)
vectorstore = 载入并构建知识库()

# ============================================================
# 三、核心逻辑:查询理解 + 面试助手(RAG 问答)
# ============================================================
# 全局对话历史(多轮记忆)
对话历史 = []


def 查询理解(question, history=None):
    """结合对话历史,把带指代/省略的新问题改写成一个独立完整的检索句;
    闲聊/无关问题则返回 [无需检索]。"""
    历史文本 = ""
    if history:
        for 轮 in history[-3:]:  # 只取最近 3 轮,防膨胀
            历史文本 += f"用户:{轮['user']}\n助手:{轮['assistant']}\n"

    理解prompt = f"""你是多轮对话的查询理解器。请结合【对话历史】,把用户的【当前问题】处理成一个用于检索面经知识库的查询句。

处理规则:
1. 消解指代和省略:把"它/这个/第二点/那"等,结合历史补全成明确的完整问题。
2. 扩写成含相关关键词的陈述句(不要用问句),补充 AI产品经理/AI产品运营 领域的相关概念、同义词。
3. 若当前问题是闲聊、感谢、寒暄,或明显与面经/面试无关,只输出四个字:[无需检索]
4. 只输出处理后的查询句(或[无需检索]),不要解释、不加引号。

【对话历史】
{历史文本 if 历史文本 else "(无,这是第一轮)"}

【当前问题】
{question}

处理后的查询句:"""

    resp = client.chat.completions.create(
        model=接入点ID,
        messages=[{"role": "user", "content": 理解prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def 面试助手(question, k=5, 阈值=1.1):
    """RAG 主流程:查询理解 → 检索 + 阈值过滤 → 拼历史 → DeepSeek 生成分层回答。
    ⚠️ 注意:阈值 1.1 是当初用本地 bge 模型采样定的。换成云端 Embedding 后
    分数分布可能变化,若发现「什么都检索不到」或「无关内容混入」,
    请重新采样几个问题的 similarity_search_with_score 分数、重新校准这个阈值。"""
    global 对话历史

    # ① 查询理解(结合历史)
    理解结果 = 查询理解(question, 对话历史)

    # ② 闲聊/无需检索 → 直接回应
    if "[无需检索]" in 理解结果:
        回答 = "我是面试问答助手,有关于 AI 产品/运营岗面试的问题都可以问我~"
        对话历史.append({"user": question, "assistant": 回答})
        return 回答

    # ③ 检索 + 相似度阈值过滤(过滤掉不相关的)
    results = vectorstore.similarity_search_with_score(理解结果, k=k)
    相关docs = [doc for doc, score in results if score < 阈值]

    if not 相关docs:
        回答 = "根据现有面经,没有检索到与你问题足够相关的内容。可以换个更具体的问法,或补充相关面经。"
        对话历史.append({"user": question, "assistant": 回答})
        return 回答

    检索内容 = "\n\n".join([d.page_content for d in 相关docs])

    # ④ 拼接最近 3 轮历史,让回答承接上文
    历史文本 = ""
    for 轮 in 对话历史[-3:]:
        历史文本 += f"用户:{轮['user']}\n助手:{轮['assistant']}\n"

    prompt = f"""你是一个基于真实面经的AI面试问答助手。

核心原则:以用户实际问题为中心,面经作为补充参考,不被面经角度带偏。若有对话历史,回答要自然承接上文。

【输出风格要求 —— 严格遵守】
- 精炼准确、直击要点,拒绝冗长铺陈和套话。
- 全文控制在 300 字以内,能一句话说清就不用两句。
- 每层只保留最有价值的信息,不逐条展开分析。

【输出格式】请严格按以下 Markdown 格式输出,三个部分标题必须用 ### 开头:

### 直接回答
(2-3 句话的核心结论,关键词用 **加粗**)

### 真题溯源
- 用无序列表逐条列出,来源用 *斜体* 标注;严格依据下面检索到的面经,出处只标明公司,禁止编造来源。

### AI 补充思路
(分点说明,关键概念用 **加粗**,并标注"以下为 AI 建议")

若检索内容与问题无关,请忽略它,不要牵强附会。

【对话历史】
{历史文本 if 历史文本 else "(无)"}

【检索到的真实面经内容】
{检索内容}

【用户当前问题】
{question}
"""
    response = client.chat.completions.create(
        model=接入点ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    回答 = response.choices[0].message.content

    # ⑤ 存进历史
    对话历史.append({"user": question, "assistant": 回答})
    return 回答


# ============================================================
# 四、Gradio 界面
# ============================================================
自定义CSS = """
* { font-family: 'Microsoft YaHei','PingFang SC','Heiti SC','Segoe UI',sans-serif !important; }

.gradio-container {
    min-height: 100vh !important;
    background: linear-gradient(135deg,#eef2f7,#f3e8ff,#e0f2fe,#fce7f3) !important;
    background-size: 300% 300% !important;
    animation: 背景流动 22s ease infinite;
    padding: 30px !important;
}
@keyframes 背景流动 { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

#光晕层 {
    position: fixed !important; inset: 0 !important;
    z-index: 0 !important; pointer-events: none !important; overflow: hidden !important;
}
#光晕层 .光斑 { position: absolute; border-radius: 50%; filter: blur(70px); opacity: 0.75; }
#光晕层 .斑1 { width:38vw; height:38vw; left:2%;  top:5%;   background: radial-gradient(circle, #a78bfa, transparent 70%); animation: 漂1 20s ease-in-out infinite; }
#光晕层 .斑2 { width:34vw; height:34vw; right:3%; top:0%;   background: radial-gradient(circle, #5eead4, transparent 70%); animation: 漂2 24s ease-in-out infinite; }
#光晕层 .斑3 { width:36vw; height:36vw; right:8%; bottom:2%;background: radial-gradient(circle, #f472b6, transparent 70%); animation: 漂3 26s ease-in-out infinite; }
#光晕层 .斑4 { width:32vw; height:32vw; left:6%;  bottom:4%;background: radial-gradient(circle, #93c5fd, transparent 70%); animation: 漂4 22s ease-in-out infinite; }
@keyframes 漂1 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(6%,8%)} }
@keyframes 漂2 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-7%,6%)} }
@keyframes 漂3 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-5%,-7%)} }
@keyframes 漂4 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(8%,-5%)} }

.gradio-container > * { position: relative; z-index: 1; }

#外框 {
    background: linear-gradient(135deg, rgba(255,255,255,0.32), rgba(255,255,255,0.16)) !important;
    backdrop-filter: blur(24px) saturate(1.5);
    border: 1.5px solid rgba(255,255,255,0.85);
    border-radius: 32px !important; padding: 32px !important;
    box-shadow: 0 20px 60px rgba(150,120,200,0.28), inset 0 1px 1px rgba(255,255,255,0.95);
    animation: 整体渐入 0.8s ease both;
}
@keyframes 整体渐入 { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }

#标题区 h1 {
    font-size: 2.5em !important; font-weight: 800 !important; letter-spacing: 2px !important;
    background: linear-gradient(90deg,#7c3aed,#a855f7,#c026d3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 6px !important;
}
#标题区 h1 .sparkle { display:inline-block; animation: 呼吸 2.5s ease-in-out infinite; }
@keyframes 呼吸 { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.15)} }
#标题区 p { color:#7c3aed !important; text-align:center; font-size:1em; letter-spacing:1px; }
#分隔线 { height:2px; border:none; margin:16px auto 24px; background: linear-gradient(90deg,transparent,#c084fc,transparent); width:60%; }

.卡片 {
    backdrop-filter: blur(20px) saturate(1.3); border-radius: 22px !important;
    padding: 18px !important; margin-bottom: 16px !important;
    box-shadow: 0 8px 28px rgba(150,120,200,0.15), inset 0 1px 1px rgba(255,255,255,0.9);
    border: 1px solid rgba(255,255,255,0.8); animation: 卡片渐入 0.7s ease both;
    transition: transform 0.45s cubic-bezier(0.22,1,0.36,1),
                box-shadow 0.45s cubic-bezier(0.22,1,0.36,1) !important;
    transform-style: preserve-3d;
    position: relative;
    overflow: hidden;
}
.卡片:hover {
    transform: perspective(900px) rotateX(3deg) rotateY(-3deg) translateY(-6px) scale(1.015);
    box-shadow: 0 22px 48px rgba(150,120,200,0.32),
                inset 0 1px 1px rgba(255,255,255,0.95);
}
.卡片::after {
    content: "";
    position: absolute;
    top: 0; left: -120%;
    width: 80%; height: 100%;
    background: linear-gradient(115deg, transparent 0%, rgba(255,255,255,0.55) 50%, transparent 100%);
    transform: skewX(-18deg);
    pointer-events: none;
    transition: left 0.7s cubic-bezier(0.22,1,0.36,1);
    z-index: 2;
}
.卡片:hover::after { left: 130%; }
@keyframes 卡片渐入 { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
#卡片覆盖 { background: linear-gradient(135deg, rgba(200,245,215,0.5), rgba(200,245,215,0.25)) !important; animation-delay:0.15s; }
#卡片示例 { background: linear-gradient(135deg, rgba(230,220,250,0.55), rgba(230,220,250,0.3)) !important; animation-delay:0.30s; }
#卡片说明 { background: linear-gradient(135deg, rgba(255,235,215,0.5), rgba(255,235,215,0.25)) !important; animation-delay:0.45s; }
.卡片 h3 { color:#6d28d9 !important; margin:0 0 10px 0 !important; font-size:1.08em; font-weight:700 !important; }
.卡片 p,.卡片 li { color:#4a3f5c !important; font-size:0.87em; line-height:1.7; }

#卡片示例 button {
    position: relative;
    background: linear-gradient(160deg, rgba(255,255,255,0.9), rgba(200,180,255,0.35)) !important;
    border: 1px solid rgba(255,255,255,0.9) !important; color:#5b21b6 !important;
    border-radius: 999px !important; text-align:center !important; margin:7px 0 !important;
    font-weight: 600 !important; transition: all 0.28s ease;
    box-shadow: 0 6px 16px rgba(150,120,200,0.22), inset 0 2px 3px rgba(255,255,255,1), inset 0 -3px 6px rgba(150,120,200,0.15);
}
#卡片示例 button:hover {
    background: linear-gradient(135deg, #a78bfa, #8b5cf6, #7c3aed) !important;
    color:#fff !important;
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 12px 26px rgba(124,58,237,0.4), inset 0 2px 4px rgba(255,255,255,0.7);
}

#聊天区 {
    background: linear-gradient(135deg, rgba(255,255,255,0.6), rgba(255,255,255,0.4)) !important;
    backdrop-filter: blur(20px) saturate(1.3); border: 1px solid rgba(255,255,255,0.85);
    border-radius: 24px !important;
    box-shadow: 0 10px 32px rgba(150,120,200,0.15), inset 0 1px 1px rgba(255,255,255,0.9);
    animation: 卡片渐入 0.7s ease both; animation-delay:0.3s;
}
#聊天区 .label-wrap, #聊天区 label { display:none !important; }
#聊天区 .message.user, #聊天区 .user {
    background: linear-gradient(135deg, rgba(168,139,250,0.28), rgba(196,181,253,0.20)) !important;
    color: #4c1d95 !important;
    border: 1px solid rgba(168,139,250,0.35) !important;
    backdrop-filter: blur(8px) saturate(1.3) !important;
    border-radius: 20px 20px 6px 20px !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.12), inset 0 1px 2px rgba(255,255,255,0.5) !important;
}
#聊天区 .message.bot, #聊天区 .bot {
    background: rgba(248,245,255,0.92) !important; color:#3b2f52 !important;
    border: 1px solid rgba(255,255,255,0.8); border-radius: 20px 20px 20px 6px !important;
    box-shadow: 0 4px 14px rgba(150,120,200,0.12);
}
#聊天区 .bot p, #聊天区 .message.bot p {
    font-size: 15px !important; line-height: 1.85 !important; color: #3b2f52 !important;
    margin: 10px 0 !important; letter-spacing: 0.3px !important;
}
#聊天区 .bot strong, #聊天区 .bot b { color: #6d28d9 !important; font-weight: 700 !important; }
#聊天区 .bot h1, #聊天区 .bot h2, #聊天区 .bot h3 {
    color: #6d28d9 !important; font-weight: 800 !important; font-size: 1.15em !important;
    margin: 18px 0 8px 0 !important; padding-left: 12px !important;
    border-left: 4px solid #a855f7 !important; line-height: 1.4 !important;
}
#聊天区 .bot ul, #聊天区 .bot ol { padding-left: 22px !important; margin: 8px 0 !important; }
#聊天区 .bot li { margin: 6px 0 !important; line-height: 1.8 !important; color: #4a3f5c !important; }
#聊天区 .bot em, #聊天区 .bot i { color: #8b7fa5 !important; font-style: normal !important; font-size: 0.9em !important; }

#输入区 button {
    background: linear-gradient(135deg, #a855f7 0%, #7c3aed 55%, #5eead4 135%) !important;
    color:#fff !important; border:none !important; border-radius:999px !important;
    font-weight: 700 !important; transition: all 0.28s;
    box-shadow: 0 8px 20px rgba(124,58,237,0.4), inset 0 2px 4px rgba(255,255,255,0.6), inset 0 -3px 6px rgba(88,28,135,0.4);
}
#输入区 button:hover {
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 14px 30px rgba(124,58,237,0.55), inset 0 2px 5px rgba(255,255,255,0.7) !important;
}

@media (max-width:768px){ #主布局{flex-direction:column !important;} #外框{padding:16px !important;} }

#输入区 .block, #输入区 .form, #输入区 > div, #输入区 .container {
    background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important;
}
#输入区 textarea {
    background: rgba(255,255,255,0.9) !important;
    border: 1px solid rgba(168,139,250,0.5) !important;
    border-radius: 999px !important;
    color:#3b2f52 !important;
    padding: 12px 18px !important;
    box-shadow: inset 0 1px 3px rgba(150,120,200,0.12) !important;
    transition: all 0.2s;
}
#输入区 textarea:focus { border-color:#a855f7 !important; box-shadow:0 0 0 3px rgba(168,139,250,0.25) !important; }
"""

欢迎语 = (
    "👋 你好,我是你的 **AI 面试题库问答助手**\n\n"
    "我基于真实面经回答,并会区分【真题溯源】和【AI 补充思路】。\n\n"
    "试试点左边的示例问题,或直接向我发问吧"
)


def 填入示例(问题):
    return 问题


with gr.Blocks() as demo:
    gr.HTML("""
    <div id="光晕层">
        <div class="光斑 斑1"></div>
        <div class="光斑 斑2"></div>
        <div class="光斑 斑3"></div>
        <div class="光斑 斑4"></div>
    </div>
    """)
    with gr.Column(elem_id="外框"):
        with gr.Column(elem_id="标题区"):
            gr.HTML("<h1><span class='sparkle'>✨</span> AI 面试小助手 <span class='sparkle'>✨</span></h1>")
            gr.Markdown("基于真实面经的 AI 问答 · 可信溯源 + AI 答题思路")
        gr.HTML("<hr id='分隔线'>")

        with gr.Row(elem_id="主布局"):
            with gr.Column(scale=1):
                with gr.Column(elem_id="卡片覆盖", elem_classes="卡片"):
                    gr.Markdown("### 覆盖范围")
                    gr.Markdown("美团 · 腾讯 · 小红书 · 字节跳动 · MiniMax\n\n岗位:AI 产品经理 / AI 产品运营")
                with gr.Column(elem_id="卡片示例", elem_classes="卡片"):
                    gr.Markdown("### 试试这些问题")
                    示例1 = gr.Button("AI 产品经理和传统产品经理的区别")
                    示例2 = gr.Button("什么是 RAG")
                    示例3 = gr.Button("如何写好一个Skill")
                    示例4 = gr.Button("美团AI 产品岗位面试的问题类别")
                with gr.Column(elem_id="卡片说明", elem_classes="卡片"):
                    gr.Markdown("### 说明")
                    gr.Markdown("真题严格溯源自面经,AI 补充思路会明确标注,请区分使用。")

            with gr.Column(scale=3):
                聊天框 = gr.Chatbot(
                    elem_id="聊天区", height=560, label="",
                    value=[{"role": "assistant", "content": 欢迎语}],
                    avatar_images=(
                        "https://api.dicebear.com/7.x/thumbs/svg?seed=user&backgroundColor=c026d3",
                        "https://api.dicebear.com/7.x/bottts/svg?seed=ai&backgroundColor=a855f7",
                    ),
                )
                with gr.Row(elem_id="输入区"):
                    输入框 = gr.Textbox(placeholder="输入你的面试问题…", scale=8, label="", container=False)
                    发送键 = gr.Button("发送", scale=1, variant="primary")

    # ---- 事件绑定(messages 字典格式)----
    def 显示问题(消息, 历史):
        # 立刻把用户的问题上屏,并清空输入框(即时反馈)
        if not 消息.strip():
            return 历史, ""
        历史 = 历史 + [{"role": "user", "content": 消息}]
        return 历史, ""

    def 回答问题(历史):
        # 取最后一条用户问题,调用 RAG 出答案,再追加
        if not 历史 or 历史[-1]["role"] != "user":
            return 历史
        问题 = 历史[-1]["content"]
        答案 = 面试助手(问题)
        历史 = 历史 + [{"role": "assistant", "content": 答案}]
        return 历史

    发送键.click(显示问题, [输入框, 聊天框], [聊天框, 输入框]).then(回答问题, 聊天框, 聊天框)
    输入框.submit(显示问题, [输入框, 聊天框], [聊天框, 输入框]).then(回答问题, 聊天框, 聊天框)

    示例1.click(填入示例, 示例1, 输入框)
    示例2.click(填入示例, 示例2, 输入框)
    示例3.click(填入示例, 示例3, 输入框)
    示例4.click(填入示例, 示例4, 输入框)


# ============================================================
# 五、启动(部署平台会给一个 PORT 环境变量;本地默认 7860)
# ============================================================
if __name__ == "__main__":
    端口 = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=端口, css=自定义CSS, theme=gr.themes.Base())
