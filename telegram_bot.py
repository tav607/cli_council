#!/usr/bin/env python3
"""
CLI Council Telegram Bot
通过 Telegram 接收问题，调用 CLI Council 处理后返回结果
"""

import os
import re
import asyncio
import logging
import html
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.request import HTTPXRequest


def convert_markdown_table(text: str) -> str:
    """将 Markdown 表格转换为列表格式（在 HTML 转义之前调用）"""
    lines = text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 检测表格：以 | 开头或包含 | 的行
        if '|' in line and line.strip().startswith('|'):
            # 收集整个表格
            table_lines = []
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1

            # 解析表格
            if len(table_lines) >= 2:
                # 解析表头
                header_cells = [c.strip() for c in table_lines[0].split('|')]
                header_cells = [c for c in header_cells if c]  # 去除空单元格

                # 跳过分隔行（包含 --- 的行）
                data_start = 1
                if len(table_lines) > 1 and re.match(r'^[\s|:-]+$', table_lines[1]):
                    data_start = 2

                # 解析数据行
                data_rows = []
                for row_line in table_lines[data_start:]:
                    cells = [c.strip() for c in row_line.split('|')]
                    cells = [c for c in cells if c or cells.index(c) > 0]  # 保留非空单元格
                    # 过滤掉空字符串
                    cells = [c for c in cells if c]
                    if cells:
                        data_rows.append(cells)

                # 转换为列表格式
                if len(header_cells) >= 2 and data_rows:
                    # 假设第一列是特性名，其余列是不同选项的值
                    result.append(f"📊 对比：{' vs '.join(header_cells[1:])}\n")

                    for row in data_rows:
                        if len(row) >= 2:
                            feature = row[0]
                            result.append(f"▸ {feature}")
                            for idx, value in enumerate(row[1:]):
                                if idx < len(header_cells) - 1:
                                    col_name = header_cells[idx + 1]
                                    result.append(f"  • {col_name}: {value}")
                            result.append("")  # 空行分隔
                else:
                    # 无法解析的表格，保持原样
                    result.extend(table_lines)
            else:
                result.extend(table_lines)
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def markdown_to_html(text: str) -> str:
    """将 Markdown 风格的文本转换为 Telegram HTML 格式

    Telegram HTML 支持: <b>, <i>, <u>, <s>, <code>, <pre>, <a>
    """
    # 先转换表格（在 HTML 转义之前）
    text = convert_markdown_table(text)

    # 转义 HTML 特殊字符
    text = html.escape(text)

    # 处理代码块 ```...``` (必须在单行 ` 之前处理)
    text = re.sub(r'```(\w*)\n?(.*?)```', r'<pre>\2</pre>', text, flags=re.DOTALL)

    # 转换 `code` 为 <code>code</code>
    text = re.sub(r'`([^`\n]+?)`', r'<code>\1</code>', text)

    # 转换 **bold** 或 __bold__ 为 <b>bold</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # 转换 *italic* 为 <i>italic</i>
    # 更严格的匹配：星号前后不能是字母数字或其他星号，且内容不能以空格开头/结尾
    text = re.sub(r'(?<![*\w])\*([^\s*][^*]*?[^\s*])\*(?![*\w])', r'<i>\1</i>', text)
    # 处理单字符斜体 *x*
    text = re.sub(r'(?<![*\w])\*([^\s*])\*(?![*\w])', r'<i>\1</i>', text)

    # 处理标题 ### / ## / # -> 加粗
    text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    # 处理列表项：把 - 或 * 开头的列表转为 • (保留缩进层级)
    def convert_list_item(match):
        indent = match.group(1)
        content = match.group(2)
        # 计算缩进层级，每2-4个空格算一层
        indent_level = len(indent) // 2
        # 用不同的符号表示层级，或者用空格缩进
        prefix = "  " * indent_level + "• "
        return prefix + content

    # 匹配行首的缩进 + [-*] + 空格 + 内容
    text = re.sub(r'^(\s*)[-*]\s+(.+)$', convert_list_item, text, flags=re.MULTILINE)

    # 清理残留的单独 * 号（非斜体标记的情况）
    # 行首的单独 * 后面跟空格和内容，但没有闭合
    # 这种情况已经被上面的列表处理了

    # 处理 > 引用块 (注意: > 已被转义为 &gt;)
    # 使用 [ \t] 而不是 \s，避免匹配换行符
    text = re.sub(r'^&gt;[ \t]?(.*)$', r'│ \1', text, flags=re.MULTILINE)

    # 处理 LaTeX 数学符号 $...$，转换为 Unicode
    latex_symbols = {
        r'\\approx': '≈',
        r'\\neq': '≠',
        r'\\leq': '≤',
        r'\\geq': '≥',
        r'\\times': '×',
        r'\\div': '÷',
        r'\\pm': '±',
        r'\\infty': '∞',
        r'\\sum': '∑',
        r'\\prod': '∏',
        r'\\sqrt': '√',
        r'\\alpha': 'α',
        r'\\beta': 'β',
        r'\\gamma': 'γ',
        r'\\delta': 'δ',
        r'\\pi': 'π',
        r'\\theta': 'θ',
        r'\\lambda': 'λ',
        r'\\mu': 'μ',
        r'\\sigma': 'σ',
        r'\\rightarrow': '→',
        r'\\leftarrow': '←',
        r'\\Rightarrow': '⇒',
        r'\\Leftarrow': '⇐',
        r'\\leftrightarrow': '↔',
        r'\\subset': '⊂',
        r'\\supset': '⊃',
        r'\\in': '∈',
        r'\\notin': '∉',
        r'\\forall': '∀',
        r'\\exists': '∃',
        r'\\cdot': '·',
        r'\\ldots': '…',
        r'\\dots': '…',
    }

    def replace_latex(match):
        content = match.group(1)
        for latex, unicode_char in latex_symbols.items():
            content = re.sub(latex, unicode_char, content)
        return content

    # 匹配 $...$ 中的内容并替换 LaTeX 符号
    text = re.sub(r'\$([^$]+)\$', replace_latex, text)

    return text

from cli_council import (
    stage1_first_opinions,
    stage2_review,
    stage3_final_response,
    CLIS,
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 从环境变量读取配置
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_IDS: set[int] = set()

# 解析允许的用户 ID
allowed_ids_str = os.getenv("ALLOWED_USER_IDS", "")
if allowed_ids_str:
    for uid in allowed_ids_str.split(","):
        uid = uid.strip()
        if uid.isdigit():
            ALLOWED_USER_IDS.add(int(uid))

# 用户设置存储（内存）
user_settings: dict[int, dict] = {}

# 用户处理状态（防止并发请求）
user_processing: dict[int, bool] = {}


def is_allowed(user_id: int) -> bool:
    """检查用户是否在白名单中"""
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def get_user_settings(user_id: int) -> dict:
    """获取用户设置，不存在则创建默认设置"""
    if user_id not in user_settings:
        user_settings[user_id] = {"quiet": True}
    return user_settings[user_id]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /start 命令"""
    if not is_allowed(update.effective_user.id):
        return

    welcome_text = """🏛️ 欢迎使用 CLI Council Bot!

这是一个多模型协作问答系统，基于 Karpathy 的三阶段机制：
1. 多个 LLM 并行回答问题
2. 互相匿名评审和排名
3. Chairman 综合生成最终答案

📝 使用方法：
• 直接发送文字提问
• /quiet - 切换到安静模式
• /verbose - 切换到详细模式
• /status - 查看当前模式
• /help - 显示帮助信息

当前模式：安静模式 🔇"""

    await update.message.reply_text(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /help 命令"""
    if not is_allowed(update.effective_user.id):
        return

    help_text = """📖 CLI Council Bot 帮助

🔧 可用命令：
/start - 开始使用
/quiet - 安静模式（只显示最终答案）
/verbose - 详细模式（显示所有阶段）
/status - 查看当前设置
/help - 显示此帮助

💡 提示：
• 详细模式会分阶段显示每个模型的回答和评审
• 安静模式更简洁，只显示最终综合答案
• 查询可能需要 1-3 分钟，请耐心等待"""

    await update.message.reply_text(help_text)


async def quiet_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /quiet 命令"""
    if not is_allowed(update.effective_user.id):
        return

    user_id = update.effective_user.id
    settings = get_user_settings(user_id)
    settings["quiet"] = True

    await update.message.reply_text("🔇 已切换到安静模式\n只显示最终答案")


async def verbose_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /verbose 命令"""
    if not is_allowed(update.effective_user.id):
        return

    user_id = update.effective_user.id
    settings = get_user_settings(user_id)
    settings["quiet"] = False

    await update.message.reply_text("📢 已切换到详细模式\n将分阶段显示详细信息")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理 /status 命令"""
    if not is_allowed(update.effective_user.id):
        return

    user_id = update.effective_user.id
    settings = get_user_settings(user_id)

    mode = "安静模式 🔇" if settings["quiet"] else "详细模式 📢"
    status_text = f"""📊 当前设置

输出模式：{mode}

使用 /quiet 或 /verbose 切换模式"""

    await update.message.reply_text(status_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理用户发送的问题"""
    if not is_allowed(update.effective_user.id):
        return

    user_id = update.effective_user.id
    question = update.message.text.strip()

    if not question:
        await update.message.reply_text("请输入您的问题")
        return

    # 检查是否正在处理上一个请求
    if user_processing.get(user_id):
        await update.message.reply_text("⏳ 上一个问题还在处理中，请稍候...")
        return

    settings = get_user_settings(user_id)
    quiet = settings["quiet"]

    # 标记为处理中
    user_processing[user_id] = True

    try:
        if quiet:
            await run_council_quiet(update, question)
        else:
            await run_council_verbose(update, question)
    except Exception as e:
        logger.error(f"处理消息时出错: {e}")
        await update.message.reply_text(f"❌ 处理出错: {str(e)}")
    finally:
        user_processing[user_id] = False


async def run_council_quiet(update: Update, question: str) -> None:
    """安静模式：只显示最终答案"""
    processing_msg = await update.message.reply_text("🏛️ Council 正在讨论...\n⏳ 请稍候...")

    loop = asyncio.get_event_loop()

    # Stage 1
    results, _ = await loop.run_in_executor(
        None,
        lambda: stage1_first_opinions(question, verbose=False)
    )

    successful = [name for name in CLIS.keys() if name in results and results[name].success]
    failed = [(name, results[name].error) for name in CLIS.keys() if name in results and not results[name].success]

    # 记录日志
    logger.info(f"Stage 1 完成: 成功={successful}, 失败={failed}")

    if len(successful) < 2:
        # 显示详细的失败信息
        error_details = "\n".join([f"• {name}: {err[:100]}" for name, err in failed]) if failed else "未知错误"
        await processing_msg.edit_text(f"❌ 有效回答不足，无法继续\n\n失败详情:\n{error_details}")
        return

    # Stage 2
    label_to_model, reviews, _ = await loop.run_in_executor(
        None,
        lambda: stage2_review(question, results, verbose=False)
    )

    # Stage 3
    final_answer, _ = await loop.run_in_executor(
        None,
        lambda: stage3_final_response(question, results, label_to_model, reviews, verbose=False)
    )

    await processing_msg.delete()
    await update.message.reply_text(
        f"<b>🏛️ CLI Council 最终答案</b>\n\n{markdown_to_html(final_answer)}",
        parse_mode=ParseMode.HTML
    )


async def run_council_verbose(update: Update, question: str) -> None:
    """详细模式：分阶段输出"""
    loop = asyncio.get_event_loop()

    # 开始提示
    start_msg = await update.message.reply_text(
        f"🏛️ CLI Council\n\n📋 问题: {question}\n\n⏳ Stage 1: 正在查询各模型..."
    )

    # Stage 1
    results, stage1_output = await loop.run_in_executor(
        None,
        lambda: stage1_first_opinions(question, verbose=True, return_output=True)
    )

    # 删除开始提示消息
    await start_msg.delete()

    successful = [name for name in CLIS.keys() if name in results and results[name].success]
    failed = [name for name in CLIS.keys() if name in results and not results[name].success]

    # 构建 Stage 1 完成状态
    status_text = f"📊 Stage 1 完成: ✅ {', '.join(successful) if successful else '无'}"
    if failed:
        status_text += f"\n❌ {', '.join(failed)}"

    # 将状态拼接到 stage1 输出后面
    if stage1_output:
        combined_stage1 = stage1_output.rstrip() + "\n\n---\n\n" + status_text
        await update.message.reply_text(markdown_to_html(combined_stage1), parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(status_text)

    if len(successful) < 2:
        await update.message.reply_text("❌ 有效回答不足2个，无法继续")
        return

    # Stage 2
    stage2_msg = await update.message.reply_text("⏳ Stage 2: 正在进行匿名互评...")

    label_to_model, reviews, stage2_output = await loop.run_in_executor(
        None,
        lambda: stage2_review(question, results, verbose=True, return_output=True)
    )

    # 删除等待消息
    await stage2_msg.delete()

    # 发送 Stage 2 输出，在各 CLI 互评之间用 --- 分隔
    if stage2_output:
        # 在每个 reviewer 的评审之间添加分隔线
        stage2_formatted = stage2_output.replace("\n**Codex 的评审**", "\n---\n\n**Codex 的评审**")
        stage2_formatted = stage2_formatted.replace("\n**Gemini 的评审**", "\n---\n\n**Gemini 的评审**")
        stage2_formatted = stage2_formatted.replace("\n**Claude Code 的评审**", "\n---\n\n**Claude Code 的评审**")
        # 移除开头可能多余的 ---
        stage2_formatted = re.sub(r'^(\s*---\s*\n\s*)+', '', stage2_formatted)
        await update.message.reply_text(markdown_to_html(stage2_formatted), parse_mode=ParseMode.HTML)

    # Stage 3
    stage3_msg = await update.message.reply_text("⏳ Stage 3: Chairman 正在综合分析...")

    final_answer, stage3_output = await loop.run_in_executor(
        None,
        lambda: stage3_final_response(question, results, label_to_model, reviews, verbose=True, return_output=True)
    )

    # 删除等待消息
    await stage3_msg.delete()

    # 将完成状态拼接到 stage3 输出后面
    if stage3_output:
        combined_stage3 = stage3_output.rstrip() + "\n\n---\n\n✅ Council 流程完成"
        await update.message.reply_text(markdown_to_html(combined_stage3), parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("✅ Council 流程完成")


def main() -> None:
    """启动 bot"""
    if not BOT_TOKEN:
        print("错误: 未设置 TELEGRAM_BOT_TOKEN 环境变量")
        print("请在 .env 文件中设置 TELEGRAM_BOT_TOKEN=your_token_here")
        return

    if ALLOWED_USER_IDS:
        print(f"白名单用户: {ALLOWED_USER_IDS}")
    else:
        print("警告: 未配置白名单，所有用户都可以使用此 bot")

    # 配置更长的 HTTP 超时（因为 Council 处理可能需要几分钟）
    request = HTTPXRequest(
        read_timeout=600,      # 10 分钟读取超时
        write_timeout=600,     # 10 分钟写入超时
        connect_timeout=30,    # 30 秒连接超时
        pool_timeout=30,       # 30 秒连接池超时
    )

    # 创建 Application
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request)
        .get_updates_request(request)
        .build()
    )

    # 注册命令处理器
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("quiet", quiet_command))
    application.add_handler(CommandHandler("verbose", verbose_command))
    application.add_handler(CommandHandler("status", status_command))

    # 注册消息处理器
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    ))

    # 添加错误处理器
    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理错误"""
        logger.error(f"异常: {context.error}")
        # 网络错误会自动重试，不需要特殊处理

    application.add_error_handler(error_handler)

    # 启动 bot
    print("🤖 Bot 已启动，等待消息...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
