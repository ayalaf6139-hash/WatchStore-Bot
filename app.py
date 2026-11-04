import os

import gradio as gr

import config
from bot import WatchStoreBot

bot = WatchStoreBot()


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        return " ".join(content_to_text(part) for part in content)
    return ""


def to_history(messages: list) -> list:
    history = []
    for message in messages:
        role = "משתמש" if message.get("role") == "user" else "בוט"
        text = content_to_text(message.get("content", "")).split("\n\n📄")[0]
        if text:
            history.append((role, text))
    return history


def respond(message: str, messages: list) -> str:
    if not message.strip():
        return "אנא כתבו שאלה."
    try:
        result = bot.answer(message, to_history(messages))
    except Exception as error:
        return f"אירעה שגיאה בפנייה לשירות ה-AI: {error}"

    if not result.sources or result.text.strip() == config.NO_ANSWER_TEXT:
        return result.text

    files = sorted({name for name, _ in result.sources})
    return result.text + "\n\n📄 מקורות: " + ", ".join(files)


demo = gr.ChatInterface(
    fn=respond,
    title="⌚ WatchStore – העוזר הווירטואלי",
    description="שאלו על הקטלוג, הזמנות, משלוחים, תשלום, אחריות ושימוש במערכת. הבוט עונה רק לפי מסמכי החברה.",
    examples=[
        "כמה עולה Omega Seamaster במחיר סיטונאי?",
        "מה קורה אם שתי חנויות מזמינות את אותו שעון?",
        "מה מדיניות ההחזרות?",
        "מתי מקבלים משלוח חינם?",
    ],
    textbox=gr.Textbox(placeholder="כתבו שאלה...", rtl=True),
    chatbot=gr.Chatbot(rtl=True, height=480),
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
