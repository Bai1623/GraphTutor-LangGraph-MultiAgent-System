"""Regression tests for the frontend multi-turn conversation contract."""

from pathlib import Path


PAGE = (
    Path(__file__).resolve().parent.parent / "frontend" / "app" / "page.tsx"
).read_text(encoding="utf-8")


def test_stream_request_reuses_existing_thread_id():
    assert "thread_id: threadIdRef.current" in PAGE


def test_sending_message_does_not_reset_thread_id():
    send_handler = PAGE.split(
        "const handleSendMessage = useCallback", maxsplit=1
    )[1].split("const handleResume = useCallback", maxsplit=1)[0]

    assert "threadIdRef.current = null" not in send_handler


def test_conversation_history_persists_messages_and_thread_id():
    assert 'const CHAT_STORAGE_KEY = "gaokao_tutor_conversations"' in PAGE
    assert "threadId: string | null" in PAGE
    assert "messages: Message[]" in PAGE
    assert "localStorage.setItem(CHAT_STORAGE_KEY" in PAGE


def test_selecting_conversation_restores_messages_and_thread_id():
    select_handler = PAGE.split(
        "const handleSelectChat = useCallback", maxsplit=1
    )[1].split("/** Process a single SSE", maxsplit=1)[0]

    assert "setMessages(chat.messages)" in select_handler
    assert "threadIdRef.current = chat.threadId" in select_handler
