"""
backend/api/routers/chat.py
===========================
Router for Persistent Multimodal Semantic Retrieval Chat System
(ChatGPT / Gemini Style Session Management with User Isolation)
"""

import uuid
import json
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Body, status
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor, Json

from backend.ingestion.db_writer import get_pg_connection

# ============================================================
# Pydantic Schemas for Chat System
# ============================================================

class ChatConversationCreate(BaseModel):
    conversation_id: Optional[str] = None
    user_id: str
    title: str = "New Semantic Search"


class ChatConversationUpdate(BaseModel):
    title: str


class ChatMessageCreate(BaseModel):
    role: str = "user"  # "user" or "assistant"
    content: str
    attached_image_name: Optional[str] = None
    attached_image_preview: Optional[str] = None
    query_context: Optional[Dict[str, Any]] = None
    results: Optional[List[Dict[str, Any]]] = None


class ChatMessageSchema(BaseModel):
    message_id: int
    conversation_id: str
    role: str
    content: str
    attached_image_name: Optional[str] = None
    attached_image_preview: Optional[str] = None
    query_context: Optional[Dict[str, Any]] = None
    results: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ChatConversationSchema(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    message_count: Optional[int] = 0

    class Config:
        from_attributes = True


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    messages: List[ChatMessageSchema] = []

    class Config:
        from_attributes = True

log = logging.getLogger("ChatRouter")
router = APIRouter(prefix="/api/v1/chat", tags=["Chat System"])


def ensure_chat_schema_migrated(conn=None):
    """
    Ensures chat_conversations and chat_messages tables exist.
    Called automatically to guarantee zero-downtime persistence.
    """
    should_close = False
    if conn is None:
        try:
            conn = get_pg_connection()
            should_close = True
        except Exception as e:
            log.warning(f"Could not connect to DB for chat migration: {e}")
            return

    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_conversations (
              conversation_id   VARCHAR(64) PRIMARY KEY,
              user_id           VARCHAR(128) NOT NULL,
              title             VARCHAR(255) NOT NULL,
              created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
              updated_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_chat_conversations_user_updated
              ON chat_conversations(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS chat_messages (
              message_id             SERIAL PRIMARY KEY,
              conversation_id        VARCHAR(64) NOT NULL REFERENCES chat_conversations(conversation_id) ON DELETE CASCADE,
              role                   VARCHAR(20) NOT NULL,
              content                TEXT NOT NULL,
              attached_image_name    TEXT,
              attached_image_preview TEXT,
              query_context          JSONB,
              results                JSONB,
              created_at             TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_chat_messages_conv_created
              ON chat_messages(conversation_id, created_at ASC);
            """)
            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        log.error(f"Chat schema migration error: {e}", exc_info=True)
    finally:
        if should_close and conn:
            conn.close()


# Run once on module import
try:
    ensure_chat_schema_migrated()
except Exception:
    pass


@router.get("/conversations", response_model=List[ChatConversationSchema], summary="List User Conversations")
def list_conversations(
    user_id: str = Query(..., description="Unique user/analyst identity for conversation isolation")
):
    """
    Retrieves all past conversations belonging to the requesting user,
    ordered by most recently updated first.
    """
    ensure_chat_schema_migrated()
    conn = None
    try:
        conn = get_pg_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    c.conversation_id,
                    c.user_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.message_id) AS message_count
                FROM chat_conversations c
                LEFT JOIN chat_messages m ON c.conversation_id = m.conversation_id
                WHERE c.user_id = %s
                GROUP BY c.conversation_id, c.user_id, c.title, c.created_at, c.updated_at
                ORDER BY c.updated_at DESC;
            """, (user_id,))
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        log.error(f"Failed to list conversations for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.post("/conversations", response_model=ChatConversationSchema, status_code=status.HTTP_201_CREATED, summary="Create New Conversation")
def create_conversation(
    payload: ChatConversationCreate = Body(...)
):
    """
    Initializes a new persistent chat session thread for the specified user.
    """
    ensure_chat_schema_migrated()
    conn = None
    try:
        conversation_id = payload.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        title = payload.title.strip() or "New Tactical Search"

        conn = get_pg_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO chat_conversations (conversation_id, user_id, title, created_at, updated_at)
                VALUES (%s, %s, %s, NOW(), NOW())
                RETURNING conversation_id, user_id, title, created_at, updated_at;
            """, (conversation_id, payload.user_id, title))
            row = cur.fetchone()
            conn.commit()
            res = dict(row)
            res["message_count"] = 0
            return res
    except Exception as e:
        if conn:
            conn.rollback()
        log.error(f"Failed to create conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse, summary="Get Full Conversation State")
def get_conversation_details(
    conversation_id: str,
    user_id: str = Query(..., description="User identity for permission and privacy isolation")
):
    """
    Retrieves full conversation metadata and every message/result turn in chronological order.
    Enforces strict user isolation.
    """
    ensure_chat_schema_migrated()
    conn = None
    try:
        conn = get_pg_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Fetch conversation header
            cur.execute("""
                SELECT conversation_id, user_id, title, created_at, updated_at
                FROM chat_conversations
                WHERE conversation_id = %s;
            """, (conversation_id,))
            conv = cur.fetchone()
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")

            if conv["user_id"] != user_id:
                raise HTTPException(status_code=403, detail="Access denied: this conversation belongs to another user.")

            # 2. Fetch all messages in order
            cur.execute("""
                SELECT 
                    message_id,
                    conversation_id,
                    role,
                    content,
                    attached_image_name,
                    attached_image_preview,
                    query_context,
                    results,
                    created_at
                FROM chat_messages
                WHERE conversation_id = %s
                ORDER BY created_at ASC, message_id ASC;
            """, (conversation_id,))
            messages = cur.fetchall()

            res = dict(conv)
            res["messages"] = [dict(m) for m in messages]
            return res
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to fetch conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.patch("/conversations/{conversation_id}", response_model=ChatConversationSchema, summary="Rename Conversation")
def rename_conversation(
    conversation_id: str,
    user_id: str = Query(..., description="User identity"),
    payload: ChatConversationUpdate = Body(...)
):
    """
    Updates the title of an existing conversation thread.
    """
    conn = None
    try:
        conn = get_pg_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT user_id FROM chat_conversations WHERE conversation_id = %s;
            """, (conversation_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Conversation not found")
            if row["user_id"] != user_id:
                raise HTTPException(status_code=403, detail="Access denied")

            cur.execute("""
                UPDATE chat_conversations
                SET title = %s, updated_at = NOW()
                WHERE conversation_id = %s
                RETURNING conversation_id, user_id, title, created_at, updated_at;
            """, (payload.title.strip(), conversation_id))
            updated = cur.fetchone()
            conn.commit()
            return dict(updated)
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        log.error(f"Failed to update conversation title {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.delete("/conversations/{conversation_id}", summary="Delete Conversation")
def delete_conversation(
    conversation_id: str,
    user_id: str = Query(..., description="User identity")
):
    """
    Permanently deletes a conversation and all of its messages.
    """
    conn = None
    try:
        conn = get_pg_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM chat_conversations WHERE conversation_id = %s;", (conversation_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Conversation not found")
            if row[0] != user_id:
                raise HTTPException(status_code=403, detail="Access denied")

            cur.execute("DELETE FROM chat_conversations WHERE conversation_id = %s;", (conversation_id,))
            conn.commit()
            return {"status": "deleted", "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        log.error(f"Failed to delete conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.post("/conversations/{conversation_id}/messages", summary="Append Messages to Conversation")
def append_messages(
    conversation_id: str,
    user_id: str = Query(..., description="User identity"),
    messages: Union[ChatMessageCreate, List[ChatMessageCreate]] = Body(...)
):
    """
    Appends one or more messages (e.g. user prompt and assistant results) to the conversation.
    Updates the conversation's updated_at timestamp.
    """
    ensure_chat_schema_migrated()
    conn = None
    try:
        msg_list = messages if isinstance(messages, list) else [messages]
        conn = get_pg_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Verify existence & ownership
            cur.execute("""
                SELECT user_id, title, (SELECT COUNT(*) FROM chat_messages WHERE conversation_id = %s) AS msg_count
                FROM chat_conversations
                WHERE conversation_id = %s;
            """, (conversation_id, conversation_id))
            conv = cur.fetchone()
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")
            if conv["user_id"] != user_id:
                raise HTTPException(status_code=403, detail="Access denied")

            inserted_messages = []
            new_title = None

            for msg in msg_list:
                # If first user message and title is default or short, auto-title it
                if msg.role == "user" and (conv["msg_count"] == 0 or conv["title"] in ("New Semantic Search", "New Tactical Search", "Untitled Search")):
                    candidate_title = msg.content.strip()
                    if not candidate_title and msg.attached_image_name:
                        candidate_title = f"Image: {msg.attached_image_name}"
                    if candidate_title:
                        new_title = candidate_title[:45] + ("..." if len(candidate_title) > 45 else "")

                cur.execute("""
                    INSERT INTO chat_messages (
                        conversation_id,
                        role,
                        content,
                        attached_image_name,
                        attached_image_preview,
                        query_context,
                        results,
                        created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING message_id, conversation_id, role, content, attached_image_name, attached_image_preview, query_context, results, created_at;
                """, (
                    conversation_id,
                    msg.role,
                    msg.content,
                    msg.attached_image_name,
                    msg.attached_image_preview,
                    Json(msg.query_context) if msg.query_context is not None else None,
                    Json(msg.results) if msg.results is not None else None
                ))
                inserted = cur.fetchone()
                inserted_messages.append(dict(inserted))

            # Update conversation timestamp & title if modified
            if new_title:
                cur.execute("""
                    UPDATE chat_conversations
                    SET updated_at = NOW(), title = %s
                    WHERE conversation_id = %s;
                """, (new_title, conversation_id))
            else:
                cur.execute("""
                    UPDATE chat_conversations
                    SET updated_at = NOW()
                    WHERE conversation_id = %s;
                """, (conversation_id,))

            conn.commit()
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "title": new_title or conv["title"],
                "inserted_count": len(inserted_messages),
                "messages": inserted_messages
            }
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        log.error(f"Failed to append messages to conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()
