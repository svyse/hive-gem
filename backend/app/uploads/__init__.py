"""Uploads (documents/images) ingestion.

Uploads are:
- saved to disk under settings.uploads_dir/<upload_id>/...
- extracted into text (best-effort)
- chunked and added into hive memory so both local and OpenAI LLMs can use them
- optionally converted into training examples for the background LoRA worker
"""
