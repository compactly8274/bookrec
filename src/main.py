@app.get("/api/list/{action}")
async def list_by_action(action: str):
    """List books the user has interacted with, filtered by action type.

    Returns books from the feedback table, ordered by most recent first.
    Each book is decorated with cover_url and a "why" field.
    """
    valid_actions = {"like", "dislike", "skip", "toread"}
    if action not in valid_actions:
        return Response(status_code=400, content=json.dumps({"error": "invalid action"}), media_type="application/json")

    state = STATE
    books = []
    async with DB_LOCK:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT book_id FROM feedback WHERE action=? ORDER BY created_at DESC",
                (action,),
            )
            for row in cur.fetchall():
                bid = row[0]
                if bid in state.id_to_idx:
                    b = dict(state.books[state.id_to_idx[bid]])
                    b["why"] = f"{action.title()}d"
                    b["cover_url"] = cover_url(b)
                    books.append(b)
        finally:
            conn.close()

    return {"books": books, "done": len(books) == 0}