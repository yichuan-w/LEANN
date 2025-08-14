import json
import sqlite3
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Annotated

import requests
import typer
from tqdm import tqdm

app = typer.Typer()


def get_safe_path(s: str) -> str:
    """
    Remove invalid characters to sanitize a path.
    :param s: str to sanitize
    :returns: sanitized str
    """
    ban_chars = "\\  /  :  *  ?  \"  '  <  >  |  $  \r  \n".replace(" ", "")
    for i in ban_chars:
        s = s.replace(i, "")
    return s


def process_history(history: str):
    if history.startswith("<?xml") or history.startswith("<msg>"):
        try:
            root = ElementTree.fromstring(history)
            title = root.find(".//title").text if root.find(".//title") is not None else None
            quoted = (
                root.find(".//refermsg/content").text
                if root.find(".//refermsg/content") is not None
                else None
            )
            if title and quoted:
                return {"title": title, "quoted": process_history(quoted)}
            if title:
                return title
        except Exception:
            return history
    return history


def get_message(history: dict | str):
    if isinstance(history, dict):
        if "title" in history:
            return history["title"]
    else:
        return history


def export_chathistory(user_id: str):
    res = requests.get(
        "http://localhost:48065/wechat/chatlog",
        params={"userId": user_id, "count": 100000},
    ).json()
    for i in range(len(res["chatLogs"])):
        res["chatLogs"][i]["content"] = process_history(res["chatLogs"][i]["content"])
        res["chatLogs"][i]["message"] = get_message(res["chatLogs"][i]["content"])
    return res["chatLogs"]


@app.command()
def export_all(dest: Annotated[Path, typer.Argument(help="Destination path to export to.")]):
    """
    Export all users' chat history to json files.
    """
    if not dest.is_dir():
        if not dest.exists():
            inp = typer.prompt("Destination path does not exist, create it? (y/n)")
            if inp.lower() == "y":
                dest.mkdir(parents=True)
            else:
                typer.echo("Aborted.", err=True)
                return
        else:
            typer.echo("Destination path is not a directory!", err=True)
            return
    all_users = requests.get("http://localhost:48065/wechat/allcontacts").json()

    exported_count = 0
    for user in tqdm(all_users):
        try:
            usr_chatlog = export_chathistory(user["arg"])

            # Only write file if there are messages
            if len(usr_chatlog) > 0:
                out_path = dest / get_safe_path((user["title"] or "") + "-" + user["arg"] + ".json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(usr_chatlog, f, ensure_ascii=False, indent=2)
                exported_count += 1
        except Exception as e:
            print(f"Error exporting {user.get('title', 'Unknown')}: {e}")
            continue

    print(f"Exported {exported_count} users' chat history to {dest} in json.")


@app.command()
def export_sqlite(
    dest: Annotated[Path, typer.Argument(help="Destination path to export to.")] = Path(
        "chatlog.db"
    ),
):
    """
    Export all users' chat history to a sqlite database.
    """
    connection = sqlite3.connect(dest)
    cursor = connection.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS chatlog (id INTEGER PRIMARY KEY AUTOINCREMENT, with_id TEXT, from_user TEXT, to_user TEXT, message TEXT, timest DATETIME, auxiliary TEXT)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS chatlog_with_id_index ON chatlog (with_id)")
    cursor.execute("CREATE TABLE iF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT)")

    all_users = requests.get("http://localhost:48065/wechat/allcontacts").json()
    for user in tqdm(all_users):
        cursor.execute(
            "INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)",
            (user["arg"], user["title"]),
        )
        usr_chatlog = export_chathistory(user["arg"])
        for msg in usr_chatlog:
            cursor.execute(
                "INSERT INTO chatlog (with_id, from_user, to_user, message, timest, auxiliary) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    user["arg"],
                    msg["fromUser"],
                    msg["toUser"],
                    msg["message"],
                    msg["createTime"],
                    str(msg["content"]),
                ),
            )
    connection.commit()


def main():
    app()


if __name__ == "__main__":
    main()
