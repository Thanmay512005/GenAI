"""JSON-backed travel notice management."""
from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import uuid

DEFAULT_NOTICE_PATH=Path(__file__).parent/"data"/"notices.json"

@dataclass
class Notice:
    id:str; title:str; content:str; category:str="Travel Updates"; priority:str="Normal"
    target_audience:str="All Travelers"; posted_date:str=""; expiry_date:str|None=None
    is_pinned:bool=False; is_active:bool=True
    @property
    def searchable_chunk(self):
        return f"Notice: {self.title}\nDetails: {self.content}\nCategory: {self.category}"

def _dict(n): return {k:getattr(n,k) for k in Notice.__dataclass_fields__}
def load_notices(file_path=DEFAULT_NOTICE_PATH):
    p=Path(file_path)
    if not p.exists(): p.parent.mkdir(parents=True,exist_ok=True); p.write_text("[]",encoding="utf-8")
    values=json.loads(p.read_text(encoding="utf-8") or "[]")
    return [Notice(**{**v,"posted_date":v.get("posted_date") or date.today().isoformat()}) for v in values]
def save_notices(items,file_path=DEFAULT_NOTICE_PATH):
    p=Path(file_path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps([_dict(n) for n in items],indent=2,ensure_ascii=False),encoding="utf-8")
def add_notice(title,content,category="Travel Updates",priority="Normal",is_pinned=False,is_active=True,target_audience="All Travelers",expiry_date=None,file_path=DEFAULT_NOTICE_PATH):
    items=load_notices(file_path)
    n=Notice(uuid.uuid4().hex[:10],title.strip(),content.strip(),category,priority,target_audience,date.today().isoformat(),expiry_date,is_pinned,is_active)
    items.insert(0,n); save_notices(items,file_path); return n
def update_notice(notice_id, title=None, content=None, category=None, priority=None, is_pinned=None, is_active=None, expiry_date=None, file_path=DEFAULT_NOTICE_PATH):
    items=load_notices(file_path)
    for n in items:
        if n.id==notice_id:
            for k,v in dict(title=title,content=content,category=category,priority=priority,is_pinned=is_pinned,is_active=is_active,expiry_date=expiry_date).items():
                if v is not None: setattr(n,k,v)
            save_notices(items,file_path); return True
    return False
def delete_notice(notice_id,file_path=DEFAULT_NOTICE_PATH):
    items=load_notices(file_path); new=[n for n in items if n.id!=notice_id]
    if len(new)==len(items): return False
    save_notices(new,file_path); return True
def get_active_notices(file_path=DEFAULT_NOTICE_PATH):
    today=date.today().isoformat()
    return [n for n in load_notices(file_path) if n.is_active and (not n.expiry_date or n.expiry_date>=today)]
def get_pinned_notices(file_path=DEFAULT_NOTICE_PATH):
    return [n for n in get_active_notices(file_path) if n.is_pinned]
